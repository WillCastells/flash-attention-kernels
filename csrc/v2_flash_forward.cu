#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================================
// V2: Flash Attention 2 Forward Pass
//
// Single kernel launch with online softmax and tiling.
// - Outer loop over query blocks (parallelized via grid)
// - Inner loop over KV blocks (sequential, with causal early-exit)
// - Shared memory for Q, K, V tiles
// - Online softmax: running max and sum in registers, rescale O each iteration
// - Outputs O (attention output) and L (logsumexp for backward pass)
// ============================================================================

constexpr int BLOCK_SIZE = 32;  // Br = Bc = 32

__global__ void flash_forward_kernel(
    const float* __restrict__ Q,   // [B, nh, N, d]
    const float* __restrict__ K,   // [B, nh, N, d]
    const float* __restrict__ V,   // [B, nh, N, d]
    float* __restrict__ O,         // [B, nh, N, d]
    float* __restrict__ L,         // [B, nh, N]  (logsumexp per row)
    int N, int d, float scale
) {
    int batch_head = blockIdx.x * gridDim.y + blockIdx.y;  // B*nh index
    int block_row  = blockIdx.z;                            // which query block
    int tx = threadIdx.x;                                   // thread within block [0, Br)

    int qr = block_row * BLOCK_SIZE + tx;  // global query row for this thread
    if (qr >= N) return;

    // Pointers for this (batch, head)
    const float* Q_bh = Q + batch_head * N * d;
    const float* K_bh = K + batch_head * N * d;
    const float* V_bh = V + batch_head * N * d;
    float* O_bh = O + batch_head * N * d;
    float* L_bh = L + batch_head * N;

    // Shared memory layout: Qi[Br][d] + Kj[Bc][d] + Vj[Bc][d] + S[Br][Bc]
    extern __shared__ float smem[];
    float* Qi = smem;                                     // [Br][d]
    float* Kj = Qi + BLOCK_SIZE * d;                      // [Bc][d]
    float* Vj = Kj + BLOCK_SIZE * d;                      // [Bc][d]
    float* S  = Vj + BLOCK_SIZE * d;                      // [Br][Bc]

    // Load Qi into shared memory (each thread loads its own row)
    for (int i = 0; i < d; i++) {
        Qi[tx * d + i] = Q_bh[qr * d + i];
    }

    // Running statistics for online softmax (in registers)
    float mi = 0.0f;  // running max
    float li = 0.0f;       // running sum of exp

    // Output accumulator (in registers)
    float oi[128];  // max d=128
    for (int i = 0; i < d; i++) {
        oi[i] = 0.0f;
    }

    // Number of KV blocks
    int Tc = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Causal: only iterate over KV blocks where j <= qr
    int max_kv_block = (qr / BLOCK_SIZE) + 1;  // last block that might have valid keys

    for (int j = 0; j < max_kv_block && j < Tc; j++) {
        int kv_start = j * BLOCK_SIZE;

        // Load Kj and Vj into shared memory (each thread loads one row)
        int kv_row = kv_start + tx;
        if (kv_row < N) {
            for (int i = 0; i < d; i++) {
                Kj[tx * d + i] = K_bh[kv_row * d + i];
                Vj[tx * d + i] = V_bh[kv_row * d + i];
            }
        } else {
            for (int i = 0; i < d; i++) {
                Kj[tx * d + i] = 0.0f;
                Vj[tx * d + i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute S[tx][c] = Qi[tx] @ Kj[c]^T * scale, with causal mask
        for (int c = 0; c < BLOCK_SIZE; c++) {
            int key_pos = kv_start + c;
            if (key_pos > qr || key_pos >= N) {
                S[tx * BLOCK_SIZE + c] = -INFINITY;
            } else {
                float dot = 0.0f;
                for (int i = 0; i < d; i++) {
                    dot += Qi[tx * d + i] * Kj[c * d + i];
                }
                S[tx * BLOCK_SIZE + c] = dot * scale;
            }
        }

        // Online softmax update
        // Find new max for this block
        float block_max = -INFINITY;
        for (int c = 0; c < BLOCK_SIZE; c++) {
            block_max = fmaxf(block_max, S[tx * BLOCK_SIZE + c]);
        }

        float new_mi = fmaxf(mi, block_max);

        // Rescale previous accumulator
        float rescale = expf(mi - new_mi);
        li *= rescale;
        for (int i = 0; i < d; i++) {
            oi[i] *= rescale;
        }

        // Compute exp(S - new_mi) and accumulate
        float block_sum = 0.0f;
        for (int c = 0; c < BLOCK_SIZE; c++) {
            float p = expf(S[tx * BLOCK_SIZE + c] - new_mi);
            block_sum += p;

            // Accumulate P @ V
            for (int i = 0; i < d; i++) {
                oi[i] += p * Vj[c * d + i];
            }
        }

        mi = new_mi;
        li += block_sum;

        __syncthreads();
    }

    // Final normalization: O = oi / li
    for (int i = 0; i < d; i++) {
        O_bh[qr * d + i] = oi[i] / li;
    }

    // Store logsumexp: L = mi + log(li)
    L_bh[qr] = mi + logf(li);
}

std::vector<torch::Tensor> flash_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    TORCH_CHECK(d <= 128, "head_dim must be <= 128");

    float scale = 1.0f / sqrtf(static_cast<float>(d));

    auto O = torch::empty_like(Q);
    auto L = torch::empty({B, nh, N}, Q.options());

    int Tr = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // number of query blocks

    dim3 grid(B, nh, Tr);
    dim3 block(BLOCK_SIZE);

    // Shared memory: Qi[Br*d] + Kj[Bc*d] + Vj[Bc*d] + S[Br*Bc]
    int smem_size = (3 * BLOCK_SIZE * d + BLOCK_SIZE * BLOCK_SIZE) * sizeof(float);

    flash_forward_kernel<<<grid, block, smem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        N, d, scale
    );

    return {O, L};
}
