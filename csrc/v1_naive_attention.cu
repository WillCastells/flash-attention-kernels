#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================================
// V1: Naive 3-pass attention (baseline)
//
// Three separate kernel launches:
//   1. QK^T  — compute full N×N attention scores in global memory
//   2. Softmax — row-wise softmax with causal masking
//   3. PV   — multiply attention weights by V
//
// This is intentionally slow: allocates full N×N matrix, no shared memory,
// no tiling. It exists purely as a correctness reference and performance
// baseline for the Flash Attention versions.
// ============================================================================

// Kernel 1: S = Q @ K^T * scale, with causal mask
// Grid: (B*nh) blocks, Block: N threads (1 thread per query row)
__global__ void naive_qk_kernel(
    const float* __restrict__ Q,   // [B, nh, N, d]
    const float* __restrict__ K,   // [B, nh, N, d]
    float* __restrict__ S,         // [B, nh, N, N]
    int N, int d, float scale
) {
    int batch_head = blockIdx.x;   // which (batch, head) pair
    int row = threadIdx.x;         // which query row

    if (row >= N) return;

    const float* q_row = Q + batch_head * N * d + row * d;
    const float* k_base = K + batch_head * N * d;
    float* s_row = S + batch_head * N * N + row * N;

    for (int col = 0; col < N; col++) {
        if (col > row) {
            // Causal mask: future positions get -inf
            s_row[col] = -INFINITY;
        } else {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += q_row[i] * k_base[col * d + i];
            }
            s_row[col] = dot * scale;
        }
    }
}

// Kernel 2: Row-wise softmax over S (in-place)
// Grid: (B*nh) blocks, Block: N threads
__global__ void naive_softmax_kernel(
    float* __restrict__ S,   // [B, nh, N, N] — modified in-place
    int N
) {
    int batch_head = blockIdx.x;
    int row = threadIdx.x;

    if (row >= N) return;

    float* s_row = S + batch_head * N * N + row * N;

    // Find row max for numerical stability
    float row_max = -INFINITY;
    for (int j = 0; j < N; j++) {
        row_max = fmaxf(row_max, s_row[j]);
    }

    // Compute exp and sum
    float row_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        s_row[j] = expf(s_row[j] - row_max);
        row_sum += s_row[j];
    }

    // Normalize
    for (int j = 0; j < N; j++) {
        s_row[j] /= row_sum;
    }
}

// Kernel 3: O = P @ V
// Grid: (B*nh) blocks, Block: N threads
__global__ void naive_pv_kernel(
    const float* __restrict__ P,   // [B, nh, N, N] (softmax output)
    const float* __restrict__ V,   // [B, nh, N, d]
    float* __restrict__ O,         // [B, nh, N, d]
    int N, int d
) {
    int batch_head = blockIdx.x;
    int row = threadIdx.x;

    if (row >= N) return;

    const float* p_row = P + batch_head * N * N + row * N;
    const float* v_base = V + batch_head * N * d;
    float* o_row = O + batch_head * N * d + row * d;

    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += p_row[k] * v_base[k * d + j];
        }
        o_row[j] = sum;
    }
}

// Host wrapper: takes torch tensors, launches kernels
torch::Tensor naive_attention(
    torch::Tensor Q,   // [B, nh, N, d]
    torch::Tensor K,   // [B, nh, N, d]
    torch::Tensor V    // [B, nh, N, d]
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    TORCH_CHECK(N <= 1024, "N must be <= 1024 for naive kernel (thread limit)");

    float scale = 1.0f / static_cast<float>(d);

    // Allocate full N×N attention matrix (this is what Flash Attention avoids)
    auto S = torch::empty({B, nh, N, N}, Q.options());
    auto O = torch::empty_like(Q);

    int grid = B * nh;
    int block = N;

    naive_qk_kernel<<<grid, block>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        S.data_ptr<float>(), N, d, scale
    );

    naive_softmax_kernel<<<grid, block>>>(
        S.data_ptr<float>(), N
    );

    naive_pv_kernel<<<grid, block>>>(
        S.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), N, d
    );

    return O;
}
