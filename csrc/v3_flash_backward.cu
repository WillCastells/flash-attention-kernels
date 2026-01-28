#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================================
// V3: Flash Attention 2 Backward Pass
//
// THE DIFFERENTIATOR — most open-source FA implementations stop at forward.
//
// Algorithm (from FlashAttention-2 paper, adapted):
// - Outer loop over KV blocks (j) — each block accumulates dKj, dVj
// - Inner loop over query blocks (i) — recomputes S, P from Q, K, stored L
// - No N×N matrices stored: S and P are recomputed on the fly
// - dV += P^T @ dO     (accumulated in shared memory per j-block)
// - dK += scale * dS^T @ Q  (accumulated in shared memory per j-block)
// - dQ += scale * dS @ K    (global atomicAdd — known bottleneck, optimized in v4)
//
// Where dS = P * (dO @ V^T - D), D[i] = rowsum(dO[i] * O[i])
// ============================================================================

constexpr int BWD_BLOCK = 16;  // Br = Bc = 16 (smaller tiles, more shared mem)

// Pre-compute D[i] = sum_j(dO[i][j] * O[i][j]) for each row
__global__ void compute_D_kernel(
    const float* __restrict__ O,    // [B, nh, N, d]
    const float* __restrict__ dO,   // [B, nh, N, d]
    float* __restrict__ D,          // [B, nh, N]
    int N, int d
) {
    int batch_head = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= N) return;

    const float* o_row = O + batch_head * N * d + row * d;
    const float* do_row = dO + batch_head * N * d + row * d;

    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        sum += o_row[i] * do_row[i];
    }

    D[batch_head * N + row] = sum;
}

// Main backward kernel
// Grid: dim3(B*nh) — one block per (batch, head)
// Block: dim3(BWD_BLOCK) — threads handle rows within tiles
__global__ void flash_backward_kernel(
    const float* __restrict__ Q,    // [B, nh, N, d]
    const float* __restrict__ K,    // [B, nh, N, d]
    const float* __restrict__ V,    // [B, nh, N, d]
    const float* __restrict__ O,    // [B, nh, N, d]
    const float* __restrict__ dO,   // [B, nh, N, d]
    const float* __restrict__ L,    // [B, nh, N]
    const float* __restrict__ D,    // [B, nh, N]
    float* __restrict__ dQ,         // [B, nh, N, d]
    float* __restrict__ dK,         // [B, nh, N, d]
    float* __restrict__ dV,         // [B, nh, N, d]
    int N, int d, float scale
) {
    int batch_head = blockIdx.x;
    int tx = threadIdx.x;   // [0, BWD_BLOCK)

    // Pointers for this (batch, head)
    const float* Q_bh  = Q  + batch_head * N * d;
    const float* K_bh  = K  + batch_head * N * d;
    const float* V_bh  = V  + batch_head * N * d;
    const float* dO_bh = dO + batch_head * N * d;
    const float* L_bh  = L  + batch_head * N;
    const float* D_bh  = D  + batch_head * N;
    float* dQ_bh = dQ + batch_head * N * d;
    float* dK_bh = dK + batch_head * N * d;
    float* dV_bh = dV + batch_head * N * d;

    int Tc = (N + BWD_BLOCK - 1) / BWD_BLOCK;
    int Tr = (N + BWD_BLOCK - 1) / BWD_BLOCK;

    // Shared memory layout:
    // Kj[Bc][d], Vj[Bc][d], dKj[Bc][d], dVj[Bc][d],
    // Qi[Br][d], dOi[Br][d], S[Br][Bc]
    extern __shared__ float smem[];
    float* Kj_s  = smem;                                          // [Bc][d]
    float* Vj_s  = Kj_s  + BWD_BLOCK * d;                         // [Bc][d]
    float* dKj_s = Vj_s  + BWD_BLOCK * d;                         // [Bc][d]
    float* dVj_s = dKj_s + BWD_BLOCK * d;                         // [Bc][d]
    float* Qi_s  = dVj_s + BWD_BLOCK * d;                         // [Br][d]
    float* dOi_s = Qi_s  + BWD_BLOCK * d;                         // [Br][d]
    float* S_s   = dOi_s + BWD_BLOCK * d;                         // [Br][Bc]

    // Outer loop over KV blocks
    for (int j = 0; j < Tc; j++) {
        int kv_start = j * BWD_BLOCK;
        int kv_row = kv_start + tx;

        // Load Kj, Vj into shared memory
        if (kv_row < N) {
            for (int i = 0; i < d; i++) {
                Kj_s[tx * d + i] = K_bh[kv_row * d + i];
                Vj_s[tx * d + i] = V_bh[kv_row * d + i];
            }
        } else {
            for (int i = 0; i < d; i++) {
                Kj_s[tx * d + i] = 0.0f;
                Vj_s[tx * d + i] = 0.0f;
            }
        }

        // Zero dKj, dVj accumulators in shared memory
        for (int i = 0; i < d; i++) {
            dKj_s[tx * d + i] = 0.0f;
            dVj_s[tx * d + i] = 0.0f;
        }
        __syncthreads();

        // Inner loop over query blocks
        // Causal: only query blocks where some qi >= kv_start (i.e., i >= j)
        for (int i = j; i < Tr; i++) {
            int q_start = i * BWD_BLOCK;
            int q_row = q_start + tx;

            // Load Qi, dOi into shared memory
            if (q_row < N) {
                for (int dd = 0; dd < d; dd++) {
                    Qi_s[tx * d + dd] = Q_bh[q_row * d + dd];
                    dOi_s[tx * d + dd] = dO_bh[q_row * d + dd];
                }
            } else {
                for (int dd = 0; dd < d; dd++) {
                    Qi_s[tx * d + dd] = 0.0f;
                    dOi_s[tx * d + dd] = 0.0f;
                }
            }
            __syncthreads();

            // Compute S = Qi @ Kj^T * scale with causal mask
            // Then P = exp(S - L)
            // Each thread tx computes one row of S (row tx in the Br tile)
            float Li = (q_row < N) ? L_bh[q_row] : 0.0f;
            float Di = (q_row < N) ? D_bh[q_row] : 0.0f;

            for (int c = 0; c < BWD_BLOCK; c++) {
                int key_pos = kv_start + c;
                if (key_pos > q_row || key_pos >= N || q_row >= N) {
                    S_s[tx * BWD_BLOCK + c] = 0.0f;  // P = 0 for masked positions
                } else {
                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        dot += Qi_s[tx * d + dd] * Kj_s[c * d + dd];
                    }
                    S_s[tx * BWD_BLOCK + c] = expf(dot * scale - Li);  // This is P
                }
            }
            __syncthreads();

            // Accumulate dVj += P^T @ dOi
            // Each thread tx handles row tx of dVj (i.e., column tx of P^T)
            if (kv_row < N) {
                for (int dd = 0; dd < d; dd++) {
                    float sum = 0.0f;
                    for (int r = 0; r < BWD_BLOCK; r++) {
                        sum += S_s[r * BWD_BLOCK + tx] * dOi_s[r * d + dd];
                    }
                    dVj_s[tx * d + dd] += sum;
                }
            }

            // Compute dS = P * (dO @ V^T - D)
            // dS[r][c] = P[r][c] * (sum_dd(dOi[r][dd] * Vj[c][dd]) - D[qr])
            // Then accumulate:
            //   dKj[c] += scale * sum_r(dS[r][c] * Qi[r])  -- each thread handles row tx of dKj
            //   dQi[r] += scale * sum_c(dS[r][c] * Kj[c])  -- each thread handles row tx of dQi
            __syncthreads();

            // Each thread computes its row of dS and accumulates dQ
            if (q_row < N) {
                float dq_local[128];  // max d=128
                for (int dd = 0; dd < d; dd++) {
                    dq_local[dd] = 0.0f;
                }

                for (int c = 0; c < BWD_BLOCK; c++) {
                    float p = S_s[tx * BWD_BLOCK + c];
                    if (p == 0.0f) continue;

                    // dO[r] @ V[c]^T
                    float doV = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        doV += dOi_s[tx * d + dd] * Vj_s[c * d + dd];
                    }

                    float ds = p * (doV - Di);

                    // dQ[r] += scale * ds * K[c]
                    for (int dd = 0; dd < d; dd++) {
                        dq_local[dd] += ds * Kj_s[c * d + dd];
                    }
                }

                // Write dQ with atomicAdd (accumulated across j-blocks)
                for (int dd = 0; dd < d; dd++) {
                    atomicAdd(&dQ_bh[q_row * d + dd], scale * dq_local[dd]);
                }
            }

            // Accumulate dKj: each thread tx handles its row of dKj
            // dKj[tx] += scale * sum_r(dS[r][tx] * Qi[r])
            __syncthreads();

            if (kv_row < N) {
                for (int r = 0; r < BWD_BLOCK; r++) {
                    int qr_inner = q_start + r;
                    float p = S_s[r * BWD_BLOCK + tx];
                    if (p == 0.0f || qr_inner >= N) continue;

                    float Dr = D_bh[qr_inner];

                    // dO[r] @ V[tx]^T
                    float doV = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        doV += dOi_s[r * d + dd] * Vj_s[tx * d + dd];
                    }

                    float ds = p * (doV - Dr);

                    for (int dd = 0; dd < d; dd++) {
                        dKj_s[tx * d + dd] += scale * ds * Qi_s[r * d + dd];
                    }
                }
            }
            __syncthreads();
        }

        // Write dKj, dVj back to global memory
        if (kv_row < N) {
            for (int dd = 0; dd < d; dd++) {
                dK_bh[kv_row * d + dd] = dKj_s[tx * d + dd];
                dV_bh[kv_row * d + dd] = dVj_s[tx * d + dd];
            }
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> flash_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    TORCH_CHECK(d <= 128, "head_dim must be <= 128");

    float scale = 1.0f / sqrtf(static_cast<float>(d));

    auto dQ = torch::empty_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    // Step 1: Compute D[i] = sum_j(O[i][j] * dO[i][j])
    auto D = torch::empty({B, nh, N}, Q.options());
    {
        int threads = 256;
        int blocks_y = (N + threads - 1) / threads;
        dim3 grid(B * nh, blocks_y);
        compute_D_kernel<<<grid, threads>>>(
            O.data_ptr<float>(), dO.data_ptr<float>(),
            D.data_ptr<float>(), N, d
        );
    }

    // Step 2: Main backward kernel
    {
        dim3 grid(B * nh);
        dim3 block(BWD_BLOCK);

        // Shared memory: Kj + Vj + dKj + dVj + Qi + dOi + S
        // = (4*Bc*d + 2*Br*d + Br*Bc) * sizeof(float)
        int smem_size = (6 * BWD_BLOCK * d + BWD_BLOCK * BWD_BLOCK) * sizeof(float);

        flash_backward_kernel<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            O.data_ptr<float>(), dO.data_ptr<float>(),
            L.data_ptr<float>(), D.data_ptr<float>(),
            dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
            N, d, scale
        );
    }

    return {dQ, dK, dV};
}
