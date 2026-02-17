#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// V5: Flash Attention 2 — fp16 + WMMA Tensor Cores
//
// Key features:
// 1. fp16 inputs/outputs, fp32 accumulators (numerical stability)
// 2. WMMA 16x16x16 tensor core matmuls for S=Q@K^T and O+=P@V (forward)
// 3. Split backward: separate dK/dV and dQ kernels (no atomics)
// 4. Block sizes Br=Bc=32 with 16x16 WMMA tiles (2x2 tiles per block)
// 5. Backward kernels use thread-level matmul with fp16 loads
// 6. Supports SM 7.0+ (Volta, Turing, Ampere, Hopper)
// ============================================================================

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// Forward Kernel — WMMA Tensor Core matmuls
// ============================================================================
//
// Block: Br=Bc=32, 4 warps (128 threads)
// Warp layout: 2x2 grid of 16x16 tiles
//   - For S=Q@K^T: each warp computes one 16x16 tile of S[Br][Bc]
//   - For O+=P@V:  each warp computes a portion of O[Br][D]
//     (tile_row selects 16 rows, tile_col selects D/2 columns)
//
// O is accumulated in shared memory (fp32) across KV blocks.
// Online softmax rescale factors are stored in rescale_smem[Br].
// Each KV block: rescale O_smem, then accumulate PV via WMMA.

template <int Br, int Bc, int D>
__global__ void flash_forward_fp16tc_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O_out,
    float* __restrict__ L_out,
    int N, float scale
) {
    const int batch_head = blockIdx.x * gridDim.y + blockIdx.y;
    const int block_row  = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    constexpr int TILES_ROW = Br / WMMA_M;         // 2
    constexpr int TILES_COL = Bc / WMMA_N;          // 2
    constexpr int NUM_WARPS = TILES_ROW * TILES_COL; // 4
    constexpr int NUM_THREADS = NUM_WARPS * 32;      // 128
    constexpr int D_TILES = D / WMMA_N;
    constexpr int D_TILES_PER_WARP = D_TILES / TILES_COL;

    const int tile_row = warp_id / TILES_COL;  // 0 or 1
    const int tile_col = warp_id % TILES_COL;  // 0 or 1
    const int my_d_start = tile_col * D_TILES_PER_WARP;

    const half* Q_bh = Q + batch_head * N * D;
    const half* K_bh = K + batch_head * N * D;
    const half* V_bh = V + batch_head * N * D;
    half* O_bh = O_out + batch_head * N * D;
    float* L_bh = L_out + batch_head * N;

    // Shared memory layout (all properly aligned):
    //   Q_smem[Br][D]       half     — loaded once
    //   K_smem[Bc][D]       half     — per KV block
    //   V_smem[Bc][D]       half     — per KV block
    //   S_smem[Br][Bc]      float    — reused for P (fp32)
    //   P_smem[Br][Bc]      half     — P in fp16 for WMMA
    //   mi_smem[Br]         float    — running row max
    //   li_smem[Br]         float    — running row sum
    //   rescale_smem[Br]    float    — exp(old_mi - new_mi)
    //   O_smem[Br][D]       float    — O accumulator

    extern __shared__ float smem_f[];
    half*  Q_smem        = reinterpret_cast<half*>(smem_f);
    half*  K_smem        = Q_smem + Br * D;
    half*  V_smem        = K_smem + Bc * D;
    float* S_smem        = reinterpret_cast<float*>(V_smem + Bc * D);
    half*  P_smem        = reinterpret_cast<half*>(S_smem + Br * Bc);
    float* mi_smem       = reinterpret_cast<float*>(P_smem + Br * Bc);
    float* li_smem       = mi_smem + Br;
    float* rescale_smem  = li_smem + Br;
    float* O_smem        = rescale_smem + Br;

    // Initialize mi, li
    if (tid < Br) {
        mi_smem[tid] = -INFINITY;
        li_smem[tid] = 0.0f;
    }

    // Zero O_smem
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS)
        O_smem[idx] = 0.0f;

    // Load Q into shared memory (persistent across KV blocks)
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        int r = idx / D, d_idx = idx % D;
        int gr = block_row * Br + r;
        Q_smem[idx] = (gr < N) ? Q_bh[gr * D + d_idx] : __float2half(0.0f);
    }
    __syncthreads();

    // KV block loop (causal: only up to the last valid KV block)
    int Tc = (N + Bc - 1) / Bc;
    int max_qr = min(block_row * Br + Br - 1, N - 1);
    int max_kv_block = (max_qr >= 0) ? min(max_qr / Bc + 1, Tc) : 0;

    for (int j = 0; j < max_kv_block; j++) {
        int kv_start = j * Bc;

        // Load K, V into shared memory
        for (int idx = tid; idx < Bc * D; idx += NUM_THREADS) {
            int r = idx / D, d_idx = idx % D;
            int gr = kv_start + r;
            K_smem[idx] = (gr < N) ? K_bh[gr * D + d_idx] : __float2half(0.0f);
            V_smem[idx] = (gr < N) ? V_bh[gr * D + d_idx] : __float2half(0.0f);
        }
        __syncthreads();

        // ---- S = Q @ K^T via WMMA ----
        // Each warp computes one 16x16 tile of S[Br][Bc]
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);

        #pragma unroll
        for (int k = 0; k < D; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            // Q[tile_row*16 : +16, k : +16], row-major stride D
            wmma::load_matrix_sync(a_frag, Q_smem + tile_row * WMMA_M * D + k, D);

            // K^T = K (col_major load). K[tile_col*16 : +16, k : +16], stride D
            wmma::load_matrix_sync(b_frag, K_smem + tile_col * WMMA_N * D + k, D);

            wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
        }

        // Apply scale
        #pragma unroll
        for (int i = 0; i < s_frag.num_elements; i++)
            s_frag.x[i] *= scale;

        // Store S tile to shared memory
        wmma::store_matrix_sync(
            S_smem + tile_row * WMMA_M * Bc + tile_col * WMMA_N,
            s_frag, Bc, wmma::mem_row_major);
        __syncthreads();

        // ---- Causal mask ----
        for (int idx = tid; idx < Br * Bc; idx += NUM_THREADS) {
            int r = idx / Bc, c = idx % Bc;
            int qr = block_row * Br + r;
            int kp = kv_start + c;
            if (kp > qr || kp >= N || qr >= N)
                S_smem[idx] = -INFINITY;
        }
        __syncthreads();

        // ---- Online softmax + store rescale factor ----
        // Each thread handles at most one row (128 threads, 32 rows)
        for (int r = tid; r < Br; r += NUM_THREADS) {
            int qr = block_row * Br + r;
            if (qr >= N) {
                rescale_smem[r] = 1.0f;
                continue;
            }

            float old_mi = mi_smem[r];

            // Row max
            float row_max = -INFINITY;
            for (int c = 0; c < Bc; c++)
                row_max = fmaxf(row_max, S_smem[r * Bc + c]);

            float new_mi = fmaxf(old_mi, row_max);
            float rescale = expf(new_mi - old_mi);

            // Compute P values and row sum
            float row_sum = 0.0f;
            for (int c = 0; c < Bc; c++) {
                float p = expf(S_smem[r * Bc + c] - new_mi);
                S_smem[r * Bc + c] = p;  // Store P (fp32) in place
                row_sum += p;
            }

            mi_smem[r] = new_mi;
            li_smem[r] = li_smem[r] * rescale + row_sum;
            rescale_smem[r] = rescale;  // Save for O_smem rescaling
        }
        __syncthreads();

        // ---- Rescale O_smem + convert P to fp16 ----
        for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
            int r = idx / D;
            O_smem[idx] *= rescale_smem[r];
        }
        for (int idx = tid; idx < Br * Bc; idx += NUM_THREADS)
            P_smem[idx] = __float2half(S_smem[idx]);
        __syncthreads();

        // ---- O += P @ V via WMMA ----
        // Each warp handles: rows [tile_row*16 : +16], D-cols [my_d_start*16 : +D_TILES_PER_WARP*16]
        constexpr int K_TILES = Bc / WMMA_K;  // 2

        #pragma unroll
        for (int dw = 0; dw < D_TILES_PER_WARP; dw++) {
            int dt = my_d_start + dw;

            // Load current O tile from O_smem into accumulator fragment
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag,
                O_smem + tile_row * WMMA_M * D + dt * WMMA_N,
                D, wmma::mem_row_major);

            // Accumulate P @ V
            #pragma unroll
            for (int kt = 0; kt < K_TILES; kt++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;

                // P[tile_row*16 : +16, kt*16 : +16], row-major stride Bc
                wmma::load_matrix_sync(p_frag,
                    P_smem + tile_row * WMMA_M * Bc + kt * WMMA_K, Bc);

                // V[kt*16 : +16, dt*16 : +16], row-major stride D
                wmma::load_matrix_sync(v_frag,
                    V_smem + kt * WMMA_K * D + dt * WMMA_N, D);

                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            // Store updated O tile back to O_smem
            wmma::store_matrix_sync(
                O_smem + tile_row * WMMA_M * D + dt * WMMA_N,
                o_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // ---- Normalize and write to global memory ----
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        int r = idx / D, d_idx = idx % D;
        int qr = block_row * Br + r;
        if (qr < N) {
            float inv_li = 1.0f / li_smem[r];
            O_bh[qr * D + d_idx] = __float2half(O_smem[idx] * inv_li);
        }
    }

    // Write logsumexp L
    if (tid < Br) {
        int qr = block_row * Br + tid;
        if (qr < N)
            L_bh[qr] = mi_smem[tid] + logf(li_smem[tid]);
    }
}

// ============================================================================
// Compute D[i] = dot(O[i], dO[i]) — fp16 version
// ============================================================================

__global__ void compute_D_fp16tc_kernel(
    const half* __restrict__ O,
    const half* __restrict__ dO,
    float* __restrict__ D_out,
    int N, int d
) {
    int batch_head = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const half* o_row = O + batch_head * N * d + row * d;
    const half* do_row = dO + batch_head * N * d + row * d;

    float sum = 0.0f;
    for (int i = 0; i < d; i++)
        sum += __half2float(o_row[i]) * __half2float(do_row[i]);
    D_out[batch_head * N + row] = sum;
}

// ============================================================================
// Backward dK/dV Kernel — thread-level matmul with fp16 loads
// ============================================================================
// Grid: dim3(B*nh, Tc). Each block owns one KV block.
// Same collaborative structure as v4, but with half-precision I/O.

template <int Br, int Bc, int D, int TPR>
__global__ void flash_backward_dkdv_fp16tc_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    half* __restrict__ dK,
    half* __restrict__ dV,
    int N, float scale
) {
    const int NUM_THREADS = Bc * TPR;
    const int batch_head = blockIdx.x;
    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const int row  = tid / TPR;
    const int lane = tid % TPR;
    const int kv_start = j * Bc;
    const int kv_row = kv_start + row;
    const int Tr = (N + Br - 1) / Br;

    const half* Q_bh  = Q  + batch_head * N * D;
    const half* K_bh  = K  + batch_head * N * D;
    const half* V_bh  = V  + batch_head * N * D;
    const half* dO_bh = dO + batch_head * N * D;
    const float* L_bh  = L  + batch_head * N;
    const float* D_bh  = D_arr + batch_head * N;
    half* dK_bh = dK + batch_head * N * D;
    half* dV_bh = dV + batch_head * N * D;

    constexpr int DPT = D / TPR;

    extern __shared__ float smem_f[];
    half* Q_smem  = reinterpret_cast<half*>(smem_f);          // [Br][D]
    half* dO_smem = Q_smem + Br * D;                          // [Br][D]
    float* S_smem = reinterpret_cast<float*>(dO_smem + Br * D); // [Br][Bc]

    // Load K, V into registers (fp32 for accumulation)
    float k_reg[DPT], v_reg[DPT], dk_reg[DPT], dv_reg[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; i++) { dk_reg[i] = 0.0f; dv_reg[i] = 0.0f; }

    if (kv_row < N) {
        #pragma unroll
        for (int i = 0; i < DPT; i++) {
            k_reg[i] = __half2float(K_bh[kv_row * D + lane * DPT + i]);
            v_reg[i] = __half2float(V_bh[kv_row * D + lane * DPT + i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < DPT; i++) { k_reg[i] = 0.0f; v_reg[i] = 0.0f; }
    }

    for (int i = j; i < Tr; i++) {
        int q_start = i * Br;

        // Load Q and dO into shared memory (fp16)
        {
            const int total = Br * D;
            const int loads = (total + NUM_THREADS - 1) / NUM_THREADS;
            for (int l = 0; l < loads; l++) {
                int idx = tid + l * NUM_THREADS;
                if (idx < total) {
                    int r = idx / D;
                    int d = idx % D;
                    int gr = q_start + r;
                    Q_smem[idx]  = (gr < N) ? Q_bh[gr * D + d]  : __float2half(0.0f);
                    dO_smem[idx] = (gr < N) ? dO_bh[gr * D + d] : __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute P[r][row] (thread-level, fp32 computation)
        #pragma unroll
        for (int r = 0; r < Br; r++) {
            int qg = q_start + r;
            float partial = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial += __half2float(Q_smem[r * D + lane * DPT + dd]) * k_reg[dd];

            // Warp reduction (all lanes participate — no divergence)
            #pragma unroll
            for (int offset = TPR / 2; offset > 0; offset >>= 1)
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);

            if (lane == 0) {
                bool masked = (qg >= N || kv_row >= N || kv_row > qg);
                S_smem[r * Bc + row] = masked ? 0.0f : expf(partial * scale - L_bh[min(qg, N-1)]);
            }
        }
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for (int r = 0; r < Br; r++) {
            float p = S_smem[r * Bc + row];
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                dv_reg[dd] += p * __half2float(dO_smem[r * D + lane * DPT + dd]);
        }

        // dK += scale * dS * Q
        #pragma unroll
        for (int r = 0; r < Br; r++) {
            int qg = q_start + r;
            float p = S_smem[r * Bc + row];

            float partial_doV = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial_doV += __half2float(dO_smem[r * D + lane * DPT + dd]) * v_reg[dd];
            #pragma unroll
            for (int offset = TPR / 2; offset > 0; offset >>= 1)
                partial_doV += __shfl_xor_sync(0xFFFFFFFF, partial_doV, offset);

            if (p == 0.0f || qg >= N) continue;

            float ds = p * (partial_doV - D_bh[qg]);
            float scaled_ds = scale * ds;

            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                dk_reg[dd] += scaled_ds * __half2float(Q_smem[r * D + lane * DPT + dd]);
        }
        __syncthreads();
    }

    if (kv_row < N) {
        #pragma unroll
        for (int dd = 0; dd < DPT; dd++) {
            dK_bh[kv_row * D + lane * DPT + dd] = __float2half(dk_reg[dd]);
            dV_bh[kv_row * D + lane * DPT + dd] = __float2half(dv_reg[dd]);
        }
    }
}

// ============================================================================
// Backward dQ Kernel — thread-level matmul with fp16 loads
// ============================================================================

template <int Br, int Bc, int D, int TPR>
__global__ void flash_backward_dq_fp16tc_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    half* __restrict__ dQ,
    int N, float scale
) {
    const int NUM_THREADS = Br * TPR;
    const int batch_head = blockIdx.x;
    const int i_block = blockIdx.y;
    const int tid = threadIdx.x;
    const int row  = tid / TPR;
    const int lane = tid % TPR;
    const int q_start = i_block * Br;
    const int qr = q_start + row;
    const int Tc = (N + Bc - 1) / Bc;

    constexpr int DPT = D / TPR;

    const half* Q_bh  = Q  + batch_head * N * D;
    const half* K_bh  = K  + batch_head * N * D;
    const half* V_bh  = V  + batch_head * N * D;
    const half* dO_bh = dO + batch_head * N * D;
    const float* L_bh  = L  + batch_head * N;
    const float* D_bh  = D_arr + batch_head * N;
    half* dQ_bh = dQ + batch_head * N * D;

    extern __shared__ float smem_f[];
    half* K_smem = reinterpret_cast<half*>(smem_f);      // [Bc][D]
    half* V_smem = K_smem + Bc * D;                       // [Bc][D]

    float q_reg[DPT], do_reg[DPT], dq_reg[DPT];
    #pragma unroll
    for (int dd = 0; dd < DPT; dd++) dq_reg[dd] = 0.0f;

    if (qr < N) {
        #pragma unroll
        for (int dd = 0; dd < DPT; dd++) {
            q_reg[dd]  = __half2float(Q_bh[qr * D + lane * DPT + dd]);
            do_reg[dd] = __half2float(dO_bh[qr * D + lane * DPT + dd]);
        }
    } else {
        #pragma unroll
        for (int dd = 0; dd < DPT; dd++) { q_reg[dd] = 0.0f; do_reg[dd] = 0.0f; }
    }

    float Li = (qr < N) ? L_bh[qr] : 0.0f;
    float Di = (qr < N) ? D_bh[qr] : 0.0f;

    int max_kv = min(i_block + 1, Tc);

    for (int j = 0; j < max_kv; j++) {
        int kv_start = j * Bc;

        // Load K and V (fp16)
        {
            const int total = Bc * D;
            const int loads = (total + NUM_THREADS - 1) / NUM_THREADS;
            for (int l = 0; l < loads; l++) {
                int idx = tid + l * NUM_THREADS;
                if (idx < total) {
                    int r = idx / D;
                    int d = idx % D;
                    int gr = kv_start + r;
                    K_smem[idx] = (gr < N) ? K_bh[gr * D + d] : __float2half(0.0f);
                    V_smem[idx] = (gr < N) ? V_bh[gr * D + d] : __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            int kp = kv_start + c;

            // Q @ K^T dot product (split across lanes)
            float partial_qk = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial_qk += q_reg[dd] * __half2float(K_smem[c * D + lane * DPT + dd]);
            #pragma unroll
            for (int offset = TPR / 2; offset > 0; offset >>= 1)
                partial_qk += __shfl_xor_sync(0xFFFFFFFF, partial_qk, offset);

            // dO @ V^T dot product
            float partial_doV = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial_doV += do_reg[dd] * __half2float(V_smem[c * D + lane * DPT + dd]);
            #pragma unroll
            for (int offset = TPR / 2; offset > 0; offset >>= 1)
                partial_doV += __shfl_xor_sync(0xFFFFFFFF, partial_doV, offset);

            float P_val = 0.0f;
            if (kp <= qr && kp < N && qr < N)
                P_val = expf(partial_qk * scale - Li);

            float ds = P_val * (partial_doV - Di);
            float scaled_ds = scale * ds;

            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                dq_reg[dd] += scaled_ds * __half2float(K_smem[c * D + lane * DPT + dd]);
        }
        __syncthreads();
    }

    if (qr < N) {
        #pragma unroll
        for (int dd = 0; dd < DPT; dd++)
            dQ_bh[qr * D + lane * DPT + dd] = __float2half(dq_reg[dd]);
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

std::vector<torch::Tensor> flash_forward_fp16_tc(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    float scale = 1.0f / sqrtf(static_cast<float>(d));
    auto O = torch::empty_like(Q);
    auto L = torch::empty({B, nh, N}, Q.options().dtype(torch::kFloat32));

    constexpr int Br = 32, Bc = 32;
    int Tr = (N + Br - 1) / Br;
    dim3 grid(B, nh, Tr);
    int num_warps = (Br / WMMA_M) * (Bc / WMMA_N);  // 4
    dim3 block(num_warps * 32);  // 128

    auto launch = [&](auto D_val) {
        constexpr int DD = decltype(D_val)::value;
        constexpr int LBr = 32, LBc = 32;  // local constexpr (not captured)
        int smem = (LBr * DD + LBc * DD + LBc * DD) * sizeof(half)  // Q, K, V
                 + LBr * LBc * sizeof(float)                         // S
                 + LBr * LBc * sizeof(half)                          // P
                 + 3 * LBr * sizeof(float)                           // mi, li, rescale
                 + LBr * DD * sizeof(float);                         // O buffer
        flash_forward_fp16tc_kernel<LBr, LBc, DD><<<grid, block, smem>>>(
            reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<half*>(O.data_ptr<at::Half>()),
            L.data_ptr<float>(), N, scale);
    };

    if (d == 32) launch(std::integral_constant<int, 32>{});
    else if (d == 64) launch(std::integral_constant<int, 64>{});
    else if (d == 128) launch(std::integral_constant<int, 128>{});
    else TORCH_CHECK(false, "V5 fp16+TC only supports d in {32, 64, 128}");

    return {O, L};
}

std::vector<torch::Tensor> flash_backward_fp16_tc(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    float scale = 1.0f / sqrtf(static_cast<float>(d));

    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);

    auto D_arr = torch::empty({B, nh, N}, Q.options().dtype(torch::kFloat32));
    {
        int threads = 256;
        int blocks_y = (N + threads - 1) / threads;
        dim3 grid_d(B * nh, blocks_y);
        compute_D_fp16tc_kernel<<<grid_d, threads>>>(
            reinterpret_cast<const half*>(O.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(dO.data_ptr<at::Half>()),
            D_arr.data_ptr<float>(), N, d);
    }

    constexpr int Br = 32, Bc = 32;

    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    auto launch_bwd = [&](auto D_val) {
        constexpr int DD = decltype(D_val)::value;
        constexpr int TPR = DD / 8;
        constexpr int LBr = 32, LBc = 32;  // local constexpr (not captured)
        constexpr int NT = LBc * TPR;

        // dK/dV kernel
        {
            dim3 grid_a(B * nh, Tc);
            dim3 block_a(NT);
            // smem: Q(half) + dO(half) + S(float)
            int smem_a = (LBr * DD + LBr * DD) * sizeof(half) + LBr * LBc * sizeof(float);
            flash_backward_dkdv_fp16tc_kernel<LBr, LBc, DD, TPR><<<grid_a, block_a, smem_a>>>(
                reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(dO.data_ptr<at::Half>()),
                L.data_ptr<float>(), D_arr.data_ptr<float>(),
                reinterpret_cast<half*>(dK.data_ptr<at::Half>()),
                reinterpret_cast<half*>(dV.data_ptr<at::Half>()),
                N, scale);
        }

        // dQ kernel
        {
            dim3 grid_b(B * nh, Tr);
            dim3 block_b(LBr * TPR);
            // smem: K(half) + V(half)
            int smem_b = (LBc * DD + LBc * DD) * sizeof(half);
            flash_backward_dq_fp16tc_kernel<LBr, LBc, DD, TPR><<<grid_b, block_b, smem_b>>>(
                reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(dO.data_ptr<at::Half>()),
                L.data_ptr<float>(), D_arr.data_ptr<float>(),
                reinterpret_cast<half*>(dQ.data_ptr<at::Half>()),
                N, scale);
        }
    };

    if (d == 32) launch_bwd(std::integral_constant<int, 32>{});
    else if (d == 64) launch_bwd(std::integral_constant<int, 64>{});
    else if (d == 128) launch_bwd(std::integral_constant<int, 128>{});
    else TORCH_CHECK(false, "V5 fp16+TC backward only supports d in {32, 64, 128}");

    return {dQ, dK, dV};
}
