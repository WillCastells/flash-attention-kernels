#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================================
// V4: Optimized Flash Attention 2
//
// Optimizations:
// 1. Collaborative matmul: TPR threads per row, split-K dot products
// 2. Warp shuffle reductions (no shared memory for reductions)
// 3. 256 threads/block for high SM occupancy
// 4. Split backward: separate dK/dV and dQ kernels (no atomicAdd)
// 5. float4 vectorized loads for memory bandwidth
// 6. Template on D for compile-time unrolling
// ============================================================================

constexpr int DPT = 8;  // D_PER_THREAD: always 8 for all head dims

// ---- Warp-level reductions (all threads must participate) ----

template <int TPR>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <int TPR>
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ---- float4 helpers ----

__device__ __forceinline__ void load_reg_f4(float* dst, const float* src) {
    float4 a = *reinterpret_cast<const float4*>(src);
    float4 b = *reinterpret_cast<const float4*>(src + 4);
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
    dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
}

__device__ __forceinline__ void store_reg_f4(float* dst, const float* src) {
    *reinterpret_cast<float4*>(dst)     = make_float4(src[0], src[1], src[2], src[3]);
    *reinterpret_cast<float4*>(dst + 4) = make_float4(src[4], src[5], src[6], src[7]);
}

// Collaborative float4 load: all threads load a [rows][D] block from global to shared
template <int D, int NUM_THREADS>
__device__ __forceinline__ void collaborative_load_f4(
    float* smem, const float* gmem, int rows, int start_row, int N, int tid
) {
    const int total_vec = (rows * D) / 4;
    const int vec_loads = (total_vec + NUM_THREADS - 1) / NUM_THREADS;
    #pragma unroll
    for (int l = 0; l < vec_loads; l++) {
        int vi = tid + l * NUM_THREADS;
        if (vi < total_vec) {
            int flat = vi * 4;
            int r = flat / D;
            int gr = start_row + r;
            float4 val;
            if (gr < N) {
                val = *reinterpret_cast<const float4*>(&gmem[gr * D + (flat % D)]);
            } else {
                val = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            *reinterpret_cast<float4*>(&smem[flat]) = val;
        }
    }
}

// Same but load two arrays simultaneously (K and V, or Q and dO)
template <int D, int NUM_THREADS>
__device__ __forceinline__ void collaborative_load_f4_dual(
    float* smem_a, const float* gmem_a,
    float* smem_b, const float* gmem_b,
    int rows, int start_row, int N, int tid
) {
    const int total_vec = (rows * D) / 4;
    const int vec_loads = (total_vec + NUM_THREADS - 1) / NUM_THREADS;
    #pragma unroll
    for (int l = 0; l < vec_loads; l++) {
        int vi = tid + l * NUM_THREADS;
        if (vi < total_vec) {
            int flat = vi * 4;
            int r = flat / D;
            int d = flat % D;
            int gr = start_row + r;
            float4 va, vb;
            if (gr < N) {
                va = *reinterpret_cast<const float4*>(&gmem_a[gr * D + d]);
                vb = *reinterpret_cast<const float4*>(&gmem_b[gr * D + d]);
            } else {
                va = make_float4(0.f, 0.f, 0.f, 0.f);
                vb = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            *reinterpret_cast<float4*>(&smem_a[flat]) = va;
            *reinterpret_cast<float4*>(&smem_b[flat]) = vb;
        }
    }
}

// ============================================================================
// Forward Kernel — Collaborative Matmul + float4 Loads
// ============================================================================

template <int Br, int Bc, int D, int TPR>
__global__ void flash_forward_opt_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_out,
    float* __restrict__ L_out,
    int N, float scale
) {
    const int NUM_THREADS = Br * TPR;
    const int batch_head = blockIdx.x * gridDim.y + blockIdx.y;
    const int block_row  = blockIdx.z;
    const int tid = threadIdx.x;
    const int row  = tid / TPR;
    const int lane = tid % TPR;
    const int qr = block_row * Br + row;

    const float* Q_bh = Q + batch_head * N * D;
    const float* K_bh = K + batch_head * N * D;
    const float* V_bh = V + batch_head * N * D;
    float* O_bh = O_out + batch_head * N * D;
    float* L_bh = L_out + batch_head * N;

    extern __shared__ float smem[];
    float* K_smem = smem;               // [Bc][D]
    float* V_smem = K_smem + Bc * D;    // [Bc][D]
    float* S_smem = V_smem + Bc * D;    // [Br][Bc]

    // Load Q slice into registers via float4
    float q_reg[DPT];
    float o_reg[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; i++) o_reg[i] = 0.0f;

    if (qr < N) {
        load_reg_f4(q_reg, &Q_bh[qr * D + lane * DPT]);
    } else {
        #pragma unroll
        for (int i = 0; i < DPT; i++) q_reg[i] = 0.0f;
    }

    float mi = -INFINITY;
    float li = 0.0f;

    int Tc = (N + Bc - 1) / Bc;
    int max_qr = min(block_row * Br + Br - 1, N - 1);
    int max_kv_block = min(max_qr / Bc + 1, Tc);

    for (int j = 0; j < max_kv_block; j++) {
        int kv_start = j * Bc;

        // float4 collaborative load K and V
        collaborative_load_f4_dual<D, NUM_THREADS>(
            K_smem, K_bh, V_smem, V_bh, Bc, kv_start, N, tid);
        __syncthreads();

        // S = Q @ K^T — all threads compute, mask at write
        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            float partial = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial += q_reg[dd] * K_smem[c * D + lane * DPT + dd];
            float dot = warp_reduce_sum<TPR>(partial);
            if (lane == 0) {
                int kp = kv_start + c;
                S_smem[row * Bc + c] = (kp > qr || kp >= N || qr >= N)
                    ? -INFINITY : (dot * scale);
            }
        }
        __syncthreads();

        // Online softmax — all threads participate
        {
            float local_max = -INFINITY;
            #pragma unroll
            for (int c = lane; c < Bc; c += TPR)
                local_max = fmaxf(local_max, S_smem[row * Bc + c]);
            float row_max = warp_reduce_max<TPR>(local_max);

            float new_mi = fmaxf(mi, row_max);
            float rescale = expf(mi - new_mi);
            li *= rescale;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++) o_reg[dd] *= rescale;

            float local_sum = 0.0f;
            #pragma unroll
            for (int c = lane; c < Bc; c += TPR) {
                float p = expf(S_smem[row * Bc + c] - new_mi);
                S_smem[row * Bc + c] = p;
                local_sum += p;
            }
            float block_sum = warp_reduce_sum<TPR>(local_sum);
            mi = new_mi;
            li += block_sum;
        }
        __syncthreads();

        // O += P @ V — split-D
        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            float p = S_smem[row * Bc + c];
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                o_reg[dd] += p * V_smem[c * D + lane * DPT + dd];
        }
        __syncthreads();
    }

    // Write output via float4
    if (qr < N) {
        float inv_li = 1.0f / li;
        float tmp[DPT];
        #pragma unroll
        for (int dd = 0; dd < DPT; dd++) tmp[dd] = o_reg[dd] * inv_li;
        store_reg_f4(&O_bh[qr * D + lane * DPT], tmp);
        if (lane == 0)
            L_bh[qr] = mi + logf(li);
    }
}

// ============================================================================
// Compute D[i] = dot(O[i], dO[i])
// ============================================================================

__global__ void compute_D_opt_kernel(
    const float* __restrict__ O,
    const float* __restrict__ dO,
    float* __restrict__ D_out,
    int N, int d
) {
    int batch_head = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* o_row = O + batch_head * N * d + row * d;
    const float* do_row = dO + batch_head * N * d + row * d;

    float sum = 0.0f;
    for (int i = 0; i < d; i++)
        sum += o_row[i] * do_row[i];
    D_out[batch_head * N + row] = sum;
}

// ============================================================================
// Backward Kernel A: dK/dV — Grid over KV blocks, no atomics
// ============================================================================
//
// Grid: dim3(B*nh, Tc)
// Each block owns one KV block, iterates over query blocks i=j..Tr-1.
// dK, dV accumulated in registers, written once at the end.
//
// Shared memory: Q_smem[Br][D] + dO_smem[Br][D] + S_smem[Br][Bc]
// K, V in registers (k_reg, v_reg).

template <int Br, int Bc, int D, int TPR>
__global__ void flash_backward_dkdv_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int N, float scale
) {
    const int NUM_THREADS = Bc * TPR;
    const int batch_head = blockIdx.x;
    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const int row  = tid / TPR;  // KV row within block (0..Bc-1)
    const int lane = tid % TPR;

    const int kv_start = j * Bc;
    const int kv_row = kv_start + row;
    const int Tr = (N + Br - 1) / Br;

    const float* Q_bh  = Q  + batch_head * N * D;
    const float* K_bh  = K  + batch_head * N * D;
    const float* V_bh  = V  + batch_head * N * D;
    const float* dO_bh = dO + batch_head * N * D;
    const float* L_bh  = L  + batch_head * N;
    const float* D_bh  = D_arr + batch_head * N;
    float* dK_bh = dK + batch_head * N * D;
    float* dV_bh = dV + batch_head * N * D;

    extern __shared__ float smem[];
    float* Q_smem  = smem;                   // [Br][D]
    float* dO_smem = Q_smem  + Br * D;       // [Br][D]
    float* S_smem  = dO_smem + Br * D;       // [Br][Bc]

    // Load K, V for this KV row into registers
    float k_reg[DPT], v_reg[DPT], dk_reg[DPT], dv_reg[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; i++) { dk_reg[i] = 0.0f; dv_reg[i] = 0.0f; }

    if (kv_row < N) {
        load_reg_f4(k_reg, &K_bh[kv_row * D + lane * DPT]);
        load_reg_f4(v_reg, &V_bh[kv_row * D + lane * DPT]);
    } else {
        #pragma unroll
        for (int i = 0; i < DPT; i++) { k_reg[i] = 0.0f; v_reg[i] = 0.0f; }
    }

    // Iterate over query blocks i = j .. Tr-1
    for (int i = j; i < Tr; i++) {
        int q_start = i * Br;

        // float4 collaborative load Q[i] and dO[i]
        collaborative_load_f4_dual<D, NUM_THREADS>(
            Q_smem, Q_bh, dO_smem, dO_bh, Br, q_start, N, tid);
        __syncthreads();

        // Compute P[r][row] = exp(Q[r] @ K[row] * scale - L[r])
        // All threads compute dot + shuffle. Mask at write.
        #pragma unroll
        for (int r = 0; r < Br; r++) {
            int qg = q_start + r;
            float partial = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial += Q_smem[r * D + lane * DPT + dd] * k_reg[dd];
            float dot = warp_reduce_sum<TPR>(partial);
            if (lane == 0) {
                bool masked = (qg >= N || kv_row >= N || kv_row > qg);
                S_smem[r * Bc + row] = masked ? 0.0f : expf(dot * scale - L_bh[min(qg, N-1)]);
            }
        }
        __syncthreads();

        // Accumulate dV += P^T @ dO  (split-D, no reduction)
        #pragma unroll
        for (int r = 0; r < Br; r++) {
            float p = S_smem[r * Bc + row];
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                dv_reg[dd] += p * dO_smem[r * D + lane * DPT + dd];
        }

        // Compute dS and accumulate dK
        #pragma unroll
        for (int r = 0; r < Br; r++) {
            int qg = q_start + r;
            float p = S_smem[r * Bc + row];

            // dO[r] @ V[row] (split-K, all threads participate)
            float partial_doV = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial_doV += dO_smem[r * D + lane * DPT + dd] * v_reg[dd];
            float doV = warp_reduce_sum<TPR>(partial_doV);

            if (p == 0.0f || qg >= N) continue;

            float ds = p * (doV - D_bh[qg]);
            float scaled_ds = scale * ds;

            // dK[row] += scaled_ds * Q[r] (split-D)
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                dk_reg[dd] += scaled_ds * Q_smem[r * D + lane * DPT + dd];
        }
        __syncthreads();
    }

    // Write dK, dV via float4
    if (kv_row < N) {
        store_reg_f4(&dK_bh[kv_row * D + lane * DPT], dk_reg);
        store_reg_f4(&dV_bh[kv_row * D + lane * DPT], dv_reg);
    }
}

// ============================================================================
// Backward Kernel B: dQ — Grid over Q blocks, no atomics
// ============================================================================
//
// Grid: dim3(B*nh, Tr)
// Each block owns one Q block, iterates over KV blocks j=0..i (causal).
// dQ accumulated in registers, written once at the end.
// No S_smem needed — P is computed in registers by all lanes.
//
// Shared memory: K_smem[Bc][D] + V_smem[Bc][D]

template <int Br, int Bc, int D, int TPR>
__global__ void flash_backward_dq_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    float* __restrict__ dQ,
    int N, float scale
) {
    const int NUM_THREADS = Br * TPR;
    const int batch_head = blockIdx.x;
    const int i = blockIdx.y;  // which Q block
    const int tid = threadIdx.x;
    const int row  = tid / TPR;  // Q row within block (0..Br-1)
    const int lane = tid % TPR;

    const int q_start = i * Br;
    const int qr = q_start + row;
    const int Tc = (N + Bc - 1) / Bc;

    const float* Q_bh  = Q  + batch_head * N * D;
    const float* K_bh  = K  + batch_head * N * D;
    const float* V_bh  = V  + batch_head * N * D;
    const float* dO_bh = dO + batch_head * N * D;
    const float* L_bh  = L  + batch_head * N;
    const float* D_bh  = D_arr + batch_head * N;
    float* dQ_bh = dQ + batch_head * N * D;

    extern __shared__ float smem[];
    float* K_smem = smem;               // [Bc][D]
    float* V_smem = K_smem + Bc * D;    // [Bc][D]

    // Load Q and dO for this row into registers
    float q_reg[DPT], do_reg[DPT], dq_reg[DPT];
    #pragma unroll
    for (int dd = 0; dd < DPT; dd++) dq_reg[dd] = 0.0f;

    if (qr < N) {
        load_reg_f4(q_reg,  &Q_bh[qr * D + lane * DPT]);
        load_reg_f4(do_reg, &dO_bh[qr * D + lane * DPT]);
    } else {
        #pragma unroll
        for (int dd = 0; dd < DPT; dd++) { q_reg[dd] = 0.0f; do_reg[dd] = 0.0f; }
    }

    float Li = (qr < N) ? L_bh[qr] : 0.0f;
    float Di = (qr < N) ? D_bh[qr] : 0.0f;

    // Causal: iterate over KV blocks j = 0 .. min(i, Tc-1)
    int max_kv = min(i + 1, Tc);  // block i can attend to KV blocks 0..i

    for (int j = 0; j < max_kv; j++) {
        int kv_start = j * Bc;

        // float4 collaborative load K[j] and V[j]
        collaborative_load_f4_dual<D, NUM_THREADS>(
            K_smem, K_bh, V_smem, V_bh, Bc, kv_start, N, tid);
        __syncthreads();

        // For each key position c, compute P and dS, accumulate dQ
        // All threads compute dot products (no warp divergence at shuffles)
        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            int kp = kv_start + c;

            // Q[row] @ K[c] — split-K dot, all lanes get result
            float partial_qk = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial_qk += q_reg[dd] * K_smem[c * D + lane * DPT + dd];
            float dot_qk = warp_reduce_sum<TPR>(partial_qk);

            // dO[row] @ V[c] — split-K dot, all lanes get result
            float partial_doV = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                partial_doV += do_reg[dd] * V_smem[c * D + lane * DPT + dd];
            float doV = warp_reduce_sum<TPR>(partial_doV);

            // P and dS (scalar, same on all lanes after reduction)
            float P_val = 0.0f;
            if (kp <= qr && kp < N && qr < N)
                P_val = expf(dot_qk * scale - Li);

            float ds = P_val * (doV - Di);
            float scaled_ds = scale * ds;

            // dQ[row] += scaled_ds * K[c] — split-D, no reduction
            #pragma unroll
            for (int dd = 0; dd < DPT; dd++)
                dq_reg[dd] += scaled_ds * K_smem[c * D + lane * DPT + dd];
        }
        __syncthreads();
    }

    // Write dQ via float4
    if (qr < N) {
        store_reg_f4(&dQ_bh[qr * D + lane * DPT], dq_reg);
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

std::vector<torch::Tensor> flash_forward_optimised(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    float scale = 1.0f / sqrtf(static_cast<float>(d));
    auto O = torch::empty_like(Q);
    auto L = torch::empty({B, nh, N}, Q.options());

    constexpr int Br = 32, Bc = 32;
    int Tr = (N + Br - 1) / Br;
    dim3 grid(B, nh, Tr);

    // smem = K[Bc*D] + V[Bc*D] + S[Br*Bc]
    #define LAUNCH_FWD(DD) { \
        constexpr int TPR = DD / DPT; \
        dim3 block(Br * TPR); \
        int smem = (2 * Bc * DD + Br * Bc) * sizeof(float); \
        flash_forward_opt_kernel<Br, Bc, DD, TPR><<<grid, block, smem>>>( \
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
            O.data_ptr<float>(), L.data_ptr<float>(), N, scale); \
    }

    if (d == 32) { LAUNCH_FWD(32); }
    else if (d == 64) { LAUNCH_FWD(64); }
    else if (d == 128) { LAUNCH_FWD(128); }
    else { TORCH_CHECK(false, "Optimised kernel only supports d in {32, 64, 128}"); }
    #undef LAUNCH_FWD

    return {O, L};
}

std::vector<torch::Tensor> flash_backward_optimised(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);

    float scale = 1.0f / sqrtf(static_cast<float>(d));

    auto dQ = torch::empty_like(Q);  // No atomics — each row written by exactly one block
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);

    // Compute D[i] = dot(O[i], dO[i])
    auto D_arr = torch::empty({B, nh, N}, Q.options());
    {
        int threads = 256;
        int blocks_y = (N + threads - 1) / threads;
        dim3 grid_d(B * nh, blocks_y);
        compute_D_opt_kernel<<<grid_d, threads>>>(
            O.data_ptr<float>(), dO.data_ptr<float>(),
            D_arr.data_ptr<float>(), N, d);
    }

    constexpr int Br = 32, Bc = 32;
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    #define LAUNCH_BWD(DD) { \
        constexpr int TPR = DD / DPT; \
        constexpr int NT = Bc * TPR; \
        /* Kernel A: dK/dV — grid over KV blocks */ \
        /* smem = Q[Br*DD] + dO[Br*DD] + S[Br*Bc] */ \
        { \
            dim3 grid_a(B * nh, Tc); \
            dim3 block_a(NT); \
            int smem_a = (2 * Br * DD + Br * Bc) * sizeof(float); \
            flash_backward_dkdv_kernel<Br, Bc, DD, TPR><<<grid_a, block_a, smem_a>>>( \
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
                dO.data_ptr<float>(), L.data_ptr<float>(), D_arr.data_ptr<float>(), \
                dK.data_ptr<float>(), dV.data_ptr<float>(), N, scale); \
        } \
        /* Kernel B: dQ — grid over Q blocks */ \
        /* smem = K[Bc*DD] + V[Bc*DD] */ \
        { \
            dim3 grid_b(B * nh, Tr); \
            dim3 block_b(Br * TPR); \
            int smem_b = (2 * Bc * DD) * sizeof(float); \
            flash_backward_dq_kernel<Br, Bc, DD, TPR><<<grid_b, block_b, smem_b>>>( \
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
                dO.data_ptr<float>(), L.data_ptr<float>(), D_arr.data_ptr<float>(), \
                dQ.data_ptr<float>(), N, scale); \
        } \
    }

    if (d == 32) { LAUNCH_BWD(32); }
    else if (d == 64) { LAUNCH_BWD(64); }
    else if (d == 128) { LAUNCH_BWD(128); }
    else { TORCH_CHECK(false, "Optimised backward only supports d in {32, 64, 128}"); }
    #undef LAUNCH_BWD

    return {dQ, dK, dV};
}
