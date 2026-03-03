# Flash Attention 2 in CUDA — From Scratch

> V5 forward within 1.7x of PyTorch SDPA, backward within 1.7x of PyTorch autograd on H100. Full forward + backward pass with fp16 tensor cores — few open-source implementations include both.

A from-scratch CUDA implementation of Flash Attention 2 with **forward AND backward pass**, progressively optimized from naive baseline to fp16 tensor core kernels.

## Kernel Versions

| Version | Description | Precision | Key Techniques |
|---------|-------------|-----------|----------------|
| **V1** | Naive 3-pass attention | fp32 | Baseline, full N×N matrix |
| **V2** | Flash Attention 2 forward | fp32 | Online softmax, tiling, shared memory |
| **V3** | Flash Attention 2 backward | fp32 | Recompute P from logsumexp |
| **V4** | Optimised forward + backward | fp32 | Collaborative matmul, split backward, float4 loads, warp shuffles |
| **V5** | Tensor core forward + backward | fp16 in / fp32 accum | WMMA 16×16×16 matmuls (forward), split backward, fp16 memory bandwidth |

## What Makes This Different

Most open-source Flash Attention implementations only cover the forward pass. This project includes:

- **Complete backward pass** across three optimization levels (V3, V4, V5)
- **Split backward** (V4, V5): separate dK/dV and dQ kernels eliminate all atomicAdd operations
- **Progressive optimization**: every kernel builds on the previous, making it easy to study each technique in isolation
- **fp16 + tensor cores** (V5): WMMA 16×16×16 matmuls for S=Q@K^T and O+=P@V with online softmax rescaling via shared memory accumulator

## Quick Start

**Requirements**: Python 3.10+, PyTorch 2.x with CUDA, NVIDIA GPU (SM 7.0+, tested on RTX 2080 SM 7.5 and H100 SXM SM 9.0)

No CMake or separate build step needed — uses PyTorch JIT compilation.

```bash
# Run forward pass tests (16 tests: V1, V2, V4, V5 × 4 configs)
python tests/test_forward.py

# Run backward pass tests (9 tests: V3, V4, V5 × 3 configs)
python tests/test_backward.py

# Run benchmarks
python benchmarks/benchmark.py
```

> **Windows**: Use the `.bat` files in `scripts/` which set up the MSVC environment automatically.

## Project Structure

```
csrc/
  v1_naive_attention.cu       # Baseline: 3 kernel launches, full N×N matrix
  v2_flash_forward.cu         # FA2 forward: online softmax, tiling
  v3_flash_backward.cu        # FA2 backward: recompute P, gradient accumulation
  v4_flash_optimised.cu       # Optimised fp32: collaborative matmul, split backward, float4
  v5_flash_fp16_tensorcore.cu # fp16 + WMMA tensor cores (forward), fp16 thread-level (backward)
  bindings.cpp                # pybind11 module exposing all kernels to Python

tests/
  test_forward.py             # Verify forward against torch SDPA (25 tests)
  test_backward.py            # Verify gradients against torch.autograd

benchmarks/
  benchmark.py                # Performance measurement + chart generation
  results/                    # Generated CSVs and PNGs

scripts/                      # Windows build helper scripts (.bat)

docs/
  algorithm.md                # Flash Attention 2 algorithm explanation
```

## API

```python
from torch.utils.cpp_extension import load

flash_attn = load(
    name="flash_attn",
    sources=["csrc/bindings.cpp", "csrc/v1_naive_attention.cu",
             "csrc/v2_flash_forward.cu", "csrc/v3_flash_backward.cu",
             "csrc/v4_flash_optimised.cu", "csrc/v5_flash_fp16_tensorcore.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# fp32 inputs: Q, K, V of shape [B, num_heads, seq_len, head_dim]

O = flash_attn.naive_attention(Q, K, V)                           # V1
O, L = flash_attn.flash_forward(Q, K, V)                          # V2
dQ, dK, dV = flash_attn.flash_backward(Q, K, V, O, dO, L)        # V3
O, L = flash_attn.flash_forward_optimised(Q, K, V)                # V4
dQ, dK, dV = flash_attn.flash_backward_optimised(Q, K, V, O, dO, L) # V4

# fp16 inputs: Q, K, V of shape [B, num_heads, seq_len, head_dim], dtype=float16

O, L = flash_attn.flash_forward_fp16_tc(Q, K, V)                  # V5
dQ, dK, dV = flash_attn.flash_backward_fp16_tc(Q, K, V, O, dO, L) # V5
```

## Correctness

All **25 tests passing** on both RTX 2080 and H100:

| Version | Forward | Backward | Max Error | Test Tolerance |
|---------|---------|----------|-----------|----------------|
| V1 Naive | 4/4 | — | < 1e-6 | atol=1e-4 |
| V2 Flash | 4/4 | — | < 1e-6 | atol=1e-4 |
| V3 Flash | — | 3/3 | < 1e-5 | atol=1e-3 |
| V4 Optimised | 4/4 | 3/3 | < 1e-5 | atol=1e-3 |
| V5 fp16 TC | 4/4 | 3/3 | < 2e-3 fwd, < 4e-3 bwd (fp16) | atol=1e-2 |

## Benchmark Results (RTX 2080, SM 7.5)

### Forward Pass (ms)

| Config (B,nh,N,d) | PyTorch SDPA | PyTorch Naive | V1 Naive | V2 Flash | V4 Optimised | V5 fp16 TC | vs SDPA |
|--------------------|-------------|---------------|----------|----------|--------------|------------|---------|
| 1,1,64,32 | 0.016 | 0.024 | 0.186 | 0.133 | 0.016 | **0.012** | 0.8x |
| 2,4,128,64 | 0.030 | 0.047 | 1.547 | 0.614 | 0.054 | **0.029** | 1.0x |
| 4,8,256,64 | 0.119 | 0.268 | 8.777 | 3.360 | 0.395 | **0.127** | 1.1x |
| 4,8,512,64 | 0.297 | 1.047 | 35.028 | 11.760 | 1.409 | **0.487** | 1.6x |
| 4,8,1024,64 | 1.056 | 4.439 | 224.701 | 43.464 | 5.566 | **1.816** | 1.7x |
| 4,8,2048,64 | 4.162 | 18.108 | — | 166.905 | 22.102 | **7.026** | 1.7x |

### Backward Pass (ms)

| Config (B,nh,N,d) | PyTorch Bwd | V3 Flash | V4 Optimised | V5 fp16 TC | vs PyTorch |
|--------------------|------------|----------|--------------|------------|------------|
| 1,1,64,32 | 0.049 | 0.785 | 0.034 | **0.042** | 0.9x |
| 2,4,128,64 | 0.114 | 5.569 | 0.115 | **0.095** | 0.8x |
| 4,8,256,64 | 0.649 | 21.735 | 0.906 | **0.414** | 0.6x |
| 4,8,512,64 | 2.166 | 85.908 | 3.277 | **1.539** | 0.7x |
| 4,8,1024,64 | 6.733 | 340.837 | 13.274 | **5.713** | 0.8x |
| 4,8,2048,64 | 23.268 | 1368.716 | 50.990 | **22.323** | 1.0x |

### Key Takeaways (RTX 2080)

- **V5 forward** within **1.7x of PyTorch SDPA** at N=1024–2048
- **V5 backward** matches PyTorch autograd at N=2048 (1.0x)
- **V4→V5 speedup**: 3.1x forward, 2.3x backward (from fp16 bandwidth + tensor core matmuls)

> Small configs (N≤128) are dominated by kernel launch overhead; meaningful performance comparisons start at N≥256.

## Benchmark Results (H100 SXM, SM 9.0)

### Forward Pass (ms)

| Config (B,nh,N,d) | PyTorch SDPA | PyTorch Naive | V1 Naive | V2 Flash | V4 Optimised | V5 fp16 TC | vs SDPA |
|--------------------|-------------|---------------|----------|----------|--------------|------------|---------|
| 1,1,64,32 | 0.019 | 0.065 | 0.218 | 0.109 | 0.017 | **0.016** | 0.8x |
| 2,4,128,64 | 0.026 | 0.068 | 1.033 | 0.410 | 0.037 | **0.028** | 1.1x |
| 4,8,256,64 | 0.040 | 0.074 | 3.740 | 0.969 | 0.088 | **0.051** | 1.3x |
| 4,8,512,64 | 0.086 | 0.173 | 14.691 | 2.793 | 0.255 | **0.119** | 1.4x |
| 4,8,1024,64 | 0.229 | 0.580 | 58.042 | 8.189 | 0.840 | **0.350** | 1.5x |
| 4,8,2048,64 | 0.699 | 2.510 | — | 28.027 | 3.032 | **1.169** | 1.7x |

### Backward Pass (ms)

| Config (B,nh,N,d) | PyTorch Bwd | V3 Flash | V4 Optimised | V5 fp16 TC | vs PyTorch |
|--------------------|------------|----------|--------------|------------|------------|
| 1,1,64,32 | 0.140 | 0.666 | 0.044 | **0.046** | 0.3x |
| 2,4,128,64 | 0.245 | 4.534 | 0.101 | **0.093** | 0.4x |
| 4,8,256,64 | 0.314 | 17.462 | 0.232 | **0.182** | 0.6x |
| 4,8,512,64 | 0.432 | 68.603 | 0.632 | **0.390** | 0.9x |
| 4,8,1024,64 | 0.925 | 273.771 | 1.811 | **1.159** | 1.3x |
| 4,8,2048,64 | 2.432 | 1094.800 | 6.790 | **4.164** | 1.7x |

### Key Takeaways (H100)

- **V5 forward** within **1.5x of PyTorch SDPA** at N=1024, **1.7x** at N=2048
- **V5 backward** within **1.7x of PyTorch autograd** at N=2048
- **H100 vs RTX 2080** (V5 at N=2048): **6.0x** faster forward, **5.4x** faster backward
- Tensor core benefits scale with sequence length — at small N, fp16 conversion overhead dominates; at N≥512 the bandwidth and compute savings take over

> "vs SDPA" / "vs PyTorch" columns show the ratio V5 time / PyTorch time. Values < 1.0x mean V5 is faster.

## Optimization Techniques by Version

### V4: Collaborative Matmul + Split Backward
- **Collaborative matmul**: 8 threads per query row (TPR=D/8), split-K dot products with warp shuffle reductions
- **Split backward**: separate dK/dV and dQ kernels — dK/dV parallelizes over KV blocks, dQ parallelizes over query blocks, zero atomicAdd operations
- **float4 vectorized loads**: 128-bit memory transactions for global→register transfers
- **256 threads/block** (8 warps) for high SM occupancy

### V5: fp16 + Tensor Cores
- **WMMA 16×16×16**: tensor core matmuls for S=Q@K^T and O+=P@V in forward kernel
- **Online softmax rescaling**: O accumulated in shared memory (fp32), rescale factors stored per-row before mi is updated, accumulator loaded into WMMA fragments via `load_matrix_sync`
- **D-tile splitting**: each warp handles non-overlapping output columns, eliminating redundant computation
- **fp16 I/O, fp32 accumulation**: halves memory bandwidth while maintaining numerical stability
- **4 warps (128 threads)**: 2×2 tile arrangement for 32×32 block with 16×16 WMMA tiles

## TODO

- [ ] Double buffering (overlap compute with memory loads)
- [ ] Swizzled shared memory layout to reduce bank conflicts
- [ ] cp.async for global→shared memory copies (SM 8.0+)

## References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023

## License

MIT
