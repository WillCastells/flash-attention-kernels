# Flash Attention 2 in CUDA — From Scratch

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

**Requirements**: Python 3.10+, PyTorch 2.x with CUDA, NVIDIA GPU (SM 7.0+)

No CMake or separate build step needed — uses PyTorch JIT compilation.

```bash
# Run forward pass tests
python tests/test_forward.py

# Run backward pass tests
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
  test_forward.py             # Verify forward against torch SDPA
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

## License

MIT
