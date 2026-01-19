# Flash Attention 2 in CUDA

A from-scratch CUDA implementation of Flash Attention 2, progressively optimized from naive baseline to tensor core kernels.

## Kernel Versions

| Version | Description |
|---------|-------------|
| **V1** | Naive 3-pass attention (baseline) |

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- NVIDIA GPU (SM 7.0+)

## Usage

```bash
python tests/test_forward.py
```

## License

MIT
