"""Benchmark all Flash Attention kernel versions and generate comparison charts."""

import sys
import csv
import torch
import torch.nn.functional as F
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ_ROOT / "benchmarks" / "results"

def load_module():
    from torch.utils.cpp_extension import load
    return load(
        name="flash_attn",
        sources=[
            str(PROJ_ROOT / "csrc" / "bindings.cpp"),
            str(PROJ_ROOT / "csrc" / "v1_naive_attention.cu"),
            str(PROJ_ROOT / "csrc" / "v2_flash_forward.cu"),
            str(PROJ_ROOT / "csrc" / "v3_flash_backward.cu"),
            str(PROJ_ROOT / "csrc" / "v4_flash_optimised.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )

def cuda_timer(fn, warmup=5, repeats=20):
    """Time a CUDA function using cuda events for accurate measurement."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return min(times), sum(times) / len(times)

def pytorch_naive_attention(Q, K, V):
    """PyTorch naive attention (manual matmul, no fused kernel)."""
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    N = Q.shape[-2]
    mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)

def benchmark_forward(mod, configs):
    """Benchmark forward pass for all versions."""
    print("\n" + "=" * 70)
    print("FORWARD PASS BENCHMARKS")
    print("=" * 70)

    rows = []
    for B, nh, N, d in configs:
        print(f"\nConfig: B={B}, nh={nh}, N={N}, d={d}")
        torch.manual_seed(0)
        Q = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)

        row = {"B": B, "nh": nh, "N": N, "d": d}

        min_ms, avg_ms = cuda_timer(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        print(f"  PyTorch SDPA:    {avg_ms:8.3f} ms (min {min_ms:.3f})")
        row["pytorch_sdpa_ms"] = f"{avg_ms:.3f}"

        if N <= 2048:
            min_ms, avg_ms = cuda_timer(lambda: pytorch_naive_attention(Q, K, V))
            print(f"  PyTorch naive:   {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["pytorch_naive_ms"] = f"{avg_ms:.3f}"
        else:
            row["pytorch_naive_ms"] = "N/A"

        if N <= 1024:
            min_ms, avg_ms = cuda_timer(lambda: mod.naive_attention(Q, K, V))
            print(f"  V1 naive:        {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v1_naive_ms"] = f"{avg_ms:.3f}"
        else:
            row["v1_naive_ms"] = "N/A"

        min_ms, avg_ms = cuda_timer(lambda: mod.flash_forward(Q, K, V))
        print(f"  V2 flash fwd:    {avg_ms:8.3f} ms (min {min_ms:.3f})")
        row["v2_flash_ms"] = f"{avg_ms:.3f}"

        if d in (32, 64, 128):
            min_ms, avg_ms = cuda_timer(lambda: mod.flash_forward_optimised(Q, K, V))
            print(f"  V4 opt fwd:      {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v4_opt_ms"] = f"{avg_ms:.3f}"
        else:
            row["v4_opt_ms"] = "N/A"

        rows.append(row)
    return rows

def save_csv(rows, filename):
    """Save benchmark results to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    if not rows:
        return
    keys = rows[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {filepath}")

def main():
    print("Compiling CUDA kernels...")
    mod = load_module()
    print("Compilation complete!\n")

    configs = [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (4, 8, 256, 64),
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
    ]

    fwd_rows = benchmark_forward(mod, configs)
    save_csv(fwd_rows, "forward_benchmark.csv")
    print("\nDone!")

if __name__ == "__main__":
    main()
