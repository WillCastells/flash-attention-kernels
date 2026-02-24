"""Benchmark all Flash Attention kernel versions and generate comparison charts."""

import sys
import csv
import time
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
            str(PROJ_ROOT / "csrc" / "v5_flash_fp16_tensorcore.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def cuda_timer(fn, warmup=5, repeats=20):
    """Time a CUDA function using cuda events for accurate measurement."""
    # Warmup
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

    return min(times), sum(times) / len(times)  # min_ms, avg_ms


def pytorch_naive_attention(Q, K, V):
    """PyTorch naive attention (manual matmul, no fused kernel)."""
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    # Causal mask
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

        # PyTorch SDPA (fused)
        min_ms, avg_ms = cuda_timer(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))
        print(f"  PyTorch SDPA:    {avg_ms:8.3f} ms (min {min_ms:.3f})")
        row["pytorch_sdpa_ms"] = f"{avg_ms:.3f}"

        # PyTorch naive (manual matmul)
        if N <= 2048:
            min_ms, avg_ms = cuda_timer(lambda: pytorch_naive_attention(Q, K, V))
            print(f"  PyTorch naive:   {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["pytorch_naive_ms"] = f"{avg_ms:.3f}"
        else:
            row["pytorch_naive_ms"] = "N/A"

        # V1: Naive CUDA kernel
        if N <= 1024:
            min_ms, avg_ms = cuda_timer(lambda: mod.naive_attention(Q, K, V))
            print(f"  V1 naive:        {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v1_naive_ms"] = f"{avg_ms:.3f}"
        else:
            row["v1_naive_ms"] = "N/A"

        # V2: Flash forward
        min_ms, avg_ms = cuda_timer(lambda: mod.flash_forward(Q, K, V))
        print(f"  V2 flash fwd:    {avg_ms:8.3f} ms (min {min_ms:.3f})")
        row["v2_flash_ms"] = f"{avg_ms:.3f}"

        # V4: Optimised forward
        if d in (32, 64, 128):
            min_ms, avg_ms = cuda_timer(lambda: mod.flash_forward_optimised(Q, K, V))
            print(f"  V4 opt fwd:      {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v4_opt_ms"] = f"{avg_ms:.3f}"
        else:
            row["v4_opt_ms"] = "N/A"

        # V5: fp16 Tensor Core forward
        if d in (32, 64, 128):
            Q_h = Q.half()
            K_h = K.half()
            V_h = V.half()
            min_ms, avg_ms = cuda_timer(lambda: mod.flash_forward_fp16_tc(Q_h, K_h, V_h))
            print(f"  V5 fp16 TC fwd:  {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v5_fp16tc_ms"] = f"{avg_ms:.3f}"
        else:
            row["v5_fp16tc_ms"] = "N/A"

        rows.append(row)

    return rows


def benchmark_backward(mod, configs):
    """Benchmark backward pass for all versions."""
    print("\n" + "=" * 70)
    print("BACKWARD PASS BENCHMARKS")
    print("=" * 70)

    rows = []

    for B, nh, N, d in configs:
        print(f"\nConfig: B={B}, nh={nh}, N={N}, d={d}")
        torch.manual_seed(0)
        Q = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        dO = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)

        row = {"B": B, "nh": nh, "N": N, "d": d}

        # PyTorch autograd backward
        def pytorch_backward():
            Qr = Q.clone().requires_grad_(True)
            Kr = K.clone().requires_grad_(True)
            Vr = V.clone().requires_grad_(True)
            O = F.scaled_dot_product_attention(Qr, Kr, Vr, is_causal=True)
            O.backward(dO)

        min_ms, avg_ms = cuda_timer(pytorch_backward)
        print(f"  PyTorch bwd:     {avg_ms:8.3f} ms (min {min_ms:.3f})")
        row["pytorch_bwd_ms"] = f"{avg_ms:.3f}"

        # V3: Flash backward
        O_v2, L_v2 = mod.flash_forward(Q, K, V)
        min_ms, avg_ms = cuda_timer(lambda: mod.flash_backward(Q, K, V, O_v2, dO, L_v2))
        print(f"  V3 flash bwd:    {avg_ms:8.3f} ms (min {min_ms:.3f})")
        row["v3_flash_bwd_ms"] = f"{avg_ms:.3f}"

        # V4: Optimised backward
        if d in (32, 64, 128):
            O_v4, L_v4 = mod.flash_forward_optimised(Q, K, V)
            min_ms, avg_ms = cuda_timer(lambda: mod.flash_backward_optimised(Q, K, V, O_v4, dO, L_v4))
            print(f"  V4 opt bwd:      {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v4_opt_bwd_ms"] = f"{avg_ms:.3f}"
        else:
            row["v4_opt_bwd_ms"] = "N/A"

        # V5: fp16 Tensor Core backward
        if d in (32, 64, 128):
            Q_h = Q.half()
            K_h = K.half()
            V_h = V.half()
            dO_h = dO.half()
            O_v5, L_v5 = mod.flash_forward_fp16_tc(Q_h, K_h, V_h)
            min_ms, avg_ms = cuda_timer(lambda: mod.flash_backward_fp16_tc(Q_h, K_h, V_h, O_v5, dO_h, L_v5))
            print(f"  V5 fp16 TC bwd:  {avg_ms:8.3f} ms (min {min_ms:.3f})")
            row["v5_fp16tc_bwd_ms"] = f"{avg_ms:.3f}"
        else:
            row["v5_fp16tc_bwd_ms"] = "N/A"

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


def generate_charts():
    """Generate comparison charts from CSV results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping chart generation")
        return

    # Forward pass chart
    fwd_csv = RESULTS_DIR / "forward_benchmark.csv"
    if fwd_csv.exists():
        rows = list(csv.DictReader(open(fwd_csv)))
        labels = [f"N={r['N']}" for r in rows]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(labels))
        width = 0.14

        for i, (key, name) in enumerate([
            ("pytorch_sdpa_ms", "PyTorch SDPA"),
            ("pytorch_naive_ms", "PyTorch Naive"),
            ("v1_naive_ms", "V1 Naive"),
            ("v2_flash_ms", "V2 Flash"),
            ("v4_opt_ms", "V4 Optimised"),
            ("v5_fp16tc_ms", "V5 fp16 TC"),
        ]):
            vals = []
            for r in rows:
                v = r.get(key, "N/A")
                vals.append(float(v) if v != "N/A" else 0)
            ax.bar([xi + i * width for xi in x], vals, width, label=name)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Forward Pass Performance Comparison")
        ax.set_xticks([xi + 2.5 * width for xi in x])
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "forward_benchmark.png", dpi=150)
        print(f"Saved: {RESULTS_DIR / 'forward_benchmark.png'}")

    # Backward pass chart
    bwd_csv = RESULTS_DIR / "backward_benchmark.csv"
    if bwd_csv.exists():
        rows = list(csv.DictReader(open(bwd_csv)))
        labels = [f"N={r['N']}" for r in rows]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(labels))
        width = 0.18

        for i, (key, name) in enumerate([
            ("pytorch_bwd_ms", "PyTorch Backward"),
            ("v3_flash_bwd_ms", "V3 Flash Backward"),
            ("v4_opt_bwd_ms", "V4 Optimised Backward"),
            ("v5_fp16tc_bwd_ms", "V5 fp16 TC Backward"),
        ]):
            vals = []
            for r in rows:
                v = r.get(key, "N/A")
                vals.append(float(v) if v != "N/A" else 0)
            ax.bar([xi + i * width for xi in x], vals, width, label=name)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Backward Pass Performance Comparison")
        ax.set_xticks([xi + 1.5 * width for xi in x])
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "backward_benchmark.png", dpi=150)
        print(f"Saved: {RESULTS_DIR / 'backward_benchmark.png'}")


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
        (4, 8, 2048, 64),
    ]

    fwd_rows = benchmark_forward(mod, configs)
    save_csv(fwd_rows, "forward_benchmark.csv")

    bwd_rows = benchmark_backward(mod, configs)
    save_csv(bwd_rows, "backward_benchmark.csv")

    generate_charts()

    print("\nDone!")


if __name__ == "__main__":
    main()
