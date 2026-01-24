"""Test forward pass correctness for all Flash Attention kernel versions."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent

def load_module():
    from torch.utils.cpp_extension import load
    return load(
        name="flash_attn",
        sources=[
            str(PROJ_ROOT / "csrc" / "bindings.cpp"),
            str(PROJ_ROOT / "csrc" / "v1_naive_attention.cu"),
            str(PROJ_ROOT / "csrc" / "v2_flash_forward.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

def reference_attention(Q, K, V):
    """PyTorch reference: scaled dot-product attention with causal mask."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

def test_config(mod, B, nh, N, d, label=""):
    """Test all forward kernels against PyTorch reference for one config."""
    print(f"\n{'='*60}")
    print(f"Config: B={B}, nh={nh}, N={N}, d={d} {label}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    Q = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
    K = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
    V = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
    ref = reference_attention(Q, K, V)

    passed = 0
    total = 0

    # V1
    if N <= 1024:
        total += 1
        out_v1 = mod.naive_attention(Q, K, V)
        err = (out_v1 - ref).abs().max().item()
        ok = torch.allclose(out_v1, ref, atol=1e-3, rtol=1e-3)
        print(f"  V1 naive:      max_err={err:.6f}  [{'PASS' if ok else 'FAIL'}]")
        if ok: passed += 1

    # V2
    total += 1
    out_v2, L_v2 = mod.flash_forward(Q, K, V)
    err = (out_v2 - ref).abs().max().item()
    ok = torch.allclose(out_v2, ref, atol=1e-3, rtol=1e-3)
    print(f"  V2 flash fwd:  max_err={err:.6f}  [{'PASS' if ok else 'FAIL'}]")
    if ok: passed += 1

    # Cross-version consistency
    if N <= 1024:
        ok_12 = torch.allclose(out_v1, out_v2, atol=1e-4, rtol=1e-3)
        print(f"  V1 vs V2:      {'MATCH' if ok_12 else 'MISMATCH'}")

    print(f"  Result: {passed}/{total} passed")
    return passed, total

def main():
    print("Compiling CUDA kernels (first run may take a minute)...")
    mod = load_module()
    print("Compilation complete!\n")

    configs = [
        (1, 1, 64, 32, "tiny"),
        (2, 4, 128, 64, "small"),
        (4, 8, 256, 64, "medium"),
    ]

    total_passed = 0
    total_tests = 0

    for B, nh, N, d, label in configs:
        p, t = test_config(mod, B, nh, N, d, label)
        total_passed += p
        total_tests += t

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print(f"{'='*60}")

    sys.exit(0 if total_passed == total_tests else 1)

if __name__ == "__main__":
    main()
