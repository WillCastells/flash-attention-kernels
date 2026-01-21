"""Test forward pass correctness for Flash Attention kernels."""

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
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

def main():
    print("Compiling CUDA kernels...")
    mod = load_module()
    print("Compilation complete!\n")

    configs = [
        (1, 1, 64, 32, "tiny"),
        (2, 4, 128, 64, "small"),
    ]

    total_passed = 0
    total_tests = 0

    for B, nh, N, d, label in configs:
        print(f"\nConfig: B={B}, nh={nh}, N={N}, d={d} {label}")
        torch.manual_seed(42)
        Q = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        ref = reference_attention(Q, K, V)

        # V1
        if N <= 1024:
            total_tests += 1
            out_v1 = mod.naive_attention(Q, K, V)
            err = (out_v1 - ref).abs().max().item()
            ok = torch.allclose(out_v1, ref, atol=1e-3, rtol=1e-3)
            print(f"  V1 naive:     max_err={err:.6f} [{'PASS' if ok else 'FAIL'}]")
            if ok: total_passed += 1

        # V2
        total_tests += 1
        out_v2, L_v2 = mod.flash_forward(Q, K, V)
        err = (out_v2 - ref).abs().max().item()
        ok = torch.allclose(out_v2, ref, atol=1e-3, rtol=1e-3)
        print(f"  V2 flash fwd: max_err={err:.6f} [{'PASS' if ok else 'FAIL'}]")
        if ok: total_passed += 1

        # Cross-version check
        if N <= 1024:
            ok_cross = torch.allclose(out_v1, out_v2, atol=1e-4, rtol=1e-3)
            print(f"  V1 vs V2:     {'MATCH' if ok_cross else 'MISMATCH'}")

    print(f"\nTOTAL: {total_passed}/{total_tests} passed")
    sys.exit(0 if total_passed == total_tests else 1)

if __name__ == "__main__":
    main()
