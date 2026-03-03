"""Test forward pass correctness for all Flash Attention kernel versions."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Use the correct Python/PyTorch path
PROJ_ROOT = Path(__file__).resolve().parent.parent

def load_module():
    """JIT-compile and load the flash attention CUDA module."""
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

    # V1: Naive attention
    if N <= 1024:
        total += 1
        try:
            out_v1 = mod.naive_attention(Q, K, V)
            err = (out_v1 - ref).abs().max().item()
            ok = torch.allclose(out_v1, ref, atol=1e-4, rtol=1e-3)
            status = "PASS" if ok else "FAIL"
            print(f"  V1 naive:      max_err={err:.6f}  [{status}]")
            if ok:
                passed += 1
        except Exception as e:
            print(f"  V1 naive:      ERROR - {e}")

    # V2: Flash forward
    total += 1
    try:
        out_v2, L_v2 = mod.flash_forward(Q, K, V)
        err = (out_v2 - ref).abs().max().item()
        ok = torch.allclose(out_v2, ref, atol=1e-4, rtol=1e-3)
        status = "PASS" if ok else "FAIL"
        print(f"  V2 flash fwd:  max_err={err:.6f}  [{status}]")
        if ok:
            passed += 1
    except Exception as e:
        print(f"  V2 flash fwd:  ERROR - {e}")

    # V4: Optimised forward
    if d in (32, 64, 128):
        total += 1
        try:
            out_v4, L_v4 = mod.flash_forward_optimised(Q, K, V)
            err = (out_v4 - ref).abs().max().item()
            ok = torch.allclose(out_v4, ref, atol=1e-4, rtol=1e-3)
            status = "PASS" if ok else "FAIL"
            print(f"  V4 opt fwd:    max_err={err:.6f}  [{status}]")
            if ok:
                passed += 1
        except Exception as e:
            print(f"  V4 opt fwd:    ERROR - {e}")

    # V5: fp16 Tensor Core forward
    if d in (32, 64, 128):
        total += 1
        try:
            Q_h = Q.half()
            K_h = K.half()
            V_h = V.half()
            out_v5, L_v5 = mod.flash_forward_fp16_tc(Q_h, K_h, V_h)
            out_v5_f32 = out_v5.float()
            err = (out_v5_f32 - ref).abs().max().item()
            # Looser tolerance for fp16 (half precision ~1e-3 representable)
            ok = torch.allclose(out_v5_f32, ref, atol=1e-2, rtol=1e-2)
            status = "PASS" if ok else "FAIL"
            print(f"  V5 fp16 TC:    max_err={err:.6f}  [{status}]")
            if ok:
                passed += 1
        except Exception as e:
            print(f"  V5 fp16 TC:    ERROR - {e}")
            import traceback
            traceback.print_exc()

    # Cross-version consistency
    if N <= 1024:
        try:
            ok_12 = torch.allclose(out_v1, out_v2, atol=1e-4, rtol=1e-3)
            print(f"  V1 vs V2:      {'MATCH' if ok_12 else 'MISMATCH'}")
        except:
            pass

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
        (4, 8, 512, 64, "large"),
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
