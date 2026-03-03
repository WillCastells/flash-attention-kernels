"""Test backward pass correctness for Flash Attention kernels."""

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
            str(PROJ_ROOT / "csrc" / "v3_flash_backward.cu"),
            str(PROJ_ROOT / "csrc" / "v4_flash_optimised.cu"),
            str(PROJ_ROOT / "csrc" / "v5_flash_fp16_tensorcore.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def reference_gradients(Q, K, V):
    """Compute reference gradients using PyTorch autograd."""
    Q = Q.clone().detach().requires_grad_(True)
    K = K.clone().detach().requires_grad_(True)
    V = V.clone().detach().requires_grad_(True)

    O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    loss = O.sum()
    loss.backward()

    return O.detach(), Q.grad.detach(), K.grad.detach(), V.grad.detach()


def test_backward_config(mod, B, nh, N, d, label=""):
    """Test backward kernels against PyTorch autograd for one config."""
    print(f"\n{'='*60}")
    print(f"Backward Config: B={B}, nh={nh}, N={N}, d={d} {label}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    Q = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
    K = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
    V = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)

    ref_O, ref_dQ, ref_dK, ref_dV = reference_gradients(Q, K, V)

    # Use dO = ones (matches loss = O.sum())
    dO = torch.ones_like(ref_O)

    passed = 0
    total = 0

    # fp32 backward tolerances
    atol = 1e-3
    rtol = 1e-2

    # V3: Flash backward
    total += 1
    try:
        # First get forward output and logsumexp
        O_v2, L_v2 = mod.flash_forward(Q, K, V)
        dQ_v3, dK_v3, dV_v3 = mod.flash_backward(Q, K, V, O_v2, dO, L_v2)

        err_dQ = (dQ_v3 - ref_dQ).abs().max().item()
        err_dK = (dK_v3 - ref_dK).abs().max().item()
        err_dV = (dV_v3 - ref_dV).abs().max().item()

        ok_dQ = torch.allclose(dQ_v3, ref_dQ, atol=atol, rtol=rtol)
        ok_dK = torch.allclose(dK_v3, ref_dK, atol=atol, rtol=rtol)
        ok_dV = torch.allclose(dV_v3, ref_dV, atol=atol, rtol=rtol)

        ok = ok_dQ and ok_dK and ok_dV
        status = "PASS" if ok else "FAIL"

        print(f"  V3 flash bwd:")
        print(f"    dQ max_err={err_dQ:.6f}  [{'PASS' if ok_dQ else 'FAIL'}]")
        print(f"    dK max_err={err_dK:.6f}  [{'PASS' if ok_dK else 'FAIL'}]")
        print(f"    dV max_err={err_dV:.6f}  [{'PASS' if ok_dV else 'FAIL'}]")
        print(f"    Overall: [{status}]")
        if ok:
            passed += 1
    except Exception as e:
        print(f"  V3 flash bwd:  ERROR - {e}")
        import traceback
        traceback.print_exc()

    # V4: Optimised backward
    if d in (32, 64, 128):
        total += 1
        try:
            O_v4, L_v4 = mod.flash_forward_optimised(Q, K, V)
            dQ_v4, dK_v4, dV_v4 = mod.flash_backward_optimised(Q, K, V, O_v4, dO, L_v4)

            err_dQ = (dQ_v4 - ref_dQ).abs().max().item()
            err_dK = (dK_v4 - ref_dK).abs().max().item()
            err_dV = (dV_v4 - ref_dV).abs().max().item()

            ok_dQ = torch.allclose(dQ_v4, ref_dQ, atol=atol, rtol=rtol)
            ok_dK = torch.allclose(dK_v4, ref_dK, atol=atol, rtol=rtol)
            ok_dV = torch.allclose(dV_v4, ref_dV, atol=atol, rtol=rtol)

            ok = ok_dQ and ok_dK and ok_dV
            status = "PASS" if ok else "FAIL"

            print(f"  V4 opt bwd:")
            print(f"    dQ max_err={err_dQ:.6f}  [{'PASS' if ok_dQ else 'FAIL'}]")
            print(f"    dK max_err={err_dK:.6f}  [{'PASS' if ok_dK else 'FAIL'}]")
            print(f"    dV max_err={err_dV:.6f}  [{'PASS' if ok_dV else 'FAIL'}]")
            print(f"    Overall: [{status}]")
            if ok:
                passed += 1
        except Exception as e:
            print(f"  V4 opt bwd:    ERROR - {e}")
            import traceback
            traceback.print_exc()

    # V5: fp16 Tensor Core backward
    if d in (32, 64, 128):
        total += 1
        try:
            Q_h = Q.half()
            K_h = K.half()
            V_h = V.half()
            dO_h = dO.half()

            O_v5, L_v5 = mod.flash_forward_fp16_tc(Q_h, K_h, V_h)
            dQ_v5, dK_v5, dV_v5 = mod.flash_backward_fp16_tc(Q_h, K_h, V_h, O_v5, dO_h, L_v5)

            # Compare in fp32
            dQ_v5_f = dQ_v5.float()
            dK_v5_f = dK_v5.float()
            dV_v5_f = dV_v5.float()

            err_dQ = (dQ_v5_f - ref_dQ).abs().max().item()
            err_dK = (dK_v5_f - ref_dK).abs().max().item()
            err_dV = (dV_v5_f - ref_dV).abs().max().item()

            # fp16 tolerance — observed max errors ~3e-3
            fp16_atol = 1e-2
            fp16_rtol = 1e-2

            ok_dQ = torch.allclose(dQ_v5_f, ref_dQ, atol=fp16_atol, rtol=fp16_rtol)
            ok_dK = torch.allclose(dK_v5_f, ref_dK, atol=fp16_atol, rtol=fp16_rtol)
            ok_dV = torch.allclose(dV_v5_f, ref_dV, atol=fp16_atol, rtol=fp16_rtol)

            ok = ok_dQ and ok_dK and ok_dV
            status = "PASS" if ok else "FAIL"

            print(f"  V5 fp16 TC bwd:")
            print(f"    dQ max_err={err_dQ:.6f}  [{'PASS' if ok_dQ else 'FAIL'}]")
            print(f"    dK max_err={err_dK:.6f}  [{'PASS' if ok_dK else 'FAIL'}]")
            print(f"    dV max_err={err_dV:.6f}  [{'PASS' if ok_dV else 'FAIL'}]")
            print(f"    Overall: [{status}]")
            if ok:
                passed += 1
        except Exception as e:
            print(f"  V5 fp16 TC bwd: ERROR - {e}")
            import traceback
            traceback.print_exc()

    print(f"  Result: {passed}/{total} passed")
    return passed, total


def main():
    print("Compiling CUDA kernels...")
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
        p, t = test_backward_config(mod, B, nh, N, d, label)
        total_passed += p
        total_tests += t

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print(f"{'='*60}")

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
