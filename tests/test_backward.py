"""Test backward pass correctness for Flash Attention V3 kernel."""

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
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
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
    atol = 5e-2
    rtol = 1e-2

    for B, nh, N, d, label in configs:
        print(f"\nConfig: B={B}, nh={nh}, N={N}, d={d} {label}")
        torch.manual_seed(42)
        Q = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, nh, N, d, device="cuda", dtype=torch.float32)
        ref_O, ref_dQ, ref_dK, ref_dV = reference_gradients(Q, K, V)
        dO = torch.ones_like(ref_O)

        total_tests += 1
        O_v2, L_v2 = mod.flash_forward(Q, K, V)
        dQ_v3, dK_v3, dV_v3 = mod.flash_backward(Q, K, V, O_v2, dO, L_v2)

        err_dQ = (dQ_v3 - ref_dQ).abs().max().item()
        err_dK = (dK_v3 - ref_dK).abs().max().item()
        err_dV = (dV_v3 - ref_dV).abs().max().item()

        ok = (torch.allclose(dQ_v3, ref_dQ, atol=atol, rtol=rtol) and
              torch.allclose(dK_v3, ref_dK, atol=atol, rtol=rtol) and
              torch.allclose(dV_v3, ref_dV, atol=atol, rtol=rtol))

        print(f"  V3 flash bwd:")
        print(f"    dQ max_err={err_dQ:.6f}")
        print(f"    dK max_err={err_dK:.6f}")
        print(f"    dV max_err={err_dV:.6f}")
        print(f"    [{'PASS' if ok else 'FAIL'}]")
        if ok: total_passed += 1

    print(f"\nTOTAL: {total_passed}/{total_tests} passed")
    sys.exit(0 if total_passed == total_tests else 1)

if __name__ == "__main__":
    main()
