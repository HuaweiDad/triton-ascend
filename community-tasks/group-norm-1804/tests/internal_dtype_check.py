"""Internal dtype robustness check (NOT the acceptance test).

For fp32: direct comparison against torch.nn.GroupNorm at 1e-4 (acceptance-level).
For bf16/fp16: compares BOTH implementations against an fp32 oracle and requires
liger's error to stay within the same order as torch's own low-precision error
(1-ulp rounding differences in bf16 are unavoidable and not meaningful).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernel"))

from liger_kernel.transformers.group_norm import LigerGroupNorm
from liger_kernel.utils import infer_device

device = infer_device()

SHAPES = [
    (1, 1, 1, 3),
    (1, 32, 32, 4),
    (16, 32, 1, 4096),
    (2, 63, 21, 2163),
    (16, 48, 12, 8192),
    (4, 64, 16, 512),
]

FLOOR = {torch.float32: 1e-4, torch.bfloat16: 2e-2, torch.float16: 5e-3}


def run(dtype):
    floor = FLOOR[dtype]
    for b, c, g, h in SHAPES:
        torch.manual_seed(0)
        base32 = torch.randn(b, c, h, dtype=torch.float32, device=device)
        grad32 = torch.randn(b, c, h, dtype=torch.float32, device=device)

        # fp32 oracle
        ox = base32.clone().requires_grad_(True)
        oracle = torch.nn.GroupNorm(num_channels=c, num_groups=g, eps=1e-6).to(device)
        oy = oracle(ox)
        oy.backward(grad32)

        def run_impl(dtype_, use_liger):
            x = base32.to(dtype_).clone().requires_grad_(True)
            if use_liger:
                layer = LigerGroupNorm(c, g, eps=1e-6).to(dtype_).to(device)
            else:
                layer = torch.nn.GroupNorm(num_channels=c, num_groups=g, eps=1e-6).to(dtype_).to(device)
            with torch.no_grad():
                layer.weight.copy_(oracle.weight)
                layer.bias.copy_(oracle.bias)
            y = layer(x)
            y.backward(grad32.to(dtype_))
            return y, x.grad, layer.weight.grad, layer.bias.grad

        if dtype == torch.float32:
            ref = run_impl(dtype, False)
            got = run_impl(dtype, True)
            for name, a, t in zip(("out", "dx", "dw", "db"), got, ref):
                assert torch.allclose(a, t, atol=1e-4, rtol=1e-4), (
                    f"{name} mismatch {dtype} {(b, c, g, h)}: {(a - t).abs().max().item()}"
                )
        else:
            oracle_parts = (oy, ox.grad, oracle.weight.grad, oracle.bias.grad)
            err_t = [ (t.float() - o).abs().max().item() for t, o in zip(run_impl(dtype, False), oracle_parts) ]
            err_l = [ (l.float() - o).abs().max().item() for l, o in zip(run_impl(dtype, True), oracle_parts) ]
            for name, el, et in zip(("out", "dx", "dw", "db"), err_l, err_t):
                limit = max(et * 2.0, floor)
                assert el <= limit, (
                    f"{name} error too large {dtype} {(b, c, g, h)}: liger={el:.5f} torch={et:.5f} limit={limit:.5f}"
                )
        print(f"OK {str(dtype):15s} {(b, c, g, h)}")


if __name__ == "__main__":
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        run(dt)
    print("ALL DTYPE CHECKS PASSED")
