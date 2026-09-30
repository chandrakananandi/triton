import os

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import triton_shadow as ts  # noqa: E402

ts.enable(report_at_exit=False)


@pytest.fixture(autouse=True)
def fresh():
    ts.LAUNCHES.clear()
    yield


@triton.jit
def mul_add(X, Y, OUT, BLOCK: tl.constexpr, FUSED: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x, y = tl.load(X + offs), tl.load(Y + offs)
    if FUSED:
        out = tl.fma(x, y, x)
    else:
        out = x * y + x  # line of interest
    tl.store(OUT + offs, out)


@triton.jit
def copy(X, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(X + offs))


@triton.jit
def to_fp16(X, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(X + offs).to(tl.float16).to(tl.float32))


@triton.jit
def total(X, OUT, BLOCK: tl.constexpr):
    tl.store(OUT, tl.sum(tl.load(X + tl.arange(0, BLOCK)), axis=0))


def cancelling(n=64):
    x = torch.randn(n, generator=torch.Generator().manual_seed(0))
    return x, -1 + 2.0**-20 * torch.randn(n, generator=torch.Generator().manual_seed(1))


def test_shadow_is_float64_of_same_formula():
    x, y = cancelling()
    out = torch.empty_like(x)
    mul_add[(1, )](x, y, out, BLOCK=64, FUSED=False)
    (name, dtype, n, (ulp, _, bad)), = ts.LAUNCHES[0].outputs
    exact = x.double() * y.double() + x.double()
    ulp32 = 2.0**(torch.floor(torch.log2(exact.abs())).clamp(min=-126) - 23)
    assert (name, dtype.name, n, bad) == ("OUT", "fp32", 64, 0)
    assert ulp > 1e4 and ulp == pytest.approx(((out.double() - exact).abs() / ulp32).max().item(), rel=1e-12)
    f, line, op, _, _ = ts.LAUNCHES[0].where()
    assert op == "fadd" and "line of interest" in open(f).read().splitlines()[line - 1]


def test_fused_is_accurate():
    x, y = cancelling()
    mul_add[(1, )](x, y, torch.empty_like(x), BLOCK=64, FUSED=True)
    assert ts.LAUNCHES[0].outputs[0][3][0] <= 0.5


def test_cast_keeps_shadow():
    to_fp16[(1, )](torch.tensor([1e5, 2.0**-30, 1.0, 3.0]), torch.empty(4), BLOCK=4)
    L = ts.LAUNCHES[0]
    assert L.outputs[0][3][2] == 2  # 1e5 overflows fp16, 2**-30 flushes to 0
    assert L.where()[2] == "fp_trunc"


def test_sum_is_exact():
    total[(1, )](torch.tensor([1e20, 1.0, -1e20, 0.0]), torch.empty(1), BLOCK=4)
    assert ts.LAUNCHES[0].outputs[0][3][2] == 1  # actual 0, exact 1
    assert ts.LAUNCHES[0].where()[2].startswith("reduce")


def test_shadow_follows_memory_across_launches():
    x, y = cancelling()
    mid, out = torch.empty_like(x), torch.empty_like(x)
    mul_add[(1, )](x, y, mid, BLOCK=64, FUSED=False)
    copy[(1, )](mid, out, BLOCK=64)
    first, second = ts.LAUNCHES
    assert second.outputs[0][3][0] == first.outputs[0][3][0]
    assert second.where()[2] == "masked_load"


def test_host_write_invalidates_shadow():
    x, y = cancelling()
    mid, out = torch.empty_like(x), torch.empty_like(x)
    mul_add[(1, )](x, y, mid, BLOCK=64, FUSED=False)
    mid.copy_(torch.ones_like(mid))
    copy[(1, )](mid, out, BLOCK=64)
    assert ts.LAUNCHES[1].outputs[0][3] == (0.0, 0.0, 0)


@triton.jit
def fma_kernel(X, Y, W, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.fma(tl.load(X + offs), tl.load(Y + offs), tl.load(W + offs)))


def test_float64_shadow_rounding_is_not_a_mismatch():
    # x * y is the float32 overflow midpoint; subtracting 1 gives the largest float32, but float64 cannot hold it
    x, y = torch.tensor([31 * 2.0**59]), torch.tensor([1082401 * 2.0**44])
    out = torch.empty(1)
    fma_kernel[(1, )](x, y, torch.tensor([-1.0]), out, BLOCK=1)
    assert out.item() == torch.finfo(torch.float32).max
    assert ts.LAUNCHES[0].outputs[0][3][2] == 0


def test_freed_tensor_loses_its_shadow():
    import gc
    x, y = cancelling()
    out = torch.empty_like(x)
    mul_add[(1, )](x, y, out, BLOCK=64, FUSED=False)
    base = out.untyped_storage().data_ptr()
    assert ts._region(np.array([base], np.uint64)) is not None
    del out
    gc.collect()
    assert ts._region(np.array([base], np.uint64)) is None


@triton.jit
def loop(X, OUT, BLOCK: tl.constexpr):
    acc = tl.load(X + tl.arange(0, BLOCK))
    for _ in range(4):
        acc = acc * 2.0
        acc = (acc + 1e8) - 1e8
    tl.store(OUT + tl.arange(0, BLOCK), acc)


def test_where_in_a_loop_points_at_the_loss():
    loop[(1, )](torch.randn(64, generator=torch.Generator().manual_seed(0)), torch.empty(64), BLOCK=64)
    assert ts.LAUNCHES[0].where()[2] == "fsub"


@triton.jit
def to_fp8e4b15(X, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(X + offs).to(tl.float8e4b15))


def test_no_kind_mismatch_without_torch_dtype():
    to_fp8e4b15[(1, )](torch.tensor([1e-10, 2.0**-20]), triton.reinterpret(torch.empty(2, dtype=torch.uint8),
                                                                           tl.float8e4b15), BLOCK=2)
    assert ts.LAUNCHES[0].outputs[0][3][2] == 0


def test_sweep_flags_only_where_worse_than_reference():
    import sweep
    g = torch.Generator().manual_seed(0)
    inputs = {"x": torch.randn(64, generator=g), "y": torch.randn(64, generator=g)}

    def mul_then_add(x, y):
        mul_add[(1, )](x, y, torch.empty_like(x), BLOCK=64, FUSED=False)

    def fused(x, y):
        mul_add[(1, )](x, y, torch.empty_like(x), BLOCK=64, FUSED=True)

    hits, _ = sweep.sweep(mul_then_add, inputs, reference=fused)
    assert hits == {"mul_add[BLOCK=64, FUSED=False]": {"near -1": {0}}}  # x * y + x cancels only for y near -1
    assert ts.LAUNCHES == []  # the sweep's launches do not reach the end-of-run report


def test_sweep_applies_tolerance_to_outputs():
    import sweep
    g = torch.Generator().manual_seed(0)
    inputs = {"x": torch.randn(64, generator=g), "y": torch.randn(64, generator=g)}

    def launch(fused):

        def run(x, y):
            out = torch.empty_like(x)
            mul_add[(1, )](x, y, out, BLOCK=64, FUSED=fused)
            return out

        return run

    exact = [("equal", torch.equal)]
    _, fails = sweep.sweep(launch(False), inputs, reference=launch(True), tolerances=exact)
    assert "baseline" in fails["equal"]  # mul then add and fma differ in the last bit on random inputs
    _, fails = sweep.sweep(launch(True), inputs, reference=launch(True), tolerances=exact)
    assert fails == {"equal": {}}
