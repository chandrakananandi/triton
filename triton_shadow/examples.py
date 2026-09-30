"""Kernels for triton_shadow: a changed and a correct form of each, selected by a constexpr switch."""
import torch
import triton
import triton.language as tl


@triton.jit
def swiglu(G, L, OUT, BLOCK: tl.constexpr, FORM: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gelu, linear = tl.load(G + offs), tl.load(L + offs)
    s = gelu / (1 + tl.exp(-0.5 * gelu))
    if FORM == "fma":
        out = tl.fma(s, linear, s)
    elif FORM == "factored":
        out = s * (linear + 1)
    else:
        out = s * linear + s
    tl.store(OUT + offs, out)


@triton.jit
def epilogue(ACC, SCALE, BIAS, OUT, BLOCK: tl.constexpr, FUSED: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc, scale, bias = tl.load(ACC + offs), tl.load(SCALE + offs), tl.load(BIAS + offs)
    if FUSED:
        out = tl.fma(acc, scale, bias)
    else:
        out = acc * scale + bias
    tl.store(OUT + offs, out)


@triton.jit
def matmul_bf16_out(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
                    STAGE_FP16: tl.constexpr):
    rm, rn = tl.arange(0, M), tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        acc = tl.dot(tl.load(A + rm[:, None] * K + rk[None, :]), tl.load(B + rk[:, None] * N + rn[None, :]), acc)
    if STAGE_FP16:
        out = acc.to(tl.float16).to(tl.bfloat16)
    else:
        out = acc.to(tl.bfloat16)
    tl.store(C + rm[:, None] * N + rn[None, :], out)


@triton.jit
def softmax(X, OUT, n_cols, BLOCK: tl.constexpr, SUBTRACT_MAX: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + tl.program_id(0) * n_cols + offs, mask=mask, other=-float("inf"))
    if SUBTRACT_MAX:
        x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(OUT + tl.program_id(0) * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def online_softmax(X, OUT, n_cols, BLOCK: tl.constexpr, SPLIT_RESCALE: tl.constexpr):
    row = tl.program_id(0) * n_cols
    m = tl.full((1, ), float("-inf"), tl.float32)
    l = tl.zeros((1, ), tl.float32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(X + row + offs, mask=offs < n_cols, other=-float("inf"))
        m_new = tl.maximum(m, tl.max(x, axis=0))
        if SPLIT_RESCALE:
            l = l * (tl.exp(m) * tl.exp(-m_new)) + tl.sum(tl.exp(x - m_new), axis=0)
        else:
            l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(X + row + offs, mask=offs < n_cols, other=-float("inf"))
        tl.store(OUT + row + offs, tl.exp(x - m) / l, mask=offs < n_cols)


@triton.jit
def maximum(X, Y, OUT, BLOCK: tl.constexpr, PROPAGATE_NAN: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x, y = tl.load(X + offs), tl.load(Y + offs)
    if PROPAGATE_NAN:
        out = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    else:
        out = tl.maximum(x, y)
    tl.store(OUT + offs, out)


@triton.jit
def total(X, OUT, BLOCK: tl.constexpr):
    tl.store(OUT, tl.sum(tl.load(X + tl.arange(0, BLOCK)), axis=0))


def randn(*shape, seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def run(correct):
    gelu, linear = randn(128, 1024), randn(128, 1024, seed=1)
    swiglu[(128, )](gelu, linear, torch.empty_like(gelu), BLOCK=1024, FORM="fma" if correct else "mul_add")

    acc, scale, bias = randn(64, 128), randn(64, 128, seed=1), randn(64, 128, seed=2)
    epilogue[(8, )](acc, scale, bias, torch.empty_like(acc), BLOCK=1024, FUSED=correct)

    a, b = randn(64, 1024), randn(1024, 128, seed=1)  # fp32: the interpreter's tl.dot is wrong for bf16 inputs
    for s in (1.0, 2.0**9, 2.0**-25):  # random, fp16 overflow, fp16 underflow
        c = torch.empty(64, 128, dtype=torch.bfloat16)
        matmul_bf16_out[(1, )](a * s, b, c, M=64, N=128, K=1024, BK=64, STAGE_FP16=not correct)

    x = randn(4, 1000)
    for s in (1.0, 2.0**7):  # random, exp overflow in fp32 but not in float64
        softmax[(4, )](x * s, torch.empty_like(x), 1000, BLOCK=1024, SUBTRACT_MAX=correct)
        online_softmax[(4, )](x * s, torch.empty_like(x), 1000, BLOCK=256, SPLIT_RESCALE=not correct)

    if not correct:
        total[(1, )](torch.tensor([1e20, 1.0, -1e20, 0.0]), torch.empty(1), BLOCK=4)


if __name__ == "__main__":
    run(correct=False)
    run(correct=True)
