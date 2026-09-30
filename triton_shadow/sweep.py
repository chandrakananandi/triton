"""Run kernels on generated edge inputs under triton_shadow and report which input rules flag them.

    sweep.sweep(launch, inputs, reference=None, tolerances=())   # see README.md
"""
import collections

import torch

import triton_shadow as ts

SCALE_EXPONENTS = range(-148, 125, 4)
INJECT_VALUES = (("nan", float("nan")), ("+inf", float("inf")), ("-inf", float("-inf")), ("+0", 0.0), ("-0", -0.0))
INJECT_FRACTION = 1 / 128
NEAR_CONSTANTS = (-1.0, 1.0)
NEAR_EXPONENTS = (8, 16, 24)  # x := c + 2**-q * x


def edge_families(inputs, seed=0):
    """Yields (rule, label, inputs) with distinct values. Only floating-point tensors are changed."""
    floats = {n: t for n, t in inputs.items() if torch.is_tensor(t) and t.is_floating_point()}
    g = torch.Generator().manual_seed(seed)
    masks = {n: torch.rand(t.shape, generator=g) < INJECT_FRACTION for n, t in floats.items()}

    def candidates():
        yield "baseline", "baseline", floats
        targets = [(n, ) for n in floats] + ([tuple(floats)] if len(floats) > 1 else [])
        for p in SCALE_EXPONENTS:
            for tgt in targets if p else ():
                fam = {n: (t.double() * 2.0**p).to(t.dtype) if n in tgt else t for n, t in floats.items()}
                yield ("scale up" if p > 0 else "scale down"), f"{'+'.join(tgt)} * 2**{p}", fam
        for n, t in floats.items():
            for vname, v in INJECT_VALUES:
                yield f"inject {vname}", f"{n}: {vname}", {**floats, n: torch.where(masks[n], torch.full_like(t, v), t)}
        for n, t in floats.items():
            for c in NEAR_CONSTANTS:
                for q in NEAR_EXPONENTS:
                    near = (c + 2.0**-q * t.double()).to(t.dtype)
                    yield f"near {c:+.0f}", f"{n} := {c:+.0f} + 2**-{q} * {n}", {**floats, n: near}

    seen = set()
    for rule, label, fam in candidates():
        key = b"".join(t.double().numpy().tobytes() for t in fam.values())
        if key not in seen:
            seen.add(key)
            yield rule, label, {**inputs, **fam}


def _run(fn, fam):
    start = len(ts.LAUNCHES)
    out = fn(**fam)
    launches = ts.LAUNCHES[start:]
    del ts.LAUNCHES[start:]  # keep the sweep out of the end-of-run report
    return launches, out if isinstance(out, tuple) else (out, )


def sweep(launch, inputs, reference=None, tolerances=(), seeds=(0, )):
    """Returns ({kernel specialization: {rule: seeds}}, {tolerance: {rule: seeds}}) for the rules that flag."""
    assert reference or not tolerances, "tolerances compare with a reference"
    hits = collections.defaultdict(lambda: collections.defaultdict(set))
    fails = {name: collections.defaultdict(set) for name, _ in tolerances}
    example, specs, n = {}, {}, 0
    for seed in seeds:
        for rule, label, fam in edge_families(inputs(seed) if callable(inputs) else inputs, seed):
            n += 1
            ref, r_out = _run(reference, fam) if reference else (None, None)
            launches, k_out = _run(launch, fam)
            for L in launches:
                key = f"{L.kernel}[{L.consts}]"
                specs.setdefault(key, L.where_defined)
                if ref is None:
                    flagged = L.flagged()
                else:
                    bad, err = L.worst()
                    rbad, rerr = sum(R.worst()[0] for R in ref), max((R.worst()[1] for R in ref), default=0.0)
                    flagged = bad > rbad or (err > ts.ULP and err > 2 * rerr + 1)
                if flagged:
                    hits[key][rule].add(seed)
                    example.setdefault(key, (label, L))
            for name, check in tolerances:
                usable = all(bool((torch.isfinite(r.float()) & (r != 0)).any()) and check(r, r) for r in r_out)
                if usable and not all(map(check, k_out, r_out)):
                    fails[name][rule].add(seed)

    def rules(h):
        return ", ".join(r if len(s) == len(seeds) else f"{r} (seeds {sorted(s)})" for r, s in h.items()) or "none"

    ref_name = f" against {getattr(reference, '__name__', 'reference')}" if reference else ""
    print(f"\nsweep of {getattr(launch, '__name__', 'launch')}{ref_name}: {n} input families, seeds {tuple(seeds)}")
    for key, where_defined in specs.items():
        if key in hits:
            label, L = example[key]
            print(f"  shadow, {key}  {where_defined}\n    flagged by: {rules(hits[key])}\n    first flagged, {label}:")
            ts.describe(L, indent="      ")
        else:
            print(f"  shadow, {key}: not flagged")
    for tol, f in fails.items():
        print(f"  tolerance {tol}: fails under {rules(f)}")
    return {k: dict(v) for k, v in hits.items()}, {k: dict(v) for k, v in fails.items()}


if __name__ == "__main__":
    import examples as ex
    import tolerances as T

    def launcher(kernel, grid, out, *args, **consts):
        """launch(**inputs): launch the kernel on the inputs, in order, then out, then args; return out."""

        def launch(**inputs):
            o = torch.empty_like(out) if torch.is_tensor(out) else torch.empty_like(next(iter(inputs.values())))
            kernel[grid](*inputs.values(), o, *args, **consts)
            return o

        launch.__name__ = f"{kernel.__name__}({', '.join(f'{k}={v}' for k, v in consts.items())})"
        return launch

    def inputs(**shapes):
        g = lambda seed, k: torch.Generator().manual_seed(1000 * seed + k)
        return lambda seed: {n: torch.randn(s, generator=g(seed, k)) for k, (n, s) in enumerate(shapes.items())}

    bf16 = torch.empty(16, 32, dtype=torch.bfloat16)
    MATMUL = [("test_matmul.py:666", T.matmul_general), ("test_matmul.py:738", T.matmul_fp8)]
    SOFTMAX = [("02-fused-softmax.py:190", T.softmax_tutorial), ("test_standard.py:182", T.softmax_standard)]
    mm = dict(M=16, N=32, K=128, BK=64)
    two_pass = launcher(ex.softmax, (2, ), None, 200, BLOCK=256, SUBTRACT_MAX=True)
    cases = [  # (kernel under test, reference, inputs, tolerances of the test that covers the kernel)
        (launcher(ex.swiglu, (4, ), None, BLOCK=256, FORM="mul_add"),
         launcher(ex.swiglu, (4, ), None, BLOCK=256, FORM="fma"), inputs(gelu=1024, linear=1024),
         [("test_swiglu.py:32", T.swiglu)]),
        (launcher(ex.swiglu, (4, ), None, BLOCK=256, FORM="factored"),
         launcher(ex.swiglu, (4, ), None, BLOCK=256, FORM="fma"), inputs(gelu=1024, linear=1024),
         [("test_swiglu.py:32", T.swiglu)]),
        (launcher(ex.epilogue, (4, ), None, BLOCK=256, FUSED=False),
         launcher(ex.epilogue, (4, ), None, BLOCK=256, FUSED=True), inputs(acc=1024, scale=1024, bias=1024), MATMUL),
        (launcher(ex.matmul_bf16_out, (1, ), bf16, **mm, STAGE_FP16=True),
         launcher(ex.matmul_bf16_out, (1, ), bf16, **mm, STAGE_FP16=False), inputs(a=(16, 128), b=(128, 32)), MATMUL),
        (launcher(ex.softmax, (2, ), None, 200, BLOCK=256, SUBTRACT_MAX=False), two_pass, inputs(x=(2, 200)), SOFTMAX),
        (launcher(ex.online_softmax, (2, ), None, 200, BLOCK=64, SPLIT_RESCALE=True), two_pass, inputs(x=(2, 200)),
         SOFTMAX),
        (launcher(ex.online_softmax, (2, ), None, 200, BLOCK=64, SPLIT_RESCALE=False), two_pass, inputs(x=(2, 200)),
         SOFTMAX),
        (launcher(ex.maximum, (4, ), None, BLOCK=256, PROPAGATE_NAN=False),
         launcher(ex.maximum, (4, ), None, BLOCK=256, PROPAGATE_NAN=True), inputs(x=1024, y=1024),
         [("test_core.py:7251", T.propagate_nan)]),
    ]
    for launch, reference, make_inputs, tols in cases:
        sweep(launch, make_inputs, reference=reference, tolerances=tols, seeds=(0, 1, 2))
