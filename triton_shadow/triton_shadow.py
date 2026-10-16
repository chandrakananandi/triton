"""Shadow execution: run Triton kernels in the interpreter with a float64 shadow of every value, and report where
outputs lose precision.

    python triton_shadow.py [--ulp N] [--strict] script.py [args ...]
    python triton_shadow.py [--ulp N] [--strict] -m pytest [args ...]
"""
import atexit
import collections
import linecache
import math
import os
import runpy
import sys
import weakref

import numpy as np
import torch

ULP = 64.0
STRICT = False  # measure errors in ULP of each entry instead of the largest entry
LAUNCHES = []
MAX_REGIONS = 256

CASTS = {"cast_impl", "create_fp_to_fp", "create_fp_ext", "create_fp_trunc", "create_si_to_fp", "create_ui_to_fp"}
UNTRACKED = {"create_dot_scaled", "create_descriptor_gather", "create_atomic_rmw", "create_atomic_cas"}
DELEGATING = {"create_load", "create_store", "create_atomic_load", "create_atomic_store", "create_descriptor_store",
              "create_descriptor_scatter"}  # they call instrumented operations themselves
TORCH_DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64", "fp8e5": "float8_e5m2",
                "fp8e4nv": "float8_e4m3fn", "fp8e4b8": "float8_e4m3fnuz", "fp8e5b16": "float8_e5m2fnuz"}

_I = tl = _TRITON_DIR = None
_depth = 0  # > 0 inside an instrumented operation: operations it calls run uninstrumented
_launch = None
_memory = collections.OrderedDict()  # storage base address -> _Region, most recently used last


class _Region:
    """Shadow of one storage while a tensor using it is alive."""

    def __init__(self, base, nbytes):
        self.base, self.nbytes, self.itemsize, self.owners = base, nbytes, None, []

    def alive(self):
        return any(o() is not None for o in self.owners)

    def reset(self, itemsize):
        n = self.nbytes // itemsize
        self.itemsize, self.shadow, self.actual = itemsize, np.zeros(n), np.zeros(n)
        self.valid, self.written = np.zeros(n, bool), np.zeros(n, bool)


class _Launch:

    def __init__(self, fn):
        self.kernel = fn.__name__
        self.where_defined = f"{os.path.relpath(fn.__code__.co_filename)}:{fn.__code__.co_firstlineno}"
        self.consts = ""
        self.peak = self.stored = 0.0  # largest gap of any value, of any stored value
        self.records = []  # (file, line, operation, input gap, output gap) each time the peak gap grew
        self.untracked = collections.Counter()
        self.outputs = []  # (argument, dtype, entries written, (ULP of the entry, ULP of the largest entry, bad))

    def finish(self):
        for r in _memory.values():
            if r.itemsize and r.written.any():
                w = r.written
                self.outputs.append((r.name, r.dtype, int(w.sum()), _errors(r.actual[w], r.shadow[w], r.dtype)))
                w[:] = False

    def worst(self):
        """(NaN/inf/0 entries, largest error in the unit that is flagged on)."""
        errs = [e for *_, e in self.outputs]
        return sum(e[2] for e in errs), max((e[0] if STRICT else e[1] for e in errs), default=0.0)

    def flagged(self):
        bad, err = self.worst()
        return bad > 0 or err > ULP

    def where(self):
        """The operation after which the gap first reached half the largest gap of a stored value."""
        return next((r for r in self.records if r[4] >= self.stored / 2), None) if self.stored else None


def _is_float(h):
    # comparison results carry the operands' float type, so the data is checked too; bf16/fp8 are stored as uint
    if not isinstance(h, _I.TensorHandle) or isinstance(h.dtype.scalar, tl.pointer_type) or not h.dtype.scalar.is_floating():
        return False
    k = h.data.dtype
    return k.kind == "f" or (k.kind == "u" and k.itemsize * 8 == h.dtype.scalar.primitive_bitwidth)


def _f64(h):
    d = h.data
    if d.dtype.kind == "f":
        return d.astype(np.float64)
    name = TORCH_DTYPES.get(h.dtype.scalar.name)  # torch, since the interpreter decodes fp8 inf/NaN as finite
    if name:
        return torch.from_numpy(np.array(d).view(f"int{8 * d.itemsize}")).view(getattr(torch, name)).double().numpy()
    f32 = _I._convert_float(np.ascontiguousarray(d), h.dtype.scalar, tl.float32, None).view(np.float32)
    return f32.reshape(d.shape).astype(np.float64)


def _shadow(h):
    s = getattr(h, "_shadow", None)
    return s[1] if s is not None and s[0] is h.data else _f64(h)


def _attach(h, s):
    h._shadow = (h.data, np.broadcast_to(np.asarray(s, dtype=np.float64), h.data.shape))


def _to_f64(x):
    if _is_float(x):
        return _I.TensorHandle(np.array(_shadow(x)), tl.float64)
    return type(x)(_to_f64(v) for v in x) if isinstance(x, (list, tuple)) else x


def _flat(x):
    return [y for v in x for y in _flat(v)] if isinstance(x, (list, tuple)) else [x]


def _gap(h):
    """Largest relative difference between a value and its shadow; inf where only one of them is finite."""
    a, s = _f64(h), _shadow(h)
    with np.errstate(all="ignore"):
        g = np.where(np.isfinite(a) & np.isfinite(s), np.abs(a - s) / np.abs(s), np.inf)
    g = np.where((a == s) | (np.isnan(a) & np.isnan(s)), 0.0, g)
    return float(g.max()) if g.size else 0.0


def _caller():
    f = sys._getframe(1)
    while f is not None and (f.f_code.co_filename.startswith(_TRITON_DIR) or f.f_code.co_filename == __file__):
        f = f.f_back
    return (f.f_code.co_filename, f.f_lineno) if f is not None else ("?", 0)


def _record(op, ins, outs):
    g = max((_gap(h) for h in outs if _is_float(h)), default=0.0)
    if op == "store":
        _launch.stored = max(_launch.stored, g)
    elif g > _launch.peak:
        g_in = max((_gap(h) for h in _flat(ins) if _is_float(h)), default=0.0)
        if g > g_in:
            _launch.records.append((*_caller(), op, g_in, g))
        _launch.peak = g


def _register(name, dev, hst):
    if isinstance(hst, tuple):
        for i, (d, h) in enumerate(zip(dev, hst)):
            _register(f"{name}[{i}]", d, h)
    elif hasattr(hst, "block_shape"):  # TensorDescriptor
        _register(name, dev.base, hst.base)
    elif hasattr(hst, "data_ptr"):
        dev, hst = _I._unwrap_tensor(dev), _I._unwrap_tensor(hst)
        st = hst.untyped_storage()
        base, nbytes = st.data_ptr(), st.nbytes()
        # a CPU argument shares its storage with the caller's tensor, so its shadow lives as long as that tensor
        owner = dev if dev.untyped_storage().data_ptr() == base else hst
        r = _memory.pop(base, None)
        if r is None or r.nbytes != nbytes or not r.alive():
            r = _Region(base, nbytes)
        if all(o() is not owner for o in r.owners):
            r.owners.append(weakref.ref(owner))
        r.name, _memory[base] = name, r
        while len(_memory) > MAX_REGIONS:
            _memory.popitem(last=False)


def _region(addr):
    if addr.size:
        lo, hi = int(addr.min()), int(addr.max())
        for r in reversed(_memory.values()):
            if r.base <= lo and hi < r.base + r.nbytes and r.alive():
                return r
    return None


def _lanes(ptrs, mask):
    return np.ones(ptrs.data.shape, bool) if mask is None else np.broadcast_to(mask.data, ptrs.data.shape).astype(bool)


def _load(ptrs, mask, other, out):
    a = _f64(out)
    s, on = a.copy(), _lanes(ptrs, mask)
    if other is not None:
        s[~on] = np.broadcast_to(_shadow(other), s.shape)[~on]
    addr = ptrs.data[on]
    r = _region(addr)
    if r is not None and r.itemsize == out.data.itemsize:
        idx = ((addr - np.uint64(r.base)) // np.uint64(r.itemsize)).astype(np.int64)
        prev, now = r.actual[idx], a[on]
        unchanged = r.valid[idx] & ((prev == now) | (np.isnan(prev) & np.isnan(now)))  # memory still holds the store
        s[on] = np.where(unchanged, r.shadow[idx], now)
    return s


def _store(ptrs, value, mask):
    on = _lanes(ptrs, mask)
    addr = ptrs.data[on]
    r = _region(addr)
    if r is None:
        return
    if r.itemsize != value.data.itemsize:
        r.reset(value.data.itemsize)
    idx = ((addr - np.uint64(r.base)) // np.uint64(r.itemsize)).astype(np.int64)
    r.shadow[idx] = np.broadcast_to(_shadow(value), on.shape)[on]
    r.actual[idx] = np.broadcast_to(_f64(value), on.shape)[on]
    r.valid[idx] = r.written[idx] = True
    r.dtype = value.dtype.scalar
    _record("store", (), (value, ))


def _shadow_op(name, orig, builder, args, kwargs, out):
    if name == "create_masked_store":
        if _is_float(args[1]):
            _store(args[0], args[1], args[2])
        return
    outs = out if isinstance(out, tuple) else (out, )
    if not any(_is_float(o) for o in outs) or name == "create_bitcast":
        return
    if name in UNTRACKED:
        _launch.untracked[name[7:]] += 1
        return
    if name in CASTS:
        shadows = [_shadow(args[0]) if _is_float(args[0]) else args[0].data.astype(np.float64)]
    elif name == "create_masked_load":
        shadows = [_load(args[0], args[1], args[2], out)]
    elif name == "create_descriptor_load":
        shadows = [_load(*args[0].materialize_pointers(args[1]), None, out)]
    else:
        try:
            res = orig(builder, *_to_f64(args), **kwargs)
        except Exception:
            _launch.untracked[name[7:]] += 1
            return
        shadows = [r.data if isinstance(r, _I.TensorHandle) else None for r in (res if isinstance(res, tuple) else (res, ))]
    for o, s in zip(outs, shadows):
        if _is_float(o) and s is not None:
            _attach(o, s)
    _record(name.replace("create_", "").replace("cast_impl", "cast"), args, outs)


def _instrument(name, orig):

    def method(self, *args, **kwargs):
        global _depth
        if _depth or _launch is None:
            return orig(self, *args, **kwargs)
        _depth += 1
        try:
            out = orig(self, *args, **kwargs)
            _shadow_op(name, orig, self, args, kwargs, out)
            return out
        finally:
            _depth -= 1

    return method


def _exact_sum(a, axis, keepdims):

    def fsum(v):
        try:
            return math.fsum(v)
        except (OverflowError, ValueError):  # inf - inf, or an overflowing sum
            return float(np.sum(v))

    if axis is None:
        a, axis = a.reshape(-1), 0
    out = np.apply_along_axis(fsum, axis, a)
    return np.expand_dims(out, axis) if keepdims else out


def _instrument_reduce(orig):

    def f64_tensor(t):
        if not (isinstance(t, tl.tensor) and _is_float(t.handle)):
            return t
        ty = tl.block_type(tl.float64, list(t.shape)) if t.type.is_block() else tl.float64
        return tl.tensor(_I.TensorHandle(np.array(_shadow(t.handle)), tl.float64), ty)

    def tensors(x):
        return [o for o in (x if isinstance(x, (list, tuple)) else [x]) if isinstance(o, tl.tensor)]

    def apply_impl(self, input):
        global _depth
        if _depth or _launch is None:
            return orig(self, input)
        _depth += 1
        try:
            out = orig(self, input)
            outs = [o.handle for o in tensors(out)]
            if any(_is_float(h) for h in outs):
                if isinstance(self, _I.ReduceOps) and self.combine_fn is tl.standard._sum_combine:
                    shadows = [_exact_sum(_shadow(input[0].handle), self.axis, self.keep_dims)]
                else:
                    shadows = [o.handle.data for o in tensors(orig(self, tuple(f64_tensor(t) for t in input)))]
                for h, s in zip(outs, shadows):
                    if _is_float(h):
                        _attach(h, s)
                kind = "reduce" if isinstance(self, _I.ReduceOps) else "scan"
                _record(f"{kind}({getattr(self.combine_fn, '__name__', 'combine').strip('_')})",
                        [t.handle for t in input], outs)
            return out
        finally:
            _depth -= 1

    return apply_impl


def _instrument_lang_tensor(orig):

    def patch(tensor, scope):
        orig(tensor, scope)
        transpose = tensor.__dict__["T"].fget

        def T(self):
            out = transpose(self)
            if _is_float(self.handle):
                _attach(out.handle, np.transpose(_shadow(self.handle)))
            return out

        scope.set_attr(tensor, "T", property(T))

    return patch


def _instrument_launch(orig_call, orig_init):

    def init_args(self, args_dev, kwargs):
        args_hst, kwargs_hst = orig_init(self, args_dev, kwargs)
        for name, dev, hst in [*zip(self.arg_names, args_dev, args_hst), *((k, kwargs[k], v) for k, v in kwargs_hst.items())]:
            _register(name, dev, hst)
        named = [*zip(self.arg_names, args_hst), *kwargs_hst.items()]
        _launch.consts = ", ".join(f"{n}={a}" for n, a in named if n in self.constexprs)
        return args_hst, kwargs_hst

    def call(self, *args, **kwargs):
        global _launch
        _launch = _Launch(self.fn)
        try:
            return orig_call(self, *args, **kwargs)
        finally:
            done, _launch = _launch, None
            done.finish()
            LAUNCHES.append(done)

    return call, init_args


def enable(report_at_exit=True):
    global _I, tl, _TRITON_DIR
    import triton
    if not triton.knobs.runtime.interpret:
        raise RuntimeError("triton_shadow needs Triton's interpreter: set TRITON_INTERPRET=1 before importing Triton")
    from triton.runtime import interpreter
    _I, tl, _TRITON_DIR = interpreter, triton.language, os.path.dirname(triton.__file__) + os.sep
    B = _I.InterpreterBuilder
    if getattr(B, "_triton_shadow", False):
        return
    for name in dir(B):
        if (name.startswith("create_") or name == "cast_impl") and name not in DELEGATING:
            setattr(B, name, _instrument(name, getattr(B, name)))
    B._triton_shadow = True
    for ops in (_I.ReduceOps, _I.ScanOps):
        ops.apply_impl = _instrument_reduce(ops.apply_impl)
    _I._patch_lang_tensor = _instrument_lang_tensor(_I._patch_lang_tensor)
    G = _I.GridExecutor
    G.__call__, G._init_args_hst = _instrument_launch(G.__call__, G._init_args_hst)
    if report_at_exit:
        atexit.register(report)


def shadow_of(t):
    """float64 shadow of a contiguous CPU tensor, as a kernel last stored it; its own value where nothing was stored."""
    a = t.double().numpy().reshape(-1)
    r = _memory.get(t.untyped_storage().data_ptr())
    if r is None or not r.alive() or r.itemsize != t.element_size():
        return a.reshape(t.shape)
    idx = t.storage_offset() + np.arange(t.numel())
    unchanged = r.valid[idx] & ((r.actual[idx] == a) | (np.isnan(r.actual[idx]) & np.isnan(a)))
    return np.where(unchanged, r.shadow[idx], a).reshape(t.shape)


def _kind(x):
    return np.select([np.isnan(x), x == np.inf, x == -np.inf, x == 0], [0, 1, 2, 3], 4)


def _errors(o, s, dtype):
    """(max error in ULP of each entry, in ULP of the largest entry, entries whose kind (NaN, +-inf, 0, other) differs
    from the shadow rounded to the output type). ULP are never finer than fp32's."""
    name = TORCH_DTYPES.get(dtype.name)
    if name:

        def rounded(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(getattr(torch, name)).double().numpy()

        # the shadow is float64, not exact: an entry is bad only if it differs from the shadow nudged either way
        rr = rounded(s)
        bad = (_kind(o) != _kind(rounded(s * (1 - 2.0**-50)))) & (_kind(o) != _kind(rounded(s * (1 + 2.0**-50))))
    else:  # no torch type to round to: compare NaN only
        rr, bad = s, np.isnan(o) != np.isnan(s)
    ok = np.isfinite(o) & np.isfinite(rr) & ~bad
    if not ok.any():
        return 0.0, 0.0, int(bad.sum())
    m, emin = min(dtype.fp_mantissa_width, 23), max(1 - dtype.exponent_bias, -126)

    def ulp(x):
        with np.errstate(divide="ignore"):
            return 2.0**(np.maximum(np.floor(np.log2(np.abs(x))), emin) - m)

    diff = np.abs(o - s)[ok]
    return float((diff / ulp(s[ok])).max()), float((diff / ulp(np.abs(s[np.isfinite(rr)]).max())).max()), int(bad.sum())


def _fmt(x):
    return f"{x:.3g}" if x == 0 or x >= 0.01 else "<0.01"


def describe(L, out=None, indent="  "):
    out = out or sys.stdout
    for name, dtype, n, (ulp, scaled, bad) in L.outputs:
        print(f"{indent}{name} ({dtype.name}, {n} entries written): error up to {_fmt(scaled)} ULP of the largest entry, "
              f"{_fmt(ulp)} ULP of the entry itself; {bad} NaN/inf/0 where the shadow is not", file=out)
    w = L.where()
    if w:
        f, line, op, g_in, g_out = w
        src = linecache.getline(f, line).strip()
        print(f"{indent}where: {os.path.relpath(f)}:{line}  {op}  {f'`{src}`  ' if src else ''}relative gap "
              f"{g_in:.2g} -> {g_out:.2g}", file=out)
    if L.untracked:
        print(f"{indent}not tracked (shadow restarts from the actual value): {dict(L.untracked)}", file=out)


def report(file=None):
    out = file or sys.stdout
    groups = collections.OrderedDict()
    for L in LAUNCHES:
        groups.setdefault((L.kernel, L.consts, L.where_defined), []).append(L)
    flagged = [k for k, Ls in groups.items() if any(L.flagged() for L in Ls)]
    print(f"\ntriton_shadow: {len(LAUNCHES)} launches, {len(groups)} kernel specializations, {len(flagged)} flagged",
          file=out)
    for key in flagged:
        Ls = groups[key]
        worst = max((L for L in Ls if L.flagged()), key=_Launch.worst)
        print(f"\n{key[0]}[{key[1]}]  {key[2]}  (launch {Ls.index(worst) + 1} of {len(Ls)} shown)", file=out)
        describe(worst, out)


def main(argv):
    global ULP, STRICT
    while argv and argv[0] in ("--strict", "--ulp") or argv and argv[0].startswith("--ulp="):
        opt = argv.pop(0)
        if opt == "--strict":
            STRICT = True
        else:
            ULP = float(opt.split("=", 1)[1] if "=" in opt else argv.pop(0))
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    os.environ["TRITON_INTERPRET"] = "1"
    sys.modules.setdefault("triton_shadow", sys.modules[__name__])
    enable()
    if argv[0] == "-m":
        sys.argv, sys.path[0] = argv[1:], os.getcwd()
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv, sys.path[0] = argv, os.path.dirname(os.path.abspath(argv[0]))
        runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
