# triton_shadow

Runs Triton kernels in Triton's interpreter (on the CPU) with a float64 shadow of every floating-point value, and
reports which outputs lost precision and at which source line. No GPU, and no change to Triton or to the kernels.

## Run

From the repository root, with Triton built into `.venv`:

    .venv/bin/python triton_shadow/triton_shadow.py [--ulp N] [--strict] my_script.py [args ...]
    .venv/bin/python triton_shadow/triton_shadow.py [--ulp N] [--strict] -m pytest tests.py [args ...]

A launch is flagged if an output entry is NaN, infinite or 0 where its shadow, rounded to the output type, is not,
or is off by more than N (default 64) ULP of the output's largest entry. With `--strict` the ULP are those of each
entry: that catches precision lost on small entries, but also flags correct sums and matrix products whose outputs
land near zero. ULP are never finer than fp32's.

For each flagged kernel specialization the report shows its worst launch. From `examples.py`:

    online_softmax[BLOCK=256, SPLIT_RESCALE=True]  examples.py:59  (launch 2 of 2 shown)
      OUT (fp32, 4000 entries written): ...; 4000 NaN/inf/0 where the shadow is not
      where: examples.py:68  exp  `l = l * (tl.exp(m) * tl.exp(-m_new)) + tl.sum(tl.exp(x - m_new), axis=0)`  relative gap 0 -> inf

    total[BLOCK=4]  examples.py:90  (launch 1 of 1 shown)
      OUT (fp32, 1 entries written): ...; 1 NaN/inf/0 where the shadow is not
      where: examples.py:91  reduce(sum_combine)  `tl.store(OUT, tl.sum(tl.load(X + tl.arange(0, BLOCK)), axis=0))`  relative gap 0 -> 1

Of the six changed kernels in `examples.py`, four are flagged by default and all six with `--strict`; their five
correct forms are flagged in neither mode.

## Rules

- Arithmetic, math functions, `dot`, `where` and reshapes: the same operation on the shadows, in float64.
- Casts change only the actual value.
- `tl.sum` adds shadows exactly; other reductions and scans run in float64.
- A load returns the shadow stored at that address while the tensor is alive and the memory still holds what was
  stored; otherwise the shadow starts from the loaded value. For CPU tensors shadows carry across launches.
- "Where" is the operation after which the gap between actual value and shadow first reached half the largest gap of
  a stored value.
- Not tracked (the shadow restarts, and the report says so): `dot_scaled`, descriptor gathers, atomic
  read-modify-write.

## Sweeps

`sweep.py` runs a kernel on input families generated from its random inputs: power-of-two scaling over the fp32
range, NaN/inf/0 injection, and values near -1 and +1.

```python
import sweep, tolerances
sweep.sweep(launch, inputs)  # launch(**inputs) launches the kernel(s); inputs: dict, or seed -> dict
sweep.sweep(launch, inputs, reference=launch_ref, tolerances=[("test_matmul.py:666", tolerances.matmul_general)])
```

Two judges:

- **The shadow.** Without a reference, correct kernels are flagged too at extreme scales, where any fp32 computation
  runs out of range. With a reference run on the same inputs, a kernel is flagged only where it is worse: more
  NaN/inf/0 entries, or an error above the threshold and above 2 * (the reference's error) + 1.
- **The tolerance of the kernel's own test** (`tolerances.py`), applied to the kernel's output against the
  reference's. A family counts only if the reference output has a finite nonzero entry and passes against itself.

`python triton_shadow.py sweep.py` runs each changed kernel of `examples.py` against its correct form, seeds 0-2
(about 50 s). Rules listed flag on every seed unless stated:

| changed kernel (reference) | test tolerance fails under | shadow | shadow, `--strict` |
|---|---|---|---|
| bf16 matmul staged through fp16 (not staged) | `test_matmul.py:666`: scale down, scale up; `:738`: scale up | scale down, scale up | same |
| softmax without max subtraction (with it) | `02-fused-softmax.py:190`, `test_standard.py:182`: scale up | scale up | same |
| online softmax, rescale `exp(m_old) * exp(-m_new)` (two-pass) | both: scale up | scale up, seed 2 | same |
| `maximum` ignoring NaN (propagating NaN) | `test_core.py:7251`: inject NaN | none | none |
| SwiGLU `s*linear + s` (`fma(s, linear, s)`) | `test_swiglu.py:32`: near -1 | near -1 | every rule |
| epilogue `acc*scale + bias` (FMA) | `test_matmul.py:666`, `:738`: none | none | near -1, near +1; every rule on seed 2 |
| SwiGLU `s*(linear + 1)`, correct online softmax | none | none | none |

The tests' own tolerances catch changes that overflow, underflow or mishandle NaN once the inputs reach those
ranges, and miss the ones that lose precision through cancellation. The shadow does not see a change of formula
(the NaN-ignoring `maximum`).

## Found in Triton's own tests

Over the interpreter tests in `python/test/unit/language/test_core.py` (10,985 launches, 1,072 kernel
specializations) the tool changes no test outcome and adds about 20% to the run time (112 s to 133 s). 90
specializations are flagged with `--strict`, mostly sums, scans and matrix products whose outputs land near zero;
3 by default:

- **The interpreter's fp8 to fp16 conversion is wrong** for e5m2 subnormals `0x01`-`0x03` and their negatives
  (0, 0, 3.05e-5 instead of 1.53e-5, 3.05e-5, 4.58e-5). In `test_scaled_dot`, the e5m2-to-fp16 launches of
  `mxfp_upcast_kernel`, which builds the test's reference, are flagged at `x_f8.to(to_type)`; the test passes because
  the kernel under test shares the conversion. A direct check of all 256 bit patterns also shows e4m3 NaN
  (`0x7f`, `0xff`) converted to +-480.
- **`tl.abs` on fp8e4b8 turns `0x80` into 0.** torch's `float8_e4m3fnuz` reads `0x80` as NaN; the test reads it as -0.
- **A false alarm:** a chained `tl.dot` that deliberately casts its intermediate to a lower precision.

The interpreter's `tl.dot` on bf16 inputs also returns wrong values instead of an error; the tests skip bf16 in the
interpreter.

## Files

`triton_shadow.py` (the tool), `sweep.py`, `tolerances.py`, `examples.py`, `test_triton_shadow.py`
(`../.venv/bin/python -m pytest test_triton_shadow.py`).

## Limits

- It runs what the kernel source says. It does not see the compiler or GPU: FMA contraction, summation order,
  approximate `exp`, tensor-core accumulation.
- Gluon kernels do not run in the interpreter, and bf16 `tl.dot` is wrong there.
- A kernel that computes a different formula agrees with its own shadow.
- The shadow is float64, not exact; mismatches within a relative 2^-50 of it are ignored, and float64 kernels are
  measured against a shadow of the same precision.
- "Where" looks at all values of the launch, not only those that reach the flagged output.
- The threshold is a choice, not a rule for what is harmful.
