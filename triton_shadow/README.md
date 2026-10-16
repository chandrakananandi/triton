# triton_shadow

A Triton kernel can lose precision without any test noticing. In Triton's interpreter, `tl.sum` of
`[1e20, 1, -1e20, 0]` in fp32 returns 0, but the exact sum is 1. triton_shadow finds such losses: it runs kernels in Triton's CPU interpreter, gives every
floating-point value a float64 shadow, and reports which outputs drifted from their shadows and at which source line.
It needs no GPU and changes neither Triton nor the kernels.

For the sum above (`total` in `examples.py`), the report names the reduction:

    total[BLOCK=4]  examples.py:90  (launch 1 of 1 shown)
      OUT (fp32, 1 entries written): ...; 1 NaN/inf/0 where the shadow is not
      where: examples.py:91  reduce(sum_combine)  `tl.store(OUT, tl.sum(tl.load(X + tl.arange(0, BLOCK)), axis=0))`  relative gap 0 -> 1

## Run

From the repository root, with Triton built into `.venv`:

    .venv/bin/python triton_shadow/triton_shadow.py [--ulp N] [--strict] my_script.py [args ...]
    .venv/bin/python triton_shadow/triton_shadow.py [--ulp N] [--strict] -m pytest tests.py [args ...]

The tool flags a launch if an output entry is NaN, infinite or 0 where its shadow, rounded to the output type, is not,
or if the entry is off by more than N units in the last place (ULP; default 64) of the output's largest entry.
`--strict` measures each entry in its own ULP; this mode catches precision lost on small entries but also flags
correct sums whose outputs land near zero. ULP are never finer than fp32's. For each flagged kernel specialization,
the report shows the worst launch.

Tests: `../.venv/bin/python -m pytest test_triton_shadow.py`.

## How the shadow is computed

- Arithmetic, math functions, `dot`, `where` and reshapes repeat the operation on the shadows in float64.
- Casts change only the actual value.
- `tl.sum` adds shadows exactly; other reductions and scans run in float64.
- A load returns the stored shadow while the tensor is alive and its memory is unchanged; otherwise the shadow
  restarts from the loaded value. Shadows of CPU tensors carry across launches.
- "Where" is the first operation after which the gap between actual value and shadow reached half the largest gap
  of any stored value.
- `dot_scaled`, descriptor gathers and atomic read-modify-write restart the shadow, and the report says so.

## Do the kernels' own tests catch these changes?

`examples.py` holds six changed kernels and their correct forms. `sweep.py` derives input families from a kernel's
random inputs by four rules (scale down, scale up, inject NaN/inf/0, near -1 and +1), runs the kernel on each family,
and judges it twice:

- **By the shadow.** A kernel is flagged only where it is worse than a reference on the same inputs: more NaN/inf/0
  entries, or an error above the threshold and above 2 × the reference's error + 1. Without a reference, extreme
  scales flag correct kernels too, because fp32 runs out of range.
- **By the kernel's own test tolerance** (`tolerances.py`), comparing its output with the reference's. A family
  counts only if the reference output has a finite nonzero entry and passes against itself.

```python
import sweep, tolerances
sweep.sweep(launch, inputs)  # launch(**inputs) runs the kernel(s); inputs: dict, or seed -> dict
sweep.sweep(launch, inputs, reference=launch_ref, tolerances=[("test_matmul.py:666", tolerances.matmul_general)])
```

`python triton_shadow.py sweep.py` compares each changed kernel with its correct form on seeds 0–2 (about 50 s). A
listed rule fails on every seed unless marked:

| changed kernel (reference) | test tolerance fails under | shadow | shadow, `--strict` |
|---|---|---|---|
| bf16 matmul staged through fp16 (not staged) | `test_matmul.py:666`: scale down, scale up; `:738`: scale up | scale down, scale up | same |
| softmax without max subtraction (with it) | `02-fused-softmax.py:190`, `test_standard.py:182`: scale up | scale up | same |
| online softmax, rescale `exp(m_old) * exp(-m_new)` (two-pass) | both: scale up | scale up, seed 2 | same |
| `maximum` ignoring NaN (propagating NaN) | `test_core.py:7251`: inject NaN | none | none |
| SwiGLU `s*linear + s` (`fma(s, linear, s)`) | `test_swiglu.py:32`: near -1 | near -1 | every rule |
| epilogue `acc*scale + bias` (FMA) | `test_matmul.py:666`, `:738`: none | none | near -1, near +1; every rule on seed 2 |
| SwiGLU `s*(linear + 1)`, correct online softmax | none | none | none |

**Answer:** the tests' tolerances catch overflow, underflow and mishandled NaN once the inputs reach those ranges,
but miss the epilogue's cancellation. The shadow flags the SwiGLU cancellation by default and the epilogue's only with
`--strict`, and misses a change of formula (the NaN-ignoring `maximum`).

## What does the tool find in Triton's own tests?

We ran the interpreter tests in `python/test/unit/language/test_core.py` (10,985 launches, 1,072 kernel
specializations) under the tool. No test outcome changed, and the run took about 20% longer. `--strict` flagged 90
specializations, mostly sums, scans and matrix products whose outputs land near zero. The default flagged 3:

- **The interpreter converts some fp8 values to fp16 wrongly.** e5m2 subnormals `0x01`–`0x03` should be 1.53e-5,
  3.05e-5 and 4.58e-5 but become 0, 0 and 3.05e-5; e4m3 NaN (`0x7f`, `0xff`) becomes ±480. `test_scaled_dot` still
  passes because its reference kernel and the kernel under test share the conversion.
- **`tl.abs` on fp8e4b8 turns `0x80` into 0.** torch's `float8_e4m3fnuz` reads `0x80` as NaN; the test reads it
  as -0.
- **A false alarm:** a chained `tl.dot` that deliberately downcasts its intermediate.

## Limits

- The tool checks the kernel source as written. It misses what the compiler and GPU change: FMA contraction,
  summation order, approximate `exp`, tensor-core accumulation.
- The interpreter does not run Gluon kernels, returns wrong values for bf16 `tl.dot`, truncates fp32-to-bf16 casts
  (a GPU rounds to nearest), and gives fp32 results that can depend on the CPU (NumPy's float32 `exp` differs
  between x86 and Arm).
- A kernel that computes a different formula agrees with its own shadow.
- The shadow is float64 and so has its own rounding error: the tool ignores mismatches within a relative 2^-50 and measures float64 kernels
  against a shadow of the same precision.
- "Where" considers every value in the launch, including values that never reach the flagged output.
- The threshold is a user choice; an error below it can still matter, and one above it can be harmless.
