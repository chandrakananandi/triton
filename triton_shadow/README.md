# triton_shadow

A Triton kernel can lose precision without any test noticing: in Triton's interpreter, `tl.sum` of
`[1e20, 1, -1e20, 0]` in fp32 returns 0 (exact: 1). triton_shadow runs kernels in the CPU interpreter, gives every
floating-point value a float64 shadow, and reports which outputs drifted from their shadows and at which source
line. It needs no GPU and changes neither Triton nor the kernels.

## Run

From the repository root, with Triton built into `.venv`:

    .venv/bin/python triton_shadow/triton_shadow.py [--ulp N] [--strict] my_script.py [args ...]
    .venv/bin/python triton_shadow/triton_shadow.py [--ulp N] [--strict] -m pytest tests.py [args ...]

A launch is flagged if an output entry is NaN, infinite or 0 where its shadow (rounded to the output type) is not,
or if it is off by more than N ULP (default 64) of the output's largest entry. `--strict` measures each entry in its
own ULP, which also catches small entries but flags correct sums that land near zero. ULP are never finer than
fp32's. The report shows the worst launch of each flagged kernel specialization and the first operation after which
the gap reached half the largest gap of any stored value.

Tests: `../.venv/bin/python -m pytest test_triton_shadow.py`.

## How the shadow is computed

- Arithmetic, math functions, `dot`, `where` and reshapes repeat the operation on the shadows in float64.
- Casts change only the actual value.
- `tl.sum` adds shadows exactly; other reductions and scans run in float64.
- A load returns the stored shadow while the tensor is alive and its memory is unchanged; otherwise the shadow
  restarts from the loaded value. Shadows of CPU tensors carry across launches.
- `dot_scaled`, descriptor gathers and atomic read-modify-write restart the shadow, and the report says so.

## Do the kernels' own tests catch these changes?

`examples.py` holds six changed kernels and their correct forms. `sweep.py` derives input families from a kernel's
random inputs (scale down, scale up, inject NaN/inf/0, near ±1), runs the kernel on each, and judges it by the
shadow against a reference on the same inputs (more NaN/inf/0 entries, or error above the threshold and above
2 × reference error + 1) and by the kernel's own test tolerance (`tolerances.py`).

```python
sweep.sweep(launch, inputs, reference=launch_ref, tolerances=[("test_matmul.py:666", tolerances.matmul_general)])
```

`python triton_shadow.py sweep.py` runs seeds 0–2 (about 50 s). The test tolerances catch overflow, underflow and
mishandled NaN once the inputs reach those ranges, but miss the epilogue's `acc*scale + bias` cancellation. The
shadow flags the SwiGLU `s*linear + s` cancellation by default and the epilogue's only with `--strict`, and misses
a change of formula (`maximum` ignoring NaN).

## What does the tool find in Triton's own tests?

Running `python/test/unit/language/test_core.py` (10,985 launches, 1,072 specializations) under the tool changed no
test outcome and took about 20% longer. `--strict` flagged 90 specializations, mostly reductions near zero. The
default flagged 3:

- **The interpreter converts some fp8 values to fp16 wrongly.** e5m2 subnormals `0x01`–`0x03` become 0, 0 and
  3.05e-5 (exact: 1.53e-5, 3.05e-5, 4.58e-5); e4m3 NaN (`0x7f`, `0xff`) becomes ±480. `test_scaled_dot` still
  passes because both kernels share the conversion.
- **`tl.abs` on fp8e4b8 turns `0x80` into 0.** torch reads `0x80` as NaN; the test reads it as -0.
- **A false alarm:** a chained `tl.dot` that deliberately downcasts its intermediate.

## Limits

- The tool checks the kernel source as written and misses what the compiler and GPU change: FMA contraction,
  summation order, approximate `exp`, tensor-core accumulation.
- The interpreter skips Gluon kernels, returns wrong values for bf16 `tl.dot`, truncates fp32-to-bf16 casts, and
  gives CPU-dependent fp32 results.
- A kernel that computes a different formula agrees with its own shadow.
- The shadow has its own rounding error: mismatches within a relative 2^-50 are ignored, and float64 kernels are
  measured against a shadow of the same precision.
- The threshold is a user choice; an error below it can still matter, and one above it can be harmless.
