"""Tolerances of Triton's own tests, ported to CPU as pass/fail functions f(kernel_out, reference_out).
Line numbers refer to triton-lang/triton at commit a3c982d24."""
import torch


def _close(actual, expected, rtol, atol):
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def kernels_assert_close(ref, tri, maxtol=2e-2, rmstol=4e-3):
    """triton_kernels.testing.assert_close (python/triton_kernels/triton_kernels/testing.py:26-87) for non-empty
    tensors of 2- and 4-byte float dtypes. Same argument order as the original."""
    ref, tri = ref.to(torch.float32), tri.to(torch.float32)
    inf_ref, inf_tri = torch.isinf(ref), torch.isinf(tri)
    if ref.shape != tri.shape or not torch.equal(inf_ref, inf_tri):
        return False
    refn, trin = torch.where(inf_ref, 0, ref), torch.where(inf_tri, 0, tri)
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn, trin = refn * multiplier, trin * multiplier
    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps
    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err, rms_err = torch.max(rel_err).item(), torch.sqrt(torch.square(rel_err).mean()).item()
    return max_err <= maxtol and rms_err <= rmstol


def matmul_general(k, r):
    """python/triton_kernels/tests/test_matmul.py:666, assert_close(ref_y, tri_y)."""
    return kernels_assert_close(r, k)


def matmul_fp8(k, r):
    """python/triton_kernels/tests/test_matmul.py:738."""
    return _close(k.float(), r.float(), rtol=torch.finfo(r.dtype).eps, atol=1e-4)


def softmax_tutorial(k, r):
    """python/tutorials/02-fused-softmax.py:190."""
    return bool(torch.allclose(k, r))


def softmax_standard(k, r):
    """python/test/unit/language/test_standard.py:182."""
    return _close(k, r, rtol=1e-5, atol=1e-6)


def swiglu(k, r):
    """python/triton_kernels/tests/test_swiglu.py:32, assert_close(tri_y, ref_y): the kernel output comes first."""
    return kernels_assert_close(k, r)


def propagate_nan(k, r):
    """python/test/unit/language/test_core.py:7251, test_propagate_nan: whether each result is NaN."""
    return torch.equal(torch.isnan(k), torch.isnan(r))
