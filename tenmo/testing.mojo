"""Test-support helpers for tensor assertions.

`do_assert` / `assert_grad` moved here from `tenmo/common_utils.mojo`
so `common_utils` stops importing `Tensor` — keep the deep
core free of tensor imports.
Only tests use these; nothing in `tenmo/` imports this module.
"""

from std.testing import assert_true

from .tensor import Tensor


# Helper
def do_assert[
    dtype: DType, //
](a: Tensor[dtype], b: Tensor[dtype], msg: String) raises:
    var shape_mismatch = String("{0}: shape mismatch {1} vs {2}")
    var tensors_not_equal = String("{}: values mismatch")
    assert_true(
        a.shape() == b.shape(), shape_mismatch.format(msg, a.shape(), b.shape())
    )
    assert_true((a == b), tensors_not_equal.format(msg))


# Helper
def assert_grad[
    dtype: DType, //
](t: Tensor[dtype], expected: Tensor[dtype], label: String) raises:
    assert_true(
        (t.grad() == expected),
        String("grad assertion failed for " + label),
    )
