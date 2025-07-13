from tensors import Tensor
from shapes import Shape
from strides import Strides
from testing import assert_true, assert_false, assert_raises


fn main() raises:
    test_view_default_strides()
    test_view_invalid_offset()
    test_view_invalid_shape()
    test_view_invalid_strides_rank()
    test_view_with_strides_basic()
    test_view_negative_stride_fails_safely()
    test_view_offset_slice()
    test_view_reuse_data_storage()
    test_view_identity()
    test_view_stride_bounds_overflow()


fn test_view_default_strides() raises:
    print("test_view_default_strides")
    var t = Tensor.d2([[1, 2], [3, 4]])  # Shape: (2, 2)
    var v = t.view(Shape.of(4))  # Reshape to (4,)
    assert_true(v.shape == Shape.of(4), "Shape mismatch")
    assert_true(v.is_contiguous(), "Expected default view to be contiguous")


fn test_view_invalid_offset() raises:
    print("test_view_invalid_offset")
    var t = Tensor.d2([[1, 2], [3, 4]])
    with assert_raises():
        _ = t.view(Shape.of(2), offset=5)  # Too large offset


fn test_view_invalid_shape() raises:
    print("test_view_invalid_shape")
    var t = Tensor.d2([[1, 2], [3, 4]])
    with assert_raises():
        _ = t.view(Shape.of(5))  # More elements than base tensor


fn test_view_invalid_strides_rank() raises:
    print("test_view_invalid_strides_rank")
    var t = Tensor.d2([[1, 2], [3, 4]])
    with assert_raises():
        _ = t.view(Shape.of(2, 2), Strides.of(1))  # Mismatched rank


fn test_view_with_strides_basic() raises:
    print("test_view_with_strides_basic")
    var t = Tensor.d2([[1, 2], [3, 4]])
    var strides = Strides.of(1, 2)  # Would result in skip reading
    var v = t.view(Shape.of(2, 2), strides)
    assert_true(v.shape == Shape.of(2, 2), "Strided view shape mismatch")
    assert_false(
        v.is_contiguous(), "Expected strided view to be non-contiguous"
    )


fn test_view_negative_stride_fails_safely() raises:
    print("test_view_negative_stride_fails_safely")
    var t = Tensor.d2([[1, 2], [3, 4]])
    var strides = Strides.of(-1, 1)
    with assert_raises():
        _ = t.view(Shape.of(2, 2), strides)


fn test_view_offset_slice() raises:
    print("test_view_offset_slice")
    var t = Tensor.d1([1, 2, 3, 4, 5, 6])
    var shape = Shape.of(2)
    var v = t.view(shape, offset=3)
    assert_true(v.shape == shape, "View shape mismatch")
    assert_true(v.offset == 3, "Offset mismatch")


fn test_view_reuse_data_storage() raises:
    print("test_view_reuse_data_storage")
    var t = Tensor.d1([1, 2, 3, 4])
    var v = t.view(Shape.of(2, 2))
    # Change data in original, should reflect in view if indexing works
    t[0] = 99
    assert_true(v[0, 0] == 99, "View didn't reflect change in base tensor")


fn test_view_identity() raises:
    print("test_view_identity")
    var t = Tensor.d2([[1, 2], [3, 4]])
    var v = t.view(t.shape)
    assert_true(v.is_contiguous(), "Identity view should be contiguous")
    assert_true(v.offset == 0, "Offset should be zero")


fn test_view_stride_bounds_overflow() raises:
    print("test_view_stride_bounds_overflow")
    # max_index = offset + ∑ (shape[i] - 1) * strides[i]
    # For single dimension max_index = offset + (shape[0] - 1) * strides[0]
    var t = Tensor.d1([1, 2, 3, 4])
    var shape = Shape.of(2)
    var strides = Strides.of(3)
    # with assert_raises():
    # _ = t.view(shape, strides)  # 0 + 3*(2-1) = 3 ✅ but add one more -> 4 ➝ overflow
    assert_true(t.view(shape, strides)[1] == 4, "get item assertion failed")
