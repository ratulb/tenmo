from tensors import Tensor
from shapes import Shape
from strides import Strides
from intlist import IntList
from testing import assert_true, assert_false, assert_raises, assert_equal


fn main() raises:
    test_negative_offset_fails()
    test_large_stride_out_of_bounds()
    test_invalid_offset()
    test_invalid_stride_overflow()
    test_valid_3d_view()
    test_valid_2d_view()
    test_view_offset_out_of_bounds()
    test_view_offset_max_boundary()
    test_view_3d_invalid_strides()
    test_view_3d_valid()
    test_view_2d_strides_overflow()
    test_view_2d_strides_valid()
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

fn test_negative_offset_fails() raises:
    print("test_negative_offset_fails")
    var t = Tensor.d1([1, 2, 3])
    var shape = Shape.of(1)
    var strides = Strides.of(1)
    with assert_raises():
        _ = t.view(shape, strides, offset=-1)  # offset < 0

fn test_large_stride_out_of_bounds() raises:
    print("test_large_stride_out_of_bounds")
    var t = Tensor.d1([0.0] * 10)
    var shape = Shape.of(2, 2)
    var strides = Strides.of(5, 5)
    with assert_raises():
        _ = t.view(shape, strides)  # max_index = 5 + 5 = 10 ❌


fn test_invalid_offset() raises:
    print("test_invalid_offset")
    var t = Tensor[DType.uint8].d1(List[UInt8](0) * 10)
    var shape = Shape.of(2, 2)
    var strides = Strides.of(2, 1)
    var offset = 9
    with assert_raises():
        _ = t.view(shape, strides, offset)  # offset 9 + (1)*2 + (1)*1 = 12 ❌


fn test_invalid_stride_overflow() raises:
    print("test_invalid_stride_overflow")
    var t = Tensor.d1([0.0] * 10)
    var _shape = Shape.of(2, 2)
    var strides = Strides.of(4, 3)  # Last access: 0 + (1)*4 + (1)*3 = 7 ✅
    # Now bump shape to (2, 3)
    shape = Shape.of(2, 3)
    with assert_raises():
        _ = t.view(shape, strides)  # Accesses up to 0 + 4 + 6 = 10 ❌ (== numels)

fn test_valid_3d_view() raises:
    print("test_valid_3d_view")
    var t = Tensor.d1(List[Float32](0) * 24)
    var shape = Shape.of(2, 3, 4)           # Total 24 elements
    var strides = Strides.of(12, 4, 1)      # Standard 3D contiguous row-major layout
    var view = t.view(shape, strides)
    assert_true(view.shape == shape)


fn test_valid_2d_view() raises:
    print("test_valid_2d_view")
    var t = Tensor.d1([0, 1, 2, 3, 4, 5])
    var shape = Shape.of(2, 3)              # 2 rows, 3 cols
    var strides = Strides.of(3, 1)          # Row major: step 3 to next row, step 1 for next col
    var view = t.view(shape, strides)
    assert_true(view.shape == shape)

fn test_view_offset_out_of_bounds() raises:
    print("test_view_offset_out_of_bounds")
    var t = Tensor.d1([0, 1, 2, 3, 4])
    var shape = Shape.of(1)
    var strides = Strides.of(1)
    var offset = 5  # one past the end

    with assert_raises():
        _ = t.view(shape, strides, offset)


fn test_view_offset_max_boundary() raises:
    print("test_view_offset_max_boundary")
    var t = Tensor.d1([0, 1, 2, 3, 4])
    var shape = Shape.of(1)
    var strides = Strides.of(1)
    var offset = 4  # last element

    var v = t.view(shape, strides, offset)
    assert_true(v.shape == shape, "Offset at last element should work")
    assert_true(v[0] == 4, "Value assertion failed")


fn test_view_2d_strides_valid() raises:
    print("test_view_2d_strides_valid")
    var t = Tensor.arange(12).reshape(Shape.of(3, 4))  # 3x4 tensor
    var shape = Shape.of(2, 2)
    var strides = Strides.of(4, 1)  # row-major
    var offset = 4  # starts at row 1, col 0

    var v = t.view(shape, strides, offset)
    assert_true(v.shape == shape, "2D view shape matches")
    v[1,1] = 100
    assert_equal(t[2,1], 100.0, "Base tensor update failed via view")

fn test_view_2d_strides_overflow() raises:
    print("test_view_2d_strides_overflow")
    var t = Tensor.arange(12)
    var shape = Shape.of(3, 3)
    var strides = Strides.of(4, 1)  # row-major
    var offset = 4  # starts at 2nd row, so largest index will be:
                    # 4 + (3-1)*4 + (3-1)*1 = 4 + 8 + 2 = 14 ❌
    with assert_raises():
        _= t.view(shape, strides, offset)

fn test_view_3d_valid() raises:
    print("test_view_3d_valid")
    var t = Tensor.arange(60).reshape(Shape.of(3, 4, 5))  # 3x4x5
    var shape = Shape.of(2, 2, 2)
    var strides = Strides.of(20, 5, 1)  # default strides for 3D contiguous
    var offset = 0

    var v = t.view(shape, strides, offset)
    assert_true(v.shape == shape, "3D valid view shape matches")

fn test_view_3d_invalid_strides() raises:
    print("test_view_3d_invalid_strides")
    var t = Tensor.arange(60).reshape(Shape.of(3, 4, 5))
    var shape = Shape.of(2, 2, 2)
    var _strides = Strides.of(40, 6, 3)  # artificially large strides
    var offset = 10

    # max_index = 10 + (2-1)*40 + (2-1)*6 + (2-1)*3 = 10 + 40 + 6 + 3 = 59 ✅
    # → Still valid! So tweak one stride...

    var strides2 = Strides.of(41, 6, 3)  # now exceeds
    with assert_raises():
        _ = t.view(shape, strides2, offset)


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
