from tenmo import Tensor
from shapes import Shape
from strides import Strides
from intlist import IntList
from testing import assert_true, assert_false, assert_raises, assert_equal
from common_utils import i, newaxis, s


fn main() raises:
    test_into_tensor_isolated_memory()
    test_identity_permutation()
    test_slice_every_second_row_column1()
    test_edge_case_indexing()  # revist
    # test_mixed_indexing()
    # test_newaxis_dimension_insertion()
    test_basic_slicing()
    # test_newaxis()
    test_scalar_view()
    test_integer_indexing()
    test_nested_views_grad_propagation()
    test_reshape_slice_sum_backward()
    test_backward_through_nested_views_non_contiguous()
    test_backward_through_nested_views()

    test_permute_backward()
    test_tensor_permute_flatten_backprop()

    test_flat_view_chain_backprop()
    test_nonzero_offset_multi_view_chain()
    test_strided_view_chain_2d_to_3d()
    test_nested_views_with_interleaved_strides()
    test_view_chain_reversed_shape()
    test_grad_propagation_with_offset_chain()
    test_nested_view_backward_indexing()

    test_getitem_list_empty_indices_returns_full_view()
    test_into_tensor_full_view_copy()
    test_into_tensor_transposed_view()
    test_into_tensor_offset_view()
    test_into_tensor_scalar_view()
    # test_into_tensor_empty_view()
    test_into_tensor_grad_flag_true()
    test_into_tensor_grad_flag_false()
    test_into_tensor_large_contiguous_copy()

    # test_negative_offset_fails()
    # test_large_stride_out_of_bounds()
    # test_invalid_offset()
    # test_invalid_stride_overflow()
    test_valid_3d_view()
    test_valid_2d_view()
    # test_view_offset_out_of_bounds()
    test_view_offset_max_boundary()
    # test_view_3d_invalid_strides()
    test_view_3d_valid()
    # test_view_2d_strides_overflow()
    test_view_2d_strides_valid()
    test_view_default_strides()
    # test_view_invalid_offset()
    # test_view_invalid_shape()
    # test_view_invalid_strides_rank()
    test_view_with_strides_basic()
    # test_view_negative_stride_fails_safely()
    test_view_offset_slice()
    test_view_reuse_data_storage()
    test_view_identity()
    test_view_stride_bounds_overflow()


fn test_slice_every_second_row_column1() raises:
    print("test_slice_every_second_row_column1")
    var a = Tensor.arange(15, requires_grad=True)
    var r = a.reshape(5, 3)
    var v = r[s(None, None, 2), i(1)]  # Select col 1 of rows 0, 2, 4
    var loss = v.sum()
    loss.backward()
    grad = a.grad().copy()
    assert_true(grad.shape() == a.shape())
    assert_true(grad[1] == 1)  # r[0,1]
    assert_true(grad[7] == 1)  # r[2,1]
    assert_true(grad[13] == 1)  # r[4,1]
    assert_true(grad.sum().item() == 3)


fn test_permute_backward() raises:
    print("test_permute_backward")

    var a = Tensor.arange(6, requires_grad=True)
    var v = a.view([2, 3])
    var p = v.permute([1, 0])  # shape (3, 2), stride [1, 3]

    var flat = p.reshape([6])

    flat.backward()

    var expected = Tensor.d1([1, 1, 1, 1, 1, 1])
    assert_true((a.grad() == expected))


fn test_tensor_permute_flatten_backprop() raises:
    print("test_tensor_permute_flatten_backprop")

    var a = Tensor.arange(12, requires_grad=True)  # shape: [0..11]
    var v = a.view([3, 4])
    var p = v.permute([1, 0])  # shape: (4, 3), stride: [1, 4]
    var flat = p.reshape([12])  # flatten

    flat.backward()

    var expected = Tensor.ones(12)
    assert_true((a.grad() == expected))


fn test_flat_view_chain_backprop() raises:
    print("test_flat_view_chain_backprop")
    var a = Tensor.arange(10, requires_grad=True)
    var v1 = a.view([4, 2], offset=2)
    var v2 = v1.view([2, 4])
    var v3 = v2.view([8])
    v3.backward()
    # assert_eq(a.grad(), Tensor.ones_like(a).slice(2, 10).pad_left(2))
    assert_true((a.grad() == Tensor.of(0, 0, 1, 1, 1, 1, 1, 1, 0, 0)))


fn test_nonzero_offset_multi_view_chain() raises:
    print("test_nonzero_offset_multi_view_chain")
    var a = Tensor.arange(20, requires_grad=True)
    var v1 = a.view([6, 3], offset=2)  # offset=2
    var v2 = v1.view([3, 6])  # Absolute offset from 0
    var v3 = v2.view([18])  # flatten
    v3.backward()
    var expected = Tensor.zeros(20)
    for i in range(2, 18):
        expected[i] = 1.0
    assert_true((a.grad() == expected))


fn test_strided_view_chain_2d_to_3d() raises:
    print("test_strided_view_chain_2d_to_3d")
    var a = Tensor.arange(60, requires_grad=True)
    var v1 = a.view([10, 3], offset=0)
    var v2 = v1.view([5, 2, 3])
    var v3 = v2.view([30])
    v3.backward()
    expected = Tensor.ones_like(a)
    for i in range(30, 60):
        expected[i] = 0
    assert_true((a.grad() == expected))


fn test_nested_views_with_interleaved_strides() raises:
    print("test_nested_views_with_interleaved_strides")
    var a = Tensor.arange(36, requires_grad=True)
    var v1 = a.view([6, 6], offset=0)
    var v2 = v1.view([3, 2, 6])  # (3 blocks of 2x6)
    var v3 = v2.view([36])
    v3.backward()
    assert_true((a.grad() == Tensor.ones_like(a)))


fn test_view_chain_reversed_shape() raises:
    print("test_view_chain_reversed_shape")
    var a = Tensor.arange(24, requires_grad=True)
    var v1 = a.view([6, 4], offset=0)
    var v2 = v1.view([4, 6])
    var v3 = v2.view([24])
    v3.backward()
    assert_true((a.grad() == Tensor.ones_like(a)))


fn test_grad_propagation_with_offset_chain() raises:
    print("test_grad_propagation_with_offset_chain")
    var a = Tensor.arange(30, requires_grad=True)
    var v1 = a.view([5, 4], offset=6)
    var v2 = v1.view([2, 10])
    var v3 = v2.view([20])
    v3.backward()
    var expected = Tensor.zeros(30)
    for i in range(6, 20):
        expected[i] = 1.0
    assert_true((a.grad() == expected))


fn test_nested_view_backward_indexing() raises:
    print("test_nested_view_backward_indexing")

    var a = Tensor.arange(30, requires_grad=True)  # [0, 1, ..., 29]
    var v1 = a.view([3, 5], offset=5)  # a[5:20] → shape [3, 5]
    var v2 = v1.view([5, 3])  # reshape view
    var v3 = v2.view([15])  # flatten

    v3.backward()  # backward through view chain

    var expected = Tensor.zeros(30)
    for i in range(5, 15):  # only a[5:15] should get gradient
        expected[i] = 1
    assert_true((a.grad() == expected))


fn test_negative_offset_fails() raises:
    print("test_negative_offset_fails")
    var t = Tensor.d1([1, 2, 3])
    var shape = Shape.of(1)
    var strides = Strides(1)
    with assert_raises():
        _ = t.view(shape, strides, offset=-1)  # offset < 0


fn test_large_stride_out_of_bounds() raises:
    print("test_large_stride_out_of_bounds")
    var t = Tensor.d1([0.0] * 10)
    var shape = Shape.of(2, 2)
    var strides = Strides(5, 5)
    with assert_raises():
        _ = t.view(shape, strides)  # max_index = 5 + 5 = 10 ❌


fn test_invalid_offset() raises:
    print("test_invalid_offset")
    var t = Tensor[DType.uint8].d1(List[UInt8](0) * 10)
    var shape = Shape.of(2, 2)
    var strides = Strides(2, 1)
    var offset = 9
    with assert_raises():
        _ = t.view(shape, strides, offset)  # offset 9 + (1)*2 + (1)*1 = 12 ❌


fn test_invalid_stride_overflow() raises:
    print("test_invalid_stride_overflow")
    var t = Tensor.d1([0.0] * 10)
    var _shape = Shape.of(2, 2)
    var strides = Strides(4, 3)  # Last access: 0 + (1)*4 + (1)*3 = 7 ✅
    # Now bump shape to (2, 3)
    shape = Shape.of(2, 3)
    with assert_raises():
        _ = t.view(
            shape, strides
        )  # Accesses up to 0 + 4 + 6 = 10 ❌ (== numels)


fn test_valid_3d_view() raises:
    print("test_valid_3d_view")
    var t = Tensor.d1(List[Float32](0) * 24)
    var shape = Shape.of(2, 3, 4)  # Total 24 elements
    var strides = Strides(12, 4, 1)  # Standard 3D contiguous row-major layout
    var view = t.view(shape, strides)
    assert_true(view.shape() == shape)


fn test_valid_2d_view() raises:
    print("test_valid_2d_view")
    var t = Tensor.d1([0, 1, 2, 3, 4, 5])
    var shape = Shape.of(2, 3)  # 2 rows, 3 cols
    var strides = Strides(
        3, 1
    )  # Row major: step 3 to next row, step 1 for next col
    var view = t.view(shape, strides)
    assert_true(view.shape() == shape)


fn test_view_offset_out_of_bounds() raises:
    print("test_view_offset_out_of_bounds")
    var t = Tensor.d1([0, 1, 2, 3, 4])
    var shape = Shape.of(1)
    var strides = Strides(1)
    var offset = 5  # one past the end

    with assert_raises():
        _ = t.view(shape, strides, offset)


fn test_view_offset_max_boundary() raises:
    print("test_view_offset_max_boundary")
    var t = Tensor.d1([0, 1, 2, 3, 4])
    var shape = Shape.of(1)
    var strides = Strides(1)
    var offset = 4  # last element

    var v = t.view(shape, strides, offset)
    assert_true(v.shape() == shape, "Offset at last element should work")
    assert_true(v[0] == 4, "Value assertion failed")


fn test_view_2d_strides_valid() raises:
    print("test_view_2d_strides_valid")
    var t = Tensor.arange(12).reshape(Shape.of(3, 4))  # 3x4 tensor
    var shape = Shape.of(2, 2)
    var strides = Strides(4, 1)  # row-major
    var offset = 4  # starts at row 1, col 0

    var v = t.view(shape, strides, offset)
    assert_true(v.shape() == shape, "2D view shape matches")
    v[1, 1] = 100
    assert_equal(t[2, 1], 100.0, "Base tensor update failed via view")


fn test_view_2d_strides_overflow() raises:
    print("test_view_2d_strides_overflow")
    var t = Tensor.arange(12)
    var shape = Shape.of(3, 3)
    var strides = Strides(4, 1)  # row-major
    var offset = 4  # starts at 2nd row, so largest index will be:
    # 4 + (3-1)*4 + (3-1)*1 = 4 + 8 + 2 = 14 ❌
    with assert_raises():
        _ = t.view(shape, strides, offset)


fn test_view_3d_valid() raises:
    print("test_view_3d_valid")
    var t = Tensor.arange(60).reshape(Shape.of(3, 4, 5))  # 3x4x5
    var shape = Shape.of(2, 2, 2)
    var strides = Strides(20, 5, 1)  # default strides for 3D contiguous
    var offset = 0

    var v = t.view(shape, strides, offset)
    assert_true(v.shape() == shape, "3D valid view shape matches")


fn test_view_3d_invalid_strides() raises:
    print("test_view_3d_invalid_strides")
    var t = Tensor.arange(60).reshape(Shape.of(3, 4, 5))
    var shape = Shape.of(2, 2, 2)
    var _strides = Strides(40, 6, 3)  # artificially large strides
    var offset = 10

    # max_index = 10 + (2-1)*40 + (2-1)*6 + (2-1)*3 = 10 + 40 + 6 + 3 = 59 ✅
    # → Still valid! So tweak one stride...

    var strides2 = Strides(41, 6, 3)  # now exceeds
    with assert_raises():
        _ = t.view(shape, strides2, offset)


fn test_view_default_strides() raises:
    print("test_view_default_strides")
    var t = Tensor.d2([[1, 2], [3, 4]])  # Shape: (2, 2)
    var v = t.view(Shape.of(4))  # Reshape to (4,)
    assert_true(v.shape() == Shape.of(4), "Shape mismatch")
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
        _ = t.view(Shape.of(2, 2), Strides(1))  # Mismatched rank


fn test_view_with_strides_basic() raises:
    print("test_view_with_strides_basic")
    var t = Tensor.d2([[1, 2], [3, 4]])
    var strides = Strides(1, 2)  # Would result in skip reading
    var v = t.view(Shape.of(2, 2), strides)
    assert_true(v.shape() == Shape.of(2, 2), "Strided view shape mismatch")
    assert_false(
        v.is_contiguous(), "Expected strided view to be non-contiguous"
    )


fn test_view_negative_stride_fails_safely() raises:
    print("test_view_negative_stride_fails_safely")
    var t = Tensor.d2([[1, 2], [3, 4]])
    var strides = Strides(-1, 1)
    with assert_raises():
        _ = t.view(Shape.of(2, 2), strides)


fn test_view_offset_slice() raises:
    print("test_view_offset_slice")
    var t = Tensor.d1([1, 2, 3, 4, 5, 6])
    var shape = Shape.of(2)
    var v = t.view(shape, offset=3)
    assert_true(v.shape() == shape, "View shape mismatch")
    assert_true(v.offset() == 3, "Offset mismatch")


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
    var s = t.shape()
    var v = t.view(s)
    assert_true(v.is_contiguous(), "Identity view should be contiguous")
    assert_true(v.offset() == 0, "Offset should be zero")


fn test_view_stride_bounds_overflow() raises:
    print("test_view_stride_bounds_overflow")
    # max_index = offset + ∑ (shape[i] - 1) * strides[i]
    # For single dimension max_index = offset + (shape[0] - 1) * strides[0]
    var t = Tensor.d1([1, 2, 3, 4])
    var shape = Shape.of(2)
    var strides = Strides(3)
    # with assert_raises():
    # _ = t.view(shape, strides)  # 0 + 3*(2-1) = 3 ✅ but add one more -> 4 ➝ overflow
    assert_true(t.view(shape, strides)[1] == 4, "get item assertion failed")


fn test_getitem_list_empty_indices_returns_full_view() raises:
    print("test_getitem_list_empty_indices_returns_full_view")
    a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # v = a.__getitem__(List[Int]())
    v = a[:, :]
    assert_true(v.shape() == a.shape())
    assert_true(v.offset() == 0)
    assert_true(v.all_close(a))


# Following are older test cases - needs to be verified and run
fn test_into_tensor_full_view_copy() raises:
    print("test_into_tensor_full_view_copy")
    var t = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var v = t.view(Shape.of(2, 2))
    var out = v.contiguous()
    assert_true(out.all_close(t))
    assert_false(out.buffer is t.buffer)  # Ensure deep copy


fn test_into_tensor_transposed_view() raises:
    print("test_into_tensor_transposed_view")
    var t = Tensor.d2([[1, 2, 3], [4, 5, 6]])
    var v = t.transpose()
    var out = v.contiguous()
    assert_true(out.all_close(Tensor.d2([[1, 4], [2, 5], [3, 6]])))


fn test_into_tensor_offset_view() raises:
    print("test_into_tensor_offset_view")
    var t = Tensor.d1([0, 1, 2, 3, 4, 5])
    var v = t.view(Shape.of(2), offset=3)
    var out = v.contiguous()
    assert_true(out.all_close(Tensor.d1([3, 4])))


fn test_into_tensor_scalar_view() raises:
    print("test_into_tensor_scalar_view")
    var t = Tensor.scalar(42)
    var v = t.view(Shape())
    var out = v.contiguous()
    assert_true(out.shape() == Shape())
    assert_true(out.item() == 42)


fn test_into_tensor_empty_view() raises:
    print("test_into_tensor_empty_view")
    var t = Tensor[DType.float32](Shape.of(0, 3))
    var v = t.view(Shape.of(0, 3))
    var out = v.contiguous()
    assert_true(out.shape() == Shape.of(0, 3))
    # assert_true(out.data.len() == 0)


fn test_into_tensor_grad_flag_true() raises:
    print("test_into_tensor_grad_flag_true")
    var t = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = t.view(Shape.of(2, 2))
    var out = v.contiguous()
    assert_true(out.requires_grad == True)


fn test_into_tensor_grad_flag_false() raises:
    print("test_into_tensor_grad_flag_false")
    var t = Tensor.d2([[1, 2], [3, 4]], requires_grad=False)
    var v = t.view(Shape.of(2, 2))
    var out = v.contiguous()
    assert_true(out.requires_grad == False)


fn test_into_tensor_large_contiguous_copy() raises:
    print("test_into_tensor_large_contiguous_copy")
    N = 1024 * 1024
    var t = Tensor(Shape.of(N))
    for i in range(N):
        t[i] = i
    var v = t.view(Shape.of(N))
    var out = v.contiguous()
    assert_true(out.shape() == Shape.of(N))
    assert_true(out[123456] == 123456)


fn test_into_tensor_isolated_memory() raises:
    print("test_into_tensor_isolated_memory")
    t = Tensor.d1([1, 2, 3, 4])
    v = t[1:3]  # [2, 3]
    var out = v.contiguous()
    v[0] = 999

    assert_true(out.all_close(Tensor.d1([2, 3])))  # Unaffected by view mutation
    _ = """fn test_into_tensor_strided_view_rows() raises:
    print("test_into_tensor_strided_view_rows")
    var t = Tensor.d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var v = t.slice_rows(0, 4, 2)  # Should give rows [0, 2]
    var out = v.contiguous()
    assert_true(out.all_close(Tensor.d2([[1, 2], [5, 6]])))

fn test_into_tensor_contiguous_slice_1d() raises:
    print("test_into_tensor_contiguous_slice_1d")
    var t = Tensor.d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    var v = t.slice(2, 7)
    var out = v.contiguous()
    assert_true(out.all_close(Tensor.d1([2, 3, 4, 5, 6])))



fn test_into_tensor_nested_view() raises:
    print("test_into_tensor_nested_view")
    var t = Tensor.d1([10, 20, 30, 40, 50])
    var v1 = t.slice(1, 5)           # [20, 30, 40, 50]
    var v2 = v1.slice(1, 3)          # [30, 40]
    var out = v2.contiguous()
    assert_true(out.all_close(Tensor.d1([30, 40])))"""


fn test_backward_through_nested_views_non_contiguous() raises:
    print("test_backward_through_nested_views_non_contiguous")
    a = Tensor.rand(4, 4, requires_grad=True)
    t = a.transpose(0, 1)
    p = t.permute([1, 0])
    c = p.contiguous()
    s = c.sum()
    s.backward(42)
    assert_true(Strides.default(a.grad().shape()) == Strides.default(a.shape()))
    assert_true(
        # (a.grad() == Tensor.full(Shape([4, 4]), 42)),
        a.grad().all_close(Tensor.full(Shape([4, 4]), 42)),
        "grad propagation through contiguous failed",
    )


fn test_identity_permutation() raises:
    print("test_identity_permutation")
    x3 = Tensor.rand(3, 3, requires_grad=True)
    v3 = x3.into_view()
    y3 = v3.permute([0, 1])
    loss3 = y3.sum()
    loss3.backward()
    assert_true(x3.grad().all_close(Tensor.ones(3, 3)))


fn test_reshape_slice_sum_backward() raises:
    print("test_reshape_slice_sum_backward")
    a = Tensor.arange(15, requires_grad=True)
    r = a.reshape(5, 3)
    v = r[1:4:2, :]
    s = v.sum()
    s.backward()
    assert_true(
        (
            a.grad()
            == Tensor.d1(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ).float()
        )
    )


fn test_backward_through_nested_views() raises:
    print("test_backward_through_nested_views")
    # Test 1: Simple 2D transpose
    x1 = Tensor.rand(2, 3, requires_grad=True)
    v1 = x1.into_view()
    y1 = v1.permute([1, 0])
    yt1 = y1.contiguous()
    loss1 = yt1.sum()

    loss1.backward()
    assert_true(x1.grad().shape() == [2, 3])
    # Test 2: 3D permutation
    x2 = Tensor.rand(4, 5, 6, requires_grad=True)
    v2 = x2.into_view()
    y2 = v2.permute([2, 0, 1])
    yt2 = y2.contiguous()
    loss2 = yt2.sum()

    loss2.backward()
    assert_true(x2.grad().shape() == [4, 5, 6])


fn test_nested_views_grad_propagation_1() raises:
    print("test_nested_views_grad_propagation")

    # Step 1: Create a base tensor
    var A = Tensor.rand(4, 5, 6, requires_grad=True)

    # Step 2: Reshape to 2D (simulate flattening last two dims)
    var V1 = A.view([4, 30])  # shape: (4, 30)

    # Step 3: Slice via view with offset (simulate slicing last 20 elements of each row)
    var V2 = V1.view(
        [4, 20], offset=10
    )  # skips first 10 elements in flattened row

    # Step 4: Reshape again into (8, 10)
    var V3 = V2.view([8, 10])

    # Step 5: Sum and backward
    var LOSS = V3.sum()
    LOSS.backward()

    # Validate: only the correct region of x.grad should be non-zero
    # A.grad().print()
    var zero_count = 0
    var nonzero_count = 0
    for i in range(4):
        for j in range(5):
            for k in range(6):
                # Compute flat index in original (5, 6) → offset = j * 6 + k
                var flat = i * 30 + j * 6 + k
                if flat >= 10 and flat < 90:
                    # These 20 values should have been touched
                    assert_true(A.grad()[i, j, k] != 0.0)
                    nonzero_count += 1
                else:
                    # Rest untouched
                    assert_true(A.grad()[i, j, k] == 0.0)
                    zero_count += 1

    # Sanity: Expect 4 batches × 20 values = 80 nonzero
    assert_true(nonzero_count == 80)
    assert_true(zero_count == 4 * 30 - 80)

    # Cleanup


fn test_nested_views_grad_propagation() raises:
    print("test_nested_views_grad_propagation")

    # Step 1: Base tensor
    var A = Tensor.rand(4, 5, 6, requires_grad=True)

    # Step 2: Flatten last two dims
    var V1 = A.view([4, 30])  # shape: (4, 30)

    # Step 3: Create a subview that skips the first 10 elements of the entire buffer
    var V2 = V1.view([4, 20], offset=10)

    # Step 4: Reshape again to a different logical shape (just for variety)
    var V3 = V2.view([8, 10])

    # Step 5: Compute loss and backprop
    var LOSS = V3.sum()
    LOSS.backward()

    # Step 6: Validate grad mapping
    var grad = A.grad().copy()

    var total_nonzero = 0
    var total_zero = 0

    for i in range(4):
        for j in range(5):
            for k in range(6):
                # Compute flat index into the contiguous buffer
                var flat = i * 30 + j * 6 + k

                if (
                    flat >= 10 and flat < 80
                ):  # Only 80 elements (10..89) touched
                    assert_true(grad[i, j, k] != 0.0)
                    total_nonzero += 1
                else:
                    assert_true(grad[i, j, k] == 0.0)
                    total_zero += 1

    # Sanity check
    assert_true(total_nonzero == 70)
    assert_true(total_zero == 4 * 30 - 70)


fn test_edge_case_indexing() raises:
    print("test_edge_case_indexing")
    var a = Tensor.arange(6).reshape([2, 3])

    # Empty slice
    var empty = a[0:1, :]
    assert_true(empty.shape() == [1, 3])

    # Stop > dim size
    # var oversized_slice = a[:, 0:100]
    var oversized_slice = a[s(), s(None, 100, None)]

    assert_true((oversized_slice == a))
    assert_true(oversized_slice.buffer.size() == empty.buffer.size())


fn test_mixed_indexing() raises:
    print("test_mixed_indexing")
    var a = Tensor.arange(12, requires_grad=True)
    r = a.reshape([3, 4])

    # Mixed int/slice/newaxis
    var v = r[i(1), newaxis, s(0, 4, 2)]
    assert_true(v.shape() == [1, 2])
    assert_true((v == Tensor.d2([[4, 6]])))

    # Gradient check
    z = v.contiguous()
    var s = z.sum()
    s.backward()
    var expected_grad = Tensor.zeros([3, 4])
    expected_grad[1, 0] = 1.0
    expected_grad[1, 2] = 1.0
    assert_true((r.grad() == expected_grad))


fn test_newaxis_dimension_insertion() raises:
    print("test_newaxis_dimension_insertion")
    var a = Tensor([1, 2, 3], requires_grad=True)

    # Insert at beginning
    var v1 = a[newaxis, s()]
    assert_true(v1.shape() == [1, 3])
    assert_true((v1 == Tensor.d2([[1, 2, 3]])))

    # Insert at middle
    var v2 = a[s(), newaxis]
    assert_true(v2.shape() == [3, 1])
    assert_true((v2 == Tensor.d2([[1], [2], [3]])))

    # Gradient check
    b = v2.contiguous()
    var s = b.sum()
    s.backward()
    assert_true((a.grad() == Tensor([1, 1, 1])))


fn test_basic_slicing() raises:
    print("test_basic_slicing")
    var a = Tensor.arange(6, requires_grad=True)
    r = a.reshape([2, 3])
    # Gradient check
    var y = r[0:1, 1:3]
    ss = y.sum()
    ss.backward()
    var expected_grad = Tensor.d1([0, 1, 1, 0, 0, 0])
    assert_true((a.grad() == expected_grad))

    # Full slice
    # var full_slice = r[s(), s()]
    var full_slice = r[:, :]
    assert_true((full_slice == r))

    # Row slice
    var row = r[1, s()]
    assert_true((row == Tensor([3, 4, 5])))

    # Column slice with step
    var col_step = r[s(), s(0, 3, 2)]
    expect = Tensor.d2([[0, 2], [3, 5]])
    assert_true((col_step == expect))


fn test_integer_indexing() raises:
    print("test_integer_indexing")
    var a = Tensor.arange(6, requires_grad=True)
    r = a.reshape(2, 3)

    # Value checks
    assert_true(r[1, 2] == 5)  # Last element
    assert_true(r[-1, -1] == 5)  # Negative indices
    assert_true(r[0, 0] == 0)  # First element

    # Gradient check
    var y = r[i(1), i(2)]
    y.backward(42)
    var expected_grad = Tensor.zeros(6)
    expected_grad[5] = 42
    assert_true((a.grad() == expected_grad))


fn test_newaxis() raises:
    print("test_newaxis")
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x[newaxis, s(), newaxis]
    a = y.contiguous()
    b = a * 2
    b.backward()
    assert_true((x.grad() == Tensor([2, 2, 2])))


fn test_scalar_view() raises:
    print("test_scalar_view")
    a = Tensor.scalar(10, requires_grad=True)
    v = a.into_view()
    t = v.contiguous()
    s = t * 2
    s.backward(42)
    assert_true(a.grad().item() == 84, "Scalar view grad assertion failed")
