from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.shapes import Shape
from std.sys import has_accelerator


# Start of old tests
fn test_tile_axis_selective_grad() raises:
    print("test_tile_axis_selective_grad")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # shape (1,2,2)
    var y = x.tile(3, 2, 1)  # depth×row×col = 3×2×1
    var loss = y.sum()
    loss.backward()
    # 3×2×1=6 total repetitions per element
    assert_true(
        x.grad().all_close(Tensor[dtype].d3([[[6.0, 6.0], [6.0, 6.0]]]))
    )


fn test_tile_nonrepeated_axis_stability() raises:
    print("test_tile_nonrepeated_axis_stability")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], requires_grad=True
    )  # shape (2,1,3)
    var y = x.tile(2, 1, 3)  # only last axis repeats strongly
    var loss = y.sum()
    loss.backward()
    # gradient factor = 2×1×3 = 6
    assert_true(
        x.grad().all_close(
            Tensor[dtype].d3([[[6.0, 6.0, 6.0]], [[6.0, 6.0, 6.0]]])
        )
    )


fn test_tile_scalar() raises:
    print("test_tile_scalar")
    comptime dtype = DType.float32
    var x = Tensor[dtype].scalar(2.0, requires_grad=True)
    var y = x.tile(3, 2)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(3, 2))
    assert_true(x.grad().all_close(Tensor[dtype].scalar(6.0)))


fn test_tile_1d_repeats() raises:
    print("test_tile_1d_repeats")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.tile(4)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(12))
    assert_true(x.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


fn test_tile_2d_nonuniform_repeats() raises:
    print("test_tile_2d_nonuniform_repeats")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.tile(2, 3)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(4, 6))
    assert_true(x.grad().all_close(Tensor[dtype].d2([[6.0, 6.0], [6.0, 6.0]])))


fn test_tile_high_dimensional() raises:
    print("test_tile_high_dimensional")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(2, 1, 3, 1, 2)
    var loss = y.sum()
    loss.backward()
    # 2×1×3×1×2 = 12 repetitions per element
    assert_true(y.shape() == Shape(2, 1, 3, 1, 4))
    assert_true(x.grad().all_close(Tensor[dtype].d1([12.0, 12.0])))


fn test_tile_in_computational_graph() raises:
    print("test_tile_in_computational_graph")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var b = Tensor[dtype].d1([10.0, 20.0], requires_grad=True)
    var c = a * b  # [10, 40]
    var d = c.tile(3)
    var loss = d.sum()
    loss.backward()
    a.grad().print()
    b.grad().print()
    assert_true(a.grad().all_close(Tensor[dtype].d1([30.0, 60.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([3.0, 6.0])))


fn test_tensor_tiles_1d() raises:
    print("test_tensor_tiles_1d")
    comptime dtype = DType.float32

    var x = Tensor[dtype].d1([1, 2, 3])

    # Repeat along dimension 0
    var y = x.tile(4)
    var expected = Tensor[dtype].d1([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert_true(y == expected)

    # Repeat with multiple dimensions
    var z = x.tile(2, 3)
    var expected_z = Tensor[dtype].d2(
        [[1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3]]
    )
    assert_true(z == expected_z)

    print("Passed 1D tiles test")


fn test_tensor_tiles_2d() raises:
    print("test_tensor_tiles_2d")
    comptime dtype = DType.float32

    var x = Tensor[dtype].d2([[1, 2], [3, 4]])

    # Repeat rows and columns
    var y = x.tile(2, 3)
    var expected = Tensor[dtype].d2(
        [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ]
    )
    assert_true(y == expected)

    print("Passed 2D tiles test")


fn test_tensor_tiles_1d_to_2d_with_gradients() raises:
    print("test_tensor_tiles_1d_to_2d_with_gradients")
    comptime dtype = DType.float32

    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)

    # Repeat 1D tensor to 2D: (3,) → (2, 6)
    var y = x.tile(2, 2)  # Should give shape (2, 6)

    # Each row: [1,2,3,1,2,3]
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]]
    )
    assert_true(y.all_close(expected))

    # Test gradients
    var loss = y.sum()
    loss.backward()

    # Each element in x appears 4 times in y (2 rows × 2 repeats)
    # So gradient for each x element should be 4.0
    var expected_grad = Tensor[dtype].d1([4.0, 4.0, 4.0])
    assert_true(x.grad().all_close(expected_grad))

    print("Passed 1D to 2D tiles with gradients test")


fn test_tensor_tiles_2d_to_3d_with_gradients() raises:
    print("test_tensor_tiles_2d_to_3d_with_gradients")
    comptime dtype = DType.float32

    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Repeat 2D tensor to 3D: (2,2) → (3, 4, 4)
    var y = x.tile(3, 2, 2)

    # Test gradients - each element in x appears 3×2×2 = 12 times
    var loss = y.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].d2([[12.0, 12.0], [12.0, 12.0]])
    assert_true(x.grad().all_close(expected_grad))

    print("Passed 2D to 3D tiles with gradients test")


fn test_scalar_tiles_1d() raises:
    print("test_scalar_tiles_1d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].scalar(2.0, requires_grad=True)
    var y = x.tile(5)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(5))
    assert_true(y.all_close(Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0])))
    assert_true(x.grad().item() == 5.0)


fn test_scalar_tiles_2d() raises:
    print("test_scalar_tiles_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].scalar(3.0, requires_grad=True)
    var y = x.tile(2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3))
    assert_true(
        y.all_close(Tensor[dtype].d2([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]))
    )
    assert_true(x.grad().item() == 6.0)


fn test_1d_tiles_single_dimension() raises:
    print("test_1d_tiles_single_dimension")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.tile(4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(12))
    var expected_data = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    )
    assert_true(y.all_close(expected_data))
    assert_true(x.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


fn test_1d_tiles_to_2d() raises:
    print("test_1d_tiles_to_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(3, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 4))
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]]
    )
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_1d_tiles_multiple_dims() raises:
    print("test_1d_tiles_multiple_dims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(2, 3, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3, 2))
    # Each element appears 2*3 = 6 times
    assert_true(x.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_2d_tiles_single_dimension() raises:
    print("test_2d_tiles_single_dimension")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.tile(2)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(2, 4))
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]
    )
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_2d_tiles_both_dims() raises:
    print("test_2d_tiles_both_dims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var y = x.tile(3, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 4))
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]]
    )
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor[dtype].d2([[6.0, 6.0]])))


fn test_2d_tiles_3d_output() raises:
    print("test_2d_tiles_3d_output")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.tile(2, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2, 2))
    # Each element appears 2 times
    assert_true(x.grad().all_close(Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_3d_tiles_single_batch() raises:
    print("test_3d_tiles_single_batch")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var y = x.tile(2, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2, 2))
    assert_true(
        x.grad().all_close(Tensor[dtype].d3([[[2.0, 2.0], [2.0, 2.0]]]))
    )


fn test_3d_tiles_all_dims() raises:
    print("test_3d_tiles_all_dims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3([[[1.0], [2.0]]], requires_grad=True)
    var y = x.tile(2, 3, 4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 6, 4))
    # Each element appears 2*3*4 = 24 times
    assert_true(x.grad().all_close(Tensor[dtype].d3([[[24.0], [24.0]]])))


fn test_tiles_with_leading_singleton_dims() raises:
    print("test_tiles_with_leading_singleton_dims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(1, 1, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 1, 6))
    assert_true(x.grad().all_close(Tensor[dtype].d1([3.0, 3.0])))


fn test_tiles_with_trailing_singleton_dims() raises:
    print("test_tiles_with_trailing_singleton_dims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(3, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 1, 2))
    assert_true(x.grad().all_close(Tensor[dtype].d1([3.0, 3.0])))


fn test_tiles_complex_broadcasting_pattern() raises:
    print("test_tiles_complex_broadcasting_pattern")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0], [2.0]], requires_grad=True)
    var y = x.tile(1, 3, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 6, 2))
    # Each element appears 1*3*2 = 6 times
    assert_true(x.grad().all_close(Tensor[dtype].d2([[6.0], [6.0]])))


fn test_tiles_gradient_accumulation() raises:
    print("test_tiles_gradient_accumulation")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var y1 = x.tile(2)
    var y2 = x.tile(3)
    var loss = y1.sum() + y2.sum()
    loss.backward()

    # Each element appears 2 times in y1 and 3 times in y2 = 5 times total
    assert_true(x.grad().all_close(Tensor[dtype].d1([5.0, 5.0, 5.0])))


fn test_tiles_in_computational_graph() raises:
    print("test_tiles_in_computational_graph")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var b = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    var c = a * b  # [3.0, 8.0]
    var d = c.tile(2, 2)  # Shape: (2, 4)
    var loss = d.sum()
    loss.backward()

    # d contains each element of c 4 times (2×2)
    # So gradient through c: [4.0, 4.0]
    # Then through multiplication: a.grad = [4*3.0, 4*4.0] = [12.0, 16.0]
    # b.grad = [4*1.0, 4*2.0] = [4.0, 8.0]
    assert_true(a.grad().all_close(Tensor[dtype].d1([12.0, 16.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([4.0, 8.0])))


fn test_tiles_with_negative_gradient_flow() raises:
    print("test_tiles_with_negative_gradient_flow")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
    var y = x.tile(2)

    var z = -y  # Negative operation
    var loss = z.sum()
    loss.backward()
    # Each element appears 2 times, then negated
    assert_true(x.grad().all_close(Tensor[dtype].d1([-2.0, -2.0])))


fn test_tiles_high_dimensional() raises:
    print("test_tiles_high_dimensional")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(2, 1, 3, 1, 2)
    var loss = y.sum()
    loss.backward()

    # tiles pattern: 2×1×3×1×2 = 12 repetitions per element
    assert_true(y.shape() == Shape(2, 1, 3, 1, 4))
    assert_true(x.grad().all_close(Tensor[dtype].d1([12.0, 12.0])))


# End of old tests


# ═════════════════════════════════════════════════════════════════════════════
# CPU tile Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_tile_cpu_1d_basic() raises:
    print("test_tile_cpu_1d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.tile([3])
    assert_true(result.shape() == Shape(9))
    assert_true(
        result.all_close(
            Tensor[dtype].d1([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        )
    )


fn test_tile_cpu_1d_once() raises:
    print("test_tile_cpu_1d_once")
    comptime dtype = DType.float32
    # Tile by 1 — identity
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.tile([1])
    assert_true(result.shape() == Shape(3))
    assert_true(result.all_close(a))


fn test_tile_cpu_2d_both_dims() raises:
    print("test_tile_cpu_2d_both_dims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.tile([2, 3])
    assert_true(result.shape() == Shape(4, 6))
    # First row should be [1,2,1,2,1,2]
    assert_true(result[[0, 0]] == Scalar[dtype](1.0))
    assert_true(result[[0, 2]] == Scalar[dtype](1.0))
    assert_true(result[[0, 4]] == Scalar[dtype](1.0))
    # Third row should match first row (tiled along dim 0)
    assert_true(result[[2, 0]] == Scalar[dtype](1.0))
    assert_true(result[[2, 1]] == Scalar[dtype](2.0))


fn test_tile_cpu_2d_rows_only() raises:
    print("test_tile_cpu_2d_rows_only")
    comptime dtype = DType.float32
    # tile([3]) on (2,3) → (2,9) — fewer repeat dims than rank
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.tile([3])
    assert_true(result.shape() == Shape(2, 9))


fn test_tile_cpu_2d_extra_repeat_dims() raises:
    print("test_tile_cpu_2d_extra_repeat_dims")
    comptime dtype = DType.float32
    # tile([2,3,4]) on (2,3) → (2,6,12) — more repeat dims than rank
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.tile([2, 3, 4])
    assert_true(result.shape() == Shape(2, 6, 12))
    assert_true(result.numels() == 144)


fn test_tile_cpu_3d_basic() raises:
    print("test_tile_cpu_3d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var result = a.tile([1, 2, 1])
    assert_true(result.shape() == Shape(2, 4, 2))
    assert_true(result.numels() == 16)


fn test_tile_cpu_values_preserved() raises:
    print("test_tile_cpu_values_preserved")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0])
    var result = a.tile([4])
    for i in range(4):
        assert_true(result[[i * 2]] == Scalar[dtype](10.0))
        assert_true(result[[i * 2 + 1]] == Scalar[dtype](20.0))


fn test_tile_cpu_no_grad() raises:
    print("test_tile_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=False)
    var result = a.tile([3])
    assert_true(not result.requires_grad)


fn test_tile_cpu_requires_grad() raises:
    print("test_tile_cpu_requires_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var result = a.tile([3])
    assert_true(result.requires_grad)


fn test_tile_cpu_suppress_grad() raises:
    print("test_tile_cpu_suppress_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var result = a.tile([3], requires_grad=False)
    assert_true(not result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# CPU tile Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_tile_cpu_backward_1d() raises:
    print("test_tile_cpu_backward_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.tile([3])
    var loss = result.sum()
    loss.backward()
    # Each element tiled 3 times — grad = 3
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(3), 3.0)))


fn test_tile_cpu_backward_2d() raises:
    print("test_tile_cpu_backward_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.tile([2, 3])
    var loss = result.sum()
    loss.backward()
    # Each element tiled 2*3=6 times — grad = 6
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 6.0)))


fn test_tile_cpu_backward_1d_once() raises:
    print("test_tile_cpu_backward_1d_once")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.tile([1])
    var loss = result.sum()
    loss.backward()
    # Tiled once — grad = 1
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


fn test_tile_cpu_backward_chain() raises:
    print("test_tile_cpu_backward_chain")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var result = a.tile([4]) * 2.0
    var loss = result.sum()
    loss.backward()
    # Each element tiled 4 times, scaled by 2 — grad = 8
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2), 8.0)))


fn test_tile_cpu_backward_nonuniform_grad() raises:
    print("test_tile_cpu_backward_nonuniform_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var result = a.tile([2])
    # weights = [1, 2, 3, 4] for [a0, a1, a0, a1]
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var loss = (result * weights).sum()
    loss.backward()
    # grad[0] = 1 + 3 = 4, grad[1] = 2 + 4 = 6
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 6.0])))


fn test_tile_cpu_backward_2d_extra_dims() raises:
    print("test_tile_cpu_backward_2d_extra_dims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.tile([2, 1, 3])
    var loss = result.sum()
    loss.backward()
    # Each element tiled 2*1*3=6 times
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 6.0)))


# ═════════════════════════════════════════════════════════════════════════════
# GPU tile Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_tile_gpu_1d_basic() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_1d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.tile([3])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(9))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
            )
        )


fn test_tile_gpu_1d_once() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_1d_once")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.tile([1])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0]))
        )


fn test_tile_gpu_2d_both_dims() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_2d_both_dims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.tile([2, 3])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 6))
        var result_cpu = result.to_cpu()
        assert_true(result_cpu[[0, 0]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[0, 2]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[0, 4]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[2, 0]] == Scalar[dtype](1.0))


fn test_tile_gpu_2d_rows_only() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_2d_rows_only")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.tile([3])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 9))


fn test_tile_gpu_2d_extra_repeat_dims() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_2d_extra_repeat_dims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.tile([2, 3, 4])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 6, 12))


fn test_tile_gpu_3d_basic() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_3d_basic")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.tile([1, 2, 1])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 4, 2))


fn test_tile_gpu_values_preserved() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_values_preserved")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([10.0, 20.0]).to_gpu()
        var result = a.tile([4])
        var result_cpu = result.to_cpu()
        for i in range(4):
            assert_true(result_cpu[[i * 2]] == Scalar[dtype](10.0))
            assert_true(result_cpu[[i * 2 + 1]] == Scalar[dtype](20.0))


fn test_tile_gpu_no_grad() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_no_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=False).to_gpu()
        var result = a.tile([3])
        assert_true(not result.requires_grad)


fn test_tile_gpu_requires_grad() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_requires_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var result = a.tile([3])
        assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU tile Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_tile_gpu_backward_1d() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_backward_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([3])
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(3), 3.0)))


fn test_tile_gpu_backward_2d() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_backward_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([2, 3])
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 6.0)))


fn test_tile_gpu_backward_chain() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_backward_chain")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([4]) * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2), 8.0)))


fn test_tile_gpu_backward_nonuniform_grad() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_backward_nonuniform_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([2])
        var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var loss = (result * weights).sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 6.0])))


fn test_tile_gpu_backward_extra_dims() raises:
    comptime if has_accelerator():
        print("test_tile_gpu_backward_extra_dims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([2, 1, 3])
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 6.0)))


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_tile_parity_1d() raises:
    comptime if has_accelerator():
        print("test_tile_parity_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.tile([3]).all_close(a_gpu.tile([3]).to_cpu()))


fn test_tile_parity_2d() raises:
    comptime if has_accelerator():
        print("test_tile_parity_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.tile([2, 3]).all_close(a_gpu.tile([2, 3]).to_cpu()))


fn test_tile_parity_extra_dims() raises:
    comptime if has_accelerator():
        print("test_tile_parity_extra_dims")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.tile([2, 3, 4]).all_close(a_gpu.tile([2, 3, 4]).to_cpu())
        )


fn test_tile_parity_backward_1d() raises:
    comptime if has_accelerator():
        print("test_tile_parity_backward_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )

        var loss_cpu = a_cpu.tile([3]).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.tile([3]).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_tile_parity_backward_2d() raises:
    comptime if has_accelerator():
        print("test_tile_parity_backward_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            .to_gpu()
        )

        var loss_cpu = a_cpu.tile([2, 3]).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.tile([2, 3]).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_tile_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        print("test_tile_parity_using_zero_grad")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.tile([3]).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.tile([3]).sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


_="""
fn main() raises:
    pass
"""
