from tenmo import Tensor
from testing import assert_true
from shapes import Shape


fn main() raises:
    print("Running comprehensive tiles functionality tests...")
    test_tile_scalar()
    test_tile_1d_repeats()
    test_tile_2d_nonuniform_repeats()
    test_tile_axis_selective_grad()
    test_tile_nonrepeated_axis_stability()
    test_tile_high_dimensional()
    test_tile_in_computational_graph()

    test_tensor_tiles_1d()
    test_tensor_tiles_2d()
    test_tensor_tiles_1d_to_2d_with_gradients()
    test_tensor_tiles_2d_to_3d_with_gradients()

    test_scalar_tiles_1d()
    test_scalar_tiles_2d()
    test_1d_tiles_single_dimension()
    test_1d_tiles_to_2d()
    test_1d_tiles_multiple_dims()
    test_2d_tiles_single_dimension()
    test_2d_tiles_both_dims()
    test_2d_tiles_3d_output()
    test_3d_tiles_single_batch()
    test_3d_tiles_all_dims()
    test_tiles_with_leading_singleton_dims()
    test_tiles_with_trailing_singleton_dims()
    test_tiles_complex_broadcasting_pattern()
    test_tiles_gradient_accumulation()
    test_tiles_in_computational_graph()
    test_tiles_with_negative_gradient_flow()
    test_tiles_high_dimensional()

    print("✓ All tiles functionality tests passed!")
    print("passes")


fn test_tile_axis_selective_grad() raises:
    print("test_tile_axis_selective_grad")
    var x = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # shape (1,2,2)
    var y = x.tile(3, 2, 1)  # depth×row×col = 3×2×1
    var loss = y.sum()
    loss.backward()
    # 3×2×1=6 total repetitions per element
    assert_true(x.grad().all_close(Tensor.d3([[[6.0, 6.0], [6.0, 6.0]]])))


fn test_tile_nonrepeated_axis_stability() raises:
    print("test_tile_nonrepeated_axis_stability")
    var x = Tensor.d3(
        [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], requires_grad=True
    )  # shape (2,1,3)
    var y = x.tile(2, 1, 3)  # only last axis repeats strongly
    var loss = y.sum()
    loss.backward()
    # gradient factor = 2×1×3 = 6
    assert_true(
        x.grad().all_close(Tensor.d3([[[6.0, 6.0, 6.0]], [[6.0, 6.0, 6.0]]]))
    )


fn test_tile_scalar() raises:
    print("test_tile_scalar")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x.tile(3, 2)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(3, 2))
    assert_true(x.grad().all_close(Tensor.scalar(6.0)))


fn test_tile_1d_repeats() raises:
    print("test_tile_1d_repeats")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.tile(4)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(12))
    assert_true(x.grad().all_close(Tensor.d1([4.0, 4.0, 4.0])))


fn test_tile_2d_nonuniform_repeats() raises:
    print("test_tile_2d_nonuniform_repeats")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.tile(2, 3)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(4, 6))
    assert_true(x.grad().all_close(Tensor.d2([[6.0, 6.0], [6.0, 6.0]])))


fn test_tile_high_dimensional() raises:
    print("test_tile_high_dimensional")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(2, 1, 3, 1, 2)
    var loss = y.sum()
    loss.backward()
    # 2×1×3×1×2 = 12 repetitions per element
    assert_true(y.shape() == Shape(2, 1, 3, 1, 4))
    assert_true(x.grad().all_close(Tensor.d1([12.0, 12.0])))


fn test_tile_in_computational_graph() raises:
    print("test_tile_in_computational_graph")
    var a = Tensor.d1([1.0, 2.0], requires_grad=True)
    var b = Tensor.d1([10.0, 20.0], requires_grad=True)
    var c = a * b  # [10, 40]
    var d = c.tile(3)
    var loss = d.sum()
    loss.backward()
    a.grad().print()
    b.grad().print()
    assert_true(a.grad().all_close(Tensor.d1([30.0, 60.0])))
    assert_true(b.grad().all_close(Tensor.d1([3.0, 6.0])))


fn test_tensor_tiles_1d() raises:
    print("test_tensor_tiles_1d")

    var x = Tensor.d1([1, 2, 3])

    # Repeat along dimension 0
    var y = x.tile(4)
    var expected = Tensor.d1([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert_true(y == expected)

    # Repeat with multiple dimensions
    var z = x.tile(2, 3)
    var expected_z = Tensor.d2(
        [[1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3]]
    )
    assert_true(z == expected_z)

    print("✓ Passed 1D tiles test")


fn test_tensor_tiles_2d() raises:
    print("test_tensor_tiles_2d")

    var x = Tensor.d2([[1, 2], [3, 4]])

    # Repeat rows and columns
    var y = x.tile(2, 3)
    var expected = Tensor.d2(
        [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ]
    )
    assert_true(y == expected)

    print("✓ Passed 2D tiles test")


fn test_tensor_tiles_1d_to_2d_with_gradients() raises:
    print("test_tensor_tiles_1d_to_2d_with_gradients")

    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)

    # Repeat 1D tensor to 2D: (3,) → (2, 6)
    var y = x.tile(2, 2)  # Should give shape (2, 6)

    # Each row: [1,2,3,1,2,3]
    var expected = Tensor.d2(
        [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]]
    )
    assert_true(y.all_close(expected))

    # Test gradients
    var loss = y.sum()
    loss.backward()

    # Each element in x appears 4 times in y (2 rows × 2 repeats)
    # So gradient for each x element should be 4.0
    var expected_grad = Tensor.d1([4.0, 4.0, 4.0])
    assert_true(x.grad().all_close(expected_grad))

    print("✓ Passed 1D to 2D tiles with gradients test")


fn test_tensor_tiles_2d_to_3d_with_gradients() raises:
    print("test_tensor_tiles_2d_to_3d_with_gradients")

    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Repeat 2D tensor to 3D: (2,2) → (3, 4, 4)
    var y = x.tile(3, 2, 2)

    # Test gradients - each element in x appears 3×2×2 = 12 times
    var loss = y.sum()
    loss.backward()

    var expected_grad = Tensor.d2([[12.0, 12.0], [12.0, 12.0]])
    assert_true(x.grad().all_close(expected_grad))

    print("✓ Passed 2D to 3D tiles with gradients test")


fn test_scalar_tiles_1d() raises:
    print("test_scalar_tiles_1d")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x.tile(5)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(5))
    assert_true(y.all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0, 2.0])))
    assert_true(x.grad().item() == 5.0)


fn test_scalar_tiles_2d() raises:
    print("test_scalar_tiles_2d")
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x.tile(2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3))
    assert_true(y.all_close(Tensor.d2([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])))
    assert_true(x.grad().item() == 6.0)


fn test_1d_tiles_single_dimension() raises:
    print("test_1d_tiles_single_dimension")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.tile(4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(12))
    var expected_data = Tensor.d1(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    )
    assert_true(y.all_close(expected_data))
    assert_true(x.grad().all_close(Tensor.d1([4.0, 4.0, 4.0])))


fn test_1d_tiles_to_2d() raises:
    print("test_1d_tiles_to_2d")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(3, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 4))
    var expected = Tensor.d2(
        [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]]
    )
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor.d1([6.0, 6.0])))


fn test_1d_tiles_multiple_dims() raises:
    print("test_1d_tiles_multiple_dims")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(2, 3, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3, 2))
    # Each element appears 2*3 = 6 times
    assert_true(x.grad().all_close(Tensor.d1([6.0, 6.0])))


fn test_2d_tiles_single_dimension() raises:
    print("test_2d_tiles_single_dimension")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.tile(2)
    var loss = y.sum()
    loss.backward()
    assert_true(y.shape() == Shape(2, 4))
    var expected = Tensor.d2([[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]])
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_2d_tiles_both_dims() raises:
    print("test_2d_tiles_both_dims")
    var x = Tensor.d2([[1.0, 2.0]], requires_grad=True)
    var y = x.tile(3, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 4))
    var expected = Tensor.d2(
        [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]]
    )
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor.d2([[6.0, 6.0]])))


fn test_2d_tiles_3d_output() raises:
    print("test_2d_tiles_3d_output")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.tile(2, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2, 2))
    # Each element appears 2 times
    assert_true(x.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_3d_tiles_single_batch() raises:
    print("test_3d_tiles_single_batch")
    var x = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var y = x.tile(2, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2, 2))
    assert_true(x.grad().all_close(Tensor.d3([[[2.0, 2.0], [2.0, 2.0]]])))


fn test_3d_tiles_all_dims() raises:
    print("test_3d_tiles_all_dims")
    var x = Tensor.d3([[[1.0], [2.0]]], requires_grad=True)
    var y = x.tile(2, 3, 4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 6, 4))
    # Each element appears 2*3*4 = 24 times
    assert_true(x.grad().all_close(Tensor.d3([[[24.0], [24.0]]])))


fn test_tiles_with_leading_singleton_dims() raises:
    print("test_tiles_with_leading_singleton_dims")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(1, 1, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 1, 6))
    assert_true(x.grad().all_close(Tensor.d1([3.0, 3.0])))


fn test_tiles_with_trailing_singleton_dims() raises:
    print("test_tiles_with_trailing_singleton_dims")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(3, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 1, 2))
    assert_true(x.grad().all_close(Tensor.d1([3.0, 3.0])))


fn test_tiles_complex_broadcasting_pattern() raises:
    print("test_tiles_complex_broadcasting_pattern")
    var x = Tensor.d2([[1.0], [2.0]], requires_grad=True)
    var y = x.tile(1, 3, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 6, 2))
    # Each element appears 1*3*2 = 6 times
    assert_true(x.grad().all_close(Tensor.d2([[6.0], [6.0]])))


fn test_tiles_gradient_accumulation() raises:
    print("test_tiles_gradient_accumulation")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y1 = x.tile(2)
    var y2 = x.tile(3)
    var loss = y1.sum() + y2.sum()
    loss.backward()

    # Each element appears 2 times in y1 and 3 times in y2 = 5 times total
    assert_true(x.grad().all_close(Tensor.d1([5.0, 5.0, 5.0])))


fn test_tiles_in_computational_graph() raises:
    print("test_tiles_in_computational_graph")
    var a = Tensor.d1([1.0, 2.0], requires_grad=True)
    var b = Tensor.d1([3.0, 4.0], requires_grad=True)

    var c = a * b  # [3.0, 8.0]
    var d = c.tile(2, 2)  # Shape: (2, 4)
    var loss = d.sum()
    loss.backward()

    # d contains each element of c 4 times (2×2)
    # So gradient through c: [4.0, 4.0]
    # Then through multiplication: a.grad = [4*3.0, 4*4.0] = [12.0, 16.0]
    # b.grad = [4*1.0, 4*2.0] = [4.0, 8.0]
    assert_true(a.grad().all_close(Tensor.d1([12.0, 16.0])))
    assert_true(b.grad().all_close(Tensor.d1([4.0, 8.0])))


fn test_tiles_with_negative_gradient_flow() raises:
    print("test_tiles_with_negative_gradient_flow")
    var x = Tensor.d1([2.0, 3.0], requires_grad=True)
    var y = x.tile(2)

    var z = -y  # Negative operation
    var loss = z.sum()
    loss.backward()
    # Each element appears 2 times, then negated
    assert_true(x.grad().all_close(Tensor.d1([-2.0, -2.0])))


fn test_tiles_high_dimensional() raises:
    print("test_tiles_high_dimensional")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.tile(2, 1, 3, 1, 2)
    var loss = y.sum()
    loss.backward()

    # tiles pattern: 2×1×3×1×2 = 12 repetitions per element
    assert_true(y.shape() == Shape(2, 1, 3, 1, 4))
    assert_true(x.grad().all_close(Tensor.d1([12.0, 12.0])))
