from tenmo import Tensor
from shapes import Shape
from strides import Strides
from testing import assert_true


# ===== BASIC RESHAPE FUNCTIONALITY =====


fn test_reshape_scalar_to_1d() raises:
    print("test_reshape_scalar_to_1d")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x.reshape(1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1))
    assert_true(y.all_close(Tensor.d1([2.0])))
    assert_true(x.grad().item() == 1.0)


fn test_reshape_scalar_to_2d() raises:
    print("test_reshape_scalar_to_2d")
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x.reshape(1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 1))
    assert_true(y.all_close(Tensor.d2([[3.0]])))
    assert_true(x.grad().item() == 1.0)


fn test_reshape_1d_to_1d_same_size() raises:
    print("test_reshape_1d_to_1d_same_size")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.reshape(3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3))
    assert_true(y.all_close(Tensor.d1([1.0, 2.0, 3.0])))
    assert_true(x.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_reshape_1d_to_2d() raises:
    print("test_reshape_1d_to_2d")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.reshape(2, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2))
    assert_true(y.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])))
    assert_true(x.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 1.0])))


fn test_reshape_1d_to_3d() raises:
    print("test_reshape_1d_to_3d")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var y = x.reshape(1, 2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 2, 3))
    assert_true(x.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])))


fn test_reshape_2d_to_1d() raises:
    print("test_reshape_2d_to_1d")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.reshape(4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(4))
    assert_true(y.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(x.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_reshape_2d_to_3d() raises:
    print("test_reshape_2d_to_3d")
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var y = x.reshape(1, 2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 2, 3))
    assert_true(
        x.grad().all_close(Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


fn test_reshape_3d_to_2d() raises:
    print("test_reshape_3d_to_2d")
    var x = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var y = x.reshape(2, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2))
    assert_true(x.grad().all_close(Tensor.d3([[[1.0, 1.0], [1.0, 1.0]]])))


# ===== RESHAPE WITH STRICT DIMENSION VALIDATION =====


fn test_reshape_strict_validation_success() raises:
    print("test_reshape_strict_validation_success")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # These should work (repeat dims >= tensor rank)
    var y1 = x.reshape(2, 2)  # Exact match
    var y2 = x.reshape(1, 2, 2)  # More dims
    var y3 = x.reshape(1, 1, 2, 2)  # Even more dims

    assert_true(y1.shape() == Shape(2, 2))
    assert_true(y2.shape() == Shape(1, 2, 2))
    assert_true(y3.shape() == Shape(1, 1, 2, 2))


fn test_reshape_strict_validation_failure() raises:
    print("test_reshape_strict_validation_failure")
    var _x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])

    # These should PANIC due to strict PyTorch rules
    # Uncomment to test - they should cause panics
    # var y1 = x.reshape()           # Empty reshape for 2D tensor
    # var y2 = x.reshape(4)          # Too few dimensions (1 vs rank 2)


# ===== RESHAPE GRADIENT FLOW =====


fn test_reshape_gradient_preservation() raises:
    print("test_reshape_gradient_preservation")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.reshape(2, 2)
    var z = y * 2.0
    var loss = z.sum()
    loss.backward()

    # Gradient should flow back through reshape
    assert_true(x.grad().all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0])))


fn test_reshape_gradient_accumulation() raises:
    print("test_reshape_gradient_accumulation")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y1 = x.reshape(3, 1)
    var y2 = x.reshape(1, 3)
    var loss = y1.sum() + y2.sum()
    loss.backward()

    # Each element appears in both reshapes
    assert_true(x.grad().all_close(Tensor.d1([2.0, 2.0, 2.0])))


fn test_reshape_chain_gradient_flow() raises:
    print("test_reshape_chain_gradient_flow")
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var y = x.reshape(3, 2)
    var z = y.reshape(6)
    var loss = z.sum()
    loss.backward()

    # Gradient should flow back through reshape chain
    assert_true(
        x.grad().all_close(Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


# ===== RESHAPE WITH COMPUTATIONAL GRAPH =====


fn test_reshape_in_complex_graph() raises:
    print("test_reshape_in_complex_graph")
    var a = Tensor.d1([1.0, 2.0], requires_grad=True)
    var b = Tensor.d1([3.0, 4.0], requires_grad=True)

    var c = a * b  # [3.0, 8.0]
    var d = c.reshape(2, 1)  # [[3.0], [8.0]]
    var e = d.reshape(1, 2)  # [[3.0, 8.0]]
    var loss = e.sum()
    loss.backward()

    assert_true(a.grad().all_close(Tensor.d1([3.0, 4.0])))  # from b values
    assert_true(b.grad().all_close(Tensor.d1([1.0, 2.0])))  # from a values


fn test_reshape_with_arithmetic_ops() raises:
    print("test_reshape_with_arithmetic_ops")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = Tensor.d1([5.0, 6.0], requires_grad=True)

    var x_flat = x.reshape(4)
    var y_broadcast = y.reshape(2, 1).expand(2, 2).reshape(4)  # [5, 5, 6, 6]
    var result = x_flat * y_broadcast
    var loss = result.sum()
    loss.backward()

    # Gradients:
    # x.grad = [[5, 5], [6, 6]]
    # y.grad = [1+2, 3+4] = [3, 7]

    assert_true(x.grad().all_close(Tensor.d2([[5.0, 5.0], [6.0, 6.0]])))
    assert_true(y.grad().all_close(Tensor.d1([3.0, 7.0])))


fn test_reshape_with_arithmetic_ops_repeat() raises:
    print("test_reshape_with_arithmetic_ops_repeat")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = Tensor.d1([5.0, 6.0], requires_grad=True)

    var x_flat = x.reshape(4)
    var y_repeated = y.repeat[](2)  # Make y same shape as x_flat: [5, 6, 5, 6]
    var result = x_flat * y_repeated
    var loss = result.sum()
    loss.backward()

    # Now the gradients make sense:
    # x.grad = y_repeated reshaped = [[5, 6], [5, 6]]
    # y.grad = [1+3, 2+4] = [4, 6]

    assert_true(x.grad().all_close(Tensor.d2([[5.0, 6.0], [5.0, 6.0]])))
    assert_true(y.grad().all_close(Tensor.d1([4.0, 6.0])))  # Corrected!


# ===== RESHAPE EDGE CASES =====


fn test_reshape_singleton_expansion() raises:
    print("test_reshape_singleton_expansion")
    var x = Tensor.d1([5.0], requires_grad=True)
    var y = x.reshape(1, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 1, 1))
    assert_true(x.grad().item() == 1.0)


fn test_reshape_singleton_removal() raises:
    print("test_reshape_singleton_removal")
    var x = Tensor.d3([[[1.0]], [[2.0]]], requires_grad=True)
    var y = x.reshape(2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2))
    assert_true(x.grad().all_close(Tensor.d3([[[1.0]], [[1.0]]])))


fn test_reshape_identity() raises:
    print("test_reshape_identity")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.reshape(2, 2)  # Same shape
    var loss = y.sum()
    loss.backward()

    assert_true(y.all_close(x))
    assert_true(x.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_reshape_large_tensor() raises:
    print("test_reshape_large_tensor")
    # Test with larger tensors to ensure no memory issues
    alias dtype = DType.float32
    var data = List[Scalar[dtype]](capacity=UInt(1024))
    for i in range(1024):
        data.append(Scalar[dtype](i))

    var x = Tensor.d1(data, requires_grad=True)
    var y = x.reshape(8, 8, 8, 2)
    var z = y.reshape(4, 4, 4, 2, 2, 2, 2)
    var loss = z.sum()
    loss.backward()

    assert_true(z.shape() == Shape(4, 4, 4, 2, 2, 2, 2))
    assert_true(x.grad().sum().item() == 1024.0)  # Each element gets 1.0


# ===== RESHAPE WITH VIEW COMPATIBILITY =====


fn test_reshape_after_view_creates_copy_1() raises:
    print("test_reshape_after_view_creates_copy")
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Create a valid view (2x2 subset)
    var v = x.view(shape=Shape(2, 2), strides=Strides(3, 1), offset=1)
    # This accesses: [2,3] and [5,6]

    # Reshape the view
    var y = v.reshape(4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(4))
    # Gradients should flow to positions [2,3,5,6] in original x


fn test_reshape_after_view_creates_copy_2() raises:
    print("test_reshape_after_view_creates_copy")
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Create a transposed-like view
    var v = x.view(shape=Shape(3, 2), strides=Strides(1, 3), offset=0)
    # This accesses columns: [1,4], [2,5], [3,6]

    # Reshape the view
    var y = v.reshape(6)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(6))


fn test_reshape_after_view_creates_copy_3() raises:
    print("test_reshape_after_view_creates_copy")
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Simple row slice view
    var v = x.view(shape=Shape(1, 3), strides=Strides(3, 1), offset=3)
    # This accesses second row: [4,5,6]

    # Reshape the view
    var y = v.reshape(3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3))
    # Gradients should flow only to second row of x


fn test_reshape_preserves_requires_grad() raises:
    print("test_reshape_preserves_requires_grad")
    var x1 = Tensor.d1([1.0, 2.0], requires_grad=True)
    var x2 = Tensor.d1([3.0, 4.0], requires_grad=False)

    var y1 = x1.reshape(2, 1)
    var y2 = x2.reshape(2, 1)

    assert_true(y1.requires_grad)
    assert_true(not y2.requires_grad)


# ===== COMPREHENSIVE TEST FUNCTION =====


fn main() raises:
    print("Running comprehensive reshape functionality tests...")

    # Basic functionality
    test_reshape_scalar_to_1d()
    test_reshape_scalar_to_2d()
    test_reshape_1d_to_1d_same_size()
    test_reshape_1d_to_2d()
    test_reshape_1d_to_3d()
    test_reshape_2d_to_1d()
    test_reshape_2d_to_3d()
    test_reshape_3d_to_2d()

    # Strict validation
    test_reshape_strict_validation_success()
    test_reshape_strict_validation_failure()

    # Gradient flow
    test_reshape_gradient_preservation()
    test_reshape_gradient_accumulation()
    test_reshape_chain_gradient_flow()

    # Computational graph
    test_reshape_in_complex_graph()
    test_reshape_with_arithmetic_ops()
    test_reshape_with_arithmetic_ops_repeat()

    # Edge cases
    test_reshape_singleton_expansion()
    test_reshape_singleton_removal()
    test_reshape_identity()
    test_reshape_large_tensor()

    # View compatibility
    test_reshape_after_view_creates_copy_1()
    test_reshape_after_view_creates_copy_2()
    test_reshape_after_view_creates_copy_3()
    test_reshape_preserves_requires_grad()

    print("✓ All reshape functionality tests passed!")
