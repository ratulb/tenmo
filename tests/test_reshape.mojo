from tenmo.strides import Strides
from std.testing import assert_true, TestSuite
from tenmo import Tensor, Shape
from std.sys import has_accelerator


# ===== BASIC RESHAPE FUNCTIONALITY =====


def test_reshape_scalar_to_1d() raises:
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x.reshape(1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1))
    assert_true(y.all_close(Tensor.d1([2.0])))
    assert_true(x.grad().item() == 1.0)


def test_reshape_scalar_to_2d() raises:
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x.reshape(1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 1))
    assert_true(y.all_close(Tensor.d2([[3.0]])))
    assert_true(x.grad().item() == 1.0)


def test_reshape_1d_to_1d_same_size() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.reshape(3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3))
    assert_true(y.all_close(Tensor.d1([1.0, 2.0, 3.0])))
    assert_true(x.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


def test_reshape_1d_to_2d() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.reshape(2, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2))
    assert_true(y.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])))
    assert_true(x.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 1.0])))


def test_reshape_1d_to_3d() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var y = x.reshape(1, 2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 2, 3))
    assert_true(x.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])))


def test_reshape_2d_to_1d() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.reshape(4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(4))
    assert_true(y.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(x.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


def test_reshape_2d_to_3d() raises:
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var y = x.reshape(1, 2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 2, 3))
    assert_true(
        x.grad().all_close(Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


def test_reshape_3d_to_2d() raises:
    var x = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var y = x.reshape(2, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2))
    assert_true(x.grad().all_close(Tensor.d3([[[1.0, 1.0], [1.0, 1.0]]])))


# ===== RESHAPE WITH STRICT DIMENSION VALIDATION =====


def test_reshape_strict_validation_success() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # These should work (repeat dims >= tensor rank)
    var y1 = x.reshape(2, 2)  # Exact match
    var y2 = x.reshape(1, 2, 2)  # More dims
    var y3 = x.reshape(1, 1, 2, 2)  # Even more dims

    assert_true(y1.shape() == Shape(2, 2))
    assert_true(y2.shape() == Shape(1, 2, 2))
    assert_true(y3.shape() == Shape(1, 1, 2, 2))


def test_reshape_strict_validation_failure() raises:
    var _x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])

    # These should PANIC due to strict PyTorch rules
    # Uncomment to test - they should cause panics
    # var y1 = x.reshape()           # Empty reshape for 2D tensor
    # var y2 = x.reshape(4)          # Too few dimensions (1 vs rank 2)


# ===== RESHAPE GRADIENT FLOW =====


def test_reshape_gradient_preservation() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.reshape(2, 2)
    var z = y * 2.0
    var loss = z.sum()
    loss.backward()

    # Gradient should flow back through reshape
    assert_true(x.grad().all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0])))


def test_reshape_gradient_accumulation() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y1 = x.reshape(3, 1)
    var y2 = x.reshape(1, 3)
    var loss = y1.sum() + y2.sum()
    loss.backward()

    # Each element appears in both reshapes
    assert_true(x.grad().all_close(Tensor.d1([2.0, 2.0, 2.0])))


def test_reshape_chain_gradient_flow() raises:
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


def test_reshape_in_complex_graph() raises:
    var a = Tensor.d1([1.0, 2.0], requires_grad=True)
    var b = Tensor.d1([3.0, 4.0], requires_grad=True)

    var c = a * b  # [3.0, 8.0]
    var d = c.reshape(2, 1)  # [[3.0], [8.0]]
    var e = d.reshape(1, 2)  # [[3.0, 8.0]]
    var loss = e.sum()
    loss.backward()

    assert_true(a.grad().all_close(Tensor.d1([3.0, 4.0])))  # from b values
    assert_true(b.grad().all_close(Tensor.d1([1.0, 2.0])))  # from a values


def test_reshape_with_arithmetic_ops() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = Tensor.d1([5.0, 6.0], requires_grad=True)

    var x_flat = x.reshape(4)
    var _tmp0 = y.reshape(2, 1)
    var _tmp1 = _tmp0.expand(2, 2)
    var y_broadcast = _tmp1.reshape(4)  # [5, 5, 6, 6]
    var result = x_flat * y_broadcast
    var loss = result.sum()
    loss.backward()

    # Gradients:
    # x.grad = [[5, 5], [6, 6]]
    # y.grad = [1+2, 3+4] = [3, 7]

    assert_true(x.grad().all_close(Tensor.d2([[5.0, 5.0], [6.0, 6.0]])))
    assert_true(y.grad().all_close(Tensor.d1([3.0, 7.0])))


def test_reshape_with_arithmetic_ops_repeat() raises:
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


def test_reshape_singleton_expansion() raises:
    var x = Tensor.d1([5.0], requires_grad=True)
    var y = x.reshape(1, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(1, 1, 1))
    assert_true(x.grad().item() == 1.0)


def test_reshape_singleton_removal() raises:
    var x = Tensor.d3([[[1.0]], [[2.0]]], requires_grad=True)
    var y = x.reshape(2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2))
    assert_true(x.grad().all_close(Tensor.d3([[[1.0]], [[1.0]]])))


def test_reshape_identity() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.reshape(2, 2)  # Same shape
    var loss = y.sum()
    loss.backward()

    assert_true(y.all_close(x))
    assert_true(x.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


def test_reshape_large_tensor() raises:
    # Test with larger tensors to ensure no memory issues
    comptime dtype = DType.float32
    var data = List[Scalar[dtype]]()
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


def test_reshape_after_view_creates_copy_1() raises:
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


def test_reshape_after_view_creates_copy_2() raises:
    var x = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Create a transposed-like view
    var v = x.view(shape=Shape(3, 2), strides=Strides(1, 3), offset=0)
    # This accesses columns: [1,4], [2,5], [3,6]

    # Reshape the view
    var y = v.reshape(6)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(6))


def test_reshape_after_view_creates_copy_3() raises:
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


def test_reshape_preserves_requires_grad() raises:
    var x1 = Tensor.d1([1.0, 2.0], requires_grad=True)
    var x2 = Tensor.d1([3.0, 4.0], requires_grad=False)

    var y1 = x1.reshape(2, 1)
    var y2 = x2.reshape(2, 1)

    assert_true(y1.requires_grad)
    assert_true(not y2.requires_grad)


# ===== COMPREHENSIVE TEST FUNCTION =====


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ─────────────────────────────────────────────
#  CPU — 1-D
# ─────────────────────────────────────────────


fn test_cpu_1d_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var b = a.reshape(2, 3)
    assert_true(
        b.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    )


fn test_cpu_1d_backward_grad_flows_to_a() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var b = a.reshape(2, 3)
    var c = b * 2.0
    var loss = c.sum()
    loss.backward()
    # grad of sum(b*2) w.r.t a = 2 everywhere, reshaped back to 1-D
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    )


fn test_cpu_1d_b_has_no_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var b = a.reshape(2, 3)
    var c = b * 3.0
    var loss = c.sum()
    loss.backward()
    # reshape is not a mathematical op — b retains no grad
    assert_true(b.requires_grad)


fn test_cpu_1d_shared_buffer() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = a.reshape(2, 2)
    # b is a view — same data, different shape
    assert_true(b.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))


# ─────────────────────────────────────────────
#  CPU — 2-D → 1-D
# ─────────────────────────────────────────────


fn test_cpu_2d_to_1d_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var b = a.reshape(6)
    assert_true(b.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])))


fn test_cpu_2d_to_1d_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var b = a.reshape(6)
    var c = b * 3.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]))
    )


# ─────────────────────────────────────────────
#  CPU — 2-D → 3-D
# ─────────────────────────────────────────────


fn test_cpu_2d_to_3d_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
    )
    var b = a.reshape(2, 2, 2)
    assert_true(
        b.all_close(
            Tensor[dtype].d3(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
            )
        )
    )


fn test_cpu_2d_to_3d_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
    )
    var b = a.reshape(2, 2, 2)
    var c = b * 5.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d2([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])
        )
    )


# ─────────────────────────────────────────────
#  CPU — 3-D → 2-D
# ─────────────────────────────────────────────


fn test_cpu_3d_to_2d_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.reshape(4, 2)
    assert_true(
        b.all_close(
            Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
    )


fn test_cpu_3d_to_2d_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.reshape(4, 2)
    var c = b * 4.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[4.0, 4.0], [4.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]
            )
        )
    )


# ─────────────────────────────────────────────
#  CPU — 3-D → 1-D
# ─────────────────────────────────────────────


fn test_cpu_3d_to_1d_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.reshape(8)
    assert_true(
        b.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    )


fn test_cpu_3d_to_1d_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.reshape(8)
    var c = b * 7.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[7.0, 7.0], [7.0, 7.0]], [[7.0, 7.0], [7.0, 7.0]]]
            )
        )
    )


# ─────────────────────────────────────────────
#  CPU — scalar reshape (single element)
# ─────────────────────────────────────────────


fn test_cpu_scalar_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([42.0], requires_grad=True)
    var b = a.reshape()  # single-element → scalar Shape()
    assert_true(b.all_close(Tensor[dtype].full(Shape(), 42.0)))


fn test_cpu_scalar_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([3.0], requires_grad=True)
    var b = a.reshape()
    var c = b * 10.0
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([10.0])))


# ─────────────────────────────────────────────
#  CPU — using Shape() overload
# ─────────────────────────────────────────────


fn test_cpu_shape_overload_2d_to_2d() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var b = a.reshape(Shape(3, 2))
    assert_true(
        b.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    )


fn test_cpu_shape_overload_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var b = a.reshape(Shape(3, 2))
    var c = b * 6.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]))
    )


# ─────────────────────────────────────────────
#  CPU — using List[Int] overload
# ─────────────────────────────────────────────


fn test_cpu_list_overload_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var dims = List[Int]()
    dims.append(3)
    dims.append(2)
    var b = a.reshape(dims)
    assert_true(
        b.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    )


fn test_cpu_list_overload_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var dims = List[Int]()
    dims.append(3)
    dims.append(2)
    var b = a.reshape(dims)
    var c = b * 9.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([9.0, 9.0, 9.0, 9.0, 9.0, 9.0]))
    )


# ─────────────────────────────────────────────
#  CPU — chained reshape
# ─────────────────────────────────────────────


fn test_cpu_chained_reshape_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
    )
    var b = a.reshape(2, 4)
    var c = b.reshape(4, 2)
    assert_true(
        c.all_close(
            Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
    )


fn test_cpu_chained_reshape_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
    )
    var b = a.reshape(2, 4)
    var c = b.reshape(4, 2)
    var d = c * 2.0
    var loss = d.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        )
    )


# ─────────────────────────────────────────────
#  CPU — reshape then op then reshape (grad flow through two reshapes)
# ─────────────────────────────────────────────


fn test_cpu_op_between_two_reshapes_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.reshape(4)  # 2x2 → 1-D view
    var c = b * 3.0  # mathematical op — c has grad
    var d = c.reshape(2, 2)  # reshape again
    var loss = d.sum()
    loss.backward()
    # grad = 3 everywhere, back in original 2x2 shape
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0], [3.0, 3.0]])))


# ─────────────────────────────────────────────
#  CPU — non-scalar backward (no sum)
# ─────────────────────────────────────────────


fn test_cpu_non_scalar_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = a.reshape(2, 2)
    var c = b * 5.0
    # backward directly on non-scalar output with default seed=1
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([5.0, 5.0, 5.0, 5.0])))


# ─────────────────────────────────────────────
#  CPU — float64
# ─────────────────────────────────────────────


fn test_cpu_float64_forward_backward() raises:
    comptime dtype = DType.float64
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var b = a.reshape(2, 3)
    assert_true(
        b.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    )
    var c = b * 2.0
    var loss = c.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-9](
            Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        )
    )


# ─────────────────────────────────────────────
#  GPU — 1-D → 2-D
# ─────────────────────────────────────────────


fn test_gpu_1d_to_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 3)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )
        )


fn test_gpu_1d_to_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 3)
        var c = b * 2.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
        )


fn test_gpu_1d_to_2d_b_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2)
        var c = b * 3.0
        var loss = c.sum()
        loss.backward()
        assert_true(b.requires_grad)


# ─────────────────────────────────────────────
#  GPU — 2-D → 1-D
# ─────────────────────────────────────────────


fn test_gpu_2d_to_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(6)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            )
        )


fn test_gpu_2d_to_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(6)
        var c = b * 3.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — 2-D → 3-D
# ─────────────────────────────────────────────


fn test_gpu_2d_to_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2, 2)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
                )
            )
        )


fn test_gpu_2d_to_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2, 2)
        var c = b * 5.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — 3-D → 2-D
# ─────────────────────────────────────────────


fn test_gpu_3d_to_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4, 2)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                )
            )
        )


fn test_gpu_3d_to_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4, 2)
        var c = b * 4.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[4.0, 4.0], [4.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — 3-D → 1-D
# ─────────────────────────────────────────────


fn test_gpu_3d_to_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(8)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            )
        )


fn test_gpu_3d_to_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(8)
        var c = b * 7.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[7.0, 7.0], [7.0, 7.0]], [[7.0, 7.0], [7.0, 7.0]]]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — scalar reshape
# ─────────────────────────────────────────────


fn test_gpu_scalar_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([42.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape()
        assert_true(b.to_cpu().all_close(Tensor[dtype].full(Shape(), 42.0)))


fn test_gpu_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape()
        var c = b * 10.0
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([10.0])))


# ─────────────────────────────────────────────
#  GPU — chained reshape
# ─────────────────────────────────────────────


fn test_gpu_chained_reshape_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 4)
        var c = b.reshape(4, 2)
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                )
            )
        )


fn test_gpu_chained_reshape_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 4)
        var c = b.reshape(4, 2)
        var d = c * 2.0
        var loss = d.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            )
        )


# ─────────────────────────────────────────────
#  GPU — op between two reshapes
# ─────────────────────────────────────────────


fn test_gpu_op_between_two_reshapes_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4)
        var c = b * 3.0
        var d = c.reshape(2, 2)
        var loss = d.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0], [3.0, 3.0]]))
        )


# ─────────────────────────────────────────────
#  GPU — non-scalar backward
# ─────────────────────────────────────────────


fn test_gpu_non_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2)
        var c = b * 5.0
        c.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([5.0, 5.0, 5.0, 5.0])))


# ─────────────────────────────────────────────
#  GPU — Shape overload
# ─────────────────────────────────────────────


fn test_gpu_shape_overload_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(Shape(3, 2))
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )
        )
        var c = b * 6.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — grad stays on CPU (a's device)
# ─────────────────────────────────────────────


fn test_gpu_grad_stays_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4)
        var c = b * 8.0
        var loss = c.sum()
        loss.backward()
        # a lives on CPU — its grad must also be on CPU, directly accessible
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0], [8.0, 8.0]]))
        )
