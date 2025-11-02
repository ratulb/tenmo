from tenmo import Tensor
from shapes import Shape
from testing import assert_true

fn main() raises:
    test_2d_transpose_no_axes()
    test_2d_transpose_explicit_axes()
    test_3d_transpose_axes_0_1()
    test_3d_transpose_axes_1_2()
    test_transpose_chain_operations()
    test_transpose_with_matmul()
    test_transpose_scalar_equivalent()
    test_transpose_1d_no_change()
    test_4d_transpose_complex_axes()


fn test_2d_transpose_no_axes() raises:
    print("test_2d_transpose_no_axes")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var b = a.transpose()

    # Forward pass validation
    var expected = Tensor.d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape.of(3, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))

fn test_2d_transpose_explicit_axes() raises:
    print("test_2d_transpose_explicit_axes")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var b = a.transpose(1, 0)

    # Forward pass validation
    var expected = Tensor.d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape.of(3, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))


fn test_3d_transpose_axes_0_1() raises:
    print("test_3d_transpose_axes_0_1")
    var a = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
    var b = a.transpose(1, 0)

    # Forward pass validation
    var expected = Tensor.d3([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]])
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape.of(2, 2, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d3([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))

fn test_3d_transpose_axes_1_2() raises:
    print("test_3d_transpose_axes_1_2")
    var a = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
    #var b = a.transpose(1, 2)
    var b = a.transpose(2, 1)

    # Forward pass validation
    var expected = Tensor.d3([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]])
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape.of(2, 2, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d3([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))


fn test_4d_transpose_complex_axes() raises:
    print("test_4d_transpose_complex_axes")
    var a = Tensor.d4([
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
    ], requires_grad=True)
    var b = a.transpose(0, 2, 3, 1)

    # Forward pass validation - check shape
    assert_true(b.shape() == Shape.of(2, 2, 2, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d4([
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    ])
    assert_true(a.grad().all_close(expected_grad))

fn test_transpose_chain_operations() raises:
    print("test_transpose_chain_operations")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.transpose()
    var c = b.transpose()

    # Forward pass validation - double transpose should return original
    assert_true(c.all_close(a))

    # Backward pass validation
    var loss = c.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0], [1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))

fn test_transpose_with_matmul() raises:
    print("test_transpose_with_matmul")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var a_t = a.transpose()
    _="""var result = a_t @ b

    # Forward pass validation
    var expected = Tensor.d2([[26.0, 30.0], [38.0, 44.0]])
    assert_true(result.all_close(expected))

    # Backward pass validation
    var loss = result.sum()
    loss.backward()
    var expected_a_grad = Tensor.d2([[11.0, 15.0], [11.0, 15.0]])
    var expected_b_grad = Tensor.d2([[4.0, 4.0], [6.0, 6.0]])
    assert_true(a.grad().all_close(expected_a_grad))
    assert_true(b.grad().all_close(expected_b_grad))"""

fn test_transpose_scalar_equivalent() raises:
    print("test_transpose_scalar_equivalent")
    var a = Tensor.scalar(5.0, requires_grad=True)
    var b = a.transpose()

    # Scalar transpose should return same scalar
    assert_true(b.item() == 5.0)

    # Backward pass
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().item() == 1.0)

fn test_transpose_1d_no_change() raises:
    print("test_transpose_1d_no_change")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.transpose()

    # 1D transpose should return same tensor
    assert_true(b.all_close(a))

    # Backward pass
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d1([1.0, 1.0, 1.0])
    assert_true(a.grad().all_close(expected_grad))
