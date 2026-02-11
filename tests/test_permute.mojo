from tenmo import Tensor
from intarray import IntArray
from shapes import Shape
from testing import assert_true
from permute import Permute


fn main() raises:
    test_tensor_permute_basic()
    test_tensor_permute_3d_axes()
    test_tensor_permute_inverse()

    print("\n=== Running Tensor.permute gradient tests ===")
    test_tensor_permute_grad_sum_2d()
    test_tensor_permute_grad_scaled_sum_3d()
    test_tensor_permute_grad_inverse_chain()
    test_tensor_permute_grad_partial_ops()
    print("=== All Tensor.permute gradient tests passed ===")


fn test_tensor_permute_basic() raises:
    comptime dtype = DType.float32
    print("Running test_tensor_permute_basic")
    a = Tensor[dtype].arange(0, 12)
    t1 = a.reshape(Shape(3, 4))
    p = t1.permute(IntArray([1, 0]))
    assert_true(p.shape() == Shape(4, 3))
    assert_true(p.strides()[0] == t1.strides()[1])
    assert_true(p.strides()[1] == t1.strides()[0])
    print("Passed basic permute test")


fn test_tensor_permute_3d_axes() raises:
    comptime dtype = DType.float32
    print("Running test_tensor_permute_3d_axes")
    a = Tensor[dtype].arange(0, 60)
    t1 = a.reshape(Shape(3, 4, 5))
    p = t1.permute([2, 0, 1])
    assert_true(p.shape() == Shape(5, 3, 4))
    assert_true(p.rank() == 3)
    print("Passed 3D permutation")


fn test_tensor_permute_inverse() raises:
    comptime dtype = DType.float32
    print("Running test_tensor_permute_inverse")
    a = Tensor[dtype].arange(0, 24)
    t1 = a.reshape(Shape(2, 3, 4))
    p = Permute[dtype].forward(t1, IntArray([2, 0, 1]))
    inv = Permute[dtype].forward(p, IntArray([1, 2, 0]))
    assert_true(inv.shape() == t1.shape())
    assert_true(inv.all_close(t1))
    print("Passed inverse permutation test")


fn test_tensor_permute_grad_sum_2d() raises:
    print("test_tensor_permute_grad_sum_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 12)
    t = a.reshape(Shape(3, 4))
    t.requires_grad_(True)
    var p = Permute[dtype].forward(t, IntArray([1, 0]))
    # p.sum() should produce gradient of ones when backpropagated
    s = p.sum()
    s.backward()
    var expected = (
        Tensor[dtype]
        .d2([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        .to_dtype[dtype]()
    )
    assert_true(t.grad().all_close(expected))
    print("Passed test_tensor_permute_grad_sum_2d")


fn test_tensor_permute_grad_inverse_chain() raises:
    print("test_tensor_permute_grad_inverse_chain")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 24)
    t = a.reshape(Shape(2, 3, 4))
    t.requires_grad_(True)
    # Apply a permutation and then its inverse (should return original layout)
    var p = Permute[dtype].forward(t, IntArray([1, 2, 0]))
    # inverse of [1,2,0] is [2,0,1] (since inverse[ permutation[i] ] = i)
    var r = Permute[dtype].forward(
        p, IntArray([2, 0, 1])
    )  # r should match t's layout
    s = r.sum()
    s.backward()
    var expected = (
        Tensor[dtype]
        .d3(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        .to_dtype[dtype]()
    )
    assert_true(t.grad().all_close(expected)
    print("Passed test_tensor_permute_grad_inverse_chain")


fn test_tensor_permute_grad_partial_ops() raises:
    print("test_tensor_permute_grad_partial_ops")
    comptime dtype = DType.float32
    print(
        "This test checks a permute followed by a partial reduction along"
        " permuted axes"
    )
    var t = Tensor[dtype].arange(0, 60, requires_grad=True)
    var r = t.reshape(Shape(3, 4, 5))
    # permute to (5,3,4) then mean over first axis (axis=0 of permuted)
    var p = r.permute([2, 0, 1])
    var m = p.mean(axes=[0], keepdims=False)  # result shape (3,4)
    # backprop: m has shape (3,4); grad of t should be broadcasted back along axis 2 (size 5)
    s = m.sum()
    s.backward()
    # since m.sum() -> sums all elements of p, gradient on t should be ones

    assert_true(t.grad().all_close(Tensor.full(Shape(60), Scalar[dtype](0.2))))
    print("Passed test_tensor_permute_grad_partial_ops")


fn test_tensor_permute_grad_scaled_sum_3d() raises:
    print("test_tensor_permute_grad_scaled_sum_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 60)
    t = a.reshape(Shape(3, 4, 5))
    t.requires_grad_(True)
    var p = Permute[dtype].forward(t, IntArray([2, 0, 1]))  # shape -> (5,3,4)
    var q = p * 2.0
    s = q.sum()
    s.backward()
    # corrected expected shape (3, 4, 5)
    var expected = (
        Tensor[dtype]
        .d3(
            [
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
            ]
        )
        .to_dtype[dtype]()
    )
    assert_true(t.grad().all_close(expected))
    print("Passed test_tensor_permute_grad_scaled_sum_3d")
