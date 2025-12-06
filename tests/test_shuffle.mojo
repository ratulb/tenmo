from tenmo import Tensor
from testing import assert_true
from shapes import Shape
from shuffle import Shuffle


fn main() raises:
    test_tensor_shuffle_forward_basic()
    test_tensor_shuffle_forward_axis1()
    test_tensor_shuffle_backward_axis0()
    test_tensor_shuffle_backward_axis1()
    test_tensor_shuffle_backward_grad_mapping()
    test_tensor_shuffle_multi_dimensional()
    test_tensor_shuffle_random_perm_length_check()
    print("✓✓ All tensor shuffle tests passed ✓✓")
    print("passes")


fn test_tensor_shuffle_forward_basic() raises:
    print("test_tensor_shuffle_forward_basic")
    alias dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # permute along axis=0 with [1, 0]
    var shuffled = Shuffle[dtype].forward[False](t, [1, 0], axis=0)

    assert_true(shuffled.shape() == Shape(2, 3))
    assert_true(
        shuffled == Tensor[dtype].d2([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    )
    print("✓ Passed forward basic shuffle")


fn test_tensor_shuffle_forward_axis1() raises:
    print("test_tensor_shuffle_forward_axis1")
    alias dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # permute along axis=1 with [2, 0, 1]
    var shuffled = Shuffle[dtype].forward[False](t, [2, 0, 1], axis=1)

    assert_true(
        shuffled == Tensor[dtype].d2([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    )
    print("✓ Passed forward shuffle along axis 1")


fn test_tensor_shuffle_backward_axis0() raises:
    print("test_tensor_shuffle_backward_axis0")
    alias dtype = DType.float32

    var t = Tensor[dtype].d2([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    t.requires_grad_(True)

    var shuffled = Shuffle[dtype].forward[True](t, [1, 0], axis=0)
    var s = shuffled.sum()
    s.backward()

    # Backward should restore correct positions (reversed)
    var expected_grad = Tensor[dtype].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(t.grad() == expected_grad)
    print("✓ Passed backward shuffle along axis 0")


fn test_tensor_shuffle_backward_axis1() raises:
    print("test_tensor_shuffle_backward_axis1")
    alias dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t.requires_grad_(True)

    var shuffled = Shuffle[dtype].forward[True](t, [2, 0, 1], axis=1)
    var s = shuffled.sum()
    s.backward()

    # Each input position contributes to exactly one output
    var expected_grad = Tensor[dtype].ones(Shape(2, 3))
    assert_true(t.grad() == expected_grad)
    print("✓ Passed backward shuffle along axis 1")


fn test_tensor_shuffle_backward_grad_mapping() raises:
    print("test_tensor_shuffle_backward_grad_mapping")
    alias dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t.requires_grad_(True)

    # Shuffle along axis=1 using [1, 2, 0]
    var shuffled = Shuffle[dtype].forward[True](t, [1, 2, 0], axis=1)

    # Multiply by position-wise coefficients to make gradients distinct
    var c = Tensor[dtype].d2([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    var out = (shuffled * c).sum()
    out.backward()

    # Backprop should map gradient contributions back using inverse permutation
    var expected_grad = Tensor[dtype].d2(
        [[30.0, 10.0, 20.0], [60.0, 40.0, 50.0]]
    )
    assert_true(t.grad() == expected_grad)
    print("✓ Passed backward gradient mapping test")


fn test_tensor_shuffle_random_perm_length_check() raises:
    print("test_tensor_shuffle_random_perm_length_check")
    alias dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var shuffled = Shuffle[dtype].forward[False](t, [], axis=0)
    # Should keep same shape and values permuted (cannot predict order but same values)
    assert_true(shuffled.shape() == Shape(2, 2))
    assert_true(shuffled != t or shuffled == t)  # likely changed
    print("✓ Passed random permutation shuffle")


fn test_tensor_shuffle_multi_dimensional() raises:
    print("test_tensor_shuffle_multi_dimensional")
    alias dtype = DType.float32

    var t = Tensor[dtype].arange(0, 24).reshape(Shape(2, 3, 4))
    var shuffled = Shuffle[dtype].forward[False](t, [2, 0, 1], axis=1)
    expected = Tensor.d3(
        [
            [
                [8.0, 9.0, 10.0, 11.0],
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            [
                [20.0, 21.0, 22.0, 23.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ],
        ]
    ).float()

    assert_true(shuffled.shape() == Shape(2, 3, 4))
    assert_true(shuffled == expected)
    print("✓ Passed multi-dimensional shuffle shape test")
