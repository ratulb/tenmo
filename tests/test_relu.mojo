from tenmo import Tensor
from testing import assert_true
from relu import ReLU

fn main() raises:
    test_relu_basic()
    test_relu_multidim()

fn test_relu_basic() raises:
    print("test_relu_basic")
    alias dtype = DType.float32

    var t = Tensor[dtype].d1([-1.0, 0.0, 1.0, 2.0])
    t.requires_grad_(True)
    var out = ReLU[dtype].forward[True](t)
    out.sum().backward()

    assert_true(out == Tensor[dtype].d1([0.0, 0.0, 1.0, 2.0]))
    assert_true(t.grad() == Tensor[dtype].d1([0.0, 0.0, 1.0, 1.0]))
    print("✓ Passed ReLU forward and backward")

fn test_relu_multidim() raises:
    print("test_relu_multidim")
    alias dtype = DType.float32

    # 2×3 input tensor
    var t = Tensor[dtype].d2([
        [-1.0, 2.0, 0.0],
        [3.0, -4.0, 5.0],
    ])
    t.requires_grad_(True)

    # Apply ReLU
    var out = t.relu()
    assert_true(out == Tensor[dtype].d2([
        [0.0, 2.0, 0.0],
        [3.0, 0.0, 5.0],
    ]))

    # Backward on sum of outputs
    out.sum().backward()

    # Gradient should be 1 where input > 0, else 0
    var expected_grad = Tensor[dtype].d2([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ])
    assert_true(t.grad() == expected_grad)

    print("✓ Passed ReLU multidimensional forward/backward test")

