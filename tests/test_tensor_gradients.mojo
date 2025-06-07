from tensors import Tensor
from testing import assert_true, assert_false
from common_utils import log_debug


fn test_tensor_additions_2() raises:
    A = Tensor.zeros(1, requires_grad=True)
    B = Tensor.zeros(1, requires_grad=True)
    D = Tensor.zeros(1, requires_grad=True)
    A.fill(2.0)
    B.fill(3.0)
    D.fill(4.0)
    C = A + B
    E = C + D
    F = E + A
    G = F * A
    # G = (A + B + D + A) * A
    #start = True
    #G.backward(start)

    Tensor.walk_backward(G)

    expected_gradients = List[Float32](15.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0)

    tensors = List(A, B, C, D, E, F, G)

    for i in range(len(tensors)):
        each = tensors[i]
        assert_false(
            each.grad[].requires_grad,
            "Gradient's requires_grad = 'False' assertion failed",
        )
        assert_true(
            each.grad[][0] == expected_gradients[i],
            "Gradient equality assertion failed",
        )


fn test_tensor_additions_1() raises:
    # log_debug("test_tensor_additions_1")
    tensor1 = Tensor.rand(4, 4, requires_grad=True)
    tensor2 = Tensor.rand(4, 4, requires_grad=True)
    tensor3 = Tensor.rand(4, 4, requires_grad=True)
    tensor4 = Tensor.rand(4, 4, requires_grad=True)
    result1 = tensor1 + tensor2 * 3 + tensor3 * tensor4

    print(
        "Ptr addresses here: tensor1: ",
        tensor1.address(),
        "tensor2: ",
        tensor2.address(),
        "result1: ",
        result1.address(),
    )
    result1.print()

    try:
        assert_true(
            tensor1.has_grad()
            and tensor2.has_grad()
            and result1.has_grad()
            and result1.grad_func() is not None,
            "Tensor addition grad initialization assertion failed",
        )
        # result1.grad[].fill(100)
        start = True
        # result1.invoke_grad_fn()
        # result1.backward(start)

        Tensor.walk_backward(result1)
        tensor1.grad[].print()
        tensor2.grad[].print()
        tensor3.grad[].print()

        Tensor.walk_backward(result1)
        tensor1.grad[].print()
        tensor2.grad[].print()
        tensor3.grad[].print()

        Tensor.walk_backward(result1)
        # print(len(result1.ancestors.value()))
        # result1.invoke_grad_fn()

        tensor1.grad[].print()
        tensor2.grad[].print()
        tensor3.grad[].print()

    finally:
        tensor1.free()
        tensor2.free()
        tensor3.free()
        tensor4.free()
        result1.free()


fn main() raises:
    test_tensor_additions_2()
    # test_tensor_additions_1()
