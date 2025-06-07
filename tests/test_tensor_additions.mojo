from tensors import Tensor
from testing import assert_true


fn test_tensor_additions_1() raises:
    tensor1 = Tensor.rand(4, 4, requires_grad=True)
    tensor2 = Tensor.rand(4, 4, requires_grad=True)
    tensor3 = Tensor.rand(4, 4, requires_grad=True)
    tensor4 = Tensor.rand(4, 4, requires_grad=True)
    result1 = tensor1 + tensor2 * 3 + tensor3 * tensor4

    print(
        "Ptr addresses here: tensor1: ",
        tensor1.pointer(),
        "tensor2: ",
        tensor2.pointer(),
        "result1: ",
        result1.pointer(),
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
    test_tensor_additions_1()
