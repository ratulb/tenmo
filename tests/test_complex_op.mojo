from tensors import Tensor
from testing import assert_true, assert_false


fn test_grad_init_and_grad_func_association_mul_2_tensors() raises:
    tensor1 = Tensor.rand(4, 5, requires_grad=True)
    tensor2 = Tensor.rand(4, 5, requires_grad=True)
    product = tensor1 + tensor2

    #tensor1.print()
    #tensor2.print()
    #product.print()
    print(tensor1.has_grad(), tensor2.has_grad(), product.has_grad(), product.grad_func() is not None)

    try:
        assert_true(
            tensor1.has_grad()
            and tensor2.has_grad()
            and product.has_grad()
            and product.grad_func() is not None,
            "2 tensors multiplication grad initialization assertion failed",
        )
        product.invoke_grad_fn()
        product.invoke_grad_fn()

        assert_true(
            tensor1.grad_func() is None
            and tensor2.grad_func() is None
            and product.grad_func() is not None,
            "2 tensors mul grad func association assertion failed",
        )

    finally:
        tensor1.free()
        tensor2.free()
        product.free()
        #pass


fn main() raises:
    test_grad_init_and_grad_func_association_mul_2_tensors()
