from tensors import Tensor
from testing import assert_true, assert_false


fn test_tensor_add_scalar() raises:
    tensor = Tensor.rand(4, 5, requires_grad=True)
    result1 = tensor + 2
    result2 = 2 + tensor
    result3 = tensor * 2
    print("Tensor:", tensor.address(), "result1: ", result1.address(), "result2: ", result2.address())
    tensor.print_grad()

    try:
        assert_true(
            tensor.has_grad() and result1.has_grad()
            # and result2.has_grad()
            and result1.grad_func() is not None,
            # and result2.grad_func() is not None,
            "tensor mul scalar grad initialization assertion failed",
        )
        result2.invoke_grad_fn()
        result2.invoke_grad_fn()
        result2.invoke_grad_fn()
        result1.print_grad()
        result1.print_grad()
        result2.invoke_grad_fn()
        result2.invoke_grad_fn()
        result1.print_grad()
        result1.print_grad()
        result1.print_grad()
        result1.grad[] += 1
        result2.grad[] += 10
        result2.print_grad()
        result1.print_grad()
        result2.invoke_grad_fn()
        tensor.print_grad()
        print("Problem here")
        result1.invoke_grad_fn()
        print("Problem here****")
        tensor.print_grad()
        result2.invoke_grad_fn()
        result1.invoke_grad_fn()

    finally:
        tensor.free()
        result1.free()
        # result2.free()
        # pass


fn main() raises:
    test_tensor_add_scalar()
