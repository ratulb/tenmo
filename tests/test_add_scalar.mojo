from tensors import Tensor
from testing import assert_true, assert_false
from memory import UnsafePointer

fn test_grad_init_and_grad_func_association_post_mul_add_by_factor() raises:
    tensor = Tensor.rand(4, 5, requires_grad=True)
    product = tensor * 2

    try:
        print("Tensor has grad and memory addr: ", tensor.has_grad(), UnsafePointer(to=tensor))
        print("Product has grad and memory addr: ",  product.has_grad(), UnsafePointer(to=product))
        assert_true(
            tensor.has_grad() == False and product.has_grad() == False,
            (
                "Multiplication by factor grad should not be initialized before"
                " grad_func invocation assertion failed"
            ),
        )
        product.invoke_grad_fn()
        assert_true(
            tensor.has_grad() and product.has_grad() == False,
            (
                "Multiplication by factor grad should not be initialized before"
                " grad_func invocation assertion failed"
            ),
        )

        assert_true(
            tensor.grad_func() is None and product.grad_func() is not None,
            "Multiplication by factor grad func association assertion failed",
        )

        summ = product + 1000
        assert_true(
            summ.has_grad() == False and summ.grad_func() is not None,
            (
                "Scalar addition lazy grad initialization and grad_func association"
                " assertion failed"
            ),
        )
    #_ = tensor
    #_ = product
    #_ = summ
    finally:
        tensor.free()
        product.free()
        #summ.free()

fn main() raises:
    test_grad_init_and_grad_func_association_post_mul_add_by_factor()
