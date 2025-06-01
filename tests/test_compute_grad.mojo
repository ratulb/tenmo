from tensors import Tensor
from testing import assert_true, assert_false
from memory import UnsafePointer


fn test_compute_grad_mul_by_scalar_backprop() raises:
    tensor = Tensor.rand(4, 5, requires_grad=True)
    product = tensor * 2
    assert_true(
        tensor.grad.__as_bool__() and product.grad.__as_bool__(),
        (
            "Tensor compute grad manual backprop post multiplication with"
            " requires_grad grad UnsafePointer 'True' failed"
        ),
    )
    tensor_ptr = UnsafePointer(to=tensor)
    _= tensor_ptr[].grad[] + 42
    _= tensor_ptr[].grad[] * 42


    product_ptr = UnsafePointer(to=product)
    _= product_ptr[].grad[] + 42
    result = tensor + product_ptr[].grad[] * 42
    result.print()

    sum = product + 1000
    assert_true(
        sum.grad.__as_bool__(),
        (
            "Tensor compute grad manual backprop summation grad init failed"
        ),
    )

    sum.print()
    sum.grad[].print()

fn test_compute_grad_mul_by_scalar_manual_use_pointers() raises:
    tensor = Tensor.rand(4)
    product = tensor * 2
    assert_false(
        tensor.grad.__as_bool__(),
        (
            "Tensor compute grad manual pointers post multiplication without"
            " requires_grad grad UnsafePointer 'False' failed"
        ),
    )
    tensor.requires_grad = True
    product.requires_grad = True
    tensor.init_grad_tensor()
    product.init_grad_tensor()
    assert_true(
        tensor.grad.__as_bool__() and product.grad.__as_bool__(),
        "Tensor compute grad manual pointers grad initialization check failed",
    )
    x = tensor * 2
    y = product * 2
    z = x + y
    tensor_ptr = UnsafePointer(to=tensor)
    product_ptr = UnsafePointer(to=product)
    a = tensor_ptr[] * 2
    b = product_ptr[] * 2
    c = a + b
    all_equal = z == c
    all_equal.print()
    assert_true(all_equal.all_true(), "Arithmetic via pointer failed")
    x = tensor.grad[] * 2
    y = product.grad[] * 2
    z = x + y
    a = tensor_ptr[].grad[] * 2
    b = product_ptr[].grad[] * 2
    c = a + b
    all_equal = z == c
    assert_true(all_equal.all_true(), "Arithmetic of grad via pointer failed")


fn test_compute_grad_mul_by_scalar_manual() raises:
    tensor = Tensor.rand(4)
    assert_false(
        tensor.requires_grad,
        "Tensor compute grad manual requires_grad 'False' failed",
    )

    tensor.init_grad_tensor()
    assert_false(
        tensor.requires_grad,
        (
            "Tensor compute grad manual post init_grad_tensor requires_grad"
            " 'False' failed"
        ),
    )

    # Setting requires_grad manually
    tensor.requires_grad = True
    assert_true(
        tensor.requires_grad,
        "Tensor compute grad manual requires_grad 'True' failed",
    )

    tensor.init_grad_tensor()
    assert_true(
        tensor.requires_grad,
        (
            "Tensor compute grad manual post init_grad_tensor requires_grad"
            " 'True' failed"
        ),
    )
    assert_true(
        tensor.grad.__as_bool__(),
        (
            "Tensor compute grad manual post init_grad_tensor grad"
            " UnsafePointer 'True' failed"
        ),
    )

    # Get a fresh tensor

    tensor = Tensor.rand(4)
    product = tensor * 2
    assert_false(
        tensor.grad.__as_bool__(),
        (
            "Tensor compute grad manual post multiplication without"
            " requires_grad grad UnsafePointer 'False' failed"
        ),
    )

    assert_false(
        product.grad.__as_bool__(),
        (
            "Tensor output(prouct) compute grad manual post multiplication"
            " without requires_grad grad UnsafePointer 'False' failed"
        ),
    )
    product.requires_grad = True
    product.init_grad_tensor()

    assert_true(
        product.shape == product.grad[].shape,
        (
            "Tensor output(prouct) compute grade manual product shape and"
            " grad's shape equality failed"
        ),
    )

    # assert_true(product.all_true(), "Tensor compute grad failed")


fn test_compute_grad_mul_by_scalar() raises:
    tensor = Tensor.rand(4, requires_grad=True)
    product = tensor * 2

    print("\nTensor\n")
    tensor.print()

    print("\nTensor grad\n")
    tensor.grad[].print()

    print("\nProduct\n")
    product.print()

    print("\nProduct grad\n")
    product.grad[].print()

    print("\nProduct grad shape\n")
    print(product.grad[].shape.__str__())

    product_grad_scaled = product.grad[] * 3

    print("\nProduct grad scaled\n")
    product_grad_scaled.print()

    # product.invoke_grad_fn()

    # assert_true(product.all_true(), "Tensor compute grad failed")


fn main() raises:
    test_compute_grad_mul_by_scalar_backprop()
    test_compute_grad_mul_by_scalar_manual_use_pointers()
    # test_compute_grad_mul_by_scalar()
    test_compute_grad_mul_by_scalar_manual()
