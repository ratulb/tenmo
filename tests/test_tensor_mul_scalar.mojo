from tensors import Tensor
from testing import assert_true, assert_false


fn test_tensor_mul_scalar() raises:
    tensor = Tensor.rand(4, 5, requires_grad=True)
    result1 = 2 + tensor
    result2 = tensor * 3
    result3 = tensor * 42
    result4 = tensor + tensor

    try:
        assert_true(
            tensor.has_grad()
            and result1.has_grad()
            and result2.has_grad()
            and result1.grad_func() is not None
            and result2.grad_func() is not None
            and result3.grad_func() is not None
            and result4.grad_func() is not None,
            "tensor mul scalar grad initialization assertion failed",
        )
        result1.grad[] += 100
        result1.open_gradbox() += 1
        result1.invoke_grad_fn()
        assert_true(
            (result1.grad[] == tensor.grad[]).all_true(),
            "tensor and result1 grad equality assertion failed",
        )
        # tensor.print_grad()
        result2.grad[].fill(999)

        result2.invoke_grad_fn()

        fn pred(elem: Scalar[DType.float32]) -> Bool:
            return elem == 3098  # 999 * 3 + 101

        assert_true(
            tensor.grad[].for_all(pred),
            "Grad for result2 and tensor assertion failed",
        )
        result3.open_gradbox() += -42
        result3.invoke_grad_fn()

        fn pred2(elem: Scalar[DType.float32]) -> Bool:
            return elem == 1334

        assert_true(
            tensor.grad[].for_all(pred2),
            "Grad for result3 and tensor assertion failed",
        )
        result4.open_gradbox() += 0.45
        result4.invoke_grad_fn()

        fn pred3(elem: Scalar[DType.float32]) -> Bool:
            return elem == 1334.90

        assert_true(
            tensor.grad[].for_all(pred3),
            "Grad for result4 and tensor assertion failed",
        )

    finally:
        tensor.free()
        result1.free()
        result2.free()
        result3.free()
        result4.free()


fn main() raises:
    test_tensor_mul_scalar()
