from std.testing import assert_true
from tensor import Tensor
from std.sys import has_accelerator
from shapes import Shape


# ═══════════════════════════════════════════════════════════════════════════════
# SQRT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_uop_sqrt_forward_1d_cpu() raises:
    print("test_uop_sqrt_forward_1d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 9.0, 16.0])
    var result = a.sqrt()
    assert_true(result.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])))


fn test_uop_sqrt_forward_2d_cpu() raises:
    print("test_uop_sqrt_forward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]])
    var result = a.sqrt()
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))


fn test_uop_sqrt_forward_3d_cpu() raises:
    print("test_uop_sqrt_forward_3d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape.of(2, 3, 4), 4.0)
    var result = a.sqrt()
    assert_true(result.all_close(Tensor[dtype].full(Shape.of(2, 3, 4), 2.0)))


fn test_uop_sqrt_backward_1d_cpu() raises:
    print("test_uop_sqrt_backward_1d_cpu")
    comptime dtype = DType.float32
    # d/dx sqrt(x) = 1 / (2 * sqrt(x))
    # x=4 → grad = 1/(2*2) = 0.25
    var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
    var loss = a.sqrt().sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](
            Tensor[dtype].d1([0.25, 0.16667, 0.125])
        )
    )


fn test_uop_sqrt_backward_2d_cpu() raises:
    print("test_uop_sqrt_backward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
    var loss = a.sqrt().sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.25, 0.16667], [0.125, 0.1]])
        )
    )


fn test_uop_sqrt_grad_flow_cpu() raises:
    print("test_uop_sqrt_grad_flow_cpu")
    comptime dtype = DType.float32
    # C = sqrt(A) * 2, dC/dA = 2 * 1/(2*sqrt(A)) = 1/sqrt(A)
    var a = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
    var loss = (a.sqrt() * 2).sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.33333]))
    )


fn test_uop_sqrt_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_sqrt_gpu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 4.0], [9.0, 16.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.sqrt()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [0.16667, 0.125]])
            )
        )


fn test_uop_sqrt_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_uop_sqrt_gpu_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape.of(2, 3, 4), 4.0, requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.sqrt()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].full(Shape.of(2, 3, 4), 2.0)
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].full(Shape.of(2, 3, 4), 0.25)
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_uop_negate_forward_1d_cpu() raises:
    print("test_uop_negate_forward_1d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, -2.0, 3.0, -4.0])
    var result = -a
    assert_true(result.all_close(Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0])))


fn test_uop_negate_forward_2d_cpu() raises:
    print("test_uop_negate_forward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, -2.0], [-3.0, 4.0]])
    var result = -a
    assert_true(result.all_close(Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]])))


fn test_uop_negate_forward_3d_cpu() raises:
    print("test_uop_negate_forward_3d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape.of(2, 3, 4), 5.0)
    var result = -a
    assert_true(result.all_close(Tensor[dtype].full(Shape.of(2, 3, 4), -5.0)))


fn test_uop_negate_backward_1d_cpu() raises:
    print("test_uop_negate_backward_1d_cpu")
    comptime dtype = DType.float32
    # d/dx (-x) = -1
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = (-a).sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape.of(3), -1.0))
    )


fn test_uop_negate_backward_2d_cpu() raises:
    print("test_uop_negate_backward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var loss = (-a).sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape.of(2, 2), -1.0))
    )


fn test_uop_negate_grad_flow_cpu() raises:
    print("test_uop_negate_grad_flow_cpu")
    comptime dtype = DType.float32
    # C = -A + A = 0, but grads: dC/dA = -1 + 1 = 0
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = (-a + a).sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape.of(3), 0.0))
    )


fn test_uop_negate_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_negate_gpu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, -2.0], [-3.0, 4.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = -a_gpu
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]])
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].full(Shape.of(2, 2), -1.0))
        )


fn test_uop_negate_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_uop_negate_gpu_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(
            Shape.of(2, 3, 4), 5.0, requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = -a_gpu
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].full(Shape.of(2, 3, 4), -5.0)
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].full(Shape.of(2, 3, 4), -1.0)
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ABS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_uop_abs_forward_1d_cpu() raises:
    print("test_uop_abs_forward_1d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0])
    var result = a.__abs__()
    assert_true(result.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])))


fn test_uop_abs_forward_2d_cpu() raises:
    print("test_uop_abs_forward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [-3.0, -4.0]])
    var result = a.__abs__()
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))


fn test_uop_abs_forward_3d_cpu() raises:
    print("test_uop_abs_forward_3d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape.of(2, 3, 4), -5.0)
    var result = a.__abs__()
    assert_true(result.all_close(Tensor[dtype].full(Shape.of(2, 3, 4), 5.0)))


fn test_uop_abs_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_abs_gpu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[-1.0, 2.0], [-3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.__abs__()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
            )
        )


fn test_uop_abs_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_uop_abs_gpu_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape.of(2, 3, 4), -5.0)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.__abs__()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].full(Shape.of(2, 3, 4), 5.0)
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RELU TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_uop_relu_forward_1d_cpu() raises:
    print("test_uop_relu_forward_1d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-2.0, -1.0, 0.0, 1.0, 2.0])
    var result = a.relu()
    assert_true(
        result.all_close(Tensor[dtype].d1([0.0, 0.0, 0.0, 1.0, 2.0]))
    )


fn test_uop_relu_forward_2d_cpu() raises:
    print("test_uop_relu_forward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [-3.0, 4.0]])
    var result = a.relu()
    assert_true(result.all_close(Tensor[dtype].d2([[0.0, 2.0], [0.0, 4.0]])))


fn test_uop_relu_forward_3d_cpu() raises:
    print("test_uop_relu_forward_3d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape.of(2, 3, 4), -1.0)
    var result = a.relu()
    assert_true(result.all_close(Tensor[dtype].full(Shape.of(2, 3, 4), 0.0)))


fn test_uop_relu_backward_1d_cpu() raises:
    print("test_uop_relu_backward_1d_cpu")
    comptime dtype = DType.float32
    # d/dx relu(x) = 1 if x > 0 else 0
    var a = Tensor[dtype].d1([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    var loss = a.relu().sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 1.0, 1.0]))
    )


fn test_uop_relu_backward_2d_cpu() raises:
    print("test_uop_relu_backward_2d_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[-1.0, 2.0], [-3.0, 4.0]], requires_grad=True
    )
    var loss = a.relu().sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [0.0, 1.0]]))
    )


fn test_uop_relu_grad_flow_cpu() raises:
    print("test_uop_relu_grad_flow_cpu")
    comptime dtype = DType.float32
    # C = relu(A) * 2
    # dC/dA = 2 where A > 0, 0 elsewhere
    var a = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    var loss = (a.relu() * 2).sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([0.0, 2.0, 0.0, 2.0]))
    )


fn test_uop_relu_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_relu_gpu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[-1.0, 2.0], [-3.0, 4.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.relu()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[0.0, 2.0], [0.0, 4.0]])
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].d2([[0.0, 1.0], [0.0, 1.0]])
            )
        )


fn test_uop_relu_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_uop_relu_gpu_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(
            Shape.of(2, 3, 4), -1.0, requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.relu()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].full(Shape.of(2, 3, 4), 0.0)
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].full(Shape.of(2, 3, 4), 0.0)
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INVERT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_uop_invert_bool_forward_1d_cpu() raises:
    print("test_uop_invert_bool_forward_1d_cpu")
    var a = Tensor[DType.bool].d1([True, False, True, True])
    var result = ~a
    assert_true(
        result == Tensor[DType.bool].d1([False, True, False, False])
    )


fn test_uop_invert_bool_forward_2d_cpu() raises:
    print("test_uop_invert_bool_forward_2d_cpu")
    var a = Tensor[DType.bool].d2([[True, False], [False, True]])
    var result = ~a
    assert_true(
        result == Tensor[DType.bool].d2([[False, True], [True, False]])
    )


fn test_uop_invert_bool_forward_3d_cpu() raises:
    print("test_uop_invert_bool_forward_3d_cpu")
    var a = Tensor[DType.bool].full(Shape.of(2, 3, 4), Scalar[DType.bool](True))
    var result = ~a
    assert_true(
        result == Tensor[DType.bool].full(
            Shape.of(2, 3, 4), Scalar[DType.bool](False)
        )
    )


fn test_uop_invert_bool_double_cpu() raises:
    print("test_uop_invert_bool_double_cpu")
    # ~~a == a
    var a = Tensor[DType.bool].d1([True, False, True, False])
    assert_true(~~a == a)


fn test_uop_invert_int_forward_1d_cpu() raises:
    print("test_uop_invert_int_forward_1d_cpu")
    comptime dtype = DType.int32
    var a = Tensor[dtype].d1([0, 1, -1, 5])
    var result = ~a
    # ~x = -(x+1) for signed integers
    assert_true(result == Tensor[dtype].d1([-1, -2, 0, -6]))


fn test_uop_invert_int_forward_2d_cpu() raises:
    print("test_uop_invert_int_forward_2d_cpu")
    comptime dtype = DType.int32
    var a = Tensor[dtype].d2([[1, 2], [3, 4]])
    var result = ~a
    assert_true(result == Tensor[dtype].d2([[-2, -3], [-4, -5]]))


fn test_uop_invert_int_forward_3d_cpu() raises:
    print("test_uop_invert_int_forward_3d_cpu")
    comptime dtype = DType.int32
    var a = Tensor[dtype].full(Shape.of(2, 3, 4), 1)
    var result = ~a
    assert_true(result == Tensor[dtype].full(Shape.of(2, 3, 4), -2))


fn test_uop_invert_bool_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_invert_bool_gpu")
        var a_cpu = Tensor[DType.bool].d2(
            [[True, False], [False, True]]
        )
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(
            result.to_cpu() == Tensor[DType.bool].d2(
                [[False, True], [True, False]]
            )
        )


fn test_uop_invert_bool_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_uop_invert_bool_gpu_3d")
        var a_cpu = Tensor[DType.bool].full(
            Shape.of(2, 3, 4), Scalar[DType.bool](True)
        )
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(
            result.to_cpu() == Tensor[DType.bool].full(
                Shape.of(2, 3, 4), Scalar[DType.bool](False)
            )
        )


fn test_uop_invert_bool_gpu_double() raises:
    comptime if has_accelerator():
        print("test_uop_invert_bool_gpu_double")
        # ~~a == a on GPU
        var a_cpu = Tensor[DType.bool].d1([True, False, True, False])
        var a_gpu = a_cpu.to_gpu()
        assert_true((~~a_gpu).to_cpu() == a_cpu)


fn test_uop_invert_int_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_invert_int_gpu")
        comptime dtype = DType.int32
        var a_cpu = Tensor[dtype].d2([[1, 2], [3, 4]])
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(
            result.to_cpu() == Tensor[dtype].d2([[-2, -3], [-4, -5]])
        )


fn test_uop_invert_int_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_uop_invert_int_gpu_3d")
        comptime dtype = DType.int32
        var a_cpu = Tensor[dtype].full(Shape.of(2, 3, 4), 1)
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(
            result.to_cpu() == Tensor[dtype].full(Shape.of(2, 3, 4), -2)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS OP GRAD FLOW — ensure INVERT did not break other ops
# ═══════════════════════════════════════════════════════════════════════════════

fn test_uop_cross_sqrt_negate_grad_flow_cpu() raises:
    print("test_uop_cross_sqrt_negate_grad_flow_cpu")
    comptime dtype = DType.float32
    # C = -sqrt(A), dC/dA = -1/(2*sqrt(A))
    var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
    var loss = (-a.sqrt()).sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](
            Tensor[dtype].d1([-0.25, -0.16667, -0.125])
        )
    )


fn test_uop_cross_relu_negate_grad_flow_cpu() raises:
    print("test_uop_cross_relu_negate_grad_flow_cpu")
    comptime dtype = DType.float32
    # C = relu(-A), dC/dA = -1 where A < 0 else 0
    var a = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    var loss = (-a).relu().sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([-1.0, 0.0, -1.0, 0.0]))
    )


fn test_uop_cross_sqrt_negate_grad_flow_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_cross_sqrt_negate_grad_flow_gpu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1(
            [4.0, 9.0, 16.0], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var loss = (-a_gpu.sqrt()).sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([-0.25, -0.16667, -0.125])
            )
        )


fn test_uop_cross_relu_negate_grad_flow_gpu() raises:
    comptime if has_accelerator():
        print("test_uop_cross_relu_negate_grad_flow_gpu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1(
            [-1.0, 2.0, -3.0, 4.0], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var loss = (-a_gpu).relu().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].d1([-1.0, 0.0, -1.0, 0.0])
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    # SQRT
    test_uop_sqrt_forward_1d_cpu()
    test_uop_sqrt_forward_2d_cpu()
    test_uop_sqrt_forward_3d_cpu()
    test_uop_sqrt_backward_1d_cpu()
    test_uop_sqrt_backward_2d_cpu()
    test_uop_sqrt_grad_flow_cpu()
    test_uop_sqrt_gpu()
    test_uop_sqrt_gpu_3d()

    # NEGATE
    test_uop_negate_forward_1d_cpu()
    test_uop_negate_forward_2d_cpu()
    test_uop_negate_forward_3d_cpu()
    test_uop_negate_backward_1d_cpu()
    test_uop_negate_backward_2d_cpu()
    test_uop_negate_grad_flow_cpu()
    test_uop_negate_gpu()
    test_uop_negate_gpu_3d()

    # ABS
    test_uop_abs_forward_1d_cpu()
    test_uop_abs_forward_2d_cpu()
    test_uop_abs_forward_3d_cpu()
    test_uop_abs_gpu()
    test_uop_abs_gpu_3d()

    # RELU
    test_uop_relu_forward_1d_cpu()
    test_uop_relu_forward_2d_cpu()
    test_uop_relu_forward_3d_cpu()
    test_uop_relu_backward_1d_cpu()
    test_uop_relu_backward_2d_cpu()
    test_uop_relu_grad_flow_cpu()
    test_uop_relu_gpu()
    test_uop_relu_gpu_3d()

    # INVERT
    test_uop_invert_bool_forward_1d_cpu()
    test_uop_invert_bool_forward_2d_cpu()
    test_uop_invert_bool_forward_3d_cpu()
    test_uop_invert_bool_double_cpu()
    test_uop_invert_int_forward_1d_cpu()
    test_uop_invert_int_forward_2d_cpu()
    test_uop_invert_int_forward_3d_cpu()
    test_uop_invert_bool_gpu()
    test_uop_invert_bool_gpu_3d()
    test_uop_invert_bool_gpu_double()
    test_uop_invert_int_gpu()
    test_uop_invert_int_gpu_3d()

    # CROSS OP GRAD FLOW
    test_uop_cross_sqrt_negate_grad_flow_cpu()
    test_uop_cross_relu_negate_grad_flow_cpu()
    test_uop_cross_sqrt_negate_grad_flow_gpu()
    test_uop_cross_relu_negate_grad_flow_gpu()

    print("All uop tests passed!")
