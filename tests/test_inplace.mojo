from tensor import Tensor
from shapes import Shape
from std.testing import assert_true
from std.sys import has_accelerator

comptime dtype = DType.float32


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

fn iop_close[atol: Scalar[dtype] = 1e-5](
    a: Tensor[dtype], b: Tensor[dtype]
) raises -> Bool:
    return a.all_close[atol=atol](b)


# ═══════════════════════════════════════════════════════════════════════════════
# IADD CPU — Forward
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_cpu_iadd_fwd_1d_same_shape() raises:
    print("test_iop_cpu_iadd_fwd_1d_same_shape")
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0])
    a += b
    assert_true(iop_close(a, Tensor[dtype].d1([5.0, 7.0, 9.0])))
    print("passed")


fn test_iop_cpu_iadd_fwd_2d_same_shape() raises:
    print("test_iop_cpu_iadd_fwd_2d_same_shape")
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]])
    a += b
    assert_true(iop_close(a, Tensor[dtype].d2([[11.0, 22.0], [33.0, 44.0]])))
    print("passed")


fn test_iop_cpu_iadd_fwd_3d_same_shape() raises:
    print("test_iop_cpu_iadd_fwd_3d_same_shape")
    var a = Tensor[dtype].ones(Shape(2, 3, 4))
    var b = Tensor[dtype].ones(Shape(2, 3, 4))
    a += b
    assert_true(iop_close(a, Tensor[dtype].full(Shape(2, 3, 4), 2.0)))
    print("passed")


fn test_iop_cpu_iadd_fwd_4d_same_shape() raises:
    print("test_iop_cpu_iadd_fwd_4d_same_shape")
    var a = Tensor[dtype].ones(Shape(2, 3, 4, 5))
    var b = Tensor[dtype].full(Shape(2, 3, 4, 5), 3.0)
    a += b
    assert_true(iop_close(a, Tensor[dtype].full(Shape(2, 3, 4, 5), 4.0)))
    print("passed")


fn test_iop_cpu_iadd_fwd_broadcast_scalar() raises:
    print("test_iop_cpu_iadd_fwd_broadcast_scalar")
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    a += Scalar[dtype](10)
    assert_true(iop_close(a, Tensor[dtype].d1([11.0, 12.0, 13.0])))
    print("passed")


fn test_iop_cpu_iadd_fwd_broadcast_row() raises:
    print("test_iop_cpu_iadd_fwd_broadcast_row")
    var a = Tensor[dtype].ones(Shape(3, 4))
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])  # (4,) → (3,4)
    a += b
    assert_true(iop_close(a, Tensor[dtype].d2([
        [2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0],
    ])))
    print("passed")


fn test_iop_cpu_iadd_fwd_broadcast_col() raises:
    print("test_iop_cpu_iadd_fwd_broadcast_col")
    var a = Tensor[dtype].ones(Shape(3, 4))
    var b = Tensor[dtype].d2([[1.0], [2.0], [3.0]])  # (3,1) → (3,4)
    a += b
    assert_true(iop_close(a, Tensor[dtype].d2([
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0],
    ])))
    print("passed")


fn test_iop_cpu_iadd_fwd_large() raises:
    print("test_iop_cpu_iadd_fwd_large")
    var a = Tensor[dtype].zeros(Shape(64, 128))
    var b = Tensor[dtype].ones(Shape(64, 128))
    a += b
    assert_true(iop_close(a, Tensor[dtype].ones(Shape(64, 128))))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# ISUB CPU — Forward
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_cpu_isub_fwd_1d_same_shape() raises:
    print("test_iop_cpu_isub_fwd_1d_same_shape")
    var a = Tensor[dtype].d1([5.0, 7.0, 9.0])
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0])
    a -= b
    assert_true(iop_close(a, Tensor[dtype].d1([4.0, 5.0, 6.0])))
    print("passed")


fn test_iop_cpu_isub_fwd_2d_same_shape() raises:
    print("test_iop_cpu_isub_fwd_2d_same_shape")
    var a = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]])
    var b = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    a -= b
    assert_true(iop_close(a, Tensor[dtype].d2([[9.0, 18.0], [27.0, 36.0]])))
    print("passed")


fn test_iop_cpu_isub_fwd_broadcast_row() raises:
    print("test_iop_cpu_isub_fwd_broadcast_row")
    var a = Tensor[dtype].full(Shape(3, 4), 5.0)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    a -= b
    assert_true(iop_close(a, Tensor[dtype].d2([
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ])))
    print("passed")


fn test_iop_cpu_isub_fwd_3d() raises:
    print("test_iop_cpu_isub_fwd_3d")
    var a = Tensor[dtype].full(Shape(2, 3, 4), 5.0)
    var b = Tensor[dtype].ones(Shape(2, 3, 4))
    a -= b
    assert_true(iop_close(a, Tensor[dtype].full(Shape(2, 3, 4), 4.0)))
    print("passed")


fn test_iop_cpu_isub_fwd_scalar() raises:
    print("test_iop_cpu_isub_fwd_scalar")
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0])
    a -= Scalar[dtype](5)
    assert_true(iop_close(a, Tensor[dtype].d1([5.0, 15.0, 25.0])))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# IMUL CPU — Forward
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_cpu_imul_fwd_1d_same_shape() raises:
    print("test_iop_cpu_imul_fwd_1d_same_shape")
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([2.0, 3.0, 4.0])
    a *= b
    assert_true(iop_close(a, Tensor[dtype].d1([2.0, 6.0, 12.0])))
    print("passed")


fn test_iop_cpu_imul_fwd_2d_same_shape() raises:
    print("test_iop_cpu_imul_fwd_2d_same_shape")
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    a *= b
    assert_true(iop_close(a, Tensor[dtype].d2([[2.0, 4.0], [6.0, 8.0]])))
    print("passed")


fn test_iop_cpu_imul_fwd_broadcast_scalar() raises:
    print("test_iop_cpu_imul_fwd_broadcast_scalar")
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    a *= Scalar[dtype](3)
    assert_true(iop_close(a, Tensor[dtype].d1([3.0, 6.0, 9.0])))
    print("passed")


fn test_iop_cpu_imul_fwd_broadcast_row() raises:
    print("test_iop_cpu_imul_fwd_broadcast_row")
    var a = Tensor[dtype].full(Shape(3, 4), 2.0)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    a *= b
    assert_true(iop_close(a, Tensor[dtype].d2([
        [2.0, 4.0, 6.0, 8.0],
        [2.0, 4.0, 6.0, 8.0],
        [2.0, 4.0, 6.0, 8.0],
    ])))
    print("passed")


fn test_iop_cpu_imul_fwd_3d() raises:
    print("test_iop_cpu_imul_fwd_3d")
    var a = Tensor[dtype].full(Shape(2, 3, 4), 3.0)
    var b = Tensor[dtype].full(Shape(2, 3, 4), 2.0)
    a *= b
    assert_true(iop_close(a, Tensor[dtype].full(Shape(2, 3, 4), 6.0)))
    print("passed")


fn test_iop_cpu_imul_fwd_zeros() raises:
    print("test_iop_cpu_imul_fwd_zeros")
    var a = Tensor[dtype].rand(Shape(4, 5))
    var b = Tensor[dtype].zeros(Shape(4, 5))
    a *= b
    assert_true(iop_close(a, Tensor[dtype].zeros(Shape(4, 5))))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# IDIV CPU — Forward
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_cpu_idiv_fwd_1d_same_shape() raises:
    print("test_iop_cpu_idiv_fwd_1d_same_shape")
    var a = Tensor[dtype].d1([4.0, 6.0, 8.0])
    var b = Tensor[dtype].d1([2.0, 3.0, 4.0])
    a /= b
    assert_true(iop_close(a, Tensor[dtype].d1([2.0, 2.0, 2.0])))
    print("passed")


fn test_iop_cpu_idiv_fwd_2d_same_shape() raises:
    print("test_iop_cpu_idiv_fwd_2d_same_shape")
    var a = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]])
    var b = Tensor[dtype].d2([[2.0, 4.0], [5.0, 8.0]])
    a /= b
    assert_true(iop_close(a, Tensor[dtype].d2([[5.0, 5.0], [6.0, 5.0]])))
    print("passed")


fn test_iop_cpu_idiv_fwd_broadcast_scalar() raises:
    print("test_iop_cpu_idiv_fwd_broadcast_scalar")
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0])
    a /= Scalar[dtype](10)
    assert_true(iop_close(a, Tensor[dtype].d1([1.0, 2.0, 3.0])))
    print("passed")


fn test_iop_cpu_idiv_fwd_broadcast_row() raises:
    print("test_iop_cpu_idiv_fwd_broadcast_row")
    var a = Tensor[dtype].full(Shape(3, 4), 12.0)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    a /= b
    assert_true(iop_close(a, Tensor[dtype].d2([
        [12.0, 6.0, 4.0, 3.0],
        [12.0, 6.0, 4.0, 3.0],
        [12.0, 6.0, 4.0, 3.0],
    ])))
    print("passed")



fn test_iop_cpu_idiv_fwd_3d() raises:
    print("test_iop_cpu_idiv_fwd_3d")
    var a = Tensor[dtype].full(Shape(2, 3, 4), 6.0)
    var b = Tensor[dtype].full(Shape(2, 3, 4), 2.0)
    a /= b
    assert_true(iop_close(a, Tensor[dtype].full(Shape(2, 3, 4), 3.0)))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU (grad flow)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_cpu_iadd_bwd_basic() raises:
    print("test_iop_cpu_iadd_bwd_basic")
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
    var c = a + b  # use non-inplace for backward tracking
    var loss = c.sum()
    loss.backward()
    # grad of sum(a+b) w.r.t a = ones, w.r.t b = ones
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(3))))
    assert_true(iop_close(b.grad().as_tensor(), Tensor[dtype].ones(Shape(3))))
    print("passed")


fn test_iop_cpu_isub_bwd_basic() raises:
    print("test_iop_cpu_isub_bwd_basic")
    var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a - b
    var loss = c.sum()
    loss.backward()
    # grad w.r.t a = ones, w.r.t b = -ones
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(3))))
    assert_true(iop_close(
        b.grad().as_tensor(), Tensor[dtype].full(Shape(3), -1.0)
    ))
    print("passed")


fn test_iop_cpu_imul_bwd_basic() raises:
    print("test_iop_cpu_imul_bwd_basic")
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var b = Tensor[dtype].d1([3.0, 4.0, 5.0], requires_grad=True)
    var c = a * b
    var loss = c.sum()
    loss.backward()
    # grad w.r.t a = b, w.r.t b = a
    assert_true(iop_close(a.grad().as_tensor(), b))
    assert_true(iop_close(b.grad().as_tensor(), a))
    print("passed")


fn test_iop_cpu_idiv_bwd_basic() raises:
    print("test_iop_cpu_idiv_bwd_basic")
    var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
    var b = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var c = a / b
    var loss = c.sum()
    loss.backward()
    # grad w.r.t a = 1/b, w.r.t b = -a/b^2
    assert_true(iop_close[atol=1e-4](
        a.grad().as_tensor(),
        Tensor[dtype].d1([0.5, 0.333333, 0.25])
    ))
    assert_true(iop_close[atol=1e-4](
        b.grad().as_tensor(),
        Tensor[dtype].d1([-1.0, -1.0, -1.0])
    ))
    print("passed")


fn test_iop_cpu_iadd_bwd_2d() raises:
    print("test_iop_cpu_iadd_bwd_2d")
    var a = Tensor[dtype].ones(Shape(3, 4), requires_grad=True)
    var b = Tensor[dtype].ones(Shape(3, 4), requires_grad=True)
    var c = a + b
    var loss = c.sum()
    loss.backward()
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))))
    assert_true(iop_close(b.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))))
    print("passed")


fn test_iop_cpu_imul_bwd_chain() raises:
    print("test_iop_cpu_imul_bwd_chain")
    # loss = sum((a * b) * c)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([2.0, 2.0, 2.0], requires_grad=True)
    var c = Tensor[dtype].d1([3.0, 3.0, 3.0], requires_grad=True)
    var ab = a * b
    var abc = ab * c
    var loss = abc.sum()
    loss.backward()
    # grad_a = b*c = 6, grad_b = a*c = 3,6,9, grad_c = a*b = 2,4,6
    assert_true(iop_close(
        a.grad().as_tensor(), Tensor[dtype].full(Shape(3), 6.0)
    ))
    assert_true(iop_close(
        b.grad().as_tensor(), Tensor[dtype].d1([3.0, 6.0, 9.0])
    ))
    assert_true(iop_close(
        c.grad().as_tensor(), Tensor[dtype].d1([2.0, 4.0, 6.0])
    ))
    print("passed")


fn test_iop_cpu_iadd_bwd_broadcast() raises:
    print("test_iop_cpu_iadd_bwd_broadcast")
    var a = Tensor[dtype].ones(Shape(3, 4), requires_grad=True)
    var b = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var c = a + b  # (3,4) + (4,) broadcast
    var loss = c.sum()
    loss.backward()
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))))
    # b grad = sum over broadcast dim = 3
    assert_true(iop_close(
        b.grad().as_tensor(), Tensor[dtype].full(Shape(4), 3.0)
    ))
    print("passed")


fn test_iop_cpu_isub_bwd_3d() raises:
    print("test_iop_cpu_isub_bwd_3d")
    var a = Tensor[dtype].ones(Shape(2, 3, 4), requires_grad=True)
    var b = Tensor[dtype].ones(Shape(2, 3, 4), requires_grad=True)
    var c = a - b
    var loss = c.sum()
    loss.backward()
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4))))
    assert_true(iop_close(
        b.grad().as_tensor(), Tensor[dtype].full(Shape(2, 3, 4), -1.0)
    ))
    print("passed")


fn test_iop_cpu_imul_bwd_2d() raises:
    print("test_iop_cpu_imul_bwd_2d")
    var a = Tensor[dtype].full(Shape(3, 4), 2.0, requires_grad=True)
    var b = Tensor[dtype].full(Shape(3, 4), 3.0, requires_grad=True)
    var c = a * b
    var loss = c.sum()
    loss.backward()
    assert_true(iop_close(
        a.grad().as_tensor(), Tensor[dtype].full(Shape(3, 4), 3.0)
    ))
    assert_true(iop_close(
        b.grad().as_tensor(), Tensor[dtype].full(Shape(3, 4), 2.0)
    ))
    print("passed")


fn test_iop_cpu_grad_flow_iadd_isub_combined() raises:
    print("test_iop_cpu_grad_flow_iadd_isub_combined")
    # loss = sum((a + b) - c)
    var a = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var b = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var c = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var ab = a + b
    var abc = ab - c
    var loss = abc.sum()
    loss.backward()
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(4))))
    assert_true(iop_close(b.grad().as_tensor(), Tensor[dtype].ones(Shape(4))))
    assert_true(iop_close(
        c.grad().as_tensor(), Tensor[dtype].full(Shape(4), -1.0)
    ))
    print("passed")


fn test_iop_cpu_grad_flow_shared_ancestor() raises:
    print("test_iop_cpu_grad_flow_shared_ancestor")
    # loss = sum(a + a) → grad_a = 2
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a + a
    var loss = c.sum()
    loss.backward()
    assert_true(iop_close(
        a.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)
    ))
    print("passed")


fn test_iop_cpu_grad_flow_deep_chain() raises:
    print("test_iop_cpu_grad_flow_deep_chain")
    # loss = sum(((a + b) * c) - d)
    var a = Tensor[dtype].ones(Shape(3), requires_grad=True)
    var b = Tensor[dtype].ones(Shape(3), requires_grad=True)
    var c = Tensor[dtype].full(Shape(3), 2.0, requires_grad=True)
    var d = Tensor[dtype].ones(Shape(3), requires_grad=True)
    var ab = a + b
    var abc = ab * c
    var abcd = abc - d
    var loss = abcd.sum()
    loss.backward()
    # grad_a = c = 2, grad_b = c = 2, grad_c = (a+b) = 2, grad_d = -1
    assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)))
    assert_true(iop_close(b.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)))
    assert_true(iop_close(c.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)))
    assert_true(iop_close(
        d.grad().as_tensor(), Tensor[dtype].full(Shape(3), -1.0)
    ))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU FORWARD — IADD
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_gpu_iadd_fwd_1d_same_shape() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_1d_same_shape")
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var b_cpu = Tensor[dtype].d1([4.0, 5.0, 6.0])
        var a = a_cpu.to_gpu()
        var b = b_cpu.to_gpu()
        a += b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d1([5.0, 7.0, 9.0])))
        print("passed")


fn test_iop_gpu_iadd_fwd_2d_same_shape() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_2d_same_shape")
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var b = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]]).to_gpu()
        a += b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].d2([[11.0, 22.0], [33.0, 44.0]])
        ))
        print("passed")


fn test_iop_gpu_iadd_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_3d")
        var a = Tensor[dtype].ones(Shape(2, 3, 4)).to_gpu()
        var b = Tensor[dtype].ones(Shape(2, 3, 4)).to_gpu()
        a += b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].full(Shape(2, 3, 4), 2.0)
        ))
        print("passed")


fn test_iop_gpu_iadd_fwd_4d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_4d")
        var a = Tensor[dtype].ones(Shape(2, 3, 4, 5)).to_gpu()
        var b = Tensor[dtype].full(Shape(2, 3, 4, 5), 3.0).to_gpu()
        a += b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].full(Shape(2, 3, 4, 5), 4.0)
        ))
        print("passed")


fn test_iop_gpu_iadd_fwd_broadcast_row() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_broadcast_row")
        var a = Tensor[dtype].ones(Shape(3, 4)).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        a += b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d2([
            [2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0],
        ])))
        print("passed")


fn test_iop_gpu_iadd_fwd_broadcast_col() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_broadcast_col")
        var a = Tensor[dtype].ones(Shape(3, 4)).to_gpu()
        var b = Tensor[dtype].d2([[1.0], [2.0], [3.0]]).to_gpu()
        a += b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d2([
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
        ])))
        print("passed")


fn test_iop_gpu_iadd_fwd_large() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_large")
        var a = Tensor[dtype].zeros(Shape(64, 128)).to_gpu()
        var b = Tensor[dtype].ones(Shape(64, 128)).to_gpu()
        a += b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].ones(Shape(64, 128))))
        print("passed")


fn test_iop_gpu_iadd_fwd_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_fwd_matches_cpu")
        var a_cpu = Tensor[dtype].rand(Shape(9, 20))
        var b_cpu = Tensor[dtype].rand(Shape(9, 20))
        var a_ref = a_cpu.copy()
        a_ref += b_cpu
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        a_gpu += b_gpu
        assert_true(iop_close(a_gpu.to_cpu(), a_ref))
        print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU FORWARD — ISUB
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_gpu_isub_fwd_1d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_fwd_1d")
        var a = Tensor[dtype].d1([5.0, 7.0, 9.0]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        a -= b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d1([4.0, 5.0, 6.0])))
        print("passed")


fn test_iop_gpu_isub_fwd_2d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_fwd_2d")
        var a = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]]).to_gpu()
        var b = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        a -= b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].d2([[9.0, 18.0], [27.0, 36.0]])
        ))
        print("passed")


fn test_iop_gpu_isub_fwd_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_fwd_matches_cpu")
        var a_cpu = Tensor[dtype].rand(Shape(8, 12))
        var b_cpu = Tensor[dtype].rand(Shape(8, 12))
        var a_ref = a_cpu.copy()
        a_ref -= b_cpu
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        a_gpu -= b_gpu
        assert_true(iop_close(a_gpu.to_cpu(), a_ref))
        print("passed")


fn test_iop_gpu_isub_fwd_broadcast_row() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_fwd_broadcast_row")
        var a = Tensor[dtype].full(Shape(3, 4), 5.0).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        a -= b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d2([
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ])))
        print("passed")


fn test_iop_gpu_isub_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_fwd_3d")
        var a = Tensor[dtype].full(Shape(2, 3, 4), 5.0).to_gpu()
        var b = Tensor[dtype].ones(Shape(2, 3, 4)).to_gpu()
        a -= b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].full(Shape(2, 3, 4), 4.0)
        ))
        print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU FORWARD — IMUL
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_gpu_imul_fwd_1d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_fwd_1d")
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        a *= b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d1([2.0, 6.0, 12.0])))
        print("passed")


fn test_iop_gpu_imul_fwd_2d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_fwd_2d")
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var b = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]]).to_gpu()
        a *= b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].d2([[2.0, 4.0], [6.0, 8.0]])
        ))
        print("passed")


fn test_iop_gpu_imul_fwd_broadcast_row() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_fwd_broadcast_row")
        var a = Tensor[dtype].full(Shape(3, 4), 2.0).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        a *= b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d2([
            [2.0, 4.0, 6.0, 8.0],
            [2.0, 4.0, 6.0, 8.0],
            [2.0, 4.0, 6.0, 8.0],
        ])))
        print("passed")


fn test_iop_gpu_imul_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_fwd_3d")
        var a = Tensor[dtype].full(Shape(2, 3, 4), 3.0).to_gpu()
        var b = Tensor[dtype].full(Shape(2, 3, 4), 2.0).to_gpu()
        a *= b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].full(Shape(2, 3, 4), 6.0)
        ))
        print("passed")


fn test_iop_gpu_imul_fwd_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_fwd_matches_cpu")
        var a_cpu = Tensor[dtype].rand(Shape(9, 20))
        var b_cpu = Tensor[dtype].rand(Shape(9, 20))
        var a_ref = a_cpu.copy()
        a_ref *= b_cpu
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        a_gpu *= b_gpu
        assert_true(iop_close(a_gpu.to_cpu(), a_ref))
        print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU FORWARD — IDIV
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_gpu_idiv_fwd_1d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_idiv_fwd_1d")
        var a = Tensor[dtype].d1([4.0, 6.0, 8.0]).to_gpu()
        var b = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        a /= b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d1([2.0, 2.0, 2.0])))
        print("passed")


fn test_iop_gpu_idiv_fwd_2d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_idiv_fwd_2d")
        var a = Tensor[dtype].full(Shape(3, 4), 12.0).to_gpu()
        var b = Tensor[dtype].full(Shape(3, 4), 4.0).to_gpu()
        a /= b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].full(Shape(3, 4), 3.0)
        ))
        print("passed")


fn test_iop_gpu_idiv_fwd_broadcast_row() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_idiv_fwd_broadcast_row")
        var a = Tensor[dtype].full(Shape(3, 4), 12.0).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        a /= b
        assert_true(iop_close(a.to_cpu(), Tensor[dtype].d2([
            [12.0, 6.0, 4.0, 3.0],
            [12.0, 6.0, 4.0, 3.0],
            [12.0, 6.0, 4.0, 3.0],
        ])))
        print("passed")


fn test_iop_gpu_idiv_fwd_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_idiv_fwd_matches_cpu")
        var a_cpu = Tensor[dtype].rand(Shape(8, 12)) + Scalar[dtype](0.1)
        var b_cpu = Tensor[dtype].rand(Shape(8, 12)) + Scalar[dtype](0.1)
        var a_ref = a_cpu.copy()
        a_ref /= b_cpu
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        a_gpu /= b_gpu
        assert_true(iop_close(a_gpu.to_cpu(), a_ref))
        print("passed")


fn test_iop_gpu_idiv_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_idiv_fwd_3d")
        var a = Tensor[dtype].full(Shape(2, 3, 4), 6.0).to_gpu()
        var b = Tensor[dtype].full(Shape(2, 3, 4), 2.0).to_gpu()
        a /= b
        assert_true(iop_close(
            a.to_cpu(), Tensor[dtype].full(Shape(2, 3, 4), 3.0)
        ))
        print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU BACKWARD — grad flow
# ═══════════════════════════════════════════════════════════════════════════════

fn test_iop_gpu_iadd_bwd_basic() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_bwd_basic")
        var a = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var b = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu + b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(3))))
        assert_true(iop_close(b.grad().as_tensor(), Tensor[dtype].ones(Shape(3))))
        print("passed")


fn test_iop_gpu_isub_bwd_basic() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_bwd_basic")
        var a = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var b = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu - b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].ones(Shape(3))))
        assert_true(iop_close(
            b.grad().as_tensor(), Tensor[dtype].full(Shape(3), -1.0)
        ))
        print("passed")


fn test_iop_gpu_imul_bwd_basic() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_bwd_basic")
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var b = Tensor[dtype].d1([3.0, 4.0, 5.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu * b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(
            a.grad().as_tensor(), Tensor[dtype].d1([3.0, 4.0, 5.0])
        ))
        assert_true(iop_close(
            b.grad().as_tensor(), Tensor[dtype].d1([2.0, 3.0, 4.0])
        ))
        print("passed")


fn test_iop_gpu_idiv_bwd_basic() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_idiv_bwd_basic")
        var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
        var b = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu / b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close[atol=1e-4](
            a.grad().as_tensor(),
            Tensor[dtype].d1([0.5, 0.333333, 0.25])
        ))
        assert_true(iop_close[atol=1e-4](
            b.grad().as_tensor(),
            Tensor[dtype].d1([-1.0, -1.0, -1.0])
        ))
        print("passed")


fn test_iop_gpu_iadd_bwd_2d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_bwd_2d")
        var a = Tensor[dtype].ones(Shape(3, 4), requires_grad=True)
        var b = Tensor[dtype].ones(Shape(3, 4), requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu + b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(
            a.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))
        ))
        assert_true(iop_close(
            b.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))
        ))
        print("passed")


fn test_iop_gpu_imul_bwd_chain() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_imul_bwd_chain")
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([2.0, 2.0, 2.0], requires_grad=True)
        var c = Tensor[dtype].d1([3.0, 3.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c_gpu = c.to_gpu()
        var ab = a_gpu * b_gpu
        var abc = ab * c_gpu
        var loss = abc.sum()
        loss.backward()
        assert_true(iop_close(
            a.grad().as_tensor(), Tensor[dtype].full(Shape(3), 6.0)
        ))
        assert_true(iop_close(
            b.grad().as_tensor(), Tensor[dtype].d1([3.0, 6.0, 9.0])
        ))
        assert_true(iop_close(
            c.grad().as_tensor(), Tensor[dtype].d1([2.0, 4.0, 6.0])
        ))
        print("passed")


fn test_iop_gpu_iadd_bwd_broadcast() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_iadd_bwd_broadcast")
        var a = Tensor[dtype].ones(Shape(3, 4), requires_grad=True)
        var b = Tensor[dtype].ones(Shape(4), requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu + b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(
            a.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))
        ))
        assert_true(iop_close(
            b.grad().as_tensor(), Tensor[dtype].full(Shape(4), 3.0)
        ))
        print("passed")


fn test_iop_gpu_grad_flow_shared_ancestor() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_grad_flow_shared_ancestor")
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu + a_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(
            a.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)
        ))
        print("passed")


fn test_iop_gpu_bwd_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_bwd_matches_cpu")
        var a = Tensor[dtype].rand(Shape(4, 5), requires_grad=True)
        var b = Tensor[dtype].rand(Shape(4, 5), requires_grad=True)

        # CPU
        var out_cpu = a * b
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        var ga_cpu = a.grad().as_tensor().copy()
        var gb_cpu = b.grad().as_tensor().copy()
        a.zero_grad()
        b.zero_grad()

        # GPU
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var out_gpu = a_gpu * b_gpu
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()

        assert_true(iop_close(a.grad().as_tensor(), ga_cpu))
        assert_true(iop_close(b.grad().as_tensor(), gb_cpu))
        print("passed")


fn test_iop_gpu_isub_bwd_3d() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_isub_bwd_3d")
        var a = Tensor[dtype].ones(Shape(2, 3, 4), requires_grad=True)
        var b = Tensor[dtype].ones(Shape(2, 3, 4), requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c = a_gpu - b_gpu
        var loss = c.sum()
        loss.backward()
        assert_true(iop_close(
            a.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4))
        ))
        assert_true(iop_close(
            b.grad().as_tensor(), Tensor[dtype].full(Shape(2, 3, 4), -1.0)
        ))
        print("passed")

fn test_iop_gpu_grad_flow_deep_chain() raises:
    comptime if has_accelerator():
        print("test_iop_gpu_grad_flow_deep_chain")
        var a = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var b = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var c = Tensor[dtype].full(Shape(3), 2.0, requires_grad=True)
        var d = Tensor[dtype].ones(Shape(3), requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c_gpu = c.to_gpu()
        var d_gpu = d.to_gpu()
        var ab = a_gpu + b_gpu
        var abc = ab * c_gpu
        var abcd = abc - d_gpu
        var loss = abcd.sum()
        loss.backward()
        assert_true(iop_close(a.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)))
        assert_true(iop_close(b.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)))
        assert_true(iop_close(c.grad().as_tensor(), Tensor[dtype].full(Shape(3), 2.0)))
        assert_true(iop_close(
            d.grad().as_tensor(), Tensor[dtype].full(Shape(3), -1.0)
        ))
        print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    print("=== inplace ops forward CPU ===")
    test_iop_cpu_iadd_fwd_1d_same_shape()
    test_iop_cpu_iadd_fwd_2d_same_shape()
    test_iop_cpu_iadd_fwd_3d_same_shape()
    test_iop_cpu_iadd_fwd_4d_same_shape()
    test_iop_cpu_iadd_fwd_broadcast_scalar()
    test_iop_cpu_iadd_fwd_broadcast_row()
    test_iop_cpu_iadd_fwd_broadcast_col()
    test_iop_cpu_iadd_fwd_large()
    test_iop_cpu_isub_fwd_1d_same_shape()
    test_iop_cpu_isub_fwd_2d_same_shape()
    test_iop_cpu_isub_fwd_broadcast_row()
    test_iop_cpu_isub_fwd_3d()
    test_iop_cpu_isub_fwd_scalar()
    test_iop_cpu_imul_fwd_1d_same_shape()
    test_iop_cpu_imul_fwd_2d_same_shape()
    test_iop_cpu_imul_fwd_broadcast_scalar()
    test_iop_cpu_imul_fwd_broadcast_row()
    test_iop_cpu_imul_fwd_3d()
    test_iop_cpu_imul_fwd_zeros()
    test_iop_cpu_idiv_fwd_1d_same_shape()
    test_iop_cpu_idiv_fwd_2d_same_shape()
    test_iop_cpu_idiv_fwd_broadcast_scalar()
    test_iop_cpu_idiv_fwd_broadcast_row()
    test_iop_cpu_idiv_fwd_3d()
    print("=== inplace ops backward CPU ===")
    test_iop_cpu_iadd_bwd_basic()
    test_iop_cpu_isub_bwd_basic()
    test_iop_cpu_imul_bwd_basic()
    test_iop_cpu_idiv_bwd_basic()
    test_iop_cpu_iadd_bwd_2d()
    test_iop_cpu_imul_bwd_chain()
    test_iop_cpu_iadd_bwd_broadcast()
    test_iop_cpu_isub_bwd_3d()
    test_iop_cpu_imul_bwd_2d()
    test_iop_cpu_grad_flow_iadd_isub_combined()
    test_iop_cpu_grad_flow_shared_ancestor()
    test_iop_cpu_grad_flow_deep_chain()
    print("=== All CPU inplace ops tests passed ===\n")

    test_iop_gpu_iadd_fwd_1d_same_shape()
    test_iop_gpu_iadd_fwd_2d_same_shape()
    test_iop_gpu_iadd_fwd_3d()
    test_iop_gpu_iadd_fwd_4d()
    test_iop_gpu_iadd_fwd_broadcast_row()
    test_iop_gpu_iadd_fwd_broadcast_col()
    test_iop_gpu_iadd_fwd_large()
    test_iop_gpu_iadd_fwd_matches_cpu()
    test_iop_gpu_isub_fwd_1d()
    test_iop_gpu_isub_fwd_2d()
    test_iop_gpu_isub_fwd_matches_cpu()
    test_iop_gpu_isub_fwd_broadcast_row()
    test_iop_gpu_isub_fwd_3d()
    test_iop_gpu_imul_fwd_1d()
    test_iop_gpu_imul_fwd_2d()
    test_iop_gpu_imul_fwd_broadcast_row()
    test_iop_gpu_imul_fwd_3d()
    test_iop_gpu_imul_fwd_matches_cpu()
    test_iop_gpu_idiv_fwd_1d()
    test_iop_gpu_idiv_fwd_2d()
    test_iop_gpu_idiv_fwd_broadcast_row()
    test_iop_gpu_idiv_fwd_matches_cpu()
    test_iop_gpu_idiv_fwd_3d()
    test_iop_gpu_iadd_bwd_basic()
    test_iop_gpu_isub_bwd_basic()
    test_iop_gpu_imul_bwd_basic()
    test_iop_gpu_idiv_bwd_basic()
    test_iop_gpu_iadd_bwd_2d()
    test_iop_gpu_imul_bwd_chain()
    test_iop_gpu_iadd_bwd_broadcast()
    test_iop_gpu_grad_flow_shared_ancestor()
    test_iop_gpu_bwd_matches_cpu()
    test_iop_gpu_isub_bwd_3d()
    test_iop_gpu_grad_flow_deep_chain()
    print("=== All GPU inplace ops tests passed (if accelerator present) ===")

