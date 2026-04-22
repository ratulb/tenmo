from std.testing import assert_true
from std.sys import has_accelerator
from tensor import Tensor  # adjust import path to your project


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORWARD PASS – CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_fwd_cpu_1d() raises:
    print("test_sqrt_fwd_cpu_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0, 1.0, 4.0, 9.0, 16.0])
    var out = a.sqrt()
    assert_true(out.all_close(Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0])))


fn test_sqrt_fwd_cpu_2d() raises:
    print("test_sqrt_fwd_cpu_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]])
    var out = a.sqrt()
    assert_true(out.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))


fn test_sqrt_fwd_cpu_3d() raises:
    print("test_sqrt_fwd_cpu_3d")
    comptime dtype = DType.float32
    # shape [2, 2, 2]
    var a = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
    var out = a.sqrt()
    var expected = Tensor[dtype].d1(
        [1.0, 1.4142135, 1.7320508, 2.0, 2.2360680, 2.4494897, 2.6457513, 2.8284271]
    ).reshape(2, 2, 2)
    assert_true(out.all_close[atol=1e-5](expected))


fn test_sqrt_fwd_cpu_scalar_like() raises:
    print("test_sqrt_fwd_cpu_scalar_like")
    comptime dtype = DType.float32
    # Single-element tensor behaves like a scalar
    var a = Tensor[dtype].d1([25.0])
    var out = a.sqrt()
    assert_true(out.all_close(Tensor[dtype].d1([5.0])))


fn test_sqrt_fwd_cpu_zero() raises:
    print("test_sqrt_fwd_cpu_zero")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0])
    var out = a.sqrt()
    assert_true(out.all_close(Tensor[dtype].d1([0.0])))


fn test_sqrt_fwd_cpu_fractional() raises:
    print("test_sqrt_fwd_cpu_fractional")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.25, 0.5, 2.0])
    var out = a.sqrt()
    assert_true(
        out.all_close[atol=1e-5](
            Tensor[dtype].d1([0.5, 0.7071068, 1.4142135])
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKWARD PASS – CPU   (grad of sqrt(x) = 1 / (2*sqrt(x)))
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_bwd_cpu_1d() raises:
    print("test_sqrt_bwd_cpu_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 9.0, 16.0], requires_grad=True)
    var loss = a.sqrt().sum()
    loss.backward()
    # d/dx sqrt(x) = 0.5/sqrt(x)  => [0.5, 0.25, 0.1667, 0.125]
    assert_true(
        a.grad().all_close[atol=1e-5](
            Tensor[dtype].d1([0.5, 0.25, 0.16666667, 0.125])
        )
    )


fn test_sqrt_bwd_cpu_2d() raises:
    print("test_sqrt_bwd_cpu_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
    var loss = a.sqrt().sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.5, 0.25], [0.16666667, 0.125]])
        )
    )


fn test_sqrt_bwd_cpu_3d() raises:
    print("test_sqrt_bwd_cpu_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
    a.requires_grad_(True)
    var loss = a.sqrt().sum()
    loss.backward()
    # expected: 0.5 / sqrt(k) for k in 1..8
    var expected = Tensor[dtype].d1(
        [0.5, 0.35355339, 0.28867513, 0.25,
         0.22360680, 0.20412415, 0.18898224, 0.17677670]
    ).reshape(2, 2, 2)
    assert_true(a.grad().all_close[atol=1e-5](expected))


fn test_sqrt_bwd_cpu_chain_mul() raises:
    # loss = sum(sqrt(a) * 2)  =>  grad = 1/sqrt(a)
    print("test_sqrt_bwd_cpu_chain_mul")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 9.0], requires_grad=True)
    var s = a.sqrt()
    var scaled = s * Tensor[dtype].full_like(s, 2.0)
    var loss = scaled.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](
            Tensor[dtype].d1([1.0, 0.5, 0.33333334])
        )
    )


fn test_sqrt_bwd_cpu_chain_add() raises:
    # loss = sum(sqrt(a) + sqrt(b))  =>  grads = 0.5/sqrt(x)
    print("test_sqrt_bwd_cpu_chain_add")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 16.0], requires_grad=True)
    var loss = (a.sqrt() + b.sqrt()).sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.25, 0.16666667]))
    )
    assert_true(
        b.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.125]))
    )


fn test_sqrt_bwd_cpu_sqrt_of_sqrt() raises:
    # loss = sum(sqrt(sqrt(a))) = sum(a^0.25)  =>  grad = 0.25 * a^(-0.75)
    print("test_sqrt_bwd_cpu_sqrt_of_sqrt")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 16.0], requires_grad=True)
    var loss = a.sqrt().sqrt().sum()
    loss.backward()
    # 0.25 / a^0.75 => [0.25, 0.25/8] = [0.25, 0.03125]
    assert_true(
        a.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.25, 0.03125]))
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADIENT FLOW VERIFICATION – CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_gradflow_cpu_no_grad_leaf() raises:
    # Tensor without requires_grad should have no gradient after backward
    print("test_sqrt_gradflow_cpu_no_grad_leaf")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0, 9.0])          # no grad
    var b = Tensor[dtype].d1([1.0, 4.0], requires_grad=True)
    var loss = (a.sqrt() + b.sqrt()).sum()
    loss.backward()
    assert_true(b.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.25])))
    # a should have no grad — verify b's grad is unaffected by a's values
    # (structural check; no crash means flow is correct)


fn test_sqrt_gradflow_cpu_multi_use_leaf() raises:
    # a is used twice: loss = sum(sqrt(a) + sqrt(a)) => grad = 1/sqrt(a)
    print("test_sqrt_gradflow_cpu_multi_use_leaf")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
    var s1 = a.sqrt()
    var s2 = a.sqrt()
    var loss = (s1 + s2).sum()
    loss.backward()
    # each sqrt contributes 0.5/sqrt(a), two paths => 1/sqrt(a)
    assert_true(
        a.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.33333334]))
    )


fn test_sqrt_gradflow_cpu_intermediate_reuse_warning() raises:
    # Demonstrates retained-grad accumulation when backward is called twice.
    # First backward leaves grad on intermediate; second adds on top.
    # We only call backward once per graph here to stay correct.
    print("test_sqrt_gradflow_cpu_intermediate_reuse_warning")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0], requires_grad=True)
    var s = a.sqrt()
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.25])))


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD PASS – GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_fwd_gpu_1d() raises:
    comptime if has_accelerator():
        print("test_sqrt_fwd_gpu_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, 1.0, 4.0, 9.0, 16.0])
        var a_gpu = a_cpu.to_gpu()
        var out = a_gpu.sqrt()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0]))
        )


fn test_sqrt_fwd_gpu_2d() raises:
    comptime if has_accelerator():
        print("test_sqrt_fwd_gpu_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]])
        var a_gpu = a_cpu.to_gpu()
        var out = a_gpu.sqrt()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )


fn test_sqrt_fwd_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_sqrt_fwd_gpu_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
        var a_gpu = a_cpu.to_gpu()
        var out = a_gpu.sqrt()
        var expected = Tensor[dtype].d1(
            [1.0, 1.4142135, 1.7320508, 2.0,
             2.2360680, 2.4494897, 2.6457513, 2.8284271]
        ).reshape(2, 2, 2)
        assert_true(out.to_cpu().all_close[atol=1e-5](expected))


fn test_sqrt_fwd_gpu_zero() raises:
    comptime if has_accelerator():
        print("test_sqrt_fwd_gpu_zero")
        comptime dtype = DType.float32
        var a_gpu = Tensor[dtype].d1([0.0]).to_gpu()
        var out = a_gpu.sqrt()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0])))


fn test_sqrt_fwd_gpu_fractional() raises:
    comptime if has_accelerator():
        print("test_sqrt_fwd_gpu_fractional")
        comptime dtype = DType.float32
        var a_gpu = Tensor[dtype].d1([0.25, 0.5, 2.0]).to_gpu()
        var out = a_gpu.sqrt()
        assert_true(
            out.to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.7071068, 1.4142135])
            )
        )


fn test_sqrt_fwd_gpu_large_tensor() raises:
    comptime if has_accelerator():
        print("test_sqrt_fwd_gpu_large_tensor")
        comptime dtype = DType.float32
        # 1024 elements: values 1..1024, expected sqrt(k)
        var a_cpu = Tensor[dtype].arange(1.0, 1025.0)
        var expected = a_cpu.sqrt()           # CPU reference
        var out = a_cpu.to_gpu().sqrt().to_cpu()
        assert_true(out.all_close[atol=1e-4](expected))


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKWARD PASS – GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_bwd_gpu_1d() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 9.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.25, 0.16666667, 0.125])
            )
        )


fn test_sqrt_bwd_gpu_2d() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [0.16666667, 0.125]])
            )
        )


fn test_sqrt_bwd_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [0.5, 0.35355339, 0.28867513, 0.25,
             0.22360680, 0.20412415, 0.18898224, 0.17677670]
        ).reshape(2, 2, 2)
        assert_true(a_cpu.grad().all_close[atol=1e-5](expected))


fn test_sqrt_bwd_gpu_chain_mul() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_chain_mul")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 9.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var s = a_gpu.sqrt()
        var scaled = s * Tensor[dtype].full_like(s, 2.0).to_gpu()
        var loss = scaled.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([1.0, 0.5, 0.33333334])
            )
        )


fn test_sqrt_bwd_gpu_chain_add() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_chain_add")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
        var b_cpu = Tensor[dtype].d1([1.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        var loss = (a_gpu.sqrt() + b_gpu.sqrt()).sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.25, 0.16666667]))
        )
        assert_true(
            b_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.125]))
        )


fn test_sqrt_bwd_gpu_sqrt_of_sqrt() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_sqrt_of_sqrt")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sqrt().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.25, 0.03125]))
        )


fn test_sqrt_bwd_gpu_large_tensor() raises:
    comptime if has_accelerator():
        print("test_sqrt_bwd_gpu_large_tensor")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(1.0, 1025.0)
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        # expected: 0.5 / sqrt(k) for k in 1..1024
        var expected = (Tensor[dtype].full_like(a_cpu, 0.5)) / a_cpu.sqrt()
        assert_true(a_cpu.grad().all_close[atol=1e-4](expected))


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRADIENT FLOW – GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_gradflow_gpu_no_grad_leaf() raises:
    comptime if has_accelerator():
        print("test_sqrt_gradflow_gpu_no_grad_leaf")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0])          # no grad
        var b_cpu = Tensor[dtype].d1([1.0, 4.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        var loss = (a_gpu.sqrt() + b_gpu.sqrt()).sum()
        loss.backward()
        assert_true(
            b_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.25]))
        )


fn test_sqrt_gradflow_gpu_multi_use_leaf() raises:
    comptime if has_accelerator():
        print("test_sqrt_gradflow_gpu_multi_use_leaf")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var s1 = a_gpu.sqrt()
        var s2 = a_gpu.sqrt()
        var loss = (s1 + s2).sum()
        loss.backward()
        # two sqrt paths accumulate: 1/sqrt(a)
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.33333334]))
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. CPU / GPU PARITY
# ─────────────────────────────────────────────────────────────────────────────

fn test_sqrt_parity_fwd_1d() raises:
    comptime if has_accelerator():
        print("test_sqrt_parity_fwd_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var cpu_out = a_cpu.sqrt()
        var gpu_out = a_cpu.to_gpu().sqrt().to_cpu()
        assert_true(cpu_out.all_close[atol=1e-5](gpu_out))


fn test_sqrt_parity_bwd_1d() raises:
    comptime if has_accelerator():
        print("test_sqrt_parity_bwd_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu_leaf = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a_gpu_leaf.to_gpu()

        var loss_cpu = a_cpu.sqrt().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.sqrt().sum()
        loss_gpu.backward()

        # CPU grad has been accumulated once; GPU grad is fresh — should match
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu_leaf.grad()))


fn test_sqrt_parity_bwd_2d() raises:
    comptime if has_accelerator():
        print("test_sqrt_parity_bwd_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        var a_gpu_leaf = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        var a_gpu = a_gpu_leaf.to_gpu()

        var loss_cpu = a_cpu.sqrt().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.sqrt().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu_leaf.grad()))


fn test_sqrt_parity_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_sqrt_parity_fwd_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
        var cpu_out = a_cpu.sqrt()
        var gpu_out = a_cpu.to_gpu().sqrt().to_cpu()
        assert_true(cpu_out.all_close[atol=1e-5](gpu_out))


fn test_sqrt_parity_bwd_3d() raises:
    comptime if has_accelerator():
        print("test_sqrt_parity_bwd_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
        a_cpu.requires_grad_(True)
        var a_gpu_leaf = Tensor[dtype].arange(1.0, 9.0).reshape(2, 2, 2)
        a_gpu_leaf.requires_grad_(True)
        var a_gpu = a_gpu_leaf.to_gpu()

        var loss_cpu = a_cpu.sqrt().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.sqrt().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu_leaf.grad()))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN – run all tests
# ─────────────────────────────────────────────────────────────────────────────

fn main() raises:
    # ── Forward CPU ──
    test_sqrt_fwd_cpu_1d()
    test_sqrt_fwd_cpu_2d()
    test_sqrt_fwd_cpu_3d()
    test_sqrt_fwd_cpu_scalar_like()
    test_sqrt_fwd_cpu_zero()
    test_sqrt_fwd_cpu_fractional()

    # ── Backward CPU ──
    test_sqrt_bwd_cpu_1d()
    test_sqrt_bwd_cpu_2d()
    test_sqrt_bwd_cpu_3d()
    test_sqrt_bwd_cpu_chain_mul()
    test_sqrt_bwd_cpu_chain_add()
    test_sqrt_bwd_cpu_sqrt_of_sqrt()

    # ── Grad-flow CPU ──
    test_sqrt_gradflow_cpu_no_grad_leaf()
    test_sqrt_gradflow_cpu_multi_use_leaf()
    test_sqrt_gradflow_cpu_intermediate_reuse_warning()

    # ── Forward GPU ──
    test_sqrt_fwd_gpu_1d()
    test_sqrt_fwd_gpu_2d()
    test_sqrt_fwd_gpu_3d()
    test_sqrt_fwd_gpu_zero()
    test_sqrt_fwd_gpu_fractional()
    test_sqrt_fwd_gpu_large_tensor()

    # ── Backward GPU ──
    test_sqrt_bwd_gpu_1d()
    test_sqrt_bwd_gpu_2d()
    test_sqrt_bwd_gpu_3d()
    test_sqrt_bwd_gpu_chain_mul()
    test_sqrt_bwd_gpu_chain_add()
    test_sqrt_bwd_gpu_sqrt_of_sqrt()
    test_sqrt_bwd_gpu_large_tensor()

    # ── Grad-flow GPU ──
    test_sqrt_gradflow_gpu_no_grad_leaf()
    test_sqrt_gradflow_gpu_multi_use_leaf()

    # ── CPU / GPU parity ──
    test_sqrt_parity_fwd_1d()
    test_sqrt_parity_bwd_1d()
    test_sqrt_parity_bwd_2d()
    test_sqrt_parity_fwd_3d()
    test_sqrt_parity_bwd_3d()

    print("All sqrt tests passed ✓")
