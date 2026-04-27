from std.testing import assert_true
from std.sys import has_accelerator
from tenmo.tensor import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1: CPU -> GPU (stop_grad=False, default)
# Grad MUST flow back to original CPU tensor
# ─────────────────────────────────────────────────────────────────────────────

fn test_devtransfer_cpu2gpu_default_stop_grad_false_1d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_default_stop_grad_false_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var b = a.to_gpu()  # stop_grad=False by default
        var loss = b.sum()
        loss.backward()
        # grad of sum w.r.t each element is 1.0, flows back to a
        assert_true(a.grad().all_close[atol=1e-5](Tensor.ones_like(a)))


fn test_devtransfer_cpu2gpu_default_stop_grad_false_2d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_default_stop_grad_false_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var b = a.to_gpu()
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close[atol=1e-5](Tensor.ones_like(a)))


fn test_devtransfer_cpu2gpu_default_stop_grad_false_with_mul() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_default_stop_grad_false_with_mul")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = a.to_gpu()
        # C = B * 3, loss = sum(C) => grad of a = 3.0
        var c = b * Tensor[dtype].full_like(b, 3.0)
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](
                Tensor[dtype].full_like(a, 3.0)
            )
        )


fn test_devtransfer_cpu2gpu_default_stop_grad_false_with_chained_ops() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_default_stop_grad_false_with_chained_ops")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        var b = a.to_gpu()
        # sqrt then sum — grad = 0.5/sqrt(a)
        var loss = b.sqrt().sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [0.16666667, 0.125]])
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2: CPU -> GPU (stop_grad=True)
# B is a new GPU leaf. Ops on B deposit grad on B, NOT on A.
# ─────────────────────────────────────────────────────────────────────────────

fn test_devtransfer_cpu2gpu_stop_grad_true_grad_on_gpu_leaf_1d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_stop_grad_true_grad_on_gpu_leaf_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)  # B is new GPU leaf
        var c = b * Tensor[dtype].full_like(b, 45.0)
        var loss = c.sum()
        loss.backward()
        # B should receive grad = 45.0 everywhere
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].full_like(a, 45.0)
            )
        )
        # A's grad must be untouched (all zeros / no grad)
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


fn test_devtransfer_cpu2gpu_stop_grad_true_grad_on_gpu_leaf_2d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_stop_grad_true_grad_on_gpu_leaf_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var c = b * Tensor[dtype].full_like(b, 45.0)
        var loss = c.sum()
        loss.backward()
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].full_like(a, 45.0)
            )
        )
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


fn test_devtransfer_cpu2gpu_stop_grad_true_sum_op() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_stop_grad_true_sum_op")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 4.0, 6.0], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var loss = b.sum()
        loss.backward()
        # B gets grad of 1.0 per element
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](Tensor.ones_like(a))
        )
        # A is completely isolated
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


fn test_devtransfer_cpu2gpu_stop_grad_true_chained_ops() raises:
    comptime if has_accelerator():
        print("test_devtransfer_cpu2gpu_stop_grad_true_chained_ops")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 9.0], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var loss = b.sqrt().sum()
        loss.backward()
        # grad deposits on B: 0.5/sqrt(b)
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.25, 0.16666667])
            )
        )
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3: CPU -> GPU -> ops -> CPU (stop_grad=False throughout)
# Grad MUST flow all the way back to A on CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_sum() raises:
    comptime if has_accelerator():
        print("test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_sum")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var b = a.to_gpu()                    # CPU->GPU, stop_grad=False
        var c = b * Tensor[dtype].full_like(b, 2.0)
        var d = c.to_cpu(stop_grad=False)     # GPU->CPU, stop_grad=False
        var loss = d.sum()
        loss.backward()
        # grad = 2.0 flows all the way back to A
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].full_like(a, 2.0))
        )


fn test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_2d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var b = a.to_gpu()
        var c = b * Tensor[dtype].full_like(b, 3.0)
        var d = c.to_cpu(stop_grad=False)
        var loss = d.sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].full_like(a, 3.0))
        )


fn test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_sqrt() raises:
    comptime if has_accelerator():
        print("test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_sqrt")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 9.0, 16.0], requires_grad=True)
        var b = a.to_gpu()
        var c = b.sqrt()
        var d = c.to_cpu(stop_grad=False)
        var loss = d.sum()
        loss.backward()
        # grad = 0.5/sqrt(a) flows back to A
        assert_true(
            a.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.25, 0.16666667, 0.125])
            )
        )


fn test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_chained_ops() raises:
    comptime if has_accelerator():
        print("test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_chained_ops")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        var b = a.to_gpu()
        # Multiple GPU ops before coming back to CPU
        var c = b.sqrt() * Tensor[dtype].full_like(b, 2.0)
        var d = c.to_cpu(stop_grad=False)
        var loss = d.sum()
        loss.backward()
        # grad = 2 * 0.5/sqrt(a) = 1/sqrt(a)
        assert_true(
            a.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[1.0, 0.5], [0.33333334, 0.25]])
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4: CPU -> GPU (stop_grad=True) -> ops -> CPU (stop_grad=False)
# Grad flows back to GPU leaf B, but NOT to A
# ─────────────────────────────────────────────────────────────────────────────

fn test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_1d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)      # B is GPU leaf, A is cut off
        var c = b * Tensor[dtype].full_like(b, 5.0)
        var d = c.to_cpu(stop_grad=False)     # comes back to CPU
        var loss = d.sum()
        loss.backward()
        # B receives grad = 5.0, A receives nothing
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].full_like(a, 5.0)
            )
        )
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


fn test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_2d() raises:
    comptime if has_accelerator():
        print("test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var c = b * Tensor[dtype].full_like(b, 7.0)
        var d = c.to_cpu(stop_grad=False)
        var loss = d.sum()
        loss.backward()
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].full_like(a, 7.0)
            )
        )
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


fn test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_sqrt() raises:
    comptime if has_accelerator():
        print("test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_sqrt")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 9.0, 16.0], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var c = b.sqrt()
        var d = c.to_cpu(stop_grad=False)
        var loss = d.sum()
        loss.backward()
        # B is GPU leaf with values same as A: grad = 0.5/sqrt(b)
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.25, 0.16666667, 0.125])
            )
        )
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


fn test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_chained() raises:
    comptime if has_accelerator():
        print("test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_chained")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var c = b.sqrt() * Tensor[dtype].full_like(b, 2.0)
        var d = c.to_cpu(stop_grad=False)
        var loss = d.sum()
        loss.backward()
        # B grad = 1/sqrt(b), A grad = zero
        assert_true(
            b.grad().to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d2([[1.0, 0.5], [0.33333334, 0.25]])
            )
        )
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA: Both transfers stop_grad=True — grad stays on GPU, never reaches CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_devtransfer_both_stop_grad_true_grad_stays_on_gpu() raises:
    comptime if has_accelerator():
        print("test_devtransfer_both_stop_grad_true_grad_stays_on_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = a.to_gpu(stop_grad=True)
        var c = b * Tensor[dtype].full_like(b, 4.0)
        var d = c.to_cpu(stop_grad=True)
        var loss = d.sum()
        loss.backward()
        # D is the only leaf backward sees — grad = 1.0 per element
        assert_true(
            d.grad().all_close[atol=1e-5](Tensor.ones_like(d))
        )
        # A is completely isolated — never reached by backward
        assert_true(a.grad().all_close[atol=1e-5](Tensor.zeros_like(a)))
        # B is also isolated — no assertion on b.grad() to avoid
        # accessing an unpopulated grad buffer  

# ─────────────────────────────────────────────────────────────────────────────
# EXTRA: Multi-hop — CPU->GPU->CPU->GPU, stop_grad=False throughout
# Grad must traverse all boundaries back to origin
# ─────────────────────────────────────────────────────────────────────────────

fn test_devtransfer_multihop_all_stop_grad_false() raises:
    comptime if has_accelerator():
        print("test_devtransfer_multihop_all_stop_grad_false")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = a.to_gpu()                        # CPU->GPU
        var c = b * Tensor[dtype].full_like(b, 2.0)
        var d = c.to_cpu(stop_grad=False)         # GPU->CPU
        var e = d * Tensor[dtype].full_like(d, 3.0)
        var f = e.to_gpu()                        # CPU->GPU again
        var loss = f.sum()
        loss.backward()
        # grad = 2 * 3 = 6 flows all the way back to A
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].full_like(a, 6.0))
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

fn main() raises:
    # Scenario 1: CPU->GPU stop_grad=False, grad reaches A
    test_devtransfer_cpu2gpu_default_stop_grad_false_1d()
    test_devtransfer_cpu2gpu_default_stop_grad_false_2d()
    test_devtransfer_cpu2gpu_default_stop_grad_false_with_mul()
    test_devtransfer_cpu2gpu_default_stop_grad_false_with_chained_ops()

    # Scenario 2: CPU->GPU stop_grad=True, grad stays on GPU leaf B
    test_devtransfer_cpu2gpu_stop_grad_true_grad_on_gpu_leaf_1d()
    test_devtransfer_cpu2gpu_stop_grad_true_grad_on_gpu_leaf_2d()
    test_devtransfer_cpu2gpu_stop_grad_true_sum_op()
    test_devtransfer_cpu2gpu_stop_grad_true_chained_ops()

    # Scenario 3: CPU->GPU->ops->CPU stop_grad=False, grad reaches A
    test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_sum()
    test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_2d()
    test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_sqrt()
    test_devtransfer_roundtrip_gpu_cpu_stop_grad_false_chained_ops()

    # Scenario 4: CPU->GPU(stop_grad=True)->ops->CPU, grad on B not A
    test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_1d()
    test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_2d()
    test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_sqrt()
    test_devtransfer_gpu_stop_grad_true_then_back_to_cpu_chained()

    # Extra: both stop_grad=True
    test_devtransfer_both_stop_grad_true_grad_stays_on_gpu()

    # Extra: multi-hop all stop_grad=False
    test_devtransfer_multihop_all_stop_grad_false()

    print("All device transfer grad flow tests passed ✓")
