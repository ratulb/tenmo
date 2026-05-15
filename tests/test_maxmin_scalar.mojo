from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator


# ── CPU Tests: Max ────────────────────────────────────────────────────────────


fn test_maxmin_cpu_max_1d_forward() raises:
    print("test_maxmin_cpu_max_1d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
    var b = a.max(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([4.0, 5.0, 4.0, 7.0, 4.0])))


fn test_maxmin_cpu_max_1d_backward() raises:
    print("test_maxmin_cpu_max_1d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
    var b = a.max(4.0)
    var loss = b.sum()
    loss.backward()
    # Grad passes through where a > 4.0
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0, 0.0])))


fn test_maxmin_cpu_max_1d_boundary() raises:
    print("test_maxmin_cpu_max_1d_boundary")
    comptime dtype = DType.float32
    # Value exactly equal to scalar — grad should be zero (not strictly greater)
    var a = Tensor[dtype].d1([4.0, 4.0, 5.0], requires_grad=True)
    var b = a.max(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([4.0, 4.0, 5.0])))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 1.0])))


fn test_maxmin_cpu_max_2d_forward() raises:
    print("test_maxmin_cpu_max_2d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
    var b = a.max(4.0)
    assert_true(b.all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 4.0]])))


fn test_maxmin_cpu_max_2d_backward() raises:
    print("test_maxmin_cpu_max_2d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
    var b = a.max(4.0)
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [0.0, 0.0]])))


fn test_maxmin_cpu_max_3d_forward() raises:
    print("test_maxmin_cpu_max_3d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
        requires_grad=True,
    )
    var b = a.max(4.0)
    assert_true(
        b.all_close(
            Tensor[dtype].d3(
                [[[4.0, 5.0], [4.0, 8.0]], [[6.0, 4.0], [4.0, 9.0]]]
            )
        )
    )


fn test_maxmin_cpu_max_3d_backward() raises:
    print("test_maxmin_cpu_max_3d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
        requires_grad=True,
    )
    var b = a.max(4.0)
    var loss = b.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
            )
        )
    )


fn test_maxmin_cpu_max_all_below_scalar() raises:
    print("test_maxmin_cpu_max_all_below_scalar")
    comptime dtype = DType.float32
    # All values below scalar — grad should be all zeros
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.max(10.0)
    assert_true(b.all_close(Tensor[dtype].d1([10.0, 10.0, 10.0])))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_maxmin_cpu_max_all_above_scalar() raises:
    print("test_maxmin_cpu_max_all_above_scalar")
    comptime dtype = DType.float32
    # All values above scalar — grad should be all ones
    var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
    var b = a.max(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([5.0, 6.0, 7.0])))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_maxmin_cpu_max_chained() raises:
    print("test_maxmin_cpu_max_chained")
    comptime dtype = DType.float32
    # Chain: max then sum — verify grad flows correctly through chain
    var a = Tensor[dtype].d2([[1.0, 8.0], [5.0, 3.0]], requires_grad=True)
    var b = a.max(4.0)
    var c = b * 2.0
    var loss = c.sum()
    loss.backward()
    # grad of a = 2.0 where a > 4.0, else 0.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[0.0, 2.0], [2.0, 0.0]])))


# ── CPU Tests: Min ────────────────────────────────────────────────────────────
fn test_maxmin_cpu_min_1d_forward() raises:
    print("test_maxmin_cpu_min_1d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
    var b = a.min(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([1.0, 4.0, 3.0, 4.0, 2.0])))


fn test_maxmin_cpu_min_1d_backward() raises:
    print("test_maxmin_cpu_min_1d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
    var b = a.min(4.0)
    var loss = b.sum()
    loss.backward()
    # Grad passes through where a < 4.0
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.0, 1.0, 0.0, 1.0])))


fn test_maxmin_cpu_min_1d_boundary() raises:
    print("test_maxmin_cpu_min_1d_boundary")
    comptime dtype = DType.float32
    # Value exactly equal to scalar — grad should be zero (not strictly less)
    var a = Tensor[dtype].d1([4.0, 4.0, 3.0], requires_grad=True)
    var b = a.min(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([4.0, 4.0, 3.0])))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 1.0])))


fn test_maxmin_cpu_min_2d_forward() raises:
    print("test_maxmin_cpu_min_2d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
    var b = a.min(4.0)
    assert_true(b.all_close(Tensor[dtype].d2([[1.0, 4.0], [3.0, 2.0]])))


fn test_maxmin_cpu_min_2d_backward() raises:
    print("test_maxmin_cpu_min_2d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
    var b = a.min(4.0)
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 0.0], [1.0, 1.0]])))


fn test_maxmin_cpu_min_3d_forward() raises:
    print("test_maxmin_cpu_min_3d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
        requires_grad=True,
    )
    var b = a.min(4.0)
    assert_true(
        b.all_close(
            Tensor[dtype].d3(
                [[[1.0, 4.0], [3.0, 4.0]], [[4.0, 2.0], [4.0, 4.0]]]
            )
        )
    )


fn test_maxmin_cpu_min_3d_backward() raises:
    print("test_maxmin_cpu_min_3d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
        requires_grad=True,
    )
    var b = a.min(4.0)
    var loss = b.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]
            )
        )
    )


fn test_maxmin_cpu_min_all_above_scalar() raises:
    print("test_maxmin_cpu_min_all_above_scalar")
    comptime dtype = DType.float32
    # All values above scalar — grad should be all zeros
    var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
    var b = a.min(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_maxmin_cpu_min_all_below_scalar() raises:
    print("test_maxmin_cpu_min_all_below_scalar")
    comptime dtype = DType.float32
    # All values below scalar — grad should be all ones
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.min(4.0)
    assert_true(b.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_maxmin_cpu_min_chained() raises:
    print("test_maxmin_cpu_min_chained")
    comptime dtype = DType.float32
    # Chain: min then multiply — verify grad flows correctly
    var a = Tensor[dtype].d2([[1.0, 8.0], [5.0, 3.0]], requires_grad=True)
    var b = a.min(4.0)
    var c = b * 2.0
    var loss = c.sum()
    loss.backward()
    # grad of a = 2.0 where a < 4.0, else 0.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[2.0, 0.0], [0.0, 2.0]])))


# ── CPU Tests: Max + Min combined in chain ────────────────────────────────────


fn test_maxmin_cpu_max_then_min_chained() raises:
    print("test_maxmin_cpu_max_then_min_chained")
    comptime dtype = DType.float32
    # clamp(x, 2.0, 6.0) = min(max(x, 2.0), 6.0)
    var a = Tensor[dtype].d1([1.0, 3.0, 5.0, 7.0, 9.0], requires_grad=True)
    var b = a.max(2.0)   # clamp low
    var c = b.min(6.0)   # clamp high
    var loss = c.sum()
    loss.backward()
    # Grad = 1.0 only where 2.0 < a < 6.0, i.e. a=3.0 and a=5.0
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0, 0.0, 0.0]))
    )


fn test_maxmin_cpu_negated_grad_flow() raises:
    print("test_maxmin_cpu_negated_grad_flow")
    comptime dtype = DType.float32
    # Verify grad is negated correctly through subtraction chain
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0], requires_grad=True)
    var b = a.max(2.0)
    var c = -b
    var loss = c.sum()
    loss.backward()
    # grad of a = -1.0 where a > 2.0, else 0.0
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, -1.0, -1.0])))


# ── GPU Tests: Max ────────────────────────────────────────────────────────────


fn test_maxmin_gpu_max_1d_forward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_1d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d1([4.0, 5.0, 4.0, 7.0, 4.0]))
        )


fn test_maxmin_gpu_max_1d_backward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_1d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0, 0.0]))
        )


fn test_maxmin_gpu_max_2d_forward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_2d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 4.0]]))
        )


fn test_maxmin_gpu_max_2d_backward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_2d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [0.0, 0.0]]))
        )


fn test_maxmin_gpu_max_3d_backward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_3d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
                )
            )
        )


fn test_maxmin_gpu_max_all_below_scalar() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_all_below_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(10.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_maxmin_gpu_max_all_above_scalar() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_all_above_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_maxmin_gpu_max_chained() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_chained")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 8.0], [5.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var c = b * 2.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.0, 2.0], [2.0, 0.0]]))
        )


# ── GPU Tests: Min ────────────────────────────────────────────────────────────


fn test_maxmin_gpu_min_1d_forward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_1d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d1([1.0, 4.0, 3.0, 4.0, 2.0]))
        )


fn test_maxmin_gpu_min_1d_backward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_1d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([1.0, 0.0, 1.0, 0.0, 1.0]))
        )


fn test_maxmin_gpu_min_2d_forward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_2d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d2([[1.0, 4.0], [3.0, 2.0]]))
        )


fn test_maxmin_gpu_min_2d_backward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_2d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 0.0], [1.0, 1.0]]))
        )


fn test_maxmin_gpu_min_3d_backward() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_3d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]
                )
            )
        )


fn test_maxmin_gpu_min_all_above_scalar() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_all_above_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_maxmin_gpu_min_all_below_scalar() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_all_below_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_maxmin_gpu_min_chained() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_min_chained")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 8.0], [5.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var c = b * 2.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[2.0, 0.0], [0.0, 2.0]]))
        )


# ── GPU Tests: Max + Min combined ─────────────────────────────────────────────


fn test_maxmin_gpu_max_then_min_chained() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_max_then_min_chained")
        comptime dtype = DType.float32
        # clamp(x, 2.0, 6.0) = min(max(x, 2.0), 6.0)
        var a = Tensor[dtype].d1(
            [1.0, 3.0, 5.0, 7.0, 9.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(2.0)
        var c = b.min(6.0)
        var loss = c.sum()
        loss.backward()
        # Grad = 1.0 only where 2.0 < a < 6.0
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0, 0.0, 0.0]))
        )


fn test_maxmin_gpu_negated_grad_flow() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_negated_grad_flow")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(2.0)
        var c = -b
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, -1.0, -1.0])))


# ── CPU/GPU parity check ──────────────────────────────────────────────────────


fn test_maxmin_gpu_parity_max() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_parity_max")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 3.0, 7.0], [8.0, 2.0, 5.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var b_cpu = a_cpu.max(4.0)
        var b_gpu = a_gpu.max(4.0)
        assert_true(b_cpu.all_close(b_gpu.to_cpu()))

        var loss_cpu = b_cpu.sum()
        loss_cpu.backward()

        var loss_gpu = b_gpu.sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


fn test_maxmin_gpu_parity_min() raises:
    comptime if has_accelerator():
        print("test_maxmin_gpu_parity_min")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 3.0, 7.0], [8.0, 2.0, 5.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var b_cpu = a_cpu.min(4.0)
        var b_gpu = a_gpu.min(4.0)
        assert_true(b_cpu.all_close(b_gpu.to_cpu()))

        var loss_cpu = b_cpu.sum()
        loss_cpu.backward()

        var loss_gpu = b_gpu.sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


# ── Main ──────────────────────────────────────────────────────────────────────


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

_ = """
fn main() raises:
    # CPU Max
    test_maxmin_cpu_max_1d_forward()
    test_maxmin_cpu_max_1d_backward()
    test_maxmin_cpu_max_1d_boundary()
    test_maxmin_cpu_max_2d_forward()
    test_maxmin_cpu_max_2d_backward()
    test_maxmin_cpu_max_3d_forward()
    test_maxmin_cpu_max_3d_backward()
    test_maxmin_cpu_max_all_below_scalar()
    test_maxmin_cpu_max_all_above_scalar()
    test_maxmin_cpu_max_chained()

    # CPU Min
    test_maxmin_cpu_min_1d_forward()
    test_maxmin_cpu_min_1d_backward()
    test_maxmin_cpu_min_1d_boundary()
    test_maxmin_cpu_min_2d_forward()
    test_maxmin_cpu_min_2d_backward()
    test_maxmin_cpu_min_3d_forward()
    test_maxmin_cpu_min_3d_backward()
    test_maxmin_cpu_min_all_above_scalar()
    test_maxmin_cpu_min_all_below_scalar()
    test_maxmin_cpu_min_chained()

    # CPU combined
    test_maxmin_cpu_max_then_min_chained()
    test_maxmin_cpu_negated_grad_flow()

    # GPU Max
    test_maxmin_gpu_max_1d_forward()
    test_maxmin_gpu_max_1d_backward()
    test_maxmin_gpu_max_2d_forward()
    test_maxmin_gpu_max_2d_backward()
    test_maxmin_gpu_max_3d_backward()
    test_maxmin_gpu_max_all_below_scalar()
    test_maxmin_gpu_max_all_above_scalar()
    test_maxmin_gpu_max_chained()

    # GPU Min
    test_maxmin_gpu_min_1d_forward()
    test_maxmin_gpu_min_1d_backward()
    test_maxmin_gpu_min_2d_forward()
    test_maxmin_gpu_min_2d_backward()
    test_maxmin_gpu_min_3d_backward()
    test_maxmin_gpu_min_all_above_scalar()
    test_maxmin_gpu_min_all_below_scalar()
    test_maxmin_gpu_min_chained()

    # GPU combined
    test_maxmin_gpu_max_then_min_chained()
    test_maxmin_gpu_negated_grad_flow()

    # CPU/GPU parity
    test_maxmin_gpu_parity_max()
    test_maxmin_gpu_parity_min()

    print("All max/min tests passed!")
"""
