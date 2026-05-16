from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from std.sys import has_accelerator
from tenmo.shapes import Shape
from std.sys import has_accelerator

comptime dtype = DType.float32



# ============================================================
# GPU SUM TESTS — forward + backward, all reduction patterns
# ============================================================


fn test_gpu_sum_full_reduction_scalar_grad() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_full_reduction_scalar_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum()  # scalar output
        s_gpu.backward()
        # Forward: 1+2+3+4 = 10
        var s_cpu = s_gpu.to_cpu()
        assert_true(s_cpu.all_close(Tensor[dtype].scalar(10.0)))
        # Backward: grad flows back to CPU a, each element gets 1.0
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_axis0_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis0_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[0])  # shape (2,)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        # Forward: [1+3, 2+4] = [4, 6]
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d1([4.0, 6.0])))
        # Backward: each element gets 1.0
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_axis1_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis1_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[1])  # shape (2,)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        # Forward: [1+2+3, 4+5+6] = [6, 15]
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d1([6.0, 15.0])))
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_axis0_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis0_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[0], keepdims=True)  # shape (1,2)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d2([[4.0, 6.0]])))
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_axis1_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis1_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[1], keepdims=True)  # shape (2,1)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d2([[3.0], [7.0]])))
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_multi_axis_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_multi_axis_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[0, 2])  # shape (2,)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        # axis0+axis2: sum over batch and last dim
        # col0: (1+2)+(5+6)=14, col1: (3+4)+(7+8)=22
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d1([14.0, 22.0])))
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_multi_axis_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_multi_axis_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[0, 2], keepdims=True)  # shape (1,2,1)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d3([[[14.0], [22.0]]])))
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_3d_axis1_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis1_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[1], keepdims=True)  # shape (2,1,2)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        assert_true(
            s_gpu.to_cpu().all_close(
                Tensor[dtype].d3([[[4.0, 6.0]], [[12.0, 14.0]]])
            )
        )
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_negative_axis() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_negative_axis")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[-1])  # same as axis=1, shape (2,)
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        assert_true(s_gpu.to_cpu().all_close(Tensor[dtype].d1([6.0, 15.0])))
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_4d_middle_axes() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_4d_middle_axes")
        comptime dtype = DType.float32
        # shape (2,3,4,5) — reduce over axes 1 and 2
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_cpu_ref = a.copy()
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[1, 2])  # shape (2,5)
        var s_cpu = s_gpu.to_cpu()
        # Cross-validate against CPU sum
        var s_ref = a_cpu_ref.sum(axes=[1, 2])
        assert_true(s_cpu.all_close(s_ref))
        var loss = s_gpu.to_cpu().sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_sum_grad_accumulation() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_grad_accumulation")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        # Two separate sum ops, both backward — grads should accumulate
        var s1 = a_gpu.sum(axes=[0])
        var s2 = a_gpu.sum(axes=[1])
        var loss1 = s1.to_cpu().sum()
        var loss2 = s2.to_cpu().sum()
        loss1.backward()
        s1.zero_grad()
        a_gpu.zero_grad()
        loss2.backward()
        # Each backward contributes 1.0 per element → total 2.0 per element
        assert_true(a.grad().all_close(Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_gpu_sum_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(8, 16)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.sum(axes=[0], keepdims=True)
        var s_cpu_ref = a_copy.sum(axes=[0], keepdims=True)
        assert_true(s_gpu.to_cpu().all_close(s_cpu_ref))
        ss = s_gpu.to_cpu().sum()
        ss.backward()
        sss = s_cpu_ref.sum()
        sss.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ============================================================
# GPU MEAN TESTS — forward + backward, all reduction patterns
# ============================================================


fn test_gpu_mean_full_reduction_scalar_grad() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_full_reduction_scalar_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean()  # scalar: (1+2+3+4)/4 = 2.5
        m_gpu.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].scalar(2.5)))
        # Each element grad = 1/4 = 0.25
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.25, 0.25], [0.25, 0.25]]))
        )


fn test_gpu_mean_axis0_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis0_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[0])  # shape (2,): [2.0, 3.0]
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d1([2.0, 3.0])))
        # Each element grad = 1/2 = 0.5
        assert_true(a.grad().all_close(Tensor[dtype].d2([[0.5, 0.5], [0.5, 0.5]])))


fn test_gpu_mean_axis1_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis1_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[1])  # shape (2,): [2.0, 5.0]
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d1([2.0, 5.0])))
        # Each element grad = 1/3
        var expected_grad = Tensor[dtype].d2(
            [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
        )
        assert_true(a.grad().all_close(expected_grad))


fn test_gpu_mean_axis0_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis0_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[0], keepdims=True)  # shape (1,2): [[2.0, 3.0]]
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d2([[2.0, 3.0]])))
        assert_true(a.grad().all_close(Tensor[dtype].d2([[0.5, 0.5], [0.5, 0.5]])))


fn test_gpu_mean_axis1_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis1_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(
            axes=[1], keepdims=True
        )  # shape (2,1): [[1.5],[3.5]]
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d2([[1.5], [3.5]])))
        assert_true(a.grad().all_close(Tensor[dtype].d2([[0.5, 0.5], [0.5, 0.5]])))


fn test_gpu_mean_3d_axis1_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis1_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[1])  # shape (2,2)
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        # axis1 mean: [[2,3],[6,7]]
        assert_true(
            m_gpu.to_cpu().all_close(Tensor[dtype].d2([[2.0, 3.0], [6.0, 7.0]]))
        )
        # Each element grad = 1/2 = 0.5
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]
                )
            )
        )


fn test_gpu_mean_3d_axis1_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis1_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[1], keepdims=True)  # shape (2,1,2)
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(
            m_gpu.to_cpu().all_close(Tensor[dtype].d3([[[2.0, 3.0]], [[6.0, 7.0]]]))
        )
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]
                )
            )
        )


fn test_gpu_mean_multi_axis_no_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_multi_axis_no_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[0, 1])  # shape (2,): mean over batch+rows
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        # mean over axis0+axis1: count=4
        # col0: (1+3+5+7)/4=4.0, col1: (2+4+6+8)/4=5.0
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d1([4.0, 5.0])))
        # Each element grad = 1/4 = 0.25
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]
                )
            )
        )


fn test_gpu_mean_multi_axis_keepdims() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_multi_axis_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[0, 1], keepdims=True)  # shape (1,1,2)
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d3([[[4.0, 5.0]]])))
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]
                )
            )
        )


fn test_gpu_mean_negative_axis() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_negative_axis")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[-1])  # same as axis=1
        var loss = m_gpu.to_cpu().sum()
        loss.backward()
        assert_true(m_gpu.to_cpu().all_close(Tensor[dtype].d1([2.0, 5.0])))
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    ]
                )
            )
        )


fn test_gpu_mean_4d_middle_axes() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_4d_middle_axes")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[1, 2])  # shape (2,5)
        var m_cpu_ref = a_copy.mean(axes=[1, 2])
        # Forward cross-validate
        assert_true(m_gpu.to_cpu().all_close(m_cpu_ref))
        # Backward cross-validate
        ss = m_gpu.to_cpu().sum()
        ss.backward()
        sss = m_cpu_ref.sum()
        sss.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_gpu_mean_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(8, 16)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m_gpu = a_gpu.mean(axes=[0], keepdims=True)
        var m_cpu_ref = a_copy.mean(axes=[0], keepdims=True)
        assert_true(m_gpu.to_cpu().all_close(m_cpu_ref))
        ss = m_gpu.to_cpu().sum()
        ss.backward()
        sss = m_cpu_ref.sum()
        sss.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_gpu_mean_grad_accumulation() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_grad_accumulation")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m1 = a_gpu.mean(axes=[0])
        var m2 = a_gpu.mean(axes=[1])
        ss = m1.to_cpu().sum()
        ss.backward()
        a_gpu.zero_grad()
        sss = m2.to_cpu().sum()
        sss.backward()
        # m1 backward: 0.5 each; m2 backward: 0.5 each → accumulated 1.0 each
        assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])))


# ============================================================
# DEVICE TRANSFER BACKWARD TESTS
# ============================================================


fn test_gpu_sum_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.sum()
        s.backward()
        # Grad must have flowed back to CPU tensor a
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_mean_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.mean()
        m.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.25, 0.25], [0.25, 0.25]]))
        )


fn test_gpu_sum_cpu_tensor_unchanged_after_transfer() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_cpu_tensor_unchanged_after_transfer")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_snapshot = a.copy()
        var a_gpu = a.to_gpu()
        var s = a_gpu.sum()
        s.backward()
        # Original CPU tensor data should be unchanged
        assert_true(a.all_close(a_snapshot))


fn test_gpu_sum_chained_ops_grad_flow() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_chained_ops_grad_flow")
        comptime dtype = DType.float32
        # sum over axis then sum to scalar — two hops back through graph
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.sum(axes=[2], keepdims=True)  # shape (2,2,1)
        var s2 = s1.sum(axes=[0], keepdims=True)  # shape (1,2,1)
        var loss = s2.to_cpu().sum()
        loss.backward()
        # Every element contributes once to each reduction → grad = 1.0 everywhere
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_gpu_mean_chained_with_sum_grad_flow() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_chained_with_sum_grad_flow")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.mean(axes=[1], keepdims=True)  # shape (2,1)
        var s = m.sum()  # scalar
        s.backward()
        # grad through mean: 1/3 per element
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    ]
                )
            )
        )


# ── Sum forward tests ─────────────────────────────────────────────────────────


fn test_gpu_sum_full_reduction_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_full_reduction_forward")
        var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum()
        var s_gpu = a_gpu.sum()
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis0_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0])
        var s_gpu = a_gpu.sum(axes=[0])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis1_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1])
        var s_gpu = a_gpu.sum(axes=[1])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis0_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis0_keepdims_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0], keepdims=True)
        var s_gpu = a_gpu.sum(axes=[0], keepdims=True)
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis1_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis1_keepdims_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1], keepdims=True)
        var s_gpu = a_gpu.sum(axes=[1], keepdims=True)
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis0_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0])
        var s_gpu = a_gpu.sum(axes=[0])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis1_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1])
        var s_gpu = a_gpu.sum(axes=[1])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis2_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis2_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[2])
        var s_gpu = a_gpu.sum(axes=[2])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis1_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis1_keepdims_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1], keepdims=True)
        var s_gpu = a_gpu.sum(axes=[1], keepdims=True)
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_multi_axis_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_multi_axis_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0, 2])
        var s_gpu = a_gpu.sum(axes=[0, 2])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_large_tensor_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_large_tensor_forward")
        var a = Tensor[dtype].rand(32, 64, requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1])
        var s_gpu = a_gpu.sum(axes=[1])
        assert_true(s_cpu.all_close(s_gpu.to_cpu()))


# ── Sum backward tests ────────────────────────────────────────────────────────


fn test_gpu_sum_full_reduction_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_full_reduction_backward")
        var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_axis0_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis0_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0]).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[0]).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis1_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1]).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[1]).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_axis0_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis0_keepdims_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0], keepdims=True).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[0], keepdims=True).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_axis1_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_axis1_keepdims_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1], keepdims=True).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[1], keepdims=True).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_3d_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis1_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1]).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[1]).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_3d_axis1_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_3d_axis1_keepdims_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1], keepdims=True).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[1], keepdims=True).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_multi_axis_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_multi_axis_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[0, 2]).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[0, 2]).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_large_tensor_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_large_tensor_backward")
        var a = Tensor[dtype].rand(32, 64, requires_grad=True)
        var a_gpu = a.to_gpu()
        var s_cpu = a.sum(axes=[1]).sum()
        s_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var s_gpu = a_gpu.sum(axes=[1]).sum()
        s_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_grad_flows_to_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_grad_flows_to_cpu")
        var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.sum()
        s.backward()
        # grad flows back to CPU tensor a
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── Mean forward tests ────────────────────────────────────────────────────────


fn test_gpu_mean_full_reduction_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_full_reduction_forward")
        var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean()
        var m_gpu = a_gpu.mean()
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis0_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0])
        var m_gpu = a_gpu.mean(axes=[0])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis1_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1])
        var m_gpu = a_gpu.mean(axes=[1])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis0_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis0_keepdims_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0], keepdims=True)
        var m_gpu = a_gpu.mean(axes=[0], keepdims=True)
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis1_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis1_keepdims_forward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1], keepdims=True)
        var m_gpu = a_gpu.mean(axes=[1], keepdims=True)
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis0_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0])
        var m_gpu = a_gpu.mean(axes=[0])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis1_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1])
        var m_gpu = a_gpu.mean(axes=[1])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis2_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis2_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[2])
        var m_gpu = a_gpu.mean(axes=[2])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis1_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis1_keepdims_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1], keepdims=True)
        var m_gpu = a_gpu.mean(axes=[1], keepdims=True)
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_multi_axis_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_multi_axis_forward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0, 2])
        var m_gpu = a_gpu.mean(axes=[0, 2])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_large_tensor_forward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_large_tensor_forward")
        var a = Tensor[dtype].rand(32, 64, requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1])
        var m_gpu = a_gpu.mean(axes=[1])
        assert_true(m_cpu.all_close(m_gpu.to_cpu()))


# ── Mean backward tests ───────────────────────────────────────────────────────


fn test_gpu_mean_full_reduction_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_full_reduction_backward")
        var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_axis0_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis0_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0]).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[0]).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis1_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1]).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[1]).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_axis0_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis0_keepdims_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0], keepdims=True).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[0], keepdims=True).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_axis1_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_axis1_keepdims_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1], keepdims=True).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[1], keepdims=True).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_3d_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis1_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1]).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[1]).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_3d_axis1_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_3d_axis1_keepdims_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1], keepdims=True).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[1], keepdims=True).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_multi_axis_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_multi_axis_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[0, 2]).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[0, 2]).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_large_tensor_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_large_tensor_backward")
        var a = Tensor[dtype].rand(32, 64, requires_grad=True)
        var a_gpu = a.to_gpu()
        var m_cpu = a.mean(axes=[1]).sum()
        m_cpu.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var m_gpu = a_gpu.mean(axes=[1]).sum()
        m_gpu.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_grad_flows_to_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_grad_flows_to_cpu")
        var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.mean()
        m.backward()
        var expected = Tensor[dtype].full(a.shape(), Scalar[dtype](0.25))
        assert_true(a.grad().all_close(expected))


# ── Combined sum+mean chained ops ─────────────────────────────────────────────


fn test_gpu_sum_then_mean_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_then_mean_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1]).mean()
        cpu_result.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var gpu_result = a_gpu.sum(axes=[1]).mean()
        gpu_result.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_then_sum_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_then_sum_backward")
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1]).sum()
        cpu_result.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var gpu_result = a_gpu.mean(axes=[1]).sum()
        gpu_result.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_sum_keepdims_then_mean_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_keepdims_then_mean_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1], keepdims=True).mean()
        cpu_result.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var gpu_result = a_gpu.sum(axes=[1], keepdims=True).mean()
        gpu_result.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn test_gpu_mean_keepdims_then_sum_backward() raises:
    comptime if has_accelerator():
        print("test_gpu_mean_keepdims_then_sum_backward")
        var a = Tensor[dtype].d3(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[2], keepdims=True).sum()
        cpu_result.backward()
        var a_cpu_grad = a.grad().copy()
        a.zero_grad()
        var gpu_result = a_gpu.mean(axes=[2], keepdims=True).sum()
        gpu_result.backward()
        assert_true(a.grad().all_close(a_cpu_grad))


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


