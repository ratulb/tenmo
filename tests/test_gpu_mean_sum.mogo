from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape

comptime dtype = DType.float32


# ── Sum forward tests ─────────────────────────────────────────────────────────


fn test_gpu_sum_full_reduction_forward() raises:
    print("test_gpu_sum_full_reduction_forward")
    var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum()
    var s_gpu = a_gpu.sum()
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis0_forward() raises:
    print("test_gpu_sum_axis0_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[0])
    var s_gpu = a_gpu.sum(axes=[0])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis1_forward() raises:
    print("test_gpu_sum_axis1_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[1])
    var s_gpu = a_gpu.sum(axes=[1])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis0_keepdims_forward() raises:
    print("test_gpu_sum_axis0_keepdims_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[0], keepdims=True)
    var s_gpu = a_gpu.sum(axes=[0], keepdims=True)
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_axis1_keepdims_forward() raises:
    print("test_gpu_sum_axis1_keepdims_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[1], keepdims=True)
    var s_gpu = a_gpu.sum(axes=[1], keepdims=True)
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis0_forward() raises:
    print("test_gpu_sum_3d_axis0_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[0])
    var s_gpu = a_gpu.sum(axes=[0])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis1_forward() raises:
    print("test_gpu_sum_3d_axis1_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[1])
    var s_gpu = a_gpu.sum(axes=[1])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis2_forward() raises:
    print("test_gpu_sum_3d_axis2_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[2])
    var s_gpu = a_gpu.sum(axes=[2])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_3d_axis1_keepdims_forward() raises:
    print("test_gpu_sum_3d_axis1_keepdims_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[1], keepdims=True)
    var s_gpu = a_gpu.sum(axes=[1], keepdims=True)
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_multi_axis_forward() raises:
    print("test_gpu_sum_multi_axis_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[0, 2])
    var s_gpu = a_gpu.sum(axes=[0, 2])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


fn test_gpu_sum_large_tensor_forward() raises:
    print("test_gpu_sum_large_tensor_forward")
    var a = Tensor[dtype].rand(32, 64, requires_grad=True)
    var a_gpu = a.to_gpu()
    var s_cpu = a.sum(axes=[1])
    var s_gpu = a_gpu.sum(axes=[1])
    assert_true(s_cpu.all_close(s_gpu.to_cpu()))


# ── Sum backward tests ────────────────────────────────────────────────────────


fn test_gpu_sum_full_reduction_backward() raises:
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
    print("test_gpu_sum_grad_flows_to_cpu")
    var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var s = a_gpu.sum()
    s.backward()
    # grad flows back to CPU tensor a
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── Mean forward tests ────────────────────────────────────────────────────────


fn test_gpu_mean_full_reduction_forward() raises:
    print("test_gpu_mean_full_reduction_forward")
    var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean()
    var m_gpu = a_gpu.mean()
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis0_forward() raises:
    print("test_gpu_mean_axis0_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[0])
    var m_gpu = a_gpu.mean(axes=[0])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis1_forward() raises:
    print("test_gpu_mean_axis1_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[1])
    var m_gpu = a_gpu.mean(axes=[1])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis0_keepdims_forward() raises:
    print("test_gpu_mean_axis0_keepdims_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[0], keepdims=True)
    var m_gpu = a_gpu.mean(axes=[0], keepdims=True)
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_axis1_keepdims_forward() raises:
    print("test_gpu_mean_axis1_keepdims_forward")
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[1], keepdims=True)
    var m_gpu = a_gpu.mean(axes=[1], keepdims=True)
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis0_forward() raises:
    print("test_gpu_mean_3d_axis0_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[0])
    var m_gpu = a_gpu.mean(axes=[0])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis1_forward() raises:
    print("test_gpu_mean_3d_axis1_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[1])
    var m_gpu = a_gpu.mean(axes=[1])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis2_forward() raises:
    print("test_gpu_mean_3d_axis2_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[2])
    var m_gpu = a_gpu.mean(axes=[2])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_3d_axis1_keepdims_forward() raises:
    print("test_gpu_mean_3d_axis1_keepdims_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[1], keepdims=True)
    var m_gpu = a_gpu.mean(axes=[1], keepdims=True)
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_multi_axis_forward() raises:
    print("test_gpu_mean_multi_axis_forward")
    var a = Tensor[dtype].d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[0, 2])
    var m_gpu = a_gpu.mean(axes=[0, 2])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


fn test_gpu_mean_large_tensor_forward() raises:
    print("test_gpu_mean_large_tensor_forward")
    var a = Tensor[dtype].rand(32, 64, requires_grad=True)
    var a_gpu = a.to_gpu()
    var m_cpu = a.mean(axes=[1])
    var m_gpu = a_gpu.mean(axes=[1])
    assert_true(m_cpu.all_close(m_gpu.to_cpu()))


# ── Mean backward tests ───────────────────────────────────────────────────────


fn test_gpu_mean_full_reduction_backward() raises:
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
    print("test_gpu_mean_grad_flows_to_cpu")
    var a = Tensor[dtype].d2([[1, 2], [3, 4]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var m = a_gpu.mean()
    m.backward()
    var expected = Tensor[dtype].full(a.shape(), Scalar[dtype](0.25))
    assert_true(a.grad().all_close(expected))


# ── Combined sum+mean chained ops ─────────────────────────────────────────────


fn test_gpu_sum_then_mean_backward() raises:
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
    @parameter
    if not has_accelerator():
        print("No GPU available — skipping GPU sum/mean tests")
        return

    # Sum forward
    test_gpu_sum_full_reduction_forward()
    test_gpu_sum_axis0_forward()
    test_gpu_sum_axis1_forward()
    test_gpu_sum_axis0_keepdims_forward()
    test_gpu_sum_axis1_keepdims_forward()
    test_gpu_sum_3d_axis0_forward()
    test_gpu_sum_3d_axis1_forward()
    test_gpu_sum_3d_axis2_forward()
    test_gpu_sum_3d_axis1_keepdims_forward()
    test_gpu_sum_multi_axis_forward()
    test_gpu_sum_large_tensor_forward()

    # Sum backward
    test_gpu_sum_full_reduction_backward()
    test_gpu_sum_axis0_backward()
    test_gpu_sum_axis1_backward()
    test_gpu_sum_axis0_keepdims_backward()
    test_gpu_sum_axis1_keepdims_backward()
    test_gpu_sum_3d_axis1_backward()
    test_gpu_sum_3d_axis1_keepdims_backward()
    test_gpu_sum_multi_axis_backward()
    test_gpu_sum_large_tensor_backward()
    test_gpu_sum_grad_flows_to_cpu()

    # Mean forward
    test_gpu_mean_full_reduction_forward()
    test_gpu_mean_axis0_forward()
    test_gpu_mean_axis1_forward()
    test_gpu_mean_axis0_keepdims_forward()
    test_gpu_mean_axis1_keepdims_forward()
    test_gpu_mean_3d_axis0_forward()
    test_gpu_mean_3d_axis1_forward()
    test_gpu_mean_3d_axis2_forward()
    test_gpu_mean_3d_axis1_keepdims_forward()
    test_gpu_mean_multi_axis_forward()
    test_gpu_mean_large_tensor_forward()

    # Mean backward
    test_gpu_mean_full_reduction_backward()
    test_gpu_mean_axis0_backward()
    test_gpu_mean_axis1_backward()
    test_gpu_mean_axis0_keepdims_backward()
    test_gpu_mean_axis1_keepdims_backward()
    test_gpu_mean_3d_axis1_backward()
    test_gpu_mean_3d_axis1_keepdims_backward()
    test_gpu_mean_multi_axis_backward()
    test_gpu_mean_large_tensor_backward()
    test_gpu_mean_grad_flows_to_cpu()

    # Chained
    test_gpu_sum_then_mean_backward()
    test_gpu_mean_then_sum_backward()
    test_gpu_sum_keepdims_then_mean_backward()
    test_gpu_mean_keepdims_then_sum_backward()

    print("\n=== ALL GPU SUM/MEAN TESTS PASSED ===")
