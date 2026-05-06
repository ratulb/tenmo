from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.device import GPU, has_accelerator
from std.sys import has_accelerator

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC TESTS - 1D
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_1d_forward_cpu() raises:
    """Test 1D sum forward on CPU."""
    print("test_sum_1d_forward_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var result = a.sum()
    assert_true(result.item() == 15.0)
    print("✓ CPU 1D sum forward passed")

fn test_sum_1d_forward_gpu() raises:
    """Test 1D sum forward on GPU."""
    comptime if has_accelerator():
        print("test_sum_1d_forward_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        assert_true(result.to_cpu().item() == 15.0)
        print("✓ GPU 1D sum forward passed")

fn test_sum_1d_backward_cpu() raises:
    """Test 1D sum backward on CPU."""
    print("test_sum_1d_backward_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = a.sum()
    loss.backward()
    var expected = Tensor[dtype].ones_like(a)
    assert_true(a.grad().all_close(expected))
    print("✓ CPU 1D sum backward passed")

fn test_sum_1d_backward_gpu() raises:
    """Test 1D sum backward on GPU."""
    comptime if has_accelerator():
        print("test_sum_1d_backward_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.sum()
        loss.backward()
        var expected = Tensor[dtype].ones_like(a)
        assert_true(a.grad().all_close(expected))
        print("✓ GPU 1D sum backward passed")

fn test_mean_1d_forward_cpu() raises:
    """Test 1D mean forward on CPU."""
    print("test_mean_1d_forward_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var result = a.mean()
    assert_true(result.item() == 3.0)
    print("✓ CPU 1D mean forward passed")

fn test_mean_1d_forward_gpu() raises:
    """Test 1D mean forward on GPU."""
    comptime if has_accelerator():
        print("test_mean_1d_forward_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean()
        assert_true(result.to_cpu().item() == 3.0)
        print("✓ GPU 1D mean forward passed")

fn test_mean_1d_backward_cpu() raises:
    """Test 1D mean backward on CPU."""
    print("test_mean_1d_backward_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = a.mean()
    loss.backward()
    var expected = Tensor[dtype].d1([1.0/3.0, 1.0/3.0, 1.0/3.0])
    assert_true(a.grad().all_close(expected))
    print("✓ CPU 1D mean backward passed")

fn test_mean_1d_backward_gpu() raises:
    """Test 1D mean backward on GPU."""
    comptime if has_accelerator():
        print("test_mean_1d_backward_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.mean()
        loss.backward()
        var expected = Tensor[dtype].d1([1.0/3.0, 1.0/3.0, 1.0/3.0])
        assert_true(a.grad().all_close(expected))
        print("✓ GPU 1D mean backward passed")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 2D TESTS - Axes and Keepdims
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_2d_axis0_cpu() raises:
    """Test sum along axis 0 (rows) on CPU."""
    print("test_sum_2d_axis0_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.sum(axes=[0])
    var expected = Tensor[dtype].d1([5.0, 7.0, 9.0])
    assert_true(result.all_close(expected))
    print("✓ CPU 2D sum axis0 passed")

fn test_sum_2d_axis0_gpu() raises:
    """Test sum along axis 0 (rows) on GPU."""
    comptime if has_accelerator():
        print("test_sum_2d_axis0_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[0])
        var expected = Tensor[dtype].d1([5.0, 7.0, 9.0])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU 2D sum axis0 passed")

fn test_sum_2d_axis1_cpu() raises:
    """Test sum along axis 1 (columns) on CPU."""
    print("test_sum_2d_axis1_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.sum(axes=[1])
    var expected = Tensor[dtype].d1([6.0, 15.0])
    assert_true(result.all_close(expected))
    print("✓ CPU 2D sum axis1 passed")

fn test_sum_2d_axis1_gpu() raises:
    """Test sum along axis 1 (columns) on GPU."""
    comptime if has_accelerator():
        print("test_sum_2d_axis1_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[1])
        var expected = Tensor[dtype].d1([6.0, 15.0])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU 2D sum axis1 passed")

fn test_sum_2d_keepdims_cpu() raises:
    """Test sum with keepdims=True on CPU."""
    print("test_sum_2d_keepdims_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.sum(axes=[1], keepdims=True)
    var expected = Tensor[dtype].d2([[6.0], [15.0]])
    assert_true(result.all_close(expected))
    print("✓ CPU 2D sum keepdims passed")

fn test_sum_2d_keepdims_gpu() raises:
    """Test sum with keepdims=True on GPU."""
    comptime if has_accelerator():
        print("test_sum_2d_keepdims_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[1], keepdims=True)
        var expected = Tensor[dtype].d2([[6.0], [15.0]])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU 2D sum keepdims passed")

fn test_sum_2d_backward_axis0_cpu() raises:
    """Test backward of sum along axis 0 on CPU."""
    print("test_sum_2d_backward_axis0_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var loss = a.sum(axes=[0])
    loss.backward()
    var expected = Tensor[dtype].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected))
    print("✓ CPU 2D sum backward axis0 passed")

fn test_sum_2d_backward_axis0_gpu() raises:
    """Test backward of sum along axis 0 on GPU."""
    comptime if has_accelerator():
        print("test_sum_2d_backward_axis0_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.sum(axes=[0])
        loss.backward()
        var expected = Tensor[dtype].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert_true(a.grad().all_close(expected))
        print("✓ GPU 2D sum backward axis0 passed")

fn test_mean_2d_axis0_cpu() raises:
    """Test mean along axis 0 on CPU."""
    print("test_mean_2d_axis0_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.mean(axes=[0])
    var expected = Tensor[dtype].d1([2.5, 3.5, 4.5])
    assert_true(result.all_close(expected))

fn test_mean_2d_axis0_gpu() raises:
    """Test mean along axis 0 on GPU."""
    comptime if has_accelerator():
        print("test_mean_2d_axis0_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean(axes=[0])
        var expected = Tensor[dtype].d1([2.5, 3.5, 4.5])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU 2D mean axis0 passed")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 3D TESTS - Higher Dimensions
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_3d_all_cpu() raises:
    """Test sum of all elements in 3D tensor on CPU."""
    print("test_sum_3d_all_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    var result = a.sum()
    assert_true(result.item() == 36.0)
    print("✓ CPU 3D sum all passed")

fn test_sum_3d_all_gpu() raises:
    """Test sum of all elements in 3D tensor on GPU."""
    comptime if has_accelerator():
        print("test_sum_3d_all_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        assert_true(result.to_cpu().item() == 36.0)
        print("✓ GPU 3D sum all passed")

fn test_sum_3d_axis12_cpu() raises:
    """Test sum over last two axes (spatial dimensions) on CPU."""
    print("test_sum_3d_axis12_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    var result = a.sum(axes=[1, 2])
    var expected = Tensor[dtype].d1([10.0, 26.0])  # 1+2+3+4=10, 5+6+7+8=26
    assert_true(result.all_close(expected))
    print("✓ CPU 3D sum axes [1,2] passed")

fn test_sum_3d_axis12_gpu() raises:
    """Test sum over last two axes on GPU."""
    comptime if has_accelerator():
        print("test_sum_3d_axis12_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[1, 2])
        var expected = Tensor[dtype].d1([10.0, 26.0])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU 3D sum axes [1,2] passed")

fn test_mean_3d_axis0_cpu() raises:
    """Test mean along first axis on CPU."""
    print("test_mean_3d_axis0_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    var result = a.mean(axes=[0])
    var expected = Tensor[dtype].d2([[3.0, 4.0], [5.0, 6.0]])  # (1+5)/2=3, etc
    assert_true(result.all_close(expected))
    print("✓ CPU 3D mean axis0 passed")

fn test_mean_3d_axis0_gpu() raises:
    """Test mean along first axis on GPU."""
    comptime if has_accelerator():
        print("test_mean_3d_axis0_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean(axes=[0])
        var expected = Tensor[dtype].d2([[3.0, 4.0], [5.0, 6.0]])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU 3D mean axis0 passed")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRADIENT FLOW VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_grad_flow_chain_cpu() raises:
    """Test gradient flow through chain of operations on CPU."""
    print("test_sum_grad_flow_chain_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a * 2.0
    var loss = b.sum()
    loss.backward()
    var expected = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    assert_true(a.grad().all_close(expected))
    print("✓ CPU sum grad flow chain passed")

fn test_sum_grad_flow_chain_gpu() raises:
    """Test gradient flow through chain of operations on GPU."""
    comptime if has_accelerator():
        print("test_sum_grad_flow_chain_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu * 2.0
        var loss = b.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
        assert_true(a.grad().all_close(expected))
        print("✓ GPU sum grad flow chain passed")

fn test_mean_grad_flow_scaling_cpu() raises:
    """Test mean gradient scaling on CPU."""
    print("test_mean_grad_flow_scaling_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var loss = a.mean()
    loss.backward()
    var expected = Tensor[dtype].d1([0.25, 0.25, 0.25, 0.25])
    assert_true(a.grad().all_close(expected))
    print("✓ CPU mean grad scaling passed")

fn test_mean_grad_flow_scaling_gpu() raises:
    """Test mean gradient scaling on GPU."""
    comptime if has_accelerator():
        print("test_mean_grad_flow_scaling_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.mean()
        loss.backward()
        var expected = Tensor[dtype].d1([0.25, 0.25, 0.25, 0.25])
        assert_true(a.grad().all_close(expected))
        print("✓ GPU mean grad scaling passed")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTIPLE BACKWARD PASSES (Grad Accumulation)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_multiple_backward_cpu() raises:
    """Test multiple backward passes with grad accumulation on CPU."""
    print("test_sum_multiple_backward_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)

    var loss1 = a.sum()
    loss1.backward()
    var grad_after_first = a.grad().copy()

    var loss2 = (a * 2.0).sum()
    loss2.backward()

    var expected = grad_after_first + Tensor[dtype].d1([2.0, 2.0, 2.0])
    assert_true(a.grad().all_close(expected))
    print("✓ CPU multiple backward passed")

fn test_sum_multiple_backward_gpu() raises:
    """Test multiple backward passes on GPU."""
    comptime if has_accelerator():
        print("test_sum_multiple_backward_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()

        var loss1 = a_gpu.sum()
        loss1.backward()
        var grad_after_first = a.grad().copy()

        var a_gpu2 = a.to_gpu()  # Re-get with accumulated grad
        var loss2 = (a_gpu2 * 2.0).sum()
        loss2.backward()

        var expected = grad_after_first + Tensor[dtype].d1([2.0, 2.0, 2.0])
        assert_true(a.grad().all_close(expected))
        print("✓ GPU multiple backward passed")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_single_element_cpu() raises:
    """Test sum of single element tensor on CPU."""
    print("test_sum_single_element_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(42.0)
    var result = a.sum()
    assert_true(result.item() == 42.0)
    print("✓ CPU single element sum passed")

fn test_sum_single_element_gpu() raises:
    """Test sum of single element tensor on GPU."""
    comptime if has_accelerator():
        print("test_sum_single_element_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(42.0)
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        assert_true(result.to_cpu().item() == 42.0)
        print("✓ GPU single element sum passed")

fn test_sum_negative_axis_cpu() raises:
    """Test sum with negative axes on CPU."""
    print("test_sum_negative_axis_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.sum(axes=[-1])
    var expected = Tensor[dtype].d1([3.0, 7.0])
    assert_true(result.all_close(expected))
    print("✓ CPU negative axis sum passed")

fn test_sum_negative_axis_gpu() raises:
    """Test sum with negative axes on GPU."""
    comptime if has_accelerator():
        print("test_sum_negative_axis_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[-1])
        var expected = Tensor[dtype].d1([3.0, 7.0])
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU negative axis sum passed")

fn test_sum_empty_axes_cpu() raises:
    """Test sum with empty axes (full reduction) on CPU."""
    print("test_sum_empty_axes_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.sum(axes=[])
    var expected = Tensor[dtype].scalar(10.0)
    assert_true(result.all_close(expected))
    print("✓ CPU empty axes sum passed")

fn test_sum_empty_axes_gpu() raises:
    """Test sum with empty axes on GPU."""
    comptime if has_accelerator():
        print("test_sum_empty_axes_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[])
        var expected = Tensor[dtype].scalar(10.0)
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU empty axes sum passed")

fn test_mean_empty_axes_cpu() raises:
    """Test mean with empty axes (full reduction) on CPU."""
    print("test_mean_empty_axes_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.mean(axes=[])
    var expected = Tensor[dtype].scalar(2.5)
    assert_true(result.all_close(expected))
    print("✓ CPU empty axes mean passed")

fn test_mean_empty_axes_gpu() raises:
    """Test mean with empty axes on GPU."""
    comptime if has_accelerator():
        print("test_mean_empty_axes_gpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean(axes=[])
        var expected = Tensor[dtype].scalar(2.5)
        assert_true(result.to_cpu().all_close(expected))
        print("✓ GPU empty axes mean passed")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LARGE TENSOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_large_1d_cpu() raises:
    """Test sum of large 1D tensor on CPU."""
    print("test_sum_large_1d_cpu")
    comptime dtype = DType.float32
    var size = 10000
    var data = List[Scalar[dtype]]()
    for i in range(size):
        data.append(Scalar[dtype](i + 1))
    var a = Tensor[dtype].d1(data)
    var result = a.sum()
    var expected = (size * (size + 1)) // 2
    assert_true(result.item() == Float32(expected))
    print("✓ CPU large 1D sum passed")

fn test_sum_large_1d_gpu() raises:
    """Test sum of large 1D tensor on GPU."""
    comptime if has_accelerator():
        print("test_sum_large_1d_gpu")
        comptime dtype = DType.float32
        var size = 10000
        var data = List[Scalar[dtype]]()
        for i in range(size):
            data.append(Scalar[dtype](i + 1))
        var a = Tensor[dtype].d1(data)
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        var expected = (size * (size + 1)) // 2
        assert_true(result.to_cpu().item() == Float32(expected))
        print("✓ GPU large 1D sum passed")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CPU-GPU CONSISTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sum_cpu_gpu_consistency_2d() raises:
    """Test CPU and GPU results match for 2D sum."""
    comptime if has_accelerator():
        print("test_sum_cpu_gpu_consistency_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        var cpu_result = a.sum()
        var gpu_result = a.to_gpu().sum()

        assert_true(cpu_result.all_close(gpu_result.to_cpu()))
        print("✓ CPU-GPU consistency passed")

fn test_mean_cpu_gpu_consistency_3d() raises:
    """Test CPU and GPU results match for 3D mean."""
    comptime if has_accelerator():
        print("test_mean_cpu_gpu_consistency_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        var cpu_result = a.mean(axes=[1, 2])
        var gpu_result = a.to_gpu().mean(axes=[1, 2])

        assert_true(cpu_result.all_close(gpu_result.to_cpu()))
        print("✓ CPU-GPU consistency passed")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll sum/mean tests passed!")
