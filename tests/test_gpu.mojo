from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.intarray import IntArray
from tenmo.shapes import Shape
from std.testing import assert_true
from std.sys import has_accelerator
from tenmo.mnemonics import vm, mv
from tenmo.gradbox import Gradbox
from tenmo.matmul_kernel import MatmulNdGpu

comptime dtype = DType.float32


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor.sum — CPU
# ═══════════════════════════════════════════════════════════════════════════════


fn test_v2_tensor_sum_scalar_input() raises:
    print("test_v2_tensor_sum_scalar_input")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(7)
    var s = a.sum()
    assert_true(s == Tensor[dtype].scalar(7))
    print("test_v2_tensor_sum_scalar_input passed")


fn test_v2_tensor_sum_1d() raises:
    print("test_v2_tensor_sum_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1, 2, 3, 4, 5])
    var s = a.sum()
    assert_true(s == Tensor[dtype].scalar(15))
    print("test_v2_tensor_sum_1d passed")


fn test_v2_tensor_sum_1d_keepdims() raises:
    print("test_v2_tensor_sum_1d_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1, 2, 3, 4, 5])
    var s = a.sum(keepdims=True)
    assert_true(s == Tensor[dtype].d1([15]))
    print("test_v2_tensor_sum_1d_keepdims passed")


fn test_v2_tensor_sum_2d_axis0() raises:
    print("test_v2_tensor_sum_2d_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var s = a.sum(axes=[0])
    assert_true(s == Tensor[dtype].d1([5, 7, 9]))
    print("test_v2_tensor_sum_2d_axis0 passed")


fn test_v2_tensor_sum_2d_axis1() raises:
    print("test_v2_tensor_sum_2d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var s = a.sum(axes=[1])
    assert_true(s == Tensor[dtype].d1([6, 15]))
    print("test_v2_tensor_sum_2d_axis1 passed")


fn test_v2_tensor_sum_2d_axis0_keepdims() raises:
    print("test_v2_tensor_sum_2d_axis0_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var s = a.sum(axes=[0], keepdims=True)
    assert_true(s == Tensor[dtype].d2([[5, 7, 9]]))
    print("test_v2_tensor_sum_2d_axis0_keepdims passed")


fn test_v2_tensor_sum_2d_axis1_keepdims() raises:
    print("test_v2_tensor_sum_2d_axis1_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var s = a.sum(axes=[1], keepdims=True)
    assert_true(s == Tensor[dtype].d2([[6], [15]]))
    print("test_v2_tensor_sum_2d_axis1_keepdims passed")


fn test_v2_tensor_sum_2d_all_axes() raises:
    print("test_v2_tensor_sum_2d_all_axes")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var s = a.sum(axes=[0, 1])
    assert_true(s == Tensor[dtype].scalar(21))
    print("test_v2_tensor_sum_2d_all_axes passed")


fn test_v2_tensor_sum_3d_axis0() raises:
    print("test_v2_tensor_sum_3d_axis0")
    comptime dtype = DType.float32
    # arange(24).reshape(2,3,4)
    var a = Tensor[dtype].arange(24)
    a = a.reshape(2, 3, 4)
    var s = a.sum(axes=[0])
    # row i: (0..3 + 12..15), (4..7 + 16..19), (8..11 + 20..23)
    assert_true(
        s
        == Tensor[dtype].d2(
            [[12, 14, 16, 18], [20, 22, 24, 26], [28, 30, 32, 34]]
        )
    )
    print("test_v2_tensor_sum_3d_axis0 passed")


fn test_v2_tensor_sum_3d_axis1() raises:
    print("test_v2_tensor_sum_3d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(24)
    a = a.reshape(2, 3, 4)
    var s = a.sum(axes=[1])
    # batch 0: cols 0+4+8, 1+5+9, 2+6+10, 3+7+11 = 12,15,18,21
    # batch 1: cols 12+16+20, 13+17+21, 14+18+22, 15+19+23 = 48,51,54,57
    assert_true(s == Tensor[dtype].d2([[12, 15, 18, 21], [48, 51, 54, 57]]))
    print("test_v2_tensor_sum_3d_axis1 passed")


fn test_v2_tensor_sum_3d_axis2() raises:
    print("test_v2_tensor_sum_3d_axis2")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(24)
    a = a.reshape(2, 3, 4)
    var s = a.sum(axes=[2])
    # batch 0: 0+1+2+3, 4+5+6+7, 8+9+10+11 = 6, 22, 38
    # batch 1: 12+13+14+15, 16+17+18+19, 20+21+22+23 = 54, 70, 86
    assert_true(s == Tensor[dtype].d2([[6, 22, 38], [54, 70, 86]]))
    print("test_v2_tensor_sum_3d_axis2 passed")


fn test_v2_tensor_sum_3d_axes_0_2() raises:
    print("test_v2_tensor_sum_3d_axes_0_2")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(24)
    a = a.reshape(2, 3, 4)
    var s = a.sum(axes=[0, 2])
    # For each of 3 rows: sum all elements across batch and columns
    # row 0: (0+1+2+3) + (12+13+14+15) = 6 + 54 = 60
    # row 1: (4+5+6+7) + (16+17+18+19) = 22 + 70 = 92
    # row 2: (8+9+10+11) + (20+21+22+23) = 38 + 86 = 124
    assert_true(s == Tensor[dtype].d1([60, 92, 124]))
    print("test_v2_tensor_sum_3d_axes_0_2 passed")


fn test_v2_tensor_sum_3d_all_axes_keepdims() raises:
    print("test_v2_tensor_sum_3d_all_axes_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].ones(2, 3, 4)
    var s = a.sum(axes=[0, 1, 2], keepdims=True)
    # shape (1,1,1), value 24
    assert_true(s == Tensor[dtype].d3([[[24]]]))
    print("test_v2_tensor_sum_3d_all_axes_keepdims passed")


fn test_v2_tensor_sum_zeros() raises:
    print("test_v2_tensor_sum_zeros")
    comptime dtype = DType.float32
    var a = Tensor[dtype].zeros(3, 4)
    var s = a.sum()
    assert_true(s == Tensor[dtype].scalar(0))
    print("test_v2_tensor_sum_zeros passed")


fn test_v2_tensor_sum_negative_values() raises:
    print("test_v2_tensor_sum_negative_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1, -2], [3, 4]])
    var s = a.sum()
    assert_true(s == Tensor[dtype].scalar(4))
    print("test_v2_tensor_sum_negative_values passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor.mean — CPU
# ═══════════════════════════════════════════════════════════════════════════════


fn test_v2_tensor_mean_1d() raises:
    print("test_v2_tensor_mean_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1, 2, 3, 4, 5])
    var m = a.mean()
    assert_true(m.all_close(Tensor[dtype].scalar(3)))
    print("test_v2_tensor_mean_1d passed")


fn test_v2_tensor_mean_2d_axis0() raises:
    print("test_v2_tensor_mean_2d_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
    var m = a.mean(axes=[0])
    assert_true(m.all_close(Tensor[dtype].d1([2, 3, 4])))
    print("test_v2_tensor_mean_2d_axis0 passed")


fn test_v2_tensor_mean_2d_axis1() raises:
    print("test_v2_tensor_mean_2d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var m = a.mean(axes=[1])
    assert_true(m.all_close(Tensor[dtype].d1([2, 5])))
    print("test_v2_tensor_mean_2d_axis1 passed")


fn test_v2_tensor_mean_2d_axis0_keepdims() raises:
    print("test_v2_tensor_mean_2d_axis0_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2], [3, 4]])
    var m = a.mean(axes=[0], keepdims=True)
    assert_true(m.all_close(Tensor[dtype].d2([[2, 3]])))
    print("test_v2_tensor_mean_2d_axis0_keepdims passed")


fn test_v2_tensor_mean_2d_axis1_keepdims() raises:
    print("test_v2_tensor_mean_2d_axis1_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1, 2], [3, 4]])
    var m = a.mean(axes=[1], keepdims=True)
    assert_true(m.all_close(Tensor[dtype].d2([[1.5], [3.5]])))
    print("test_v2_tensor_mean_2d_axis1_keepdims passed")


fn test_v2_tensor_mean_3d_axis1() raises:
    print("test_v2_tensor_mean_3d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(24)
    a = a.reshape(2, 3, 4)
    var m = a.mean(axes=[1])
    # batch 0: mean over 3 rows -> [4,5,6,7]
    # batch 1: mean over 3 rows -> [16,17,18,19]
    assert_true(m.all_close(Tensor[dtype].d2([[4, 5, 6, 7], [16, 17, 18, 19]])))
    print("test_v2_tensor_mean_3d_axis1 passed")


fn test_v2_tensor_mean_3d_all_axes() raises:
    print("test_v2_tensor_mean_3d_all_axes")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(24)
    a = a.reshape(2, 3, 4)
    var m = a.mean()
    # mean of 0..23 = 11.5
    assert_true(m.all_close(Tensor[dtype].scalar(11.5)))
    print("test_v2_tensor_mean_3d_all_axes passed")


fn test_v2_tensor_mean_uniform() raises:
    print("test_v2_tensor_mean_uniform")
    comptime dtype = DType.float32
    var a = Tensor[dtype].ones(4, 5) * 3
    var m = a.mean()
    assert_true(m.all_close(Tensor[dtype].scalar(3)))
    print("test_v2_tensor_mean_uniform passed")


# ═══════════════════════════════════════════════════════════════════════════════
# NDBuffer.reduce (CPU) — sum and mean via reduce[mean=False/True]
# ═══════════════════════════════════════════════════════════════════════════════


fn test_v2_ndbuffer_sum_1d() raises:
    print("test_v2_ndbuffer_sum_1d")
    var ndb = NDBuffer[DType.float32](Shape(5))
    for i in range(5):
        ndb[IntArray(i)] = Float32(i + 1)  # 1,2,3,4,5
    var result = ndb.reduce(normalized_axes=IntArray(0))
    assert_true(result[IntArray()] == 15)
    print("test_v2_ndbuffer_sum_1d passed")


fn test_v2_ndbuffer_sum_2d_axis0() raises:
    print("test_v2_ndbuffer_sum_2d_axis0")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = Float32(val)
            val += 1
    # [[1,2,3],[4,5,6]] -> axis0 -> [5,7,9]
    var result = ndb.reduce(normalized_axes=IntArray(0))
    assert_true(result[IntArray(0)] == 5)
    assert_true(result[IntArray(1)] == 7)
    assert_true(result[IntArray(2)] == 9)
    print("test_v2_ndbuffer_sum_2d_axis0 passed")


fn test_v2_ndbuffer_sum_2d_axis1() raises:
    print("test_v2_ndbuffer_sum_2d_axis1")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = Float32(val)
            val += 1
    # [[1,2,3],[4,5,6]] -> axis1 -> [6,15]
    var result = ndb.reduce(normalized_axes=IntArray(1))
    assert_true(result[IntArray(0)] == 6)
    assert_true(result[IntArray(1)] == 15)
    print("test_v2_ndbuffer_sum_2d_axis1 passed")


fn test_v2_ndbuffer_mean_2d_axis0() raises:
    print("test_v2_ndbuffer_mean_2d_axis0")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = Float32(val)
            val += 1
    # [[1,2,3],[4,5,6]] -> mean axis0 -> [2.5, 3.5, 4.5]
    var result = ndb.reduce[mean=True](normalized_axes=IntArray(0))
    assert_true(result[IntArray(0)] == 2.5)
    assert_true(result[IntArray(1)] == 3.5)
    assert_true(result[IntArray(2)] == 4.5)
    print("test_v2_ndbuffer_mean_2d_axis0 passed")


fn test_v2_ndbuffer_mean_2d_axis1() raises:
    print("test_v2_ndbuffer_mean_2d_axis1")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = Float32(val)
            val += 1
    # [[1,2,3],[4,5,6]] -> mean axis1 -> [2.0, 5.0]
    var result = ndb.reduce[mean=True](normalized_axes=IntArray(1))
    assert_true(result[IntArray(0)] == 2.0)
    assert_true(result[IntArray(1)] == 5.0)
    print("test_v2_ndbuffer_mean_2d_axis1 passed")


fn test_v2_ndbuffer_sum_keepdims() raises:
    print("test_v2_ndbuffer_sum_keepdims")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = Float32(val)
            val += 1
    # keepdims -> shape (1, 3)
    var result = ndb.reduce(normalized_axes=IntArray(0), keepdims=True)
    assert_true(result[IntArray(0, 0)] == 5)
    assert_true(result[IntArray(0, 1)] == 7)
    assert_true(result[IntArray(0, 2)] == 9)
    print("test_v2_ndbuffer_sum_keepdims passed")


fn test_v2_ndbuffer_mean_keepdims() raises:
    print("test_v2_ndbuffer_mean_keepdims")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = Float32(val)
            val += 1
    # keepdims -> shape (2, 1)
    var result = ndb.reduce[mean=True](
        normalized_axes=IntArray(1), keepdims=True
    )
    assert_true(result[IntArray(0, 0)] == 2.0)
    assert_true(result[IntArray(1, 0)] == 5.0)
    print("test_v2_ndbuffer_mean_keepdims passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU tests — Tensor.sum and Tensor.mean
# ═══════════════════════════════════════════════════════════════════════════════


fn test_v2_gpu_tensor_sum_1d() raises:
    print("test_v2_gpu_tensor_sum_1d")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1, 2, 3, 4, 5])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum()
        var gpu_result = a_gpu.sum()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_1d passed")


fn test_v2_gpu_tensor_sum_2d_axis0() raises:
    print("test_v2_gpu_tensor_sum_2d_axis0")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[0])
        var gpu_result = a_gpu.sum(axes=[0])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_2d_axis0 passed")


fn test_v2_gpu_tensor_sum_2d_axis1() raises:
    print("test_v2_gpu_tensor_sum_2d_axis1")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1])
        var gpu_result = a_gpu.sum(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_2d_axis1 passed")


fn test_v2_gpu_tensor_sum_3d_axis1() raises:
    print("test_v2_gpu_tensor_sum_3d_axis1")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1])
        var gpu_result = a_gpu.sum(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_3d_axis1 passed")


fn test_v2_gpu_tensor_sum_3d_axes_0_2() raises:
    print("test_v2_gpu_tensor_sum_3d_axes_0_2")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[0, 2])
        var gpu_result = a_gpu.sum(axes=[0, 2])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_3d_axes_0_2 passed")


fn test_v2_gpu_tensor_sum_keepdims() raises:
    print("test_v2_gpu_tensor_sum_keepdims")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1], keepdims=True)
        var gpu_result = a_gpu.sum(axes=[1], keepdims=True)
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_keepdims passed")


fn test_v2_gpu_tensor_sum_all_axes() raises:
    print("test_v2_gpu_tensor_sum_all_axes")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum()
        var gpu_result = a_gpu.sum()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_sum_all_axes passed")


fn test_v2_gpu_tensor_mean_2d_axis0() raises:
    print("test_v2_gpu_tensor_mean_2d_axis0")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[0])
        var gpu_result = a_gpu.mean(axes=[0])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_mean_2d_axis0 passed")


fn test_v2_gpu_tensor_mean_2d_axis1() raises:
    print("test_v2_gpu_tensor_mean_2d_axis1")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1])
        var gpu_result = a_gpu.mean(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_mean_2d_axis1 passed")


fn test_v2_gpu_tensor_mean_3d_axis1() raises:
    print("test_v2_gpu_tensor_mean_3d_axis1")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1])
        var gpu_result = a_gpu.mean(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_mean_3d_axis1 passed")


fn test_v2_gpu_tensor_mean_keepdims() raises:
    print("test_v2_gpu_tensor_mean_keepdims")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1], keepdims=True)
        var gpu_result = a_gpu.mean(axes=[1], keepdims=True)
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_mean_keepdims passed")


fn test_v2_gpu_tensor_mean_all_axes() raises:
    print("test_v2_gpu_tensor_mean_all_axes")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean()
        var gpu_result = a_gpu.mean()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
        print("test_v2_gpu_tensor_mean_all_axes passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU tests — NDBuffer.reduce
# ═══════════════════════════════════════════════════════════════════════════════


fn test_v2_gpu_ndbuffer_sum_2d_axis1() raises:
    print("test_v2_gpu_ndbuffer_sum_2d_axis1")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var gpu_ndb = a_gpu.buffer
        var gpu_result = gpu_ndb.reduce[mean=False](IntArray(1), keepdims=False)
        var tensor = Tensor[dtype].d1([6, 15])
        var g_tensor = tensor.to_gpu()
        assert_true(g_tensor.buffer.all_close(gpu_result))
        print("test_v2_gpu_ndbuffer_sum_2d_axis1 passed")


fn test_v2_gpu_ndbuffer_mean_2d_axis0() raises:
    print("test_v2_gpu_ndbuffer_mean_2d_axis0")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
        var a_gpu = a.to_gpu()
        var gpu_ndb = a_gpu.buffer
        var gpu_result = gpu_ndb.reduce[mean=True](IntArray(0), keepdims=False)
        var expected = Tensor[dtype].d1([2, 3, 4])
        expected = expected.to_gpu()
        assert_true(expected.buffer.all_close(gpu_result))
        print("test_v2_gpu_ndbuffer_mean_2d_axis0 passed")


fn test_cpu_grad_flow() raises:
    print("=== Test : Backward grad flow CPU ===")
    var A = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var B = Tensor[dtype].arange(30 * 5)

    var A_reshaped = A.reshape(Shape(1, 9, 30))
    var B_reshaped = B.reshape(Shape(30, 5))

    var C = A_reshaped.matmul(B_reshaped)

    C.backward()
    var A_grad = A.grad().copy()

    var grad_out = Gradbox[dtype].full(C.shape(), 1)
    # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
    var A_grad_expected = grad_out.matmul(B_reshaped.transpose(-1, -2))
    A_grad_expected = A_grad_expected.reshape(Shape(1 * 9 * 30))

    assert_true(A_grad.all_close(A_grad_expected))

    print("PASSED: CPU grad flow")


fn test_gpu_grad_flow() raises:
    print("=== Test : Backward grad flow GPU ===")
    var A = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var B = Tensor[dtype].arange(30 * 5)

    var A_reshaped = A.reshape(Shape(1, 9, 30))
    var B_reshaped = B.reshape(Shape(30, 5))

    var A_gpu = A_reshaped.to_gpu()
    var A_r = A_gpu.reshape(3, 3, 1, 30)
    var A_rr = A_r.reshape(3, 2, 3, 1, 15)
    var B_gpu = B_reshaped.to_gpu()
    var A_gpu_reshaped = A_rr.reshape(Shape(9, 1, 30))
    var C_gpu = A_gpu_reshaped.matmul(B_gpu)

    C_gpu.backward()
    var A_grad = A.grad().copy()

    var grad_out = Gradbox[dtype].full(C_gpu.shape(), 1)
    # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
    var A_grad_expected = grad_out.matmul(B_reshaped.transpose(-1, -2))
    A_grad_expected = A_grad_expected.reshape(Shape(9 * 30))

    assert_true(A_grad.all_close(A_grad_expected))

    print("PASSED: GPU backward flow")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


fn main() raises:
    # CPU — sum
    test_v2_tensor_sum_scalar_input()
    test_v2_tensor_sum_1d()
    test_v2_tensor_sum_1d_keepdims()
    test_v2_tensor_sum_2d_axis0()
    test_v2_tensor_sum_2d_axis1()
    test_v2_tensor_sum_2d_axis0_keepdims()
    test_v2_tensor_sum_2d_axis1_keepdims()
    test_v2_tensor_sum_2d_all_axes()
    test_v2_tensor_sum_3d_axis0()
    test_v2_tensor_sum_3d_axis1()
    test_v2_tensor_sum_3d_axis2()
    test_v2_tensor_sum_3d_axes_0_2()
    test_v2_tensor_sum_3d_all_axes_keepdims()
    test_v2_tensor_sum_zeros()
    test_v2_tensor_sum_negative_values()

    # CPU — mean
    test_v2_tensor_mean_1d()
    test_v2_tensor_mean_2d_axis0()
    test_v2_tensor_mean_2d_axis1()
    test_v2_tensor_mean_2d_axis0_keepdims()
    test_v2_tensor_mean_2d_axis1_keepdims()
    test_v2_tensor_mean_3d_axis1()
    test_v2_tensor_mean_3d_all_axes()
    test_v2_tensor_mean_uniform()

    # CPU — NDBuffer
    test_v2_ndbuffer_sum_1d()
    test_v2_ndbuffer_sum_2d_axis0()
    test_v2_ndbuffer_sum_2d_axis1()
    test_v2_ndbuffer_mean_2d_axis0()
    test_v2_ndbuffer_mean_2d_axis1()
    test_v2_ndbuffer_sum_keepdims()
    test_v2_ndbuffer_mean_keepdims()

    test_cpu_grad_flow()

    # GPU — Tensor sum
    test_v2_gpu_tensor_sum_1d()
    test_v2_gpu_tensor_sum_2d_axis0()
    test_v2_gpu_tensor_sum_2d_axis1()
    test_v2_gpu_tensor_sum_3d_axis1()
    test_v2_gpu_tensor_sum_3d_axes_0_2()
    test_v2_gpu_tensor_sum_keepdims()
    test_v2_gpu_tensor_sum_all_axes()

    # GPU — Tensor mean
    test_v2_gpu_tensor_mean_2d_axis0()
    test_v2_gpu_tensor_mean_2d_axis1()
    test_v2_gpu_tensor_mean_3d_axis1()
    test_v2_gpu_tensor_mean_keepdims()
    test_v2_gpu_tensor_mean_all_axes()

    # GPU — NDBuffer
    test_v2_gpu_ndbuffer_sum_2d_axis1()
    test_v2_gpu_ndbuffer_mean_2d_axis0()

    # vector matrix multiplication
    test_vector_matrix_mul_tests()

    # matrix vector multiplication
    test_matrix_vector_multiplications()

    # Tensor tensor multiplication
    test_tensor_tensor_multiplications()

    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return
    else:
        test_gpu_transfer_fidelity()
        test_ancestry_storage_fidelity()
        test_forward_matmul_fidelity()
        test_ancestry_transposed_matmul_fidelity()
        test_transposed_matmul_fidelity()
        test_backward_grad_A_fidelity()

        test_gpu_grad_flow()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


fn close_enough[
    dtype: DType
](mut a: Tensor[dtype], b: Tensor[dtype]) raises -> Bool:
    var a_gpu = a.to_gpu()
    # return a_gpu.all_close(b.to_gpu())
    return a_gpu.all_close(b)


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════


fn test_vmnd_1d_v_2d_M() raises:
    """V[k] @ M[k, n] → out[n]. Simplest case, no batch dims."""
    print("test_vmnd_1d_v_2d_M")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 2, 3],  M = [[1,0],[0,1],[1,1]]
        # out = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        var v = Tensor[dtype].d1([1, 2, 3])
        var M = Tensor[dtype].d2([[1, 0], [0, 1], [1, 1]])
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_1d_v_2d_M passed")


fn test_vmnd_identity_matrix() raises:
    """V @ I = v."""
    print("test_vmnd_identity_matrix")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([3, 1, 4, 1, 5])
        var I = Tensor[dtype].eye(5)
        var cpu_result = v.matmul[mode=vm](I)
        var v_gpu = v.to_gpu()
        var I_gpu = I.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](I_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_identity_matrix passed")


fn test_vmnd_zero_vector() raises:
    """Zero vector gives zero output."""
    print("test_vmnd_zero_vector")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].zeros(4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_zero_vector passed")


fn test_vmnd_ones_vector() raises:
    """Ones vector sums columns of M."""
    print("test_vmnd_ones_vector")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(3)
        var M = Tensor[dtype].d2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        # out = [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_ones_vector passed")


fn test_vmnd_single_output_element() raises:
    """N=1: output is a scalar-like vector."""
    print("test_vmnd_single_output_element")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([2, 3, 4])
        var M = Tensor[dtype].d2([[1], [2], [3]])
        # out = [2*1 + 3*2 + 4*3] = [20]
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_single_output_element passed")


fn test_vmnd_large_k() raises:
    """Large k to stress the dot product loop."""
    print("test_vmnd_large_k")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 512
        var n = 64
        var v = Tensor[dtype].ones(k)
        var M = Tensor[dtype].ones(k, n)
        # out[j] = k for all j
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_large_k passed")


fn test_vmnd_large_n() raises:
    """N > block_size to exercise multi-block coverage."""
    print("test_vmnd_large_n")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 8
        var n = 1024  # larger than default block_size=256
        var v = Tensor[dtype].ones(k)
        var M = Tensor[dtype].ones(k, n)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_large_n passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — v and M same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════


fn test_vmnd_batched_2d_v_3d_M() raises:
    """V[b, k] @ M[b, k, n] → out[b, n]."""
    print("test_vmnd_batched_2d_v_3d_M")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)  # (2, 3)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        # result_cpu = gpu_result.to_cpu()
        cpu_result.print()
        gpu_result.print()
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_2d_v_3d_M passed")


fn test_vmnd_batched_3d_v_4d_M() raises:
    """V[a, b, k] @ M[a, b, k, n] → out[a, b, n]."""
    print("test_vmnd_batched_3d_v_4d_M")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(2, 3, 4)  # (2, 3, 4)
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        # out[a,b,j] = 4.0 for all (a,b,j)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_3d_v_4d_M passed")


fn test_vmnd_batched_arange_values() raises:
    """Batched with non-trivial values to catch index mapping errors."""
    print("test_vmnd_batched_arange_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v: (3, 4),  M: (3, 4, 2)
        var v = Tensor[dtype].arange(12)
        v = v.reshape(3, 4)
        var M = Tensor[dtype].arange(24)
        M = M.reshape(3, 4, 2)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_arange_values passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — v and M have different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════


fn test_vmnd_broadcast_v1d_M3d() raises:
    """V[k] broadcast against M[b, k, n] → out[b, n]."""
    print("test_vmnd_broadcast_v1d_M3d")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([1, 0, 1])  # (3,)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v1d_M3d passed")


fn test_vmnd_broadcast_v2d_M3d() raises:
    """V[1, k] broadcast against M[b, k, n] → out[b, n]."""
    print("test_vmnd_broadcast_v2d_M3d")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)  # (1, 4)
        var M = Tensor[dtype].arange(48)
        M = M.reshape(3, 4, 4)  # (3, 4, 4)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v2d_M3d passed")


fn test_vmnd_broadcast_v3d_M2d() raises:
    """V[a, b, k] broadcast against M[k, n] → out[a, b, n]."""
    print("test_vmnd_broadcast_v3d_M2d")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(24)
        v = v.reshape(2, 3, 4)  # (2, 3, 4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)  # (4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v3d_M2d passed")


fn test_vmnd_broadcast_both_size1() raises:
    """Both v and M have a size-1 batch dim that broadcasts."""
    print("test_vmnd_broadcast_both_size1")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)  # (1, 4) → broadcasts to (3, 4)
        var M = Tensor[dtype].ones(3, 4, 5)  # (3, 4, 5)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_both_size1 passed")


fn test_vmnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""
    print("test_vmnd_broadcast_large_batch")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v: (32, k), M: (k, n) — M broadcast across 32 batch elements
        var k = 64
        var n = 128
        var v = Tensor[dtype].ones(32, k)
        var M = Tensor[dtype].ones(k, n)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_large_batch passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical correctness — spot-check known values
# ═══════════════════════════════════════════════════════════════════════════════


fn test_vmnd_known_values_no_batch() raises:
    """Hand-computed result verified against GPU."""
    print("test_vmnd_known_values_no_batch")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 2],  M = [[3, 4, 5], [6, 7, 8]]
        # out = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        var v = Tensor[dtype].d1([1, 2])
        var M = Tensor[dtype].d2([[3, 4, 5], [6, 7, 8]])
        var expected = Tensor[dtype].d1([15, 18, 21])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))
    print("test_vmnd_known_values_no_batch passed")


fn test_vmnd_known_values_batched() raises:
    """Hand-computed batched result verified against GPU."""
    print("test_vmnd_known_values_batched")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: v=[1,0] @ M=[[1,2],[3,4]] = [1, 2]
        # batch 1: v=[0,1] @ M=[[5,6],[7,8]] = [7, 8]
        var v = Tensor[dtype].d2([[1, 0], [0, 1]])
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var expected = Tensor[dtype].d2([[1, 2], [7, 8]])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))
    print("test_vmnd_known_values_batched passed")


fn test_vmnd_known_values_broadcast() raises:
    """Hand-computed broadcast result verified against GPU."""
    print("test_vmnd_known_values_broadcast")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 1]  (no batch)
        # M[0] = [[1,2],[3,4]] → out[0] = [4, 6]
        # M[1] = [[5,6],[7,8]] → out[1] = [12, 14]
        var v = Tensor[dtype].d1([1, 1])
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var expected = Tensor[dtype].d2([[4, 6], [12, 14]])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))
    print("test_vmnd_known_values_broadcast passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point for vector matrix multiplication
# ═══════════════════════════════════════════════════════════════════════════════


fn test_vector_matrix_mul_tests() raises:
    # Basic correctness
    test_vmnd_1d_v_2d_M()
    test_vmnd_identity_matrix()
    test_vmnd_zero_vector()
    test_vmnd_ones_vector()
    test_vmnd_single_output_element()
    test_vmnd_large_k()
    test_vmnd_large_n()

    # Batched — same batch shape
    test_vmnd_batched_2d_v_3d_M()
    test_vmnd_batched_3d_v_4d_M()
    test_vmnd_batched_arange_values()

    # Broadcast — different batch ranks
    test_vmnd_broadcast_v1d_M3d()
    test_vmnd_broadcast_v2d_M3d()
    test_vmnd_broadcast_v3d_M2d()
    test_vmnd_broadcast_both_size1()
    test_vmnd_broadcast_large_batch()

    # Known values — spot checks
    test_vmnd_known_values_no_batch()
    test_vmnd_known_values_batched()
    test_vmnd_known_values_broadcast()


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mvnd_2d_M_1d_v() raises:
    """M[m, k] @ v[k] → out[m]. Simplest case, no batch dims."""
    print("test_mvnd_2d_M_1d_v")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M = [[1,2,3],[4,5,6]], v = [1,1,1]
        # out = [1+2+3, 4+5+6] = [6, 15]
        var M = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var v = Tensor[dtype].d1([1, 1, 1])
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_2d_M_1d_v passed")


fn test_mvnd_known_values() raises:
    """Hand-computed result verified directly against GPU output."""
    print("test_mvnd_known_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M = [[1,0],[0,1],[1,1]], v = [3, 4]
        # out = [1*3+0*4, 0*3+1*4, 1*3+1*4] = [3, 4, 7]
        var M = Tensor[dtype].d2([[1, 0], [0, 1], [1, 1]])
        var v = Tensor[dtype].d1([3, 4])
        var expected = Tensor[dtype].d1([3, 4, 7])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_known_values passed")


fn test_mvnd_identity_matrix() raises:
    """I @ v = v."""
    print("test_mvnd_identity_matrix")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].eye(4)
        var v = Tensor[dtype].d1([2, 5, 1, 8])
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_identity_matrix passed")


fn test_mvnd_zero_vector() raises:
    """M @ zero_vector = zero output."""
    print("test_mvnd_zero_vector")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(3, 4)
        var v = Tensor[dtype].zeros(4)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_zero_vector passed")


fn test_mvnd_ones_vector() raises:
    """M @ ones = row sums of M."""
    print("test_mvnd_ones_vector")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
        var M = Tensor[dtype].d2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        var v = Tensor[dtype].ones(4)
        var expected = Tensor[dtype].d1([10, 26, 42])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_ones_vector passed")


fn test_mvnd_single_row_matrix() raises:
    """M[1, k] @ v[k] → out[1]. Single row edge case."""
    print("test_mvnd_single_row_matrix")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2, 3, 4]])  # (1, 3)
        var v = Tensor[dtype].d1([1, 2, 3])
        # out = [2*1 + 3*2 + 4*3] = [20]
        var expected = Tensor[dtype].d1([20])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_single_row_matrix passed")


fn test_mvnd_single_col_matrix() raises:
    """M[m, 1] @ v[1] → out[m]. k=1 edge case."""
    print("test_mvnd_single_col_matrix")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2], [3], [5]])  # (3, 1)
        var v = Tensor[dtype].d1([4])
        # out = [8, 12, 20]
        var expected = Tensor[dtype].d1([8, 12, 20])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_single_col_matrix passed")


fn test_mvnd_large_k() raises:
    """Large k to stress the dot product loop."""
    print("test_mvnd_large_k")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 512
        var m = 32
        var M = Tensor[dtype].ones(m, k)
        var v = Tensor[dtype].ones(k)
        # each output element = k = 512
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_large_k passed")


fn test_mvnd_large_m() raises:
    """M > block_size to exercise multi-block coverage."""
    print("test_mvnd_large_m")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var m = 1024  # larger than default block_size=256
        var k = 8
        var M = Tensor[dtype].ones(m, k)
        var v = Tensor[dtype].ones(k)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_large_m passed")


fn test_mvnd_negative_values() raises:
    """Negative values in both M and v."""
    print("test_mvnd_negative_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[-1, 2], [3, -4]])
        var v = Tensor[dtype].d1([-1, 2])
        # out = [(-1*-1 + 2*2), (3*-1 + -4*2)] = [5, -11]
        var expected = Tensor[dtype].d1([5, -11])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_negative_values passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — M and v same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mvnd_batched_3d_M_2d_v() raises:
    """M[b, m, k] @ v[b, k] → out[b, m]."""
    print("test_mvnd_batched_3d_M_2d_v")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)  # (2, 3)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_3d_M_2d_v passed")


fn test_mvnd_batched_4d_M_3d_v() raises:
    """M[a, b, m, k] @ v[a, b, k] → out[a, b, m]."""
    print("test_mvnd_batched_4d_M_3d_v")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(2, 3, 5)  # (2, 3, 5)
        # each output element = k = 5
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_4d_M_3d_v passed")


fn test_mvnd_batched_arange_values() raises:
    """Batched with non-trivial arange values to catch index mapping errors."""
    print("test_mvnd_batched_arange_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M: (3, 4, 2),  v: (3, 2)
        var M = Tensor[dtype].arange(24)
        M = M.reshape(3, 4, 2)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(3, 2)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_arange_values passed")


fn test_mvnd_known_values_batched() raises:
    """Hand-computed batched result verified directly against GPU."""
    print("test_mvnd_known_values_batched")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: [[1,0],[0,1]] @ [1,2] = [1, 2]
        # batch 1: [[2,0],[0,2]] @ [3,4] = [6, 8]
        var M = Tensor[dtype].d3([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
        var v = Tensor[dtype].d2([[1, 2], [3, 4]])
        var expected = Tensor[dtype].d2([[1, 2], [6, 8]])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_known_values_batched passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — M and v have different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mvnd_broadcast_3d_M_1d_v() raises:
    """M[b, m, k] broadcast against v[k] → out[b, m]."""
    print("test_mvnd_broadcast_3d_M_1d_v")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var v = Tensor[dtype].d1([1, 0, 1])  # (3,) broadcasts over batch
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_3d_M_1d_v passed")


fn test_mvnd_broadcast_2d_M_3d_v() raises:
    """M[m, k] broadcast against v[b, k] → out[b, m]."""
    print("test_mvnd_broadcast_2d_M_3d_v")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)  # (4, 3) — no batch
        var v = Tensor[dtype].arange(9)
        v = v.reshape(3, 3)  # (3, 3) — batch of 3
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_2d_M_3d_v passed")


fn test_mvnd_broadcast_4d_M_2d_v() raises:
    """M[a, b, m, k] broadcast against v[b, k] → out[a, b, m]."""
    print("test_mvnd_broadcast_4d_M_2d_v")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(3, 5)  # (3, 5) — broadcasts over dim 0
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_4d_M_2d_v passed")


fn test_mvnd_broadcast_size1_batch() raises:
    """V with size-1 batch dim that broadcasts across M's batch."""
    print("test_mvnd_broadcast_size1_batch")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(4, 3, 5)  # (4, 3, 5)
        var v = Tensor[dtype].ones(1, 5)  # (1, 5) → broadcasts to (4, 5)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_size1_batch passed")


fn test_mvnd_broadcast_known_values() raises:
    """Hand-computed broadcast result verified directly against GPU."""
    print("test_mvnd_broadcast_known_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M[0] = [[1,2],[3,4]], M[1] = [[5,6],[7,8]]
        # v = [1, 1]  (no batch — broadcasts across both)
        # out[0] = [3, 7],  out[1] = [11, 15]
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var v = Tensor[dtype].d1([1, 1])
        var expected = Tensor[dtype].d2([[3, 7], [11, 15]])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_broadcast_known_values passed")


fn test_mvnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""
    print("test_mvnd_broadcast_large_batch")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var m = 64
        var k = 32
        var M = Tensor[dtype].ones(128, m, k)  # large batch on M side
        var v = Tensor[dtype].ones(k)  # no batch — broadcasts
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_large_batch passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point for matrix vector multiplication
# ═══════════════════════════════════════════════════════════════════════════════


fn test_matrix_vector_multiplications() raises:
    # Basic correctness
    test_mvnd_2d_M_1d_v()
    test_mvnd_known_values()
    test_mvnd_identity_matrix()
    test_mvnd_zero_vector()
    test_mvnd_ones_vector()
    test_mvnd_single_row_matrix()
    test_mvnd_single_col_matrix()
    test_mvnd_large_k()
    test_mvnd_large_m()
    test_mvnd_negative_values()

    # Batched — same batch shape
    test_mvnd_batched_3d_M_2d_v()
    test_mvnd_batched_4d_M_3d_v()
    test_mvnd_batched_arange_values()
    test_mvnd_known_values_batched()

    # Broadcast — different batch ranks
    test_mvnd_broadcast_3d_M_1d_v()
    test_mvnd_broadcast_2d_M_3d_v()
    test_mvnd_broadcast_4d_M_2d_v()
    test_mvnd_broadcast_size1_batch()
    test_mvnd_broadcast_known_values()
    test_mvnd_broadcast_large_batch()


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mmnd_2d_known_values() raises:
    """Hand-computed 2D matmul verified directly against GPU."""
    print("test_mmnd_2d_known_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # C[0,0]=1*5+2*7=19, C[0,1]=1*6+2*8=22
        # C[1,0]=3*5+4*7=43, C[1,1]=3*6+4*8=50
        var A = Tensor[dtype].d2([[1, 2], [3, 4]])
        var B = Tensor[dtype].d2([[5, 6], [7, 8]])
        var expected = Tensor[dtype].d2([[19, 22], [43, 50]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mmnd_2d_known_values passed")


fn test_mmnd_2d_identity() raises:
    """A @ I = A."""
    print("test_mmnd_2d_identity")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(9)
        A = A.reshape(3, 3)
        var I = Tensor[dtype].eye(3)
        var cpu_result = A.matmul(I)
        var a_gpu = A.to_gpu()
        var i_gpu = I.to_gpu()
        var gpu_result = a_gpu.matmul(i_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_identity passed")


fn test_mmnd_2d_zero_matrix() raises:
    """A @ zeros = zeros."""
    print("test_mmnd_2d_zero_matrix")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(12)
        A = A.reshape(3, 4)
        var B = Tensor[dtype].zeros(4, 5)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_zero_matrix passed")


fn test_mmnd_2d_ones() raises:
    """Ones @ ones = matrix of k."""
    print("test_mmnd_2d_ones")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 8
        var A = Tensor[dtype].ones(4, k)
        var B = Tensor[dtype].ones(k, 5)
        # every output element = k
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_ones passed")


fn test_mmnd_2d_rectangular() raises:
    """Non-square matrices: (m, k) @ (k, n) where m != k != n."""
    print("test_mmnd_2d_rectangular")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(15)
        A = A.reshape(3, 5)  # (3, 5)
        var B = Tensor[dtype].arange(20)
        B = B.reshape(5, 4)  # (5, 4)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_rectangular passed")


fn test_mmnd_2d_negative_values() raises:
    """Negative values in both A and B."""
    print("test_mmnd_2d_negative_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[-1, 2], [3, -4]])
        var B = Tensor[dtype].d2([[1, -2], [-3, 4]])
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_negative_values passed")


fn test_mmnd_2d_single_element() raises:
    """(1, k) @ (k, 1) → (1, 1): inner product as matmul."""
    print("test_mmnd_2d_single_element")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1, 2, 3, 4]])  # (1, 4)
        var B = Tensor[dtype].d2([[1], [2], [3], [4]])  # (4, 1)
        # result = [[1+4+9+16]] = [[30]]
        var expected = Tensor[dtype].d2([[30]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mmnd_2d_single_element passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Tile boundary stress — sizes not multiples of TILE_SIZE
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mmnd_2d_non_tile_multiple_m() raises:
    """M(m) not a multiple of TILE_SIZE."""
    print("test_mmnd_2d_non_tile_multiple_m")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(17, 16)  # m=17, not multiple of 16
        var B = Tensor[dtype].ones(16, 16)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_non_tile_multiple_m passed")


fn test_mmnd_2d_non_tile_multiple_n() raises:
    """N(n) not a multiple of TILE_SIZE."""
    print("test_mmnd_2d_non_tile_multiple_n")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(16, 16)
        var B = Tensor[dtype].ones(16, 19)  # n=19, not multiple of 16
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_non_tile_multiple_n passed")


fn test_mmnd_2d_non_tile_multiple_k() raises:
    """K(k) not a multiple of TILE_SIZE."""
    print("test_mmnd_2d_non_tile_multiple_k")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(16, 13)  # k=13, not multiple of 16
        var B = Tensor[dtype].ones(13, 16)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_non_tile_multiple_k passed")


fn test_mmnd_2d_all_non_tile_multiples() raises:
    """M(m), k, n all non-multiples of TILE_SIZE simultaneously."""
    print("test_mmnd_2d_all_non_tile_multiples")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(17 * 13)
        A = A.reshape(17, 13)
        var B = Tensor[dtype].arange(13 * 19)
        B = B.reshape(13, 19)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_all_non_tile_multiples passed")


fn test_mmnd_2d_smaller_than_tile() raises:
    """M(m), k, n all smaller than TILE_SIZE."""
    print("test_mmnd_2d_smaller_than_tile")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(12)
        A = A.reshape(3, 4)  # both < 16
        var B = Tensor[dtype].arange(8)
        B = B.reshape(4, 2)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_smaller_than_tile passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Large matrices — stress multi-block coverage
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mmnd_2d_large_square() raises:
    """Large square matrices well beyond tile size."""
    print("test_mmnd_2d_large_square")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(128, 128)
        var B = Tensor[dtype].ones(128, 128)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_large_square passed")


fn test_mmnd_2d_large_rectangular() raises:
    """Large rectangular matrices with non-tile-multiple dimensions."""
    print("test_mmnd_2d_large_rectangular")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(65, 100)
        var B = Tensor[dtype].ones(100, 70)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_2d_large_rectangular passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mmnd_batched_3d_known_values() raises:
    """Hand-computed batched matmul verified directly against GPU."""
    print("test_mmnd_batched_3d_known_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: [[1,0],[0,1]] @ [[2,3],[4,5]] = [[2,3],[4,5]]
        # batch 1: [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
        var A = Tensor[dtype].d3([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
        var B = Tensor[dtype].d3([[[2, 3], [4, 5]], [[1, 0], [0, 1]]])
        var expected = Tensor[dtype].d3([[[2, 3], [4, 5]], [[1, 1], [1, 1]]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mmnd_batched_3d_known_values passed")


fn test_mmnd_batched_3d_arange() raises:
    """A[b, m, k] @ B[b, k, n] with arange values."""
    print("test_mmnd_batched_3d_arange")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(24)
        A = A.reshape(2, 3, 4)  # (2, 3, 4)
        var B = Tensor[dtype].arange(24)
        B = B.reshape(2, 4, 3)  # (2, 4, 3)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_batched_3d_arange passed")


fn test_mmnd_batched_4d() raises:
    """A[a, b, m, k] @ B[a, b, k, n] — 4D batch."""
    print("test_mmnd_batched_4d")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var B = Tensor[dtype].ones(2, 3, 5, 4)  # (2, 3, 5, 4)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_batched_4d passed")


fn test_mmnd_batched_large_batch() raises:
    """Many batch elements to stress grid.z coverage."""
    print("test_mmnd_batched_large_batch")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(32, 16, 8)  # 32 batch elements
        var B = Tensor[dtype].ones(32, 8, 16)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_batched_large_batch passed")


fn test_mmnd_batched_non_tile_multiples() raises:
    """Batched with m, k, n not multiples of TILE_SIZE."""
    print("test_mmnd_batched_non_tile_multiples")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(2 * 11 * 7)
        A = A.reshape(2, 11, 7)
        var B = Tensor[dtype].arange(2 * 7 * 13)
        B = B.reshape(2, 7, 13)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_batched_non_tile_multiples passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════


fn test_mmnd_broadcast_3d_A_2d_B() raises:
    """A[b, m, k] @ B[k, n] — B broadcasts across batch."""
    print("test_mmnd_broadcast_3d_A_2d_B")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(24)
        A = A.reshape(2, 3, 4)  # (2, 3, 4)
        var B = Tensor[dtype].arange(12)
        B = B.reshape(4, 3)  # (4, 3) — no batch, broadcasts
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_broadcast_3d_A_2d_B passed")


fn test_mmnd_broadcast_2d_A_3d_B() raises:
    """A[m, k] @ B[b, k, n] — A broadcasts across batch."""
    print("test_mmnd_broadcast_2d_A_3d_B")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(12)
        A = A.reshape(3, 4)  # (3, 4) — no batch, broadcasts
        var B = Tensor[dtype].arange(24)
        B = B.reshape(2, 4, 3)  # (2, 4, 3)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_broadcast_2d_A_3d_B passed")


fn test_mmnd_broadcast_4d_A_3d_B() raises:
    """A[a, b, m, k] @ B[b, k, n] — B missing leading batch dim."""
    print("test_mmnd_broadcast_4d_A_3d_B")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var B = Tensor[dtype].ones(3, 5, 4)  # (3, 5, 4) — broadcasts over dim 0
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_broadcast_4d_A_3d_B passed")


fn test_mmnd_broadcast_size1_batch_dim() raises:
    """Size-1 batch dim in A broadcasts across B's batch."""
    print("test_mmnd_broadcast_size1_batch_dim")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(
            1, 4, 5
        )  # (1, 4, 5) → broadcasts to (3, 4, 5)
        var B = Tensor[dtype].ones(3, 5, 4)  # (3, 5, 4)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_broadcast_size1_batch_dim passed")


fn test_mmnd_broadcast_known_values() raises:
    """Hand-computed broadcast result verified directly against GPU."""
    print("test_mmnd_broadcast_known_values")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # A = [[1,0],[0,1]] (identity, no batch)
        # B[0] = [[2,3],[4,5]], B[1] = [[6,7],[8,9]]
        # out[0] = I @ B[0] = [[2,3],[4,5]]
        # out[1] = I @ B[1] = [[6,7],[8,9]]
        var A = Tensor[dtype].d2([[1, 0], [0, 1]])
        var B = Tensor[dtype].d3([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        var expected = Tensor[dtype].d3([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mmnd_broadcast_known_values passed")


fn test_mmnd_broadcast_large() raises:
    """Large broadcast batch to stress multi-block and multi-z coverage."""
    print("test_mmnd_broadcast_large")

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(16, 32, 32)  # (16, 32, 32)
        var B = Tensor[dtype].ones(32, 32)  # (32, 32) — broadcasts over 16
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mmnd_broadcast_large passed")


fn test_gpu_transfer_fidelity() raises:
    print("=== Test 1: GPU transfer fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var B_back = B_gpu.to_cpu()
    assert_true(B.all_close(B_back))
    print("PASSED: B == B_gpu.to_cpu()")


fn test_ancestry_storage_fidelity() raises:
    print("=== Test 2: Ancestry storage fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var A_gpu = A.to_gpu()
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)
    var B_from_ancestry = C_gpu.ancestry().tensor(1)
    var B_ancestry_back = B_from_ancestry.to_cpu()
    assert_true(B.all_close(B_ancestry_back))
    print("PASSED: B from ancestry == original B")


fn test_forward_matmul_fidelity() raises:
    print("=== Test 3: Forward matmul fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)
    assert_true(C_cpu.all_close(C_gpu.to_cpu()))
    print("PASSED: CPU matmul == GPU matmul")


fn test_backward_grad_A_fidelity() raises:
    print("=== Test 4: Backward grad_A fidelity ===")
    var AA = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var A = AA.reshape(9, 30)
    var B = Tensor[dtype].arange(30 * 5).reshape(30, 5)

    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)

    C_cpu.backward()

    assert_true(A_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(9, 30))))

    C_gpu.backward()

    A_gpu.grad().print()
    assert_true(AA.grad().to_gpu().reshape(Shape(9, 30)).all_close(A_gpu.grad() * 2))
    print("PASSED: GPU backward grad_A == CPU backward grad_A")


fn test_transposed_matmul_fidelity() raises:
    print("=== Test 5: Transposed matmul fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var BT_cpu = B.transpose(axes=IntArray(-1, -2))
    var BT_gpu = B_gpu.buffer.transpose(axes=IntArray(-1, -2))

    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    var grad_A_cpu = grad_out.matmul(BT_cpu)

    var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_gpu
    )
    var grad_A_GPU = Tensor[dtype](grad_A_ndb^)
    var grad_A_gpu = grad_A_GPU.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A row 0:")
    for i in range(min(8, grad_A_gpu.shape()[-1])):
        print(grad_A_gpu.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_gpu))
    print("PASSED: GPU transposed matmul == CPU")


fn test_ancestry_transposed_matmul_fidelity() raises:
    print("=== Test 6: B from ancestry transposed matmul ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)

    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    var BT_cpu = B.transpose(axes=IntArray(-1, -2))
    var grad_A_cpu = grad_out.matmul(BT_cpu)

    var B_anc = C_gpu.ancestry().tensor(1)
    var BT_anc = B_anc.buffer.transpose(axes=IntArray(-1, -2))

    var grad_A_anc_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_anc
    )
    var grad_A_ANC = Tensor[dtype](grad_A_anc_ndb^)
    var grad_A_anc = grad_A_ANC.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A from ancestry row 0:")
    for i in range(min(8, grad_A_anc.shape()[-1])):
        print(grad_A_anc.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_anc))
    print("PASSED: B from ancestry transposed matmul == CPU")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


fn test_tensor_tensor_multiplications() raises:
    # Basic correctness
    test_mmnd_2d_known_values()
    test_mmnd_2d_identity()
    test_mmnd_2d_zero_matrix()
    test_mmnd_2d_ones()
    test_mmnd_2d_rectangular()
    test_mmnd_2d_negative_values()
    test_mmnd_2d_single_element()

    # Tile boundary stress
    test_mmnd_2d_non_tile_multiple_m()
    test_mmnd_2d_non_tile_multiple_n()
    test_mmnd_2d_non_tile_multiple_k()
    test_mmnd_2d_all_non_tile_multiples()
    test_mmnd_2d_smaller_than_tile()

    # Large matrices
    test_mmnd_2d_large_square()
    test_mmnd_2d_large_rectangular()

    # Batched same shape
    test_mmnd_batched_3d_known_values()
    test_mmnd_batched_3d_arange()
    test_mmnd_batched_4d()
    test_mmnd_batched_large_batch()
    test_mmnd_batched_non_tile_multiples()

    # Broadcast
    test_mmnd_broadcast_3d_A_2d_B()
    test_mmnd_broadcast_2d_A_3d_B()
    test_mmnd_broadcast_4d_A_3d_B()
    test_mmnd_broadcast_size1_batch_dim()
    test_mmnd_broadcast_known_values()
    test_mmnd_broadcast_large()
