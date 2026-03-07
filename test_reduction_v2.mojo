from tenmo import Tensor
from ndbuffer import NDBuffer
from intarray import IntArray
from shapes import Shape
from testing import assert_true
from sys import has_accelerator


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
    assert_true(s == Tensor[dtype].d2(
        [[12, 14, 16, 18], [20, 22, 24, 26], [28, 30, 32, 34]]
    ))
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
        ndb[IntArray(i)] = i + 1  # 1,2,3,4,5
    var result = ndb.reduce(normalized_axes=IntArray(0))
    assert_true(result[IntArray()] == 15)
    print("test_v2_ndbuffer_sum_1d passed")


fn test_v2_ndbuffer_sum_2d_axis0() raises:
    print("test_v2_ndbuffer_sum_2d_axis0")
    var ndb = NDBuffer[DType.float32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = val
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
            ndb[IntArray(i, j)] = val
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
            ndb[IntArray(i, j)] = val
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
            ndb[IntArray(i, j)] = val
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
            ndb[IntArray(i, j)] = val
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
            ndb[IntArray(i, j)] = val
            val += 1
    # keepdims -> shape (2, 1)
    var result = ndb.reduce[mean=True](normalized_axes=IntArray(1), keepdims=True)
    assert_true(result[IntArray(0, 0)] == 2.0)
    assert_true(result[IntArray(1, 0)] == 5.0)
    print("test_v2_ndbuffer_mean_keepdims passed")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU tests — Tensor.sum and Tensor.mean
# ═══════════════════════════════════════════════════════════════════════════════

fn test_v2_gpu_tensor_sum_1d() raises:
    print("test_v2_gpu_tensor_sum_1d")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1, 2, 3, 4, 5])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum()
        var gpu_result = a_gpu.sum()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
    print("test_v2_gpu_tensor_sum_1d passed")


fn test_v2_gpu_tensor_sum_2d_axis0() raises:
    print("test_v2_gpu_tensor_sum_2d_axis0")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[0])
        var gpu_result = a_gpu.sum(axes=[0])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
    print("test_v2_gpu_tensor_sum_2d_axis0 passed")


fn test_v2_gpu_tensor_sum_2d_axis1() raises:
    print("test_v2_gpu_tensor_sum_2d_axis1")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1])
        var gpu_result = a_gpu.sum(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
    print("test_v2_gpu_tensor_sum_2d_axis1 passed")


fn test_v2_gpu_tensor_sum_3d_axis1() raises:
    print("test_v2_gpu_tensor_sum_3d_axis1")
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum()
        var gpu_result = a_gpu.sum()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
    print("test_v2_gpu_tensor_sum_all_axes passed")


fn test_v2_gpu_tensor_mean_2d_axis0() raises:
    print("test_v2_gpu_tensor_mean_2d_axis0")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[0])
        var gpu_result = a_gpu.mean(axes=[0])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
    print("test_v2_gpu_tensor_mean_2d_axis0 passed")


fn test_v2_gpu_tensor_mean_2d_axis1() raises:
    print("test_v2_gpu_tensor_mean_2d_axis1")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1])
        var gpu_result = a_gpu.mean(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))
    print("test_v2_gpu_tensor_mean_2d_axis1 passed")


fn test_v2_gpu_tensor_mean_3d_axis1() raises:
    print("test_v2_gpu_tensor_mean_3d_axis1")
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        #var cpu_ndb = a.buffer
        var gpu_ndb = a_gpu.buffer
        #var cpu_result = cpu_ndb.reduce(normalized_axes=IntArray(1))
        var gpu_result = gpu_ndb.reduce[mean=False](IntArray(1), keepdims=False)
        var tensor = Tensor[dtype].d1([6, 15])
        var g_tensor = tensor.to_gpu()
        assert_true(g_tensor.buffer.all_close(gpu_result))
    print("test_v2_gpu_ndbuffer_sum_2d_axis1 passed")


fn test_v2_gpu_ndbuffer_mean_2d_axis0() raises:
    print("test_v2_gpu_ndbuffer_mean_2d_axis0")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
        var a_gpu = a.to_gpu()
        var gpu_ndb = a_gpu.buffer
        var gpu_result = gpu_ndb.reduce[mean=True](IntArray(0), keepdims=False)
        var expected = Tensor[dtype].d1([2, 3, 4])
        expected = expected.to_gpu()
        assert_true(
            expected.buffer.all_close(gpu_result)
        )
    print("test_v2_gpu_ndbuffer_mean_2d_axis0 passed")


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
