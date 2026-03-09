from tenmo import Tensor
from ndbuffer import NDBuffer
from intarray import IntArray
from shapes import Shape
from testing import assert_true
from sys import has_accelerator
from mnemonics import vm, mv


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
        var gpu_ndb = a_gpu.buffer
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
        assert_true(expected.buffer.all_close(gpu_result))
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

    # vector matrix multiplication
    test_vector_matrix_mul_tests()

    # matrix vector multiplication
    test_matrix_vector_multiplications()

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

fn close_enough[dtype: DType](
    mut a: Tensor[dtype], b: Tensor[dtype]
) raises -> Bool:
    var a_gpu = a.to_gpu()
    #return a_gpu.all_close(b.to_gpu())
    return a_gpu.all_close(b)


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════

fn test_vmnd_1d_v_2d_M() raises:
    """V[k] @ M[k, n] → out[n]. Simplest case, no batch dims."""
    print("test_vmnd_1d_v_2d_M")
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var k = 8
        var n = 1024   # larger than default block_size=256
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)          # (2, 3)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)      # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        #result_cpu = gpu_result.to_cpu()
        cpu_result.print()
        gpu_result.print()
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_2d_v_3d_M passed")


fn test_vmnd_batched_3d_v_4d_M() raises:
    """V[a, b, k] @ M[a, b, k, n] → out[a, b, n]."""
    print("test_vmnd_batched_3d_v_4d_M")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(2, 3, 4)      # (2, 3, 4)
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([1, 0, 1])          # (3,)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)                        # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v1d_M3d passed")


fn test_vmnd_broadcast_v2d_M3d() raises:
    """V[1, k] broadcast against M[b, k, n] → out[b, n]."""
    print("test_vmnd_broadcast_v2d_M3d")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)             # (1, 4)
        var M = Tensor[dtype].arange(48)
        M = M.reshape(3, 4, 4)                        # (3, 4, 4)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v2d_M3d passed")


fn test_vmnd_broadcast_v3d_M2d() raises:
    """V[a, b, k] broadcast against M[k, n] → out[a, b, n]."""
    print("test_vmnd_broadcast_v3d_M2d")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(24)
        v = v.reshape(2, 3, 4)                        # (2, 3, 4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)                            # (4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v3d_M2d passed")


fn test_vmnd_broadcast_both_size1() raises:
    """Both v and M have a size-1 batch dim that broadcasts."""
    print("test_vmnd_broadcast_both_size1")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)             # (1, 4) → broadcasts to (3, 4)
        var M = Tensor[dtype].ones(3, 4, 5)          # (3, 4, 5)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_both_size1 passed")


fn test_vmnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""
    print("test_vmnd_broadcast_large_batch")
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
        var M = Tensor[dtype].d2([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2, 3, 4]])   # (1, 3)
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2], [3], [5]])   # (3, 1)
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var m = 1024   # larger than default block_size=256
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)   # (2, 3, 3)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)      # (2, 3)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_3d_M_2d_v passed")


fn test_mvnd_batched_4d_M_3d_v() raises:
    """M[a, b, m, k] @ v[a, b, k] → out[a, b, m]."""
    print("test_mvnd_batched_4d_M_3d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)   # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(2, 3, 5)       # (2, 3, 5)
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
    @parameter
    if has_accelerator():
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: [[1,0],[0,1]] @ [1,2] = [1, 2]
        # batch 1: [[2,0],[0,2]] @ [3,4] = [6, 8]
        var M = Tensor[dtype].d3([[[1,0],[0,1]], [[2,0],[0,2]]])
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)           # (2, 3, 3)
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)              # (4, 3) — no batch
        var v = Tensor[dtype].arange(9)
        v = v.reshape(3, 3)              # (3, 3) — batch of 3
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_2d_M_3d_v passed")


fn test_mvnd_broadcast_4d_M_2d_v() raises:
    """M[a, b, m, k] broadcast against v[b, k] → out[a, b, m]."""
    print("test_mvnd_broadcast_4d_M_2d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)   # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(3, 5)          # (3, 5) — broadcasts over dim 0
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_4d_M_2d_v passed")


fn test_mvnd_broadcast_size1_batch() raises:
    """V with size-1 batch dim that broadcasts across M's batch."""
    print("test_mvnd_broadcast_size1_batch")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(4, 3, 5)   # (4, 3, 5)
        var v = Tensor[dtype].ones(1, 5)       # (1, 5) → broadcasts to (4, 5)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_size1_batch passed")


fn test_mvnd_broadcast_known_values() raises:
    """Hand-computed broadcast result verified directly against GPU."""
    print("test_mvnd_broadcast_known_values")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # M[0] = [[1,2],[3,4]], M[1] = [[5,6],[7,8]]
        # v = [1, 1]  (no batch — broadcasts across both)
        # out[0] = [3, 7],  out[1] = [11, 15]
        var M = Tensor[dtype].d3([[[1,2],[3,4]], [[5,6],[7,8]]])
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
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var m = 64
        var k = 32
        var M = Tensor[dtype].ones(128, m, k)   # large batch on M side
        var v = Tensor[dtype].ones(k)            # no batch — broadcasts
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
