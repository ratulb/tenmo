from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.shapes import Shape


# ═════════════════════════════════════════════════════════════════════════════
# CPU count Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_countuniq_cpu_count_1d_basic() raises:
    print("test_countuniq_cpu_count_1d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 1.0, 3.0, 1.0])
    assert_true(a.count(Scalar[dtype](1.0)) == 3)
    assert_true(a.count(Scalar[dtype](2.0)) == 1)
    assert_true(a.count(Scalar[dtype](3.0)) == 1)
    assert_true(a.count(Scalar[dtype](4.0)) == 0)


fn test_countuniq_cpu_count_1d_all_same() raises:
    print("test_countuniq_cpu_count_1d_all_same")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape(10), 5.0)
    assert_true(a.count(Scalar[dtype](5.0)) == 10)
    assert_true(a.count(Scalar[dtype](0.0)) == 0)


fn test_countuniq_cpu_count_1d_none() raises:
    print("test_countuniq_cpu_count_1d_none")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    assert_true(a.count(Scalar[dtype](9.0)) == 0)


fn test_countuniq_cpu_count_2d_basic() raises:
    print("test_countuniq_cpu_count_2d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    assert_true(a.count(Scalar[dtype](1.0)) == 3)
    assert_true(a.count(Scalar[dtype](2.0)) == 2)
    assert_true(a.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_cpu_count_3d_basic() raises:
    print("test_countuniq_cpu_count_3d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [1.0, 3.0]], [[2.0, 1.0], [4.0, 1.0]]]
    )
    assert_true(a.count(Scalar[dtype](1.0)) == 4)
    assert_true(a.count(Scalar[dtype](2.0)) == 2)
    assert_true(a.count(Scalar[dtype](4.0)) == 1)
    assert_true(a.count(Scalar[dtype](9.0)) == 0)


fn test_countuniq_cpu_count_zeros() raises:
    print("test_countuniq_cpu_count_zeros")
    comptime dtype = DType.float32
    var a = Tensor[dtype].zeros(Shape(5))
    assert_true(a.count(Scalar[dtype](0.0)) == 5)


fn test_countuniq_cpu_count_large() raises:
    print("test_countuniq_cpu_count_large")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape(10000), 3.0)
    assert_true(a.count(Scalar[dtype](3.0)) == 10000)
    assert_true(a.count(Scalar[dtype](0.0)) == 0)


fn test_countuniq_cpu_count_int() raises:
    print("test_countuniq_cpu_count_int")
    comptime dtype = DType.int32
    var a = Tensor[dtype].d1([1, 2, 3, 2, 1, 2])
    assert_true(a.count(Scalar[dtype](2)) == 3)
    assert_true(a.count(Scalar[dtype](1)) == 2)
    assert_true(a.count(Scalar[dtype](3)) == 1)


fn test_countuniq_cpu_count_bool() raises:
    print("test_countuniq_cpu_count_bool")
    var a = Tensor[DType.bool].d1([True, False, True, True, False])
    assert_true(a.count(Scalar[DType.bool](True)) == 3)
    assert_true(a.count(Scalar[DType.bool](False)) == 2)


fn test_countuniq_cpu_count_noncontiguous() raises:
    print("test_countuniq_cpu_count_noncontiguous")
    comptime dtype = DType.float32
    # Create non-contiguous via transpose
    var a = Tensor[dtype].d2([[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]])
    var t = a.transpose()  # non-contiguous view
    # t has shape (2,3) — values [1,1,2,2,3,1]
    assert_true(t.count(Scalar[dtype](1.0)) == 3)
    assert_true(t.count(Scalar[dtype](2.0)) == 2)
    assert_true(t.count(Scalar[dtype](3.0)) == 1)


# ═════════════════════════════════════════════════════════════════════════════
# CPU unique Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_countuniq_cpu_unique_1d_basic() raises:
    print("test_countuniq_cpu_unique_1d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 1.0, 3.0, 2.0])
    var result = a.unique()
    # unique returns sorted or unordered — just check count and membership
    assert_true(result.numels() == 3)
    assert_true(result.count(Scalar[dtype](1.0)) == 1)
    assert_true(result.count(Scalar[dtype](2.0)) == 1)
    assert_true(result.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_cpu_unique_1d_all_same() raises:
    print("test_countuniq_cpu_unique_1d_all_same")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape(10), 5.0)
    var result = a.unique()
    assert_true(result.numels() == 1)
    assert_true(result[[0]] == Scalar[dtype](5.0))


fn test_countuniq_cpu_unique_1d_all_different() raises:
    print("test_countuniq_cpu_unique_1d_all_different")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var result = a.unique()
    assert_true(result.numels() == 5)


fn test_countuniq_cpu_unique_2d_basic() raises:
    print("test_countuniq_cpu_unique_2d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [2.0, 3.0], [1.0, 3.0]])
    var result = a.unique()
    assert_true(result.numels() == 3)
    assert_true(result.count(Scalar[dtype](1.0)) == 1)
    assert_true(result.count(Scalar[dtype](2.0)) == 1)
    assert_true(result.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_cpu_unique_3d_basic() raises:
    print("test_countuniq_cpu_unique_3d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 1.0]], [[2.0, 3.0], [4.0, 1.0]]]
    )
    var result = a.unique()
    assert_true(result.numels() == 4)
    assert_true(result.count(Scalar[dtype](1.0)) == 1)
    assert_true(result.count(Scalar[dtype](4.0)) == 1)


fn test_countuniq_cpu_unique_single_element() raises:
    print("test_countuniq_cpu_unique_single_element")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([42.0])
    var result = a.unique()
    assert_true(result.numels() == 1)
    assert_true(result[[0]] == Scalar[dtype](42.0))


fn test_countuniq_cpu_unique_int() raises:
    print("test_countuniq_cpu_unique_int")
    comptime dtype = DType.int32
    var a = Tensor[dtype].d1([1, 2, 3, 2, 1, 4])
    var result = a.unique()
    assert_true(result.numels() == 4)
    assert_true(result.count(Scalar[dtype](1)) == 1)
    assert_true(result.count(Scalar[dtype](4)) == 1)


fn test_countuniq_cpu_unique_bool() raises:
    print("test_countuniq_cpu_unique_bool")
    # bool has at most 2 unique values
    var a = Tensor[DType.bool].d1([True, False, True, True])
    var result = a.unique()
    assert_true(result.numels() == 2)


fn test_countuniq_cpu_unique_bool_all_true() raises:
    print("test_countuniq_cpu_unique_bool_all_true")
    var a = Tensor[DType.bool].d1([True, True, True])
    var result = a.unique()
    assert_true(result.numels() == 1)


fn test_countuniq_cpu_unique_noncontiguous() raises:
    print("test_countuniq_cpu_unique_noncontiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]])
    var t = a.transpose()
    var result = t.unique()
    assert_true(result.numels() == 3)
    assert_true(result.count(Scalar[dtype](1.0)) == 1)
    assert_true(result.count(Scalar[dtype](2.0)) == 1)
    assert_true(result.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_cpu_unique_result_is_cpu() raises:
    print("test_countuniq_cpu_unique_result_is_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 1.0])
    var result = a.unique()
    assert_true(result.is_on_cpu())


# ═════════════════════════════════════════════════════════════════════════════
# GPU count Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_countuniq_gpu_count_1d_basic() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_1d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 1.0, 3.0, 1.0]).to_gpu()
        assert_true(a.count(Scalar[dtype](1.0)) == 3)
        assert_true(a.count(Scalar[dtype](2.0)) == 1)
        assert_true(a.count(Scalar[dtype](4.0)) == 0)


fn test_countuniq_gpu_count_1d_all_same() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_1d_all_same")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full(Shape(10), 5.0).to_gpu()
        assert_true(a.count(Scalar[dtype](5.0)) == 10)
        assert_true(a.count(Scalar[dtype](0.0)) == 0)


fn test_countuniq_gpu_count_2d_basic() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_2d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]]
        ).to_gpu()
        assert_true(a.count(Scalar[dtype](1.0)) == 3)
        assert_true(a.count(Scalar[dtype](2.0)) == 2)
        assert_true(a.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_gpu_count_3d_basic() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_3d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [1.0, 3.0]], [[2.0, 1.0], [4.0, 1.0]]]
        ).to_gpu()
        assert_true(a.count(Scalar[dtype](1.0)) == 4)
        assert_true(a.count(Scalar[dtype](2.0)) == 2)
        assert_true(a.count(Scalar[dtype](9.0)) == 0)


fn test_countuniq_gpu_count_zeros() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_zeros")
        comptime dtype = DType.float32
        var a = Tensor[dtype].zeros(Shape(5)).to_gpu()
        assert_true(a.count(Scalar[dtype](0.0)) == 5)


fn test_countuniq_gpu_count_large() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_large")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full(Shape(10000), 3.0).to_gpu()
        assert_true(a.count(Scalar[dtype](3.0)) == 10000)
        assert_true(a.count(Scalar[dtype](0.0)) == 0)


fn test_countuniq_gpu_count_int() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_int")
        comptime dtype = DType.int32
        var a = Tensor[dtype].d1([1, 2, 3, 2, 1, 2]).to_gpu()
        assert_true(a.count(Scalar[dtype](2)) == 3)
        assert_true(a.count(Scalar[dtype](1)) == 2)


fn test_countuniq_gpu_count_noncontiguous() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_count_noncontiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]]
        ).to_gpu()
        var t = a.transpose()
        assert_true(t.count(Scalar[dtype](1.0)) == 3)
        assert_true(t.count(Scalar[dtype](2.0)) == 2)
        assert_true(t.count(Scalar[dtype](3.0)) == 1)


# ═════════════════════════════════════════════════════════════════════════════
# GPU unique Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_countuniq_gpu_unique_1d_basic() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_1d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 1.0, 3.0, 2.0]).to_gpu()
        var result = a.unique()
        # unique always returns CPU NDBuffer
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 3)
        assert_true(result.count(Scalar[dtype](1.0)) == 1)
        assert_true(result.count(Scalar[dtype](2.0)) == 1)
        assert_true(result.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_gpu_unique_1d_all_same() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_1d_all_same")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full(Shape(10), 5.0).to_gpu()
        var result = a.unique()
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 1)
        assert_true(result[[0]] == Scalar[dtype](5.0))


fn test_countuniq_gpu_unique_2d_basic() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_2d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [2.0, 3.0], [1.0, 3.0]]
        ).to_gpu()
        var result = a.unique()
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 3)
        assert_true(result.count(Scalar[dtype](1.0)) == 1)
        assert_true(result.count(Scalar[dtype](2.0)) == 1)
        assert_true(result.count(Scalar[dtype](3.0)) == 1)


fn test_countuniq_gpu_unique_3d_basic() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_3d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 1.0]], [[2.0, 3.0], [4.0, 1.0]]]
        ).to_gpu()
        var result = a.unique()
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 4)
        assert_true(result.count(Scalar[dtype](1.0)) == 1)
        assert_true(result.count(Scalar[dtype](4.0)) == 1)


fn test_countuniq_gpu_unique_all_different() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_all_different")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var result = a.unique()
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 5)


fn test_countuniq_gpu_unique_int() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_int")
        comptime dtype = DType.int32
        var a = Tensor[dtype].d1([1, 2, 3, 2, 1, 4]).to_gpu()
        var result = a.unique()
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 4)


fn test_countuniq_gpu_unique_noncontiguous() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_noncontiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]]
        ).to_gpu()
        var t = a.transpose()
        var result = t.unique()
        assert_true(result.is_on_cpu())
        assert_true(result.numels() == 3)


fn test_countuniq_gpu_unique_result_always_cpu() raises:
    comptime if has_accelerator():
        print("test_countuniq_gpu_unique_result_always_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 1.0]).to_gpu()
        var result = a.unique()
        # unique always returns CPU NDBuffer regardless of input device
        assert_true(result.is_on_cpu())


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_countuniq_parity_count_1d() raises:
    comptime if has_accelerator():
        print("test_countuniq_parity_count_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.count(Scalar[dtype](1.0))
            == a_gpu.count(Scalar[dtype](1.0))
        )
        assert_true(
            a_cpu.count(Scalar[dtype](2.0))
            == a_gpu.count(Scalar[dtype](2.0))
        )
        assert_true(
            a_cpu.count(Scalar[dtype](9.0))
            == a_gpu.count(Scalar[dtype](9.0))
        )


fn test_countuniq_parity_count_2d() raises:
    comptime if has_accelerator():
        print("test_countuniq_parity_count_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
        var a_gpu = a_cpu.to_gpu()
        var values: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        for val in values:
            assert_true(
                a_cpu.count(Scalar[dtype](val))
                == a_gpu.count(Scalar[dtype](val))
            )


fn test_countuniq_parity_count_large() raises:
    comptime if has_accelerator():
        print("test_countuniq_parity_count_large")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape(10000), 7.0)
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.count(Scalar[dtype](7.0))
            == a_gpu.count(Scalar[dtype](7.0))
        )


fn test_countuniq_parity_unique_count_1d() raises:
    comptime if has_accelerator():
        print("test_countuniq_parity_unique_count_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 1.0, 3.0, 2.0, 4.0])
        var a_gpu = a_cpu.to_gpu()
        var uniq_cpu = a_cpu.unique()
        var uniq_gpu = a_gpu.unique()
        # Both should have same number of unique values
        assert_true(uniq_cpu.numels() == uniq_gpu.numels())
        # And same membership
        for i in range(uniq_cpu.numels()):
            var val = uniq_cpu[[i]]
            assert_true(uniq_gpu.count(val) == 1)


fn test_countuniq_parity_unique_2d() raises:
    comptime if has_accelerator():
        print("test_countuniq_parity_unique_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]]
        )
        var a_gpu = a_cpu.to_gpu()
        var uniq_cpu = a_cpu.unique()
        var uniq_gpu = a_gpu.unique()
        assert_true(uniq_cpu.numels() == uniq_gpu.numels())


fn test_countuniq_parity_noncontiguous() raises:
    comptime if has_accelerator():
        print("test_countuniq_parity_noncontiguous")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]]
        )
        var a_gpu = a_cpu.to_gpu()
        var t_cpu = a_cpu.transpose()
        var t_gpu = a_gpu.transpose()
        assert_true(
            t_cpu.count(Scalar[dtype](1.0))
            == t_gpu.count(Scalar[dtype](1.0))
        )
        assert_true(
            t_cpu.unique().numels() == t_gpu.unique().numels()
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    _ = """
    # CPU count
    test_countuniq_cpu_count_1d_basic()
    test_countuniq_cpu_count_1d_all_same()
    test_countuniq_cpu_count_1d_none()
    test_countuniq_cpu_count_2d_basic()
    test_countuniq_cpu_count_3d_basic()
    test_countuniq_cpu_count_zeros()
    test_countuniq_cpu_count_large()
    test_countuniq_cpu_count_int()
    test_countuniq_cpu_count_bool()
    test_countuniq_cpu_count_noncontiguous()
    print("CPU count passed!")

    # CPU unique
    test_countuniq_cpu_unique_1d_basic()
    test_countuniq_cpu_unique_1d_all_same()
    test_countuniq_cpu_unique_1d_all_different()
    test_countuniq_cpu_unique_2d_basic()
    test_countuniq_cpu_unique_3d_basic()
    test_countuniq_cpu_unique_single_element()
    test_countuniq_cpu_unique_int()
    test_countuniq_cpu_unique_bool()
    test_countuniq_cpu_unique_bool_all_true()
    test_countuniq_cpu_unique_noncontiguous()
    test_countuniq_cpu_unique_result_is_cpu()
    print("CPU unique passed!")

    # GPU count
    test_countuniq_gpu_count_1d_basic()
    test_countuniq_gpu_count_1d_all_same()
    test_countuniq_gpu_count_2d_basic()
    test_countuniq_gpu_count_3d_basic()
    test_countuniq_gpu_count_zeros()
    test_countuniq_gpu_count_large()
    test_countuniq_gpu_count_int()
    test_countuniq_gpu_count_noncontiguous()
    comptime if has_accelerator():
        print("GPU count passed!")

    # GPU unique
    test_countuniq_gpu_unique_1d_basic()
    test_countuniq_gpu_unique_1d_all_same()
    test_countuniq_gpu_unique_2d_basic()
    test_countuniq_gpu_unique_3d_basic()
    test_countuniq_gpu_unique_all_different()
    test_countuniq_gpu_unique_int()
    test_countuniq_gpu_unique_noncontiguous()
    test_countuniq_gpu_unique_result_always_cpu()
    comptime if has_accelerator():
        print("GPU unique passed!")

    # Parity
    test_countuniq_parity_count_1d()
    test_countuniq_parity_count_2d()
    test_countuniq_parity_count_large()
    test_countuniq_parity_unique_count_1d()
    test_countuniq_parity_unique_2d()
    test_countuniq_parity_noncontiguous()
    print("Parity passed!")

    print("All count/unique tests passed!")
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
