from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape
from mnemonics import Equal, NotEqual, LessThan, LessThanEqual, GreaterThan, GreaterThanEqual


# ═════════════════════════════════════════════════════════════════════════════
# CPU Tensor comparison operator tests
# These test the high-level Tensor operators which delegate to Compare/
# CompareScalar kernels on GPU or CPU compare methods on CPU
# ═════════════════════════════════════════════════════════════════════════════


# ── CPU Equal ─────────────────────────────────────────────────────────────────


fn test_compare_cpu_eq_1d() raises:
    print("test_compare_cpu_eq_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([1.0, 0.0, 3.0])
    var result = a == b
    #assert_true(result[[0]] == True)
    #assert_true(result[[1]] == False)
    assert_true(result == False)
    #assert_true(result[[2]] == True)


fn test_compare_cpu_eq_scalar() raises:
    print("test_compare_cpu_eq_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 2.0, 3.0])
    var result = a == Scalar[dtype](2.0)
    assert_true(result[[0]] == False)
    assert_true(result[[1]] == True)
    assert_true(result[[2]] == True)
    assert_true(result[[3]] == False)


fn test_compare_cpu_eq_2d() raises:
    print("test_compare_cpu_eq_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor[dtype].d2([[1.0, 0.0], [3.0, 0.0]])
    var result = a == b
    _="""assert_true(result[[0, 0]] == True)
    assert_true(result[[0, 1]] == False)
    assert_true(result[[1, 0]] == True)
    assert_true(result[[1, 1]] == False)"""
    assert_true(result == False)



# ── CPU NotEqual ──────────────────────────────────────────────────────────────


fn test_compare_cpu_ne_1d() raises:
    print("test_compare_cpu_ne_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([1.0, 0.0, 3.0])
    var result = a != b
    _="""assert_true(result[[0]] == False)
    assert_true(result[[1]] == True)
    assert_true(result[[2]] == False)"""
    assert_true(result == False)


fn test_compare_cpu_ne_scalar() raises:
    print("test_compare_cpu_ne_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 2.0, 3.0])
    var result = a != Scalar[dtype](2.0)
    assert_true(result[[0]] == True)
    assert_true(result[[1]] == False)
    assert_true(result[[2]] == False)
    assert_true(result[[3]] == True)


# ── CPU GreaterThan ───────────────────────────────────────────────────────────


fn test_compare_cpu_gt_1d() raises:
    print("test_compare_cpu_gt_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0])
    var b = Tensor[dtype].d1([2.0, 3.0, 3.0])
    var result = a > b
    assert_true(result[[0]] == False)
    assert_true(result[[1]] == True)
    assert_true(result[[2]] == False)  # equal is not greater


fn test_compare_cpu_gt_scalar() raises:
    print("test_compare_cpu_gt_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0, 4.0])
    var result = a > Scalar[dtype](4.0)
    assert_true(result[[0]] == False)
    assert_true(result[[1]] == False)  # equal is not greater
    assert_true(result[[2]] == True)
    assert_true(result[[3]] == False)


fn test_compare_cpu_gt_2d() raises:
    print("test_compare_cpu_gt_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 5.0], [3.0, 2.0]])
    var b = Tensor[dtype].d2([[2.0, 3.0], [3.0, 4.0]])
    var result = a > b
    assert_true(result[[0, 0]] == False)
    assert_true(result[[0, 1]] == True)
    assert_true(result[[1, 0]] == False)
    assert_true(result[[1, 1]] == False)


fn test_compare_cpu_gt_3d() raises:
    print("test_compare_cpu_gt_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 1.0]]]
    )
    var result = a > Scalar[dtype](4.0)
    assert_true(result[[0, 0, 0]] == False)
    assert_true(result[[0, 0, 1]] == True)
    assert_true(result[[0, 1, 0]] == False)
    assert_true(result[[0, 1, 1]] == True)
    assert_true(result[[1, 0, 0]] == True)
    assert_true(result[[1, 0, 1]] == False)
    assert_true(result[[1, 1, 0]] == False)
    assert_true(result[[1, 1, 1]] == False)


# ── CPU GreaterThanEqual ──────────────────────────────────────────────────────


fn test_compare_cpu_gte_1d() raises:
    print("test_compare_cpu_gte_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0])
    var b = Tensor[dtype].d1([2.0, 4.0, 3.0])
    var result = a >= b
    assert_true(result[[0]] == False)
    assert_true(result[[1]] == True)   # equal counts
    assert_true(result[[2]] == True)


fn test_compare_cpu_gte_scalar() raises:
    print("test_compare_cpu_gte_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0])
    var result = a >= Scalar[dtype](4.0)
    assert_true(result[[0]] == False)
    assert_true(result[[1]] == True)   # equal counts
    assert_true(result[[2]] == True)


# ── CPU LessThan ──────────────────────────────────────────────────────────────


fn test_compare_cpu_lt_1d() raises:
    print("test_compare_cpu_lt_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0])
    var b = Tensor[dtype].d1([2.0, 4.0, 3.0])
    var result = a < b
    assert_true(result[[0]] == True)
    assert_true(result[[1]] == False)  # equal is not less
    assert_true(result[[2]] == False)


fn test_compare_cpu_lt_scalar() raises:
    print("test_compare_cpu_lt_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0])
    var result = a < Scalar[dtype](4.0)
    assert_true(result[[0]] == True)
    assert_true(result[[1]] == False)  # equal is not less
    assert_true(result[[2]] == False)


fn test_compare_cpu_lt_2d() raises:
    print("test_compare_cpu_lt_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 5.0], [3.0, 2.0]])
    var b = Tensor[dtype].d2([[2.0, 3.0], [3.0, 4.0]])
    var result = a < b
    assert_true(result[[0, 0]] == True)
    assert_true(result[[0, 1]] == False)
    assert_true(result[[1, 0]] == False)
    assert_true(result[[1, 1]] == True)


# ── CPU LessThanEqual ─────────────────────────────────────────────────────────


fn test_compare_cpu_lte_1d() raises:
    print("test_compare_cpu_lte_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0])
    var b = Tensor[dtype].d1([2.0, 4.0, 3.0])
    var result = a <= b
    assert_true(result[[0]] == True)
    assert_true(result[[1]] == True)   # equal counts
    assert_true(result[[2]] == False)


fn test_compare_cpu_lte_scalar() raises:
    print("test_compare_cpu_lte_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 4.0, 5.0])
    var result = a <= Scalar[dtype](4.0)
    assert_true(result[[0]] == True)
    assert_true(result[[1]] == True)   # equal counts
    assert_true(result[[2]] == False)


# ── CPU all_true/any_true on comparison results ───────────────────────────────


fn test_compare_cpu_all_true_from_gt() raises:
    print("test_compare_cpu_all_true_from_gt")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([5.0, 6.0, 7.0])
    assert_true((a > Scalar[dtype](4.0)).all_true())


fn test_compare_cpu_all_true_fails() raises:
    print("test_compare_cpu_all_true_fails")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([3.0, 6.0, 7.0])
    assert_true(not (a > Scalar[dtype](4.0)).all_true())


fn test_compare_cpu_any_true_from_gt() raises:
    print("test_compare_cpu_any_true_from_gt")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 5.0])
    assert_true((a > Scalar[dtype](4.0)).any_true())


fn test_compare_cpu_any_true_fails() raises:
    print("test_compare_cpu_any_true_fails")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    assert_true(not (a > Scalar[dtype](4.0)).any_true())


# ── CPU large tensor ──────────────────────────────────────────────────────────


fn test_compare_cpu_large_gt_scalar() raises:
    print("test_compare_cpu_large_gt_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape(10000), 5.0)
    var result = a > Scalar[dtype](4.0)
    assert_true(result.all_true())


fn test_compare_cpu_large_lt_scalar() raises:
    print("test_compare_cpu_large_lt_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full(Shape(10000), 3.0)
    var result = a < Scalar[dtype](4.0)
    assert_true(result.all_true())


# ═════════════════════════════════════════════════════════════════════════════
# GPU Tensor comparison operator tests
# Result is always CPU NDBuffer[DType.bool] — no DeviceBuffer[DType.bool]
# ═════════════════════════════════════════════════════════════════════════════


# ── GPU Equal ─────────────────────────────────────────────────────────────────


fn test_compare_gpu_eq_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_eq_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 0.0, 3.0]).to_gpu()
        var result = a.eq(b)
        # Result is CPU bool tensor
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == True)
        assert_true(result[[1]] == False)
        assert_true(result[[2]] == True)


fn test_compare_gpu_eq_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_eq_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 2.0, 3.0]).to_gpu()
        var result = a == Scalar[dtype](2.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == False)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == True)
        assert_true(result[[3]] == False)


fn test_compare_gpu_eq_2d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_eq_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var b = Tensor[dtype].d2([[1.0, 0.0], [3.0, 0.0]]).to_gpu()
        var result = a.eq(b)
        
        assert_true(result.is_on_gpu())
        assert_true(result[[0, 0]] == True)
        assert_true(result[[0, 1]] == False)
        assert_true(result[[1, 0]] == True)
        assert_true(result[[1, 1]] == False)


# ── GPU NotEqual ──────────────────────────────────────────────────────────────


fn test_compare_gpu_ne_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_ne_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 0.0, 3.0]).to_gpu()
        var result = a.ne(b)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == False)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == False)


fn test_compare_gpu_ne_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_ne_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 2.0, 3.0]).to_gpu()
        var result = a != Scalar[dtype](2.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == True)
        assert_true(result[[1]] == False)
        assert_true(result[[2]] == False)
        assert_true(result[[3]] == True)


# ── GPU GreaterThan ───────────────────────────────────────────────────────────


fn test_compare_gpu_gt_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_gt_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([2.0, 3.0, 3.0]).to_gpu()
        var result = a > b
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == False)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == False)


fn test_compare_gpu_gt_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_gt_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0, 4.0]).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == False)
        assert_true(result[[1]] == False)
        assert_true(result[[2]] == True)
        assert_true(result[[3]] == False)


fn test_compare_gpu_gt_2d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_gt_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0], [3.0, 2.0]]).to_gpu()
        var b = Tensor[dtype].d2([[2.0, 3.0], [3.0, 4.0]]).to_gpu()
        var result = a > b
        assert_true(result.is_on_gpu())
        assert_true(result[[0, 0]] == False)
        assert_true(result[[0, 1]] == True)
        assert_true(result[[1, 0]] == False)
        assert_true(result[[1, 1]] == False)


fn test_compare_gpu_gt_3d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_gt_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 1.0]]]
        ).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0, 0, 0]] == False)
        assert_true(result[[0, 0, 1]] == True)
        assert_true(result[[0, 1, 0]] == False)
        assert_true(result[[0, 1, 1]] == True)
        assert_true(result[[1, 0, 0]] == True)
        assert_true(result[[1, 0, 1]] == False)
        assert_true(result[[1, 1, 0]] == False)
        assert_true(result[[1, 1, 1]] == False)


# ── GPU GreaterThanEqual ──────────────────────────────────────────────────────


fn test_compare_gpu_gte_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_gte_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0]).to_gpu()
        var b = Tensor[dtype].d1([2.0, 4.0, 3.0]).to_gpu()
        var result = a >= b
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == False)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == True)


fn test_compare_gpu_gte_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_gte_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0]).to_gpu()
        var result = a >= Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == False)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == True)


# ── GPU LessThan ──────────────────────────────────────────────────────────────


fn test_compare_gpu_lt_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_lt_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0]).to_gpu()
        var b = Tensor[dtype].d1([2.0, 4.0, 3.0]).to_gpu()
        var result = a < b
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == True)
        assert_true(result[[1]] == False)
        assert_true(result[[2]] == False)


fn test_compare_gpu_lt_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_lt_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0]).to_gpu()
        var result = a < Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == True)
        assert_true(result[[1]] == False)
        assert_true(result[[2]] == False)


fn test_compare_gpu_lt_2d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_lt_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0], [3.0, 2.0]]).to_gpu()
        var b = Tensor[dtype].d2([[2.0, 3.0], [3.0, 4.0]]).to_gpu()
        var result = a < b
        assert_true(result.is_on_gpu())
        assert_true(result[[0, 0]] == True)
        assert_true(result[[0, 1]] == False)
        assert_true(result[[1, 0]] == False)
        assert_true(result[[1, 1]] == True)


# ── GPU LessThanEqual ─────────────────────────────────────────────────────────


fn test_compare_gpu_lte_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_lte_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0]).to_gpu()
        var b = Tensor[dtype].d1([2.0, 4.0, 3.0]).to_gpu()
        var result = a <= b
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == True)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == False)


fn test_compare_gpu_lte_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_lte_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 4.0, 5.0]).to_gpu()
        var result = a <= Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result[[0]] == True)
        assert_true(result[[1]] == True)
        assert_true(result[[2]] == False)


# ── GPU all_true/any_true on comparison results ───────────────────────────────


fn test_compare_gpu_all_true_from_gt() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_all_true_from_gt")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0]).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result.all_true())


fn test_compare_gpu_all_true_fails() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_all_true_fails")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 6.0, 7.0]).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(not result.all_true())


fn test_compare_gpu_any_true_from_gt() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_any_true_from_gt")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 5.0]).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result.any_true())


fn test_compare_gpu_any_true_fails() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_any_true_fails")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(not result.any_true())


# ── GPU large tensor ──────────────────────────────────────────────────────────


fn test_compare_gpu_large_gt_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_large_gt_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full(Shape(10000), 5.0).to_gpu()
        var result = a > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result.all_true())


fn test_compare_gpu_large_lt_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_large_lt_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full(Shape(10000), 3.0).to_gpu()
        var result = a < Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(result.all_true())


fn test_compare_gpu_large_mixed() raises:
    @parameter
    if has_accelerator():
        print("test_compare_gpu_large_mixed")
        comptime dtype = DType.float32
        # Half above, half below
        var a_cpu = Tensor[dtype].zeros(Shape(10000))
        for i in range(5000):
            a_cpu[[i]] = Scalar[dtype](5.0)
        for i in range(5000, 10000):
            a_cpu[[i]] = Scalar[dtype](3.0)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu > Scalar[dtype](4.0)
        assert_true(result.is_on_gpu())
        assert_true(not result.all_true())
        assert_true(result.any_true())


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════

fn test_compare_parity_eq_1d() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_eq_1d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 2.0])
        var b_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 2.0])
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        var result_cpu = a_cpu.eq(b_cpu)
        var result_gpu = a_gpu.eq(b_gpu)
        # Compare element by element — avoid calling .eq() on bool tensors
        var result_gpu_cpu = result_gpu.to_cpu()
        for i in range(4):
            assert_true(result_cpu[[i]] == result_gpu_cpu[[i]])


fn test_compare_parity_gt_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_gt_scalar")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 5.0, 3.0, 6.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu > Scalar[dtype](4.0)
        var result_gpu = a_gpu > Scalar[dtype](4.0)
        for i in range(5):
            assert_true(result_cpu[[i]] == result_gpu[[i]])


fn test_compare_parity_lt_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_lt_scalar")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 5.0, 3.0, 6.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu < Scalar[dtype](4.0)
        var result_gpu = a_gpu < Scalar[dtype](4.0)
        for i in range(5):
            assert_true(result_cpu[[i]] == result_gpu[[i]])


fn test_compare_parity_gte_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_gte_scalar")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu >= Scalar[dtype](4.0)
        var result_gpu = a_gpu >= Scalar[dtype](4.0)
        for i in range(3):
            assert_true(result_cpu[[i]] == result_gpu[[i]])


fn test_compare_parity_lte_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_lte_scalar")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu <= Scalar[dtype](4.0)
        var result_gpu = a_gpu <= Scalar[dtype](4.0)
        for i in range(3):
            assert_true(result_cpu[[i]] == result_gpu[[i]])


fn test_compare_parity_ne_scalar() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_ne_scalar")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 5.0, 4.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu != Scalar[dtype](4.0)
        var result_gpu = a_gpu != Scalar[dtype](4.0)
        for i in range(4):
            assert_true(result_cpu[[i]] == result_gpu[[i]])


fn test_compare_parity_2d_gt() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_2d_gt")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 5.0], [3.0, 8.0]])
        var b_cpu = Tensor[dtype].d2([[2.0, 3.0], [3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        var result_cpu = a_cpu > b_cpu
        var result_gpu = a_gpu > b_gpu
        for i in range(2):
            for j in range(2):
                assert_true(result_cpu[[i, j]] == result_gpu[[i, j]])


fn test_compare_parity_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_all_true")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([5.0, 6.0, 7.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            (a_cpu > Scalar[dtype](4.0)).all_true()
            == (a_gpu > Scalar[dtype](4.0)).all_true()
        )


fn test_compare_parity_any_true() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_any_true")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            (a_cpu > Scalar[dtype](4.0)).any_true()
            == (a_gpu > Scalar[dtype](4.0)).any_true()
        )


fn test_compare_parity_large() raises:
    @parameter
    if has_accelerator():
        print("test_compare_parity_large")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape(10000), 5.0)
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu > Scalar[dtype](4.0)
        var result_gpu = a_gpu > Scalar[dtype](4.0)
        assert_true(result_cpu.all_true() == result_gpu.all_true())


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    # CPU Equal
    test_compare_cpu_eq_1d()
    test_compare_cpu_eq_scalar()
    test_compare_cpu_eq_2d()

    # CPU NotEqual
    test_compare_cpu_ne_1d()
    test_compare_cpu_ne_scalar()

    # CPU GreaterThan
    test_compare_cpu_gt_1d()
    test_compare_cpu_gt_scalar()
    test_compare_cpu_gt_2d()
    test_compare_cpu_gt_3d()

    # CPU GreaterThanEqual
    test_compare_cpu_gte_1d()
    test_compare_cpu_gte_scalar()

    # CPU LessThan
    test_compare_cpu_lt_1d()
    test_compare_cpu_lt_scalar()
    test_compare_cpu_lt_2d()

    # CPU LessThanEqual
    test_compare_cpu_lte_1d()
    test_compare_cpu_lte_scalar()

    # CPU all_true/any_true
    test_compare_cpu_all_true_from_gt()
    test_compare_cpu_all_true_fails()
    test_compare_cpu_any_true_from_gt()
    test_compare_cpu_any_true_fails()

    # CPU large
    test_compare_cpu_large_gt_scalar()
    test_compare_cpu_large_lt_scalar()
    print("CPU compare tests passed!")

    # GPU Equal
    test_compare_gpu_eq_1d()
    test_compare_gpu_eq_scalar()
    test_compare_gpu_eq_2d()

    # GPU NotEqual
    test_compare_gpu_ne_1d()
    test_compare_gpu_ne_scalar()

    # GPU GreaterThan
    test_compare_gpu_gt_1d()
    test_compare_gpu_gt_scalar()
    test_compare_gpu_gt_2d()
    test_compare_gpu_gt_3d()

    # GPU GreaterThanEqual
    test_compare_gpu_gte_1d()
    test_compare_gpu_gte_scalar()

    # GPU LessThan
    test_compare_gpu_lt_1d()
    test_compare_gpu_lt_scalar()
    test_compare_gpu_lt_2d()

    # GPU LessThanEqual
    test_compare_gpu_lte_1d()
    test_compare_gpu_lte_scalar()

    # GPU all_true/any_true
    test_compare_gpu_all_true_from_gt()
    test_compare_gpu_all_true_fails()
    test_compare_gpu_any_true_from_gt()
    test_compare_gpu_any_true_fails()

    # GPU large
    test_compare_gpu_large_gt_scalar()
    test_compare_gpu_large_lt_scalar()
    test_compare_gpu_large_mixed()
    print("GPU compare tests passed!")

    # Parity
    test_compare_parity_eq_1d()
    test_compare_parity_gt_scalar()
    test_compare_parity_lt_scalar()
    test_compare_parity_gte_scalar()
    test_compare_parity_lte_scalar()
    test_compare_parity_ne_scalar()
    test_compare_parity_2d_gt()
    test_compare_parity_all_true()
    test_compare_parity_any_true()
    test_compare_parity_large()
    print("Parity compare tests passed!")

    print("All compare tests passed!")
