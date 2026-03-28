from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape


# ═════════════════════════════════════════════════════════════════════════════
# CPU all_true Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_alltrue_cpu_1d_all_true() raises:
    print("test_alltrue_cpu_1d_all_true")
    var a = Tensor[DType.bool].d1([True, True, True, True])
    assert_true(a.all_true())


fn test_alltrue_cpu_1d_all_false() raises:
    print("test_alltrue_cpu_1d_all_false")
    var a = Tensor[DType.bool].d1([False, False, False])
    assert_true(not a.all_true())


fn test_alltrue_cpu_1d_mixed() raises:
    print("test_alltrue_cpu_1d_mixed")
    var a = Tensor[DType.bool].d1([True, False, True])
    assert_true(not a.all_true())


fn test_alltrue_cpu_1d_single_true() raises:
    print("test_alltrue_cpu_1d_single_true")
    var a = Tensor[DType.bool].d1([True])
    assert_true(a.all_true())


fn test_alltrue_cpu_1d_single_false() raises:
    print("test_alltrue_cpu_1d_single_false")
    var a = Tensor[DType.bool].d1([False])
    assert_true(not a.all_true())


fn test_alltrue_cpu_1d_last_false() raises:
    print("test_alltrue_cpu_1d_last_false")
    # Only last element is False
    var a = Tensor[DType.bool].d1([True, True, True, False])
    assert_true(not a.all_true())


fn test_alltrue_cpu_1d_first_false() raises:
    print("test_alltrue_cpu_1d_first_false")
    # Only first element is False
    var a = Tensor[DType.bool].d1([False, True, True, True])
    assert_true(not a.all_true())


fn test_alltrue_cpu_2d_all_true() raises:
    print("test_alltrue_cpu_2d_all_true")
    var a = Tensor[DType.bool].d2([[True, True], [True, True]])
    assert_true(a.all_true())


fn test_alltrue_cpu_2d_mixed() raises:
    print("test_alltrue_cpu_2d_mixed")
    var a = Tensor[DType.bool].d2([[True, True], [True, False]])
    assert_true(not a.all_true())


fn test_alltrue_cpu_3d_all_true() raises:
    print("test_alltrue_cpu_3d_all_true")
    var a = Tensor[DType.bool].d3(
        [[[True, True], [True, True]], [[True, True], [True, True]]]
    )
    assert_true(a.all_true())


fn test_alltrue_cpu_3d_one_false() raises:
    print("test_alltrue_cpu_3d_one_false")
    var a = Tensor[DType.bool].d3(
        [[[True, True], [True, True]], [[True, True], [True, False]]]
    )
    assert_true(not a.all_true())


fn test_alltrue_cpu_from_comparison() raises:
    print("test_alltrue_cpu_from_comparison")
    comptime dtype = DType.float32
    # Generate bool tensor from comparison
    var a = Tensor[dtype].d1([5.0, 6.0, 7.0])
    var mask = a > Tensor[dtype].full(Shape(3), 4.0)
    assert_true(mask.all_true())


fn test_alltrue_cpu_from_comparison_false() raises:
    print("test_alltrue_cpu_from_comparison_false")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([3.0, 6.0, 7.0])
    var mask = a > Tensor[dtype].full(Shape(3), 4.0)
    assert_true(not mask.all_true())


fn test_alltrue_cpu_large_tensor() raises:
    print("test_alltrue_cpu_large_tensor")
    # Large tensor — all True
    var a = Tensor[DType.bool].full(Shape(10000), True)
    assert_true(a.all_true())


fn test_alltrue_cpu_large_tensor_one_false() raises:
    print("test_alltrue_cpu_large_tensor_one_false")
    # Large tensor with one False in middle
    var a = Tensor[DType.bool].full(Shape(10000), True)
    a[[5000]] = False
    assert_true(not a.all_true())


# ═════════════════════════════════════════════════════════════════════════════
# CPU any_true Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_anytrue_cpu_1d_all_true() raises:
    print("test_anytrue_cpu_1d_all_true")
    var a = Tensor[DType.bool].d1([True, True, True])
    assert_true(a.any_true())


fn test_anytrue_cpu_1d_all_false() raises:
    print("test_anytrue_cpu_1d_all_false")
    var a = Tensor[DType.bool].d1([False, False, False])
    assert_true(not a.any_true())


fn test_anytrue_cpu_1d_mixed() raises:
    print("test_anytrue_cpu_1d_mixed")
    var a = Tensor[DType.bool].d1([False, True, False])
    assert_true(a.any_true())


fn test_anytrue_cpu_1d_single_true() raises:
    print("test_anytrue_cpu_1d_single_true")
    var a = Tensor[DType.bool].d1([True])
    assert_true(a.any_true())


fn test_anytrue_cpu_1d_single_false() raises:
    print("test_anytrue_cpu_1d_single_false")
    var a = Tensor[DType.bool].d1([False])
    assert_true(not a.any_true())


fn test_anytrue_cpu_1d_only_last_true() raises:
    print("test_anytrue_cpu_1d_only_last_true")
    var a = Tensor[DType.bool].d1([False, False, False, True])
    assert_true(a.any_true())


fn test_anytrue_cpu_1d_only_first_true() raises:
    print("test_anytrue_cpu_1d_only_first_true")
    var a = Tensor[DType.bool].d1([True, False, False, False])
    assert_true(a.any_true())


fn test_anytrue_cpu_2d_all_false() raises:
    print("test_anytrue_cpu_2d_all_false")
    var a = Tensor[DType.bool].d2([[False, False], [False, False]])
    assert_true(not a.any_true())


fn test_anytrue_cpu_2d_one_true() raises:
    print("test_anytrue_cpu_2d_one_true")
    var a = Tensor[DType.bool].d2([[False, False], [False, True]])
    assert_true(a.any_true())


fn test_anytrue_cpu_3d_all_false() raises:
    print("test_anytrue_cpu_3d_all_false")
    var a = Tensor[DType.bool].d3(
        [[[False, False], [False, False]], [[False, False], [False, False]]]
    )
    assert_true(not a.any_true())


fn test_anytrue_cpu_3d_one_true() raises:
    print("test_anytrue_cpu_3d_one_true")
    var a = Tensor[DType.bool].d3(
        [[[False, False], [False, False]], [[False, False], [False, True]]]
    )
    assert_true(a.any_true())


fn test_anytrue_cpu_from_comparison() raises:
    print("test_anytrue_cpu_from_comparison")
    comptime dtype = DType.float32
    # Only one element satisfies condition
    var a = Tensor[dtype].d1([1.0, 2.0, 5.0])
    var mask = a > Tensor[dtype].full(Shape(3), 4.0)
    assert_true(mask.any_true())


fn test_anytrue_cpu_from_comparison_none() raises:
    print("test_anytrue_cpu_from_comparison_none")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var mask = a > Tensor[dtype].full(Shape(3), 4.0)
    assert_true(not mask.any_true())


fn test_anytrue_cpu_large_all_false() raises:
    print("test_anytrue_cpu_large_all_false")
    var a = Tensor[DType.bool].full(Shape(10000), False)
    assert_true(not a.any_true())


fn test_anytrue_cpu_large_one_true() raises:
    print("test_anytrue_cpu_large_one_true")
    var a = Tensor[DType.bool].full(Shape(10000), False)
    a[[5000]] = True
    assert_true(a.any_true())


# ═════════════════════════════════════════════════════════════════════════════
# GPU all_true Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_alltrue_gpu_1d_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_1d_all_true")
        var a = Tensor[DType.bool].d1([True, True, True, True]).to_gpu()
        assert_true(a.all_true())


fn test_alltrue_gpu_1d_all_false() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_1d_all_false")
        var a = Tensor[DType.bool].d1([False, False, False]).to_gpu()
        assert_true(not a.all_true())


fn test_alltrue_gpu_1d_mixed() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_1d_mixed")
        var a = Tensor[DType.bool].d1([True, False, True]).to_gpu()
        assert_true(not a.all_true())


fn test_alltrue_gpu_1d_single_true() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_1d_single_true")
        var a = Tensor[DType.bool].d1([True]).to_gpu()
        assert_true(a.all_true())


fn test_alltrue_gpu_1d_single_false() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_1d_single_false")
        var a = Tensor[DType.bool].d1([False]).to_gpu()
        assert_true(not a.all_true())


fn test_alltrue_gpu_1d_last_false() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_1d_last_false")
        var a = Tensor[DType.bool].d1([True, True, True, False]).to_gpu()
        assert_true(not a.all_true())


fn test_alltrue_gpu_2d_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_2d_all_true")
        var a = Tensor[DType.bool].d2([[True, True], [True, True]]).to_gpu()
        assert_true(a.all_true())


fn test_alltrue_gpu_2d_mixed() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_2d_mixed")
        var a = Tensor[DType.bool].d2([[True, True], [True, False]]).to_gpu()
        assert_true(not a.all_true())


fn test_alltrue_gpu_3d_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_3d_all_true")
        var a = (
            Tensor[DType.bool]
            .d3([[[True, True], [True, True]], [[True, True], [True, True]]])
            .to_gpu()
        )
        assert_true(a.all_true())


fn test_alltrue_gpu_3d_one_false() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_3d_one_false")
        var a = (
            Tensor[DType.bool]
            .d3([[[True, True], [True, True]], [[True, True], [True, False]]])
            .to_gpu()
        )
        assert_true(not a.all_true())


fn test_alltrue_gpu_from_comparison() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_from_comparison")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(mask.all_true())


fn test_alltrue_gpu_from_comparison_false() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_from_comparison_false")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 6.0, 7.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(not mask.all_true())


fn test_alltrue_gpu_large_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_large_all_true")
        var a = Tensor[DType.bool].full(Shape(10000), True).to_gpu()
        assert_true(a.all_true())


fn test_alltrue_gpu_large_one_false() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_gpu_large_one_false")
        var a = Tensor[DType.bool].full(Shape(10000), True)
        a[[5000]] = False
        var a_gpu = a.to_gpu()
        assert_true(not a_gpu.all_true())


# ═════════════════════════════════════════════════════════════════════════════
# GPU any_true Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_anytrue_gpu_1d_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_1d_all_true")
        var a = Tensor[DType.bool].d1([True, True, True]).to_gpu()
        assert_true(a.any_true())


fn test_anytrue_gpu_1d_all_false() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_1d_all_false")
        var a = Tensor[DType.bool].d1([False, False, False]).to_gpu()
        assert_true(not a.any_true())


fn test_anytrue_gpu_1d_mixed() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_1d_mixed")
        var a = Tensor[DType.bool].d1([False, True, False]).to_gpu()
        assert_true(a.any_true())


fn test_anytrue_gpu_1d_single_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_1d_single_true")
        var a = Tensor[DType.bool].d1([True]).to_gpu()
        assert_true(a.any_true())


fn test_anytrue_gpu_1d_single_false() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_1d_single_false")
        var a = Tensor[DType.bool].d1([False]).to_gpu()
        assert_true(not a.any_true())


fn test_anytrue_gpu_1d_only_last_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_1d_only_last_true")
        var a = Tensor[DType.bool].d1([False, False, False, True]).to_gpu()
        assert_true(a.any_true())


fn test_anytrue_gpu_2d_all_false() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_2d_all_false")
        var a = Tensor[DType.bool].d2([[False, False], [False, False]]).to_gpu()
        assert_true(not a.any_true())


fn test_anytrue_gpu_2d_one_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_2d_one_true")
        var a = Tensor[DType.bool].d2([[False, False], [False, True]]).to_gpu()
        assert_true(a.any_true())


fn test_anytrue_gpu_3d_all_false() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_3d_all_false")
        var a = (
            Tensor[DType.bool]
            .d3(
                [
                    [[False, False], [False, False]],
                    [[False, False], [False, False]],
                ]
            )
            .to_gpu()
        )
        assert_true(not a.any_true())


fn test_anytrue_gpu_3d_one_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_3d_one_true")
        var a = (
            Tensor[DType.bool]
            .d3(
                [
                    [[False, False], [False, False]],
                    [[False, False], [False, True]],
                ]
            )
            .to_gpu()
        )
        assert_true(a.any_true())


fn test_anytrue_gpu_from_comparison() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_from_comparison")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 5.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(mask.any_true())


fn test_anytrue_gpu_from_comparison_none() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_from_comparison_none")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(not mask.any_true())


fn test_anytrue_gpu_large_all_false() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_large_all_false")
        var a = Tensor[DType.bool].full(Shape(10000), False).to_gpu()
        assert_true(not a.any_true())


fn test_anytrue_gpu_large_one_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_gpu_large_one_true")
        var a = Tensor[DType.bool].full(Shape(10000), False)
        a[[5000]] = True
        var a_gpu = a.to_gpu()
        assert_true(a_gpu.any_true())


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_alltrue_parity_all_true() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_parity_all_true")
        var a_cpu = Tensor[DType.bool].d2([[True, True], [True, True]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.all_true() == a_gpu.all_true())


fn test_alltrue_parity_mixed() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_parity_mixed")
        var a_cpu = Tensor[DType.bool].d2([[True, False], [True, True]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.all_true() == a_gpu.all_true())


fn test_anytrue_parity_all_false() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_parity_all_false")
        var a_cpu = Tensor[DType.bool].d2([[False, False], [False, False]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.any_true() == a_gpu.any_true())


fn test_anytrue_parity_one_true() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_parity_one_true")
        var a_cpu = Tensor[DType.bool].d2([[False, False], [False, True]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.any_true() == a_gpu.any_true())


fn test_alltrue_parity_from_comparison() raises:
    @parameter
    if has_accelerator():
        print("test_alltrue_parity_from_comparison")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([5.0, 6.0, 7.0])
        var a_gpu = a_cpu.to_gpu()
        var threshold_cpu = Tensor[dtype].full(Shape(3), 4.0)
        var threshold_gpu = threshold_cpu.to_gpu()
        assert_true(
            (a_cpu > threshold_cpu).all_true()
            == (a_gpu > threshold_gpu).all_true()
        )


fn test_anytrue_parity_from_comparison() raises:
    @parameter
    if has_accelerator():
        print("test_anytrue_parity_from_comparison")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var threshold_cpu = Tensor[dtype].full(Shape(3), 4.0)
        var threshold_gpu = threshold_cpu.to_gpu()
        assert_true(
            (a_cpu > threshold_cpu).any_true()
            == (a_gpu > threshold_gpu).any_true()
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    # CPU all_true
    test_alltrue_cpu_1d_all_true()
    test_alltrue_cpu_1d_all_false()
    test_alltrue_cpu_1d_mixed()
    test_alltrue_cpu_1d_single_true()
    test_alltrue_cpu_1d_single_false()
    test_alltrue_cpu_1d_last_false()
    test_alltrue_cpu_1d_first_false()
    test_alltrue_cpu_2d_all_true()
    test_alltrue_cpu_2d_mixed()
    test_alltrue_cpu_3d_all_true()
    test_alltrue_cpu_3d_one_false()
    test_alltrue_cpu_from_comparison()
    test_alltrue_cpu_from_comparison_false()
    test_alltrue_cpu_large_tensor()
    test_alltrue_cpu_large_tensor_one_false()
    print("CPU all_true passed!")

    # CPU any_true
    test_anytrue_cpu_1d_all_true()
    test_anytrue_cpu_1d_all_false()
    test_anytrue_cpu_1d_mixed()
    test_anytrue_cpu_1d_single_true()
    test_anytrue_cpu_1d_single_false()
    test_anytrue_cpu_1d_only_last_true()
    test_anytrue_cpu_1d_only_first_true()
    test_anytrue_cpu_2d_all_false()
    test_anytrue_cpu_2d_one_true()
    test_anytrue_cpu_3d_all_false()
    test_anytrue_cpu_3d_one_true()
    test_anytrue_cpu_from_comparison()
    test_anytrue_cpu_from_comparison_none()
    test_anytrue_cpu_large_all_false()
    test_anytrue_cpu_large_one_true()
    print("CPU any_true passed!")

    # GPU all_true
    test_alltrue_gpu_1d_all_true()
    test_alltrue_gpu_1d_all_false()
    test_alltrue_gpu_1d_mixed()
    test_alltrue_gpu_1d_single_true()
    test_alltrue_gpu_1d_single_false()
    test_alltrue_gpu_1d_last_false()
    test_alltrue_gpu_2d_all_true()
    test_alltrue_gpu_2d_mixed()
    test_alltrue_gpu_3d_all_true()
    test_alltrue_gpu_3d_one_false()
    test_alltrue_gpu_from_comparison()
    test_alltrue_gpu_from_comparison_false()
    test_alltrue_gpu_large_all_true()
    test_alltrue_gpu_large_one_false()
    print("GPU all_true passed!")

    # GPU any_true
    test_anytrue_gpu_1d_all_true()
    test_anytrue_gpu_1d_all_false()
    test_anytrue_gpu_1d_mixed()
    test_anytrue_gpu_1d_single_true()
    test_anytrue_gpu_1d_single_false()
    test_anytrue_gpu_1d_only_last_true()
    test_anytrue_gpu_2d_all_false()
    test_anytrue_gpu_2d_one_true()
    test_anytrue_gpu_3d_all_false()
    test_anytrue_gpu_3d_one_true()
    test_anytrue_gpu_from_comparison()
    test_anytrue_gpu_from_comparison_none()
    test_anytrue_gpu_large_all_false()
    test_anytrue_gpu_large_one_true()
    print("GPU any_true passed!")

    # Parity
    test_alltrue_parity_all_true()
    test_alltrue_parity_mixed()
    test_anytrue_parity_all_false()
    test_anytrue_parity_one_true()
    test_alltrue_parity_from_comparison()
    test_anytrue_parity_from_comparison()
    print("Parity passed!")

    print("All all_true/any_true tests passed!")
