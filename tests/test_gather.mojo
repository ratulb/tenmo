from std.testing import assert_true, assert_false, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from std.sys import has_accelerator

# =============================================================================
# Exhaustive tests for Tensor.gather()
# Prefix: gather_  on all test names to avoid collision with existing tests
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · 1-D tensor
# ─────────────────────────────────────────────────────────────────────────────


fn test_gather_cpu_1d_single_index() raises:
    print("gather_cpu_1d_single_index")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([30.0])))


fn test_gather_cpu_1d_contiguous_range() raises:
    print("gather_cpu_1d_contiguous_range")
    comptime dtype = DType.float32
    # indices [1,2,3] → regular step=1 → view path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(1)
    idx.append(2)
    idx.append(3)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([20.0, 30.0, 40.0])))


fn test_gather_cpu_1d_strided() raises:
    print("gather_cpu_1d_strided")
    comptime dtype = DType.float32
    # indices [0,2,4] → regular step=2 → view path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    idx.append(4)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([10.0, 30.0, 50.0])))


fn test_gather_cpu_1d_irregular() raises:
    print("gather_cpu_1d_irregular")
    comptime dtype = DType.float32
    # indices [0,1,4] → irregular → copy path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(4)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([10.0, 20.0, 50.0])))


fn test_gather_cpu_1d_negative_indices() raises:
    print("gather_cpu_1d_negative_indices")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(-1)
    idx.append(-3)  # → 4, 2
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([50.0, 30.0])))


fn test_gather_cpu_1d_reverse() raises:
    print("test_gather_cpu_1d_reverse")
    comptime dtype = DType.float32
    # indices [4,3,2,1,0] → regular step=-1 → view path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(4)
    idx.append(3)
    idx.append(2)
    idx.append(1)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(Tensor[dtype].d1([50.0, 40.0, 30.0, 20.0, 10.0]))
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · 2-D tensor · axis=0 (row selection)
# ─────────────────────────────────────────────────────────────────────────────


fn test_gather_cpu_2d_axis0_single_row() raises:
    print("gather_cpu_2d_axis0_single_row")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]])))


fn test_gather_cpu_2d_axis0_contiguous() raises:
    print("gather_cpu_2d_axis0_contiguous")
    comptime dtype = DType.float32
    # rows 0,1,2 → view path
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            )
        )
    )


fn test_gather_cpu_2d_axis0_irregular() raises:
    print("gather_cpu_2d_axis0_irregular")
    comptime dtype = DType.float32
    # Original motivating example: rows 0,1,5
    var a = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(5)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [16.0, 17.0, 18.0]]
            )
        )
    )


fn test_gather_cpu_2d_axis0_negative_indices() raises:
    print("gather_cpu_2d_axis0_negative_indices")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(-1)
    idx.append(-3)  # → rows 2, 0
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(Tensor[dtype].d2([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]))
    )


fn test_gather_cpu_2d_axis0_strided() raises:
    print("gather_cpu_2d_axis0_strided")
    comptime dtype = DType.float32
    # rows 0,2 → step=2 → view path
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0], [5.0, 6.0]])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · 2-D tensor · axis=1 (column selection)
# ─────────────────────────────────────────────────────────────────────────────


fn test_gather_cpu_2d_axis1_single_col() raises:
    print("gather_cpu_2d_axis1_single_col")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(2)
    var result = a.gather(idx, axis=1)
    assert_true(result.all_close(Tensor[dtype].d2([[3.0], [6.0], [9.0]])))


fn test_gather_cpu_2d_axis1_contiguous() raises:
    print("gather_cpu_2d_axis1_contiguous")
    comptime dtype = DType.float32
    # cols 0,1 → view path
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    var result = a.gather(idx, axis=1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0], [4.0, 5.0]])))


fn test_gather_cpu_2d_axis1_irregular() raises:
    print("gather_cpu_2d_axis1_irregular")
    comptime dtype = DType.float32
    # cols 0,2 → step=2 → view path (happens to be regular)
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 3.0], [4.0, 6.0]])))


fn test_gather_cpu_2d_axis1_noncontiguous_copy() raises:
    print("gather_cpu_2d_axis1_noncontiguous_copy")
    comptime dtype = DType.float32
    # cols 0,2,3 → irregular → copy path
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    idx.append(3)
    var result = a.gather(idx, axis=1)
    assert_true(
        result.all_close(Tensor[dtype].d2([[1.0, 3.0, 4.0], [5.0, 7.0, 8.0]]))
    )


fn test_gather_cpu_2d_axis1_negative_axis() raises:
    print("gather_cpu_2d_axis1_negative_axis")
    comptime dtype = DType.float32
    # axis=-1 should behave like axis=1 for a 2-D tensor
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=-1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 3.0], [4.0, 6.0]])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · 3-D tensor · all three axes
# ─────────────────────────────────────────────────────────────────────────────


fn test_gather_cpu_3d_axis0_irregular() raises:
    print("gather_cpu_3d_axis0_irregular")
    comptime dtype = DType.float32
    # shape (4,2,2), select slabs 0 and 3
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(3)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [[[1.0, 2.0], [3.0, 4.0]], [[13.0, 14.0], [15.0, 16.0]]]
            )
        )
    )


fn test_gather_cpu_3d_axis1_contiguous() raises:
    print("gather_cpu_3d_axis1_contiguous")
    comptime dtype = DType.float32
    # shape (2,4,2), select rows 1,2 along axis=1 → view path
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        ]
    )
    var idx = IntArray()
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=1)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [[[3.0, 4.0], [5.0, 6.0]], [[11.0, 12.0], [13.0, 14.0]]]
            )
        )
    )


fn test_gather_cpu_3d_axis2_irregular() raises:
    print("gather_cpu_3d_axis2_irregular")
    comptime dtype = DType.float32
    # shape (2,2,4), pick depth indices 0,3 (step=3, regular) → view path
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(3)
    var result = a.gather(idx, axis=2)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [[[1.0, 4.0], [5.0, 8.0]], [[9.0, 12.0], [13.0, 16.0]]]
            )
        )
    )


fn test_gather_cpu_3d_axis2_copy_path() raises:
    print("gather_cpu_3d_axis2_copy_path")
    comptime dtype = DType.float32
    # shape (2,2,5), pick indices 0,2,4 (step=2, but let's use 0,1,4 for copy path)
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
            [[11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0]],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(4)  # irregular → copy
    var result = a.gather(idx, axis=2)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 2.0, 5.0], [6.0, 7.0, 10.0]],
                    [[11.0, 12.0, 15.0], [16.0, 17.0, 20.0]],
                ]
            )
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CPU · Copy path · no grad (irregular indices)
# Irregular gather copies data → grad does NOT flow back through it.
# ─────────────────────────────────────────────────────────────────────────────


fn test_gather_cpu_nograd_copy_2d_irregular() raises:
    print("gather_cpu_nograd_copy_2d_irregular")
    comptime dtype = DType.float32
    # Verify correct VALUES are gathered (no grad check)
    var a = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(5)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [16.0, 17.0, 18.0]]
            )
        )
    )
    # Copy path: result does not require grad
    assert_false(result.requires_grad)


fn test_gather_cpu_nograd_copy_3d_irregular() raises:
    print("gather_cpu_nograd_copy_3d_irregular")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    idx.append(3)  # step varies → copy
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ]
            )
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — GPU · Copy path (irregular) · no grad
# ─────────────────────────────────────────────────────────────────────────────


fn test_gather_gpu_2d_axis0_irregular_copy() raises:
    comptime if has_accelerator():
        print("gather_gpu_2d_axis0_irregular_copy")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(5)  # irregular → copy
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [16.0, 17.0, 18.0]]
                )
            )
        )
        assert_false(result.requires_grad)


fn test_gather_gpu_3d_axis0_irregular_copy() raises:
    comptime if has_accelerator():
        print("gather_gpu_3d_axis0_irregular_copy")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(3)  # step=3 → actually regular! test 0,2,3 for copy
        idx = IntArray()
        idx.append(0)
        idx.append(2)
        idx.append(3)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[9.0, 10.0], [11.0, 12.0]],
                        [[13.0, 14.0], [15.0, 16.0]],
                    ]
                )
            )
        )


fn test_gather_gpu_2d_axis1_irregular_copy() raises:
    comptime if has_accelerator():
        print("gather_gpu_2d_axis1_irregular_copy")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        idx.append(3)  # irregular → copy
        var result = a_gpu.gather(idx, axis=1)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 3.0, 4.0], [5.0, 7.0, 8.0]])
            )
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll gather tests passed!")
