from std.testing import assert_true, assert_false, assert_equal, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.common_utils import i, s
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


# =============================================================================
# SECTION 9 — MEMCPY FAST PATH: rank==2, axis=0, unit column stride
# Prefix: mcpy_ to avoid name collision with existing gather_ tests
# =============================================================================

# ── CPU / Forward ─────────────────────────────────────────────────────────────

fn test_mcpy_cpu_2d_single_row_first() raises:
    print("mcpy_cpu_2d_single_row_first")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    var idx = IntArray()
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0]])))


fn test_mcpy_cpu_2d_single_row_last() raises:
    print("mcpy_cpu_2d_single_row_last")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    var idx = IntArray()
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[7.0, 8.0, 9.0]])))


fn test_mcpy_cpu_2d_single_row_middle() raises:
    print("mcpy_cpu_2d_single_row_middle")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    var idx = IntArray()
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]])))


fn test_mcpy_cpu_2d_multi_row_ordered() raises:
    print("mcpy_cpu_2d_multi_row_ordered")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))


fn test_mcpy_cpu_2d_multi_row_reversed() raises:
    print("mcpy_cpu_2d_multi_row_reversed")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(2)
    idx.append(1)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])))


fn test_mcpy_cpu_2d_duplicate_indices() raises:
    print("mcpy_cpu_2d_duplicate_indices")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(1)
    idx.append(1)
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])))


fn test_mcpy_cpu_2d_all_rows_identity() raises:
    print("mcpy_cpu_2d_all_rows_identity")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(a))


fn test_mcpy_cpu_2d_wide_cols() raises:
    print("mcpy_cpu_2d_wide_cols")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    ])
    var idx = IntArray()
    idx.append(2)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([
        [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ])))


fn test_mcpy_cpu_2d_single_col() raises:
    print("mcpy_cpu_2d_single_col")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0], [4.0]])
    var idx = IntArray()
    idx.append(3)
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0], [2.0]])))


fn test_mcpy_cpu_2d_float64() raises:
    print("mcpy_cpu_2d_float64")
    comptime dtype = DType.float64
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(1)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])))


fn test_mcpy_cpu_2d_float16() raises:
    print("mcpy_cpu_2d_float16")
    comptime dtype = DType.float16
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close[atol=1e-2](Tensor[dtype].d2([[1.0, 2.0], [5.0, 6.0]])))


fn test_mcpy_cpu_2d_more_output_rows_than_input() raises:
    print("mcpy_cpu_2d_more_output_rows_than_input")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(0)
    idx.append(0)
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([
        [10.0, 20.0], [30.0, 40.0], [10.0, 20.0], [10.0, 20.0], [30.0, 40.0]
    ])))


# ── CPU / Grad flow ───────────────────────────────────────────────────────────

fn test_mcpy_cpu_2d_no_grad_flows_to_source() raises:
    print("mcpy_cpu_2d_no_grad_flows_to_source")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


fn test_mcpy_cpu_2d_result_requires_grad() raises:
    print("mcpy_cpu_2d_result_requires_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var idx = IntArray()
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_equal(result.requires_grad, True)


fn test_mcpy_cpu_2d_grad_through_downstream_op() raises:
    print("mcpy_cpu_2d_grad_through_downstream_op")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var b = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    var gathered = a.gather(idx, axis=0)
    var out = gathered + b
    var loss = out.sum()
    loss.backward()
    assert_true(b.grad().all_close(Tensor[dtype].ones_like(b)))
    var grad = a.grad().detach(share=True)
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))


# ── CPU / fuse_sum ────────────────────────────────────────────────────────────

fn test_mcpy_cpu_2d_fuse_sum_basic() raises:
    print("mcpy_cpu_2d_fuse_sum_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, fuse_sum=True)
    assert_true(result.all_close(Tensor[dtype].d1([8.0, 10.0, 12.0])))


fn test_mcpy_cpu_2d_fuse_sum_single_index() raises:
    print("mcpy_cpu_2d_fuse_sum_single_index")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(1)
    var result = a.gather(idx, axis=0, fuse_sum=True)
    assert_true(result.all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


fn test_mcpy_cpu_2d_fuse_sum_all_rows() raises:
    print("mcpy_cpu_2d_fuse_sum_all_rows")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0, fuse_sum=True)
    assert_true(result.all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_mcpy_cpu_2d_fuse_sum_duplicate_indices() raises:
    print("mcpy_cpu_2d_fuse_sum_duplicate_indices")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(0)
    idx.append(0)
    var result = a.gather(idx, axis=0, fuse_sum=True)
    assert_true(result.all_close(Tensor[dtype].d1([3.0, 6.0])))


fn test_mcpy_cpu_2d_fuse_sum_no_grad_flows() raises:
    print("mcpy_cpu_2d_fuse_sum_no_grad_flows")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, fuse_sum=True)
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


# ── GPU / Forward ─────────────────────────────────────────────────────────────

fn test_mcpy_gpu_2d_single_row() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_single_row")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(1)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]])))


fn test_mcpy_gpu_2d_multi_row_reversed() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_multi_row_reversed")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(2)
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(
            Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
        ))


fn test_mcpy_gpu_2d_duplicate_indices() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_duplicate_indices")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(0)
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [10.0, 20.0], [10.0, 20.0], [30.0, 40.0], [10.0, 20.0]
        ])))


fn test_mcpy_gpu_2d_wide_cols() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_wide_cols")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ])))


fn test_mcpy_gpu_2d_single_col() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_single_col")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0], [4.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(3)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([[4.0], [1.0]])))


fn test_mcpy_gpu_2d_all_rows_identity() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_all_rows_identity")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(a))


# ── GPU / Grad flow ───────────────────────────────────────────────────────────

fn test_mcpy_gpu_2d_no_grad_flows_to_source() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_no_grad_flows_to_source")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0)
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


fn test_mcpy_gpu_2d_result_requires_grad() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_result_requires_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_equal(result.requires_grad, True)


fn test_mcpy_gpu_2d_grad_through_downstream_op() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_grad_through_downstream_op")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var b = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
        var b_gpu = b.to_gpu()
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        var gathered = a_gpu.gather(idx, axis=0)
        var out = gathered + b_gpu
        var loss = out.sum()
        loss.backward()
        assert_true(b.grad().all_close(Tensor[dtype].ones_like(b)))
        var grad = a.grad().detach(share=True)
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))


# ── GPU / fuse_sum ────────────────────────────────────────────────────────────

fn test_mcpy_gpu_2d_fuse_sum_basic() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_fuse_sum_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, fuse_sum=True)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([8.0, 10.0, 12.0])))


fn test_mcpy_gpu_2d_fuse_sum_duplicate_indices() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_fuse_sum_duplicate_indices")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(0)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0, fuse_sum=True)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([3.0, 6.0])))


fn test_mcpy_gpu_2d_fuse_sum_no_grad_flows() raises:
    comptime if has_accelerator():
        print("mcpy_gpu_2d_fuse_sum_no_grad_flows")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, fuse_sum=True)
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


# ── CPU / Higher dimensions ───────────────────────────────────────────────────

fn test_mcpy_cpu_3d_axis0_scalar_path_consistency() raises:
    print("mcpy_cpu_3d_axis0_scalar_path_consistency")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    ).reshape(3, 2, 2)
    var idx = IntArray()
    idx.append(2)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    var expected = Tensor[dtype].d1(
        [9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]
    ).reshape(2, 2, 2)
    assert_true(result.all_close(expected))


fn test_mcpy_cpu_3d_axis1_scalar_path_consistency() raises:
    print("mcpy_cpu_3d_axis1_scalar_path_consistency")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    ).reshape(2, 3, 2)
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1)
    var expected = Tensor[dtype].d1(
        [1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0]
    ).reshape(2, 2, 2)
    assert_true(result.all_close(expected))


# ── CPU / CPU↔GPU value parity ────────────────────────────────────────────────

fn test_mcpy_cpu_gpu_parity_2d_axis0() raises:
    comptime if has_accelerator():
        print("mcpy_cpu_gpu_parity_2d_axis0")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ])
        var idx = IntArray()
        idx.append(3)
        idx.append(1)
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0)
        var gpu_result = a.to_gpu().gather(idx, axis=0).to_cpu()
        assert_true(cpu_result.all_close(gpu_result))


fn test_mcpy_cpu_gpu_parity_2d_fuse_sum() raises:
    comptime if has_accelerator():
        print("mcpy_cpu_gpu_parity_2d_fuse_sum")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, fuse_sum=True)
        var gpu_result = a.to_gpu().gather(idx, axis=0, fuse_sum=True).to_cpu()
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll gather tests passed!")
