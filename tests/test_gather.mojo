from std.testing import assert_true, assert_false, assert_equal, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.common_utils import i, s
from std.sys import has_accelerator
from tenmo.shared import Reduction

# =============================================================================
# Exhaustive tests for Tensor.gather()
# Prefix: gather_  on all test names to avoid collision with existing tests
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · 1-D tensor
# ─────────────────────────────────────────────────────────────────────────────


def test_gather_cpu_1d_single_index() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([30.0])))


def test_gather_cpu_1d_contiguous_range() raises:
    comptime dtype = DType.float32
    # indices [1,2,3] → regular step=1 → view path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(1)
    idx.append(2)
    idx.append(3)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([20.0, 30.0, 40.0])))


def test_gather_cpu_1d_strided() raises:
    comptime dtype = DType.float32
    # indices [0,2,4] → regular step=2 → view path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    idx.append(4)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([10.0, 30.0, 50.0])))


def test_gather_cpu_1d_irregular() raises:
    comptime dtype = DType.float32
    # indices [0,1,4] → irregular → copy path
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(4)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([10.0, 20.0, 50.0])))


def test_gather_cpu_1d_negative_indices() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0])
    var idx = IntArray()
    idx.append(-1)
    idx.append(-3)  # → 4, 2
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d1([50.0, 30.0])))


def test_gather_cpu_1d_reverse() raises:
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


def test_gather_cpu_2d_axis0_single_row() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]])))


def test_gather_cpu_2d_axis0_contiguous() raises:
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


def test_gather_cpu_2d_axis0_irregular() raises:
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


def test_gather_cpu_2d_axis0_negative_indices() raises:
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


def test_gather_cpu_2d_axis0_strided() raises:
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


def test_gather_cpu_2d_axis1_single_col() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(2)
    var result = a.gather(idx, axis=1)
    assert_true(result.all_close(Tensor[dtype].d2([[3.0], [6.0], [9.0]])))


def test_gather_cpu_2d_axis1_contiguous() raises:
    comptime dtype = DType.float32
    # cols 0,1 → view path
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    var result = a.gather(idx, axis=1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0], [4.0, 5.0]])))


def test_gather_cpu_2d_axis1_irregular() raises:
    comptime dtype = DType.float32
    # cols 0,2 → step=2 → view path (happens to be regular)
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 3.0], [4.0, 6.0]])))


def test_gather_cpu_2d_axis1_noncontiguous_copy() raises:
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


def test_gather_cpu_2d_axis1_negative_axis() raises:
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


def test_gather_cpu_3d_axis0_irregular() raises:
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


def test_gather_cpu_3d_axis1_contiguous() raises:
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


def test_gather_cpu_3d_axis2_irregular() raises:
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


def test_gather_cpu_3d_axis2_copy_path() raises:
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


def test_gather_cpu_nograd_copy_2d_irregular() raises:
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


def test_gather_cpu_nograd_copy_3d_irregular() raises:
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


def test_gather_gpu_2d_axis0_irregular_copy() raises:
    comptime if has_accelerator():
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


def test_gather_gpu_3d_axis0_irregular_copy() raises:
    comptime if has_accelerator():
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


def test_gather_gpu_2d_axis1_irregular_copy() raises:
    comptime if has_accelerator():
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


def test_mcpy_cpu_2d_single_row_first() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0]])))


def test_mcpy_cpu_2d_single_row_last() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[7.0, 8.0, 9.0]])))


def test_mcpy_cpu_2d_single_row_middle() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]])))


def test_mcpy_cpu_2d_multi_row_ordered() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    )


def test_mcpy_cpu_2d_multi_row_reversed() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(2)
    idx.append(1)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]))
    )


def test_mcpy_cpu_2d_duplicate_indices() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(1)
    idx.append(1)
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(Tensor[dtype].d2([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]))
    )


def test_mcpy_cpu_2d_all_rows_identity() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(a))


def test_mcpy_cpu_2d_wide_cols() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        ]
    )
    var idx = IntArray()
    idx.append(2)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ]
            )
        )
    )


def test_mcpy_cpu_2d_single_col() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0], [4.0]])
    var idx = IntArray()
    idx.append(3)
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(result.all_close(Tensor[dtype].d2([[4.0], [2.0]])))


def test_mcpy_cpu_2d_float64() raises:
    comptime dtype = DType.float64
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(1)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]))
    )


def test_mcpy_cpu_2d_float16() raises:
    comptime dtype = DType.float16
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close[atol=1e-2](Tensor[dtype].d2([[1.0, 2.0], [5.0, 6.0]]))
    )


def test_mcpy_cpu_2d_more_output_rows_than_input() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[10.0, 20.0], [30.0, 40.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(0)
    idx.append(0)
    idx.append(1)
    var result = a.gather(idx, axis=0)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                    [10.0, 20.0],
                    [10.0, 20.0],
                    [30.0, 40.0],
                ]
            )
        )
    )


# ── CPU / Grad flow ───────────────────────────────────────────────────────────


def test_mcpy_cpu_2d_no_grad_flows_to_source() raises:
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


def test_mcpy_cpu_2d_result_requires_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var idx = IntArray()
    idx.append(0)
    var result = a.gather(idx, axis=0)
    assert_equal(result.requires_grad, True)


def test_mcpy_cpu_2d_grad_through_downstream_op() raises:
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


def test_mcpy_cpu_2d_fuse_sum_basic() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    assert_true(result.all_close(Tensor[dtype].d1([8.0, 10.0, 12.0])))


def test_mcpy_cpu_2d_fuse_sum_single_index() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(1)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    assert_true(result.all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


def test_mcpy_cpu_2d_fuse_sum_all_rows() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    assert_true(result.all_close(Tensor[dtype].d1([6.0, 6.0])))


def test_mcpy_cpu_2d_fuse_sum_duplicate_indices() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(0)
    idx.append(0)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    assert_true(result.all_close(Tensor[dtype].d1([3.0, 6.0])))


def test_mcpy_cpu_2d_fuse_sum_no_grad_flows() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


# ── CPU / MEAN forward ──────────────────────────────────────────────────────────


def test_mcpy_cpu_2d_fuse_mean_basic() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(1)
    var result = a.gather(idx, axis=0, reduction=Reduction(0))
    # mean of rows 0 and 1: [2.5, 3.5, 4.5]
    assert_true(result.shape() == Shape(3))
    assert_true(result.all_close(Tensor[dtype].d1([2.5, 3.5, 4.5])))


def test_mcpy_cpu_2d_fuse_mean_single_index() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    var result = a.gather(idx, axis=0, reduction=Reduction(0))
    # mean of single row 0: [1.0, 2.0, 3.0]
    assert_true(result.shape() == Shape(3))
    assert_true(result.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_mcpy_cpu_2d_fuse_mean_duplicate_indices() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(0)
    idx.append(1)
    var result = a.gather(idx, axis=0, reduction=Reduction(0))
    # mean of rows 0,0,1: [(10+10+30)/3, (20+20+40)/3] = [50/3, 80/3]
    assert_true(result.shape() == Shape(2))
    assert_true(result.all_close(Tensor[dtype].d1([50.0 / 3.0, 80.0 / 3.0])))


def test_mcpy_cpu_2d_fuse_mean_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(0))
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    # MEAN backward: grad = upstream / n_indices = 1.0 / 2 = 0.5
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([0.5, 0.5])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0.5, 0.5])))


# ── GPU / Forward ─────────────────────────────────────────────────────────────


def test_mcpy_gpu_2d_single_row() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(1)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]]))
        )


def test_mcpy_gpu_2d_multi_row_reversed() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(2)
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
            )
        )


def test_mcpy_gpu_2d_duplicate_indices() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            ]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    ]
                )
            )
        )


def test_mcpy_gpu_2d_single_col() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0], [4.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(3)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([[4.0], [1.0]])))


def test_mcpy_gpu_2d_all_rows_identity() raises:
    comptime if has_accelerator():
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


def test_mcpy_gpu_2d_no_grad_flows_to_source() raises:
    comptime if has_accelerator():
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


def test_mcpy_gpu_2d_result_requires_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_equal(result.requires_grad, True)


def test_mcpy_gpu_2d_grad_through_downstream_op() raises:
    comptime if has_accelerator():
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


def test_mcpy_gpu_2d_fuse_sum_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(1))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([8.0, 10.0, 12.0]))
        )


def test_mcpy_gpu_2d_fuse_sum_duplicate_indices() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(0)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(1))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([3.0, 6.0])))


def test_mcpy_gpu_2d_fuse_sum_no_grad_flows() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(1))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


# ── GPU / MEAN forward ──────────────────────────────────────────────────────────


def test_mcpy_gpu_2d_fuse_mean_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        assert_true(result.shape() == Shape(3))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([2.5, 3.5, 4.5]))
        )


def test_mcpy_gpu_2d_fuse_mean_duplicate_indices() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2(
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ]
            )
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(0)
        idx.append(1)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        assert_true(result.shape() == Shape(2))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([50.0 / 3.0, 80.0 / 3.0])
            )
        )


def test_mcpy_gpu_2d_fuse_mean_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(0))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([0.5, 0.5])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0.5, 0.5])))


# ── CPU / MEAN / CPU–GPU parity ───────────────────────────────────────────────


def test_mcpy_cpu_gpu_parity_2d_fuse_mean() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(0))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(0)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


# ── CPU / Higher dimensions ───────────────────────────────────────────────────


def test_mcpy_cpu_3d_axis0_scalar_path_consistency() raises:
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    var a = _tmp0.reshape(3, 2, 2)
    var idx = IntArray()
    idx.append(2)
    idx.append(0)
    var result = a.gather(idx, axis=0)
    var _tmp1 = Tensor[dtype].d1([9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0])
    var expected = _tmp1.reshape(2, 2, 2)
    assert_true(result.all_close(expected))


def test_mcpy_cpu_3d_axis1_scalar_path_consistency() raises:
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    var a = _tmp0.reshape(2, 3, 2)
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1)
    var _tmp1 = Tensor[dtype].d1([1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0])
    var expected = _tmp1.reshape(2, 2, 2)
    assert_true(result.all_close(expected))


# ── CPU / CPU↔GPU value parity ────────────────────────────────────────────────


def test_mcpy_cpu_gpu_parity_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
        var idx = IntArray()
        idx.append(3)
        idx.append(1)
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0)
        var gpu_result = a.to_gpu().gather(idx, axis=0).to_cpu()
        assert_true(cpu_result.all_close(gpu_result))


def test_mcpy_cpu_gpu_parity_2d_fuse_sum() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(1))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(1)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


# ─── General-case reduction / 3D axis0 SUM forward ────────────


def test_mcpy_cpu_3d_axis0_sum_general() raises:
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].d4(
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]]
    )
    var a = _tmp0.reshape(3, 2, 2)
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    # SUM of rows 0 and 2: [[10, 12], [14, 16]]
    assert_true(result.shape() == Shape(2, 2))
    var _tmp1 = Tensor[dtype].d1([10, 12, 14, 16])
    var expected = _tmp1.reshape(2, 2)
    assert_true(result.all_close(expected))


# ─── General-case reduction / 3D axis0 MEAN forward ───────────


def test_mcpy_cpu_3d_axis0_mean_general() raises:
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    var a = _tmp0.reshape(3, 2, 2)
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(0))
    # MEAN of rows 0 and 2: [[5, 6], [7, 8]]
    assert_true(result.shape() == Shape(2, 2))
    var _tmp1 = Tensor[dtype].d1([5, 6, 7, 8])
    var expected = _tmp1.reshape(2, 2)
    assert_true(result.all_close(expected))


# ─── General-case reduction / 2D axis1 SUM forward ────────────


def test_mcpy_cpu_2d_axis1_sum_general() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1, reduction=Reduction(1))
    # gather cols 0,2 → [[1,3],[4,6]] → sum axis=1 → [4, 10]
    assert_true(result.shape() == Shape(2))
    assert_true(result.all_close(Tensor[dtype].d1([4.0, 10.0])))


# ─── General-case reduction / 2D axis1 MEAN forward ───────────


def test_mcpy_cpu_2d_axis1_mean_general() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1, reduction=Reduction(0))
    # gather cols 0,2 → [[1,3],[4,6]] → mean axis=1 → [2, 5]
    assert_true(result.shape() == Shape(2))
    assert_true(result.all_close(Tensor[dtype].d1([2.0, 5.0])))


# ─── General-case reduction / 3D axis0 SUM backward ───────────


def test_mcpy_cpu_3d_axis0_sum_general_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(1))
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    # SUM backward: each gathered row gets d_output
    # d_output = ones(2,2), scattered to rows 0 and 2
    var _tmp0 = Tensor[dtype].d1(
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    )
    var expected_grad = _tmp0.reshape(3, 2, 2)
    assert_true(grad.all_close(expected_grad))


# ─── General-case reduction / 3D axis0 MEAN backward ──────────


def test_mcpy_cpu_3d_axis0_mean_general_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=0, reduction=Reduction(0))
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    # MEAN backward: gradient scaled by 1/n_indices = 1/2
    var _tmp1 = Tensor[dtype].d1(
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
    )
    var expected_grad = _tmp1.reshape(3, 2, 2)
    assert_true(grad.all_close(expected_grad))


# ─── General-case reduction / 2D axis1 SUM backward ───────────


def test_mcpy_cpu_2d_axis1_sum_general_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1, reduction=Reduction(1))
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    # SUM backward: each gathered column gets d_output[row]
    # d_output = [1, 1]; cols 0 and 2 get 1.0
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 0.0, 1.0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 0.0, 1.0])))


# ─── General-case reduction / 2D axis1 MEAN backward ──────────


def test_mcpy_cpu_2d_axis1_mean_general_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var idx = IntArray()
    idx.append(0)
    idx.append(2)
    var result = a.gather(idx, axis=1, reduction=Reduction(0))
    var loss = result.sum()
    loss.backward()
    var grad = a.grad().detach(share=True)
    # MEAN backward: gradient scaled by 1/2
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([0.5, 0.0, 0.5])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.5, 0.0, 0.5])))


# ─── GPU: General-case reduction SUM forward ──────────────────


def test_mcpy_gpu_3d_axis0_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(1))
        var _tmp2 = Tensor[dtype].d1([10, 12, 14, 16])
        var expected = _tmp2.reshape(2, 2)
        assert_true(result.to_cpu().all_close(expected))


def test_mcpy_gpu_2d_axis1_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(1))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([4.0, 10.0])))


# ─── GPU: General-case reduction MEAN forward ─────────────────


def test_mcpy_gpu_3d_axis0_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        var _tmp2 = Tensor[dtype].d1([5, 6, 7, 8])
        var expected = _tmp2.reshape(2, 2)
        assert_true(result.to_cpu().all_close(expected))


def test_mcpy_gpu_2d_axis1_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(0))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([2.0, 5.0])))


# ─── GPU: General-case reduction SUM backward ─────────────────


def test_mcpy_gpu_3d_axis0_sum_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            requires_grad=True,
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(1))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        var _tmp2 = Tensor[dtype].d1(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        )
        var expected_grad = _tmp2.reshape(3, 2, 2)
        assert_true(grad.all_close[atol=1e-5](expected_grad))


def test_mcpy_gpu_2d_axis1_sum_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(1))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        assert_true(
            grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 0.0, 1.0]))
        )
        assert_true(
            grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 0.0, 1.0]))
        )


# ─── GPU: General-case reduction MEAN backward ────────────────


def test_mcpy_gpu_3d_axis0_mean_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            requires_grad=True,
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        var _tmp2 = Tensor[dtype].d1(
            [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        )
        var expected_grad = _tmp2.reshape(3, 2, 2)
        assert_true(grad.all_close[atol=1e-5](expected_grad))


def test_mcpy_gpu_2d_axis1_mean_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(0))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad().detach(share=True)
        assert_true(
            grad[i(0), s()].all_close(Tensor[dtype].d1([0.5, 0.0, 0.5]))
        )
        assert_true(
            grad[i(1), s()].all_close(Tensor[dtype].d1([0.5, 0.0, 0.5]))
        )


# ─── CPU–GPU parity: general-case reduction ────────────────────


def test_mcpy_cpu_gpu_parity_3d_axis0_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var a = _tmp0.reshape(3, 2, 2)
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(1))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(1)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def test_mcpy_cpu_gpu_parity_2d_axis1_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=1, reduction=Reduction(1))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=1, reduction=Reduction(1)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def test_mcpy_cpu_gpu_parity_3d_axis0_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var a = _tmp0.reshape(3, 2, 2)
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(0))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(0)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def test_mcpy_cpu_gpu_parity_2d_axis1_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=1, reduction=Reduction(0))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=1, reduction=Reduction(0)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll gather tests passed!")
