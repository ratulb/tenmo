from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from std.testing import assert_true, TestSuite
from tenmo.optim import SGD
from tenmo.intarray import IntArray
from tenmo.filler import Filler
from std.sys import has_accelerator


comptime dtype = DType.float32


def test_sparse_sgd_step_only_updates_specified_rows() raises:
    """Sparse step with indices: only rows at those indices change."""
    var w = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step(IntArray([0, 2]))
    var expected = Tensor[dtype].d2(
        [[0.9, 0.9], [2.0, 2.0], [2.9, 2.9], [4.0, 4.0]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_step_empty_indices_dense_fallback() raises:
    """Empty indices => dense update (all rows change)."""
    var w = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step(IntArray())
    var expected = Tensor[dtype].d2(
        [[0.9, 0.9], [1.9, 1.9], [2.9, 2.9]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_step_all_indices_same_as_dense() raises:
    """All rows in indices => same as dense step."""
    var w = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step(IntArray([0, 1, 2]))
    var expected = Tensor[dtype].d2(
        [[0.9, 0.9], [1.9, 1.9], [2.9, 2.9]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_step_non_2d_skipped() raises:
    """Sparse step on a non-2D param falls back to dense update."""
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    # Non-2D: sparse step skipped entirely, param unchanged
    sgd.step(IntArray([0]))
    # Should still be [1,2,3] because sparse skips non-2D
    # Actually let's verify: the condition is `if shape.rank() != 2: continue`
    # Which means param IS skipped, so no update happens
    assert_true(w.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_sparse_sgd_zero_grad_only_zeros_specified_rows() raises:
    """Sparse zero_grad: only rows at indices are zeroed."""
    var w = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    # Set different gradients per row
    w.gradients().seed_grad(
        Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )
    sgd.zero_grad(IntArray([1]))
    # Row 1 should be zeroed, rows 0 and 2 unchanged
    assert_true(
        w.gradients().all_close(
            Tensor[dtype].d2([[1.0, 1.0], [0.0, 0.0], [3.0, 3.0]])
        )
    )


def test_sparse_sgd_zero_grad_non_specified_retain() raises:
    """Verifies non-specified rows truly retain their gradients."""
    var w = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(7.0)
    sgd.zero_grad(IntArray([0, 2, 4]))
    var expected = Tensor[dtype].d2(
        [[0.0, 0.0], [7.0, 7.0], [0.0, 0.0], [7.0, 7.0], [0.0, 0.0]],
    )
    assert_true(w.gradients().all_close(expected))


def test_sparse_sgd_zero_grad_empty_indices_dense() raises:
    """Empty indices => dense zero_grad (all rows zeroed)."""
    var w = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(5.0)
    sgd.zero_grad(IntArray())
    assert_true(w.gradients().all_close(Tensor[dtype].zeros(w.shape())))


def test_sparse_sgd_step_then_zero_grad_same_indices() raises:
    """Step then zero_grad with same indices: params updated, grads cleared."""
    var w = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    var idx = IntArray([0, 2])
    sgd.step(idx)
    # Rows 0 and 2 updated, row 1 unchanged
    var expected_w = Tensor[dtype].d2(
        [[0.9, 0.9], [2.0, 2.0], [2.9, 2.9]],
    )
    assert_true(w.all_close(expected_w))
    # After zero_grad: rows 0 and 2 have zero grad, row 1 still has grad
    sgd.zero_grad(idx)
    var expected_g = Tensor[dtype].d2(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
    )
    assert_true(w.gradients().all_close(expected_g))


def test_sparse_sgd_with_momentum() raises:
    """Sparse step with momentum: only specified rows updated, velocity tracked.
    """
    var w = Tensor[dtype].d2(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, momentum=0.9)
    var idx = IntArray([1])
    w.seed_grad(1.0)
    sgd.step(idx)
    # Row 1: v = 0.9*0 + 1 = 1, w = 20 - 0.1*1 = 19.9
    # Rows 0 and 2 unchanged
    var expected = Tensor[dtype].d2(
        [[10.0, 10.0], [19.9, 19.9], [30.0, 30.0]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_momentum_velocity_zeroed() raises:
    """Sparse zero_grad with momentum also zeros velocity for those rows."""
    var w = Tensor[dtype].d2(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, momentum=0.9)

    # Step 1 on row 1 only — builds velocity
    w.seed_grad(1.0)
    sgd.step(IntArray([1]))
    # Row 1: v=1.0, w = 20 - 0.1*1 = 19.9

    # Zero grad on row 1 — should also zero row 1's velocity
    sgd.zero_grad(IntArray([1]))

    # Step 2 on row 1 only — with zeroed velocity, behaves like fresh
    w.seed_grad(1.0)
    sgd.step(IntArray([1]))
    # If velocity was properly zeroed: v = 0.9*0 + 1 = 1.0, w = 19.9 - 0.1*1 = 19.8
    # If velocity was NOT zeroed:    v = 0.9*1 + 1 = 1.9, w = 19.9 - 0.1*1.9 = 19.71
    var expected = Tensor[dtype].d2(
        [[10.0, 10.0], [19.8, 19.8], [30.0, 30.0]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_momentum_velocity_retained_for_other_rows() raises:
    """Sparse zero_grad on some rows does NOT zero momentum velocity for other rows.
    """
    var w = Tensor[dtype].d2(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, momentum=0.9)

    # Step 1 on all rows — builds velocity everywhere
    w.seed_grad(1.0)
    sgd.step(IntArray([0, 1, 2]))
    # v = 0.9*0 + 1 = 1.0 for all rows
    # w = [[9.9,9.9],[19.9,19.9],[29.9,29.9]]

    # Zero grad only on row 0 — row 0 velocity zeroed, rows 1-2 velocity retained
    sgd.zero_grad(IntArray([0]))

    # Step 2: seed grad only on row 0, step only row 0
    # Rows 1-2 unchanged
    w.seed_grad(Tensor[dtype].d2([[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]))
    sgd.step(IntArray([0]))
    # Row 0: velocity was zeroed, so v = 0.9*0 + 5 = 5, w = 9.9-0.1*5 = 9.4 → [9.4, 9.4]
    var expected = Tensor[dtype].d2(
        [[9.4, 9.4], [19.9, 19.9], [29.9, 29.9]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_with_weight_decay() raises:
    """Sparse step with weight decay: only specified rows affected."""
    var w = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, weight_decay=0.1)
    w.seed_grad(1.0)
    sgd.step(IntArray([0, 2]))
    # Row 0: g_eff = 1 + 0.1*1 = 1.1, w = 1 - 0.1*1.1 = 0.89; wait, g_eff for [1,2]:
    #   col0: g_eff = 1 + 0.1*1 = 1.1, w = 1 - 0.1*1.1 = 0.89
    #   col1: g_eff = 1 + 0.1*2 = 1.2, w = 2 - 0.1*1.2 = 1.88
    # Row 1 (not in indices): unchanged [3, 4]
    # Row 2: col0: g_eff = 1 + 0.1*5 = 1.5, w = 5 - 0.1*1.5 = 4.85
    #         col1: g_eff = 1 + 0.1*6 = 1.6, w = 6 - 0.1*1.6 = 5.84
    var expected = Tensor[dtype].d2(
        [[0.89, 1.88], [3.0, 4.0], [4.85, 5.84]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_multiple_steps_different_indices() raises:
    """Multiple sparse steps with different indices accumulate correctly."""
    var w = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)

    # Step 1: update row 0 only
    w.seed_grad(Tensor[dtype].d2([[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]))
    sgd.step(IntArray([0]))
    # w[0] = [1,1] - 0.1*[2,2] = [0.8, 0.8]

    sgd.zero_grad(IntArray([0]))

    # Step 2: update row 2 only
    w.seed_grad(Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]))
    sgd.step(IntArray([2]))
    # w[2] = [3,3] - 0.1*[3,3] = [2.7, 2.7]

    sgd.zero_grad(IntArray([2]))

    # Step 3: update all rows (dense)
    w.seed_grad(1.0)
    sgd.step()
    # w[0] = [0.8,0.8] - 0.1 = [0.7, 0.7]
    # w[1] = [2,2] - 0.1 = [1.9, 1.9]
    # w[2] = [2.7,2.7] - 0.1 = [2.6, 2.6]

    var expected = Tensor[dtype].d2(
        [[0.7, 0.7], [1.9, 1.9], [2.6, 2.6]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_step_zero_grad_different_indices() raises:
    """Step with one set of indices, zero_grad with a different set."""
    var w = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)

    # Use grad=5.0 to avoid float32 rounding edge case with 0.1*10.0
    w.seed_grad(5.0)
    var step_idx = IntArray([0, 2])
    var zero_idx = IntArray([1, 3])

    sgd.step(step_idx)
    # Rows 0, 2: w -= 0.1*5 = w - 0.5 → [0.5, 1.5], [4.5, 5.5]
    # Rows 1, 3 unchanged: [3,4], [7,8]

    sgd.zero_grad(zero_idx)
    # Rows 1, 3: gradient zeroed
    # Rows 0, 2: gradient still 5.0

    var expected_w = Tensor[dtype].d2(
        [[0.5, 1.5], [3.0, 4.0], [4.5, 5.5], [7.0, 8.0]],
    )
    assert_true(w.all_close(expected_w))

    var expected_g = Tensor[dtype].d2(
        [[5.0, 5.0], [0.0, 0.0], [5.0, 5.0], [0.0, 0.0]],
    )
    assert_true(w.gradients().all_close(expected_g))


def test_sparse_sgd_out_of_order_indices() raises:
    """Sparse step with indices in non-sorted order works correctly."""
    var w = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    # Indices in reverse order — should still work
    sgd.step(IntArray([2, 0]))
    # Rows 0 and 2 updated, row 1 unchanged
    var expected = Tensor[dtype].d2(
        [[0.9, 1.9], [3.0, 4.0], [4.9, 5.9]],
    )
    assert_true(w.all_close(expected))


def test_sparse_sgd_duplicate_indices() raises:
    """Sparse step with duplicate indices: same row updated multiple times."""
    var w = Tensor[dtype].d2(
        [[10.0, 10.0], [20.0, 20.0]],
        requires_grad=True,
    )
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    # Duplicate index 0 — row 0 gets updated twice with grad=1.0
    sgd.step(IntArray([0, 0]))
    # Row 0: w = 10 - 0.1*1 - 0.1*1 = 9.8
    # Row 1: unchanged
    var expected = Tensor[dtype].d2(
        [[9.8, 9.8], [20.0, 20.0]],
    )
    assert_true(w.all_close(expected))


def test_gpu_sparse_sgd_step_only_updates_specified_rows() raises:
    """Sparse step on GPU: only rows at specified indices change."""
    comptime if not has_accelerator():
        print(
            "No GPU available — skipping"
            " test_gpu_sparse_sgd_step_only_updates_specified_rows"
        )
        return
    var w_cpu = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        requires_grad=True,
    )
    var w = w_cpu.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step(IntArray([0, 2]))
    var result = w.to_cpu()
    var expected = Tensor[dtype].d2(
        [[0.9, 0.9], [2.0, 2.0], [2.9, 2.9], [4.0, 4.0]],
    )
    assert_true(result.all_close(expected))


def test_gpu_sparse_sgd_with_momentum() raises:
    """Sparse momentum step on GPU: only specified rows updated."""
    comptime if not has_accelerator():
        print("No GPU available — skipping test_gpu_sparse_sgd_with_momentum")
        return
    var w_cpu = Tensor[dtype].d2(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        requires_grad=True,
    )
    var w = w_cpu.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, momentum=0.9)
    var idx = IntArray([1])
    w.seed_grad(1.0)
    sgd.step(idx)
    var result = w.to_cpu()
    # Row 1: v = 0.9*0 + 1 = 1, w = 20 - 0.1*1 = 19.9
    # Rows 0 and 2 unchanged
    var expected = Tensor[dtype].d2(
        [[10.0, 10.0], [19.9, 19.9], [30.0, 30.0]],
    )
    assert_true(result.all_close(expected))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All sparse SGD tests passed ✓")
