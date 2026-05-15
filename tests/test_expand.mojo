from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
# ============================================================
# EXPAND TESTS — forward shape, values, and backward grad flow
# ============================================================


# ------------------------------------------------------------
# Basic 1D → 2D expansion (prepend new dimension via broadcast)
# ------------------------------------------------------------

fn test_expand_1d_to_2d_new_batch_dim() raises:
    print("test_expand_1d_to_2d_new_batch_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)  # shape (3,)
    var e = a.expand(4, 3)                                           # shape (4,3)
    assert_true(e.shape() == Shape.of(4, 3))
    # Every row is [1, 2, 3]
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    assert_true(e.all_close(expected))
    es = e.sum()
    es.backward()
    # Each element of a was broadcast 4 times → grad = 4.0
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


fn test_expand_1d_to_3d() raises:
    print("test_expand_1d_to_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)   # shape (2,)
    var e = a.expand(3, 4, 2)                                    # shape (3,4,2)
    assert_true(e.shape() == Shape.of(3, 4, 2))
    # All slices identical
    var e_cpu = e.sum(axes=[0, 1])                               # shape (2,)
    assert_true(e_cpu.all_close(Tensor[dtype].d1([12.0, 24.0])))
    es = e.sum()
    es.backward()
    # broadcast count = 3*4 = 12
    assert_true(a.grad().all_close(Tensor[dtype].d1([12.0, 12.0])))


# ------------------------------------------------------------
# 2D → 2D expansion (size-1 dim expanded)
# ------------------------------------------------------------

fn test_expand_2d_row_vector_to_matrix() raises:
    print("test_expand_2d_row_vector_to_matrix")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)  # shape (1,3)
    var e = a.expand(4, 3)                                             # shape (4,3)
    assert_true(e.shape() == Shape.of(4, 3))
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    assert_true(e.all_close(expected))
    es = e.sum()
    es.backward()
    # row was broadcast 4 times
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


fn test_expand_2d_col_vector_to_matrix() raises:
    print("test_expand_2d_col_vector_to_matrix")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)  # shape (3,1)
    var e = a.expand(3, 4)                                                # shape (3,4)
    assert_true(e.shape() == Shape.of(3, 4))
    var expected = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [3.0, 3.0, 3.0, 3.0]]
    )
    assert_true(e.all_close(expected))
    es = e.sum()
    es.backward()
    # each element broadcast 4 times
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


fn test_expand_2d_both_dims_size1() raises:
    print("test_expand_2d_both_dims_size1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[5.0]], requires_grad=True)   # shape (1,1)
    var e = a.expand(3, 4)                                   # shape (3,4)
    assert_true(e.shape() == Shape.of(3, 4))
    # All 12 elements equal 5.0
    assert_true(e.all_close(Tensor[dtype].full(Shape.of(3, 4), 5.0)))
    es = e.sum()
    es.backward()
    # broadcast count = 12
    assert_true(a.grad().all_close(Tensor[dtype].d2([[12.0]])))


fn test_expand_2d_no_op_same_shape() raises:
    print("test_expand_2d_no_op_same_shape")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    var e = a.expand(2, 2)                                                    # no-op
    assert_true(e.shape() == Shape.of(2, 2))
    assert_true(e.all_close(a))
    es = e.sum()
    es.backward()
    # No broadcast → grad = 1.0 everywhere
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 3D expansions
# ------------------------------------------------------------

fn test_expand_3d_first_dim() raises:
    print("test_expand_3d_first_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1,2,2)
    var e = a.expand(5, 2, 2)                                                   # (5,2,2)
    assert_true(e.shape() == Shape.of(5, 2, 2))
    # Sum over expanded dim should give 5x original values
    var s = e.sum(axes=[0])                  # (2,2)
    assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0], [15.0, 20.0]])))
    es = e.sum()
    es.backward()
    # Each element broadcast 5 times
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[5.0, 5.0], [5.0, 5.0]]])))


fn test_expand_3d_last_dim() raises:
    print("test_expand_3d_last_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)  # (2,2,1)
    var e = a.expand(2, 2, 6)                                                         # (2,2,6)
    assert_true(e.shape() == Shape.of(2, 2, 6))
    # Each slice should repeat 6 times
    var s = e.sum(axes=[-1])                 # (2,2)
    assert_true(s.all_close(Tensor[dtype].d2([[6.0, 12.0], [18.0, 24.0]])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[6.0], [6.0]], [[6.0], [6.0]]])))


fn test_expand_3d_middle_dim() raises:
    print("test_expand_3d_middle_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)  # (2,1,2)
    var e = a.expand(2, 5, 2)                                                     # (2,5,2)
    assert_true(e.shape() == Shape.of(2, 5, 2))
    var s = e.sum(axes=[1])                  # (2,2)
    assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0], [15.0, 20.0]])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[5.0, 5.0]], [[5.0, 5.0]]])))


fn test_expand_3d_two_dims_broadcast() raises:
    print("test_expand_3d_two_dims_broadcast")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)  # (1,1,2)
    var e = a.expand(3, 4, 2)                                       # (3,4,2)
    assert_true(e.shape() == Shape.of(3, 4, 2))
    # col sums: 1*12=12, 2*12=24
    var s = e.sum(axes=[0, 1])               # (2,)
    assert_true(s.all_close(Tensor[dtype].d1([12.0, 24.0])))
    es = e.sum()
    es.backward()
    # broadcast count = 3*4 = 12
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[12.0, 12.0]]])))


fn test_expand_3d_all_dims_size1() raises:
    print("test_expand_3d_all_dims_size1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[7.0]]], requires_grad=True)  # (1,1,1)
    var e = a.expand(2, 3, 4)                                  # (2,3,4)
    assert_true(e.shape() == Shape.of(2, 3, 4))
    assert_true(e.all_close(Tensor[dtype].full(Shape.of(2, 3, 4), 7.0)))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[24.0]]])))


# ------------------------------------------------------------
# Shape API overload (Shape object, not variadic ints)
# ------------------------------------------------------------

fn test_expand_shape_api_overload() raises:
    print("test_expand_shape_api_overload")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)  # (1,2)
    var e = a.expand(Shape.of(3, 2))                              # (3,2)
    assert_true(e.shape() == Shape.of(3, 2))
    var expected = Tensor[dtype].d2([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert_true(e.all_close(expected))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0]])))


# ------------------------------------------------------------
# Grad correctness: non-uniform values
# ------------------------------------------------------------

fn test_expand_grad_non_uniform_values() raises:
    print("test_expand_grad_non_uniform_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    # Expand to (2,3,2): prepend batch of 3, no size-1 dim needed — use unsqueeze pattern
    var a_unsqueezed = a.unsqueeze(0)                     # (1,2,2) — no grad
    var e = a_unsqueezed.expand(3, 2, 2)                  # (3,2,2)
    assert_true(e.shape() == Shape.of(3, 2, 2))
    es = e.sum()
    es.backward()
    # a_unsqueezed was broadcast 3 times, grad flows to a
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0], [3.0, 3.0]])))


fn test_expand_grad_weighted_loss() raises:
    print("test_expand_grad_weighted_loss")
    comptime dtype = DType.float32
    # Simulate bias broadcast: bias (1,4) expanded to (3,4) then used in a loss
    var bias = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # (1,4)
    var e = bias.expand(3, 4)                                                   # (3,4)
    # Weighted sum: multiply by weights before summing
    var weights = Tensor[dtype].d2(
        [[1.0, 2.0, 1.0, 2.0],
         [2.0, 1.0, 2.0, 1.0],
         [1.0, 1.0, 1.0, 1.0]]
    )
    var loss = (e * weights).sum()
    loss.backward()
    # grad for bias[0,j] = sum of weights[:,j]
    # col0: 1+2+1=4, col1: 2+1+1=4, col2: 1+2+1=4, col3: 2+1+1=4
    assert_true(bias.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0, 4.0]])))


# ------------------------------------------------------------
# Expand then reduce — round-trip grad check
# ------------------------------------------------------------

fn test_expand_then_sum_axis_round_trip() raises:
    print("test_expand_then_sum_axis_round_trip")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)  # (1,3)
    var e = a.expand(5, 3)                                             # (5,3)
    var s = e.sum(axes=[0], keepdims=True)                             # (1,3)
    # s[0,j] = 5 * a[0,j]
    assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0, 15.0]])))
    ss = s.sum()
    ss.backward()
    # Each element broadcast 5 times, summed once → grad = 5.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 5.0, 5.0]])))


fn test_expand_then_mean_grad() raises:
    print("test_expand_then_mean_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[2.0, 4.0]], requires_grad=True)  # (1,2)
    var e = a.expand(4, 2)                                        # (4,2)
    var m = e.mean(axes=[0])                                      # (2,)
    # mean: [2.0, 4.0] (same as a, since all rows identical)
    assert_true(m.all_close(Tensor[dtype].d1([2.0, 4.0])))
    ms = m.sum()
    ms.backward()
    # grad through mean (÷4) then sum over 4 broadcast rows = 4*(1/4) = 1.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 1.0]])))


# ------------------------------------------------------------
# Expand in matmul-like broadcast scenario
# ------------------------------------------------------------

fn test_expand_bias_broadcast_pattern() raises:
    print("test_expand_bias_broadcast_pattern")
    comptime dtype = DType.float32
    # Common pattern: bias (1, out) expanded to match batch output (B, out)
    var bias = Tensor[dtype].d2([[0.5, 1.0, 1.5]], requires_grad=True)  # (1,3)
    var e = bias.expand(6, 3)                                             # (6,3)
    assert_true(e.shape() == Shape.of(6, 3))
    # All rows should equal bias
    var row_sum = e.sum(axes=[0])   # (3,) each col summed = 6 * bias_val
    assert_true(row_sum.all_close(Tensor[dtype].d1([3.0, 6.0, 9.0])))
    es = e.sum()
    es.backward()
    assert_true(bias.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0]])))


# ------------------------------------------------------------
# Grad accumulation across two expand ops on the same tensor
# ------------------------------------------------------------

fn test_expand_grad_accumulation_two_expands() raises:
    print("test_expand_grad_accumulation_two_expands")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)  # (1,2)
    var e1 = a.expand(3, 2)                                       # (3,2)
    var e2 = a.expand(5, 2)                                       # (5,2)
    es1 = e1.sum()
    es1.backward()
    es2 = e2.sum()
    es2.backward()
    # e1 contributes grad 3.0, e2 contributes 5.0 → accumulated 8.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0]])))


# ------------------------------------------------------------
# Expand with track_grad=False — no backward registered
# ------------------------------------------------------------

fn test_expand_no_grad_tracking() raises:
    print("test_expand_no_grad_tracking")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var e = a.expand[track_grad=False](4, 2)
    assert_true(e.shape() == Shape.of(4, 2))
    assert_true(not e.requires_grad)


# ------------------------------------------------------------
# 4D expansion
# ------------------------------------------------------------

fn test_expand_4d_first_two_dims() raises:
    print("test_expand_4d_first_two_dims")
    comptime dtype = DType.float32
    # (1,1,3,4) → (2,5,3,4)
    var a = Tensor[dtype].randn(1, 1, 3, 4)
    a.requires_grad_(True)
    var a_ref = a.copy()
    var e = a.expand(2, 5, 3, 4)
    assert_true(e.shape() == Shape.of(2, 5, 3, 4))
    # sum over broadcast dims should equal 10 * original
    var s = e.sum(axes=[0, 1])   # (3,4)
    var s_ref1 = a_ref.squeeze([0])
    s_ref2 = s_ref1.squeeze([0])
    s_ref = s_ref2 * Scalar[dtype](10)
    assert_true(s.all_close(s_ref))
    es = e.sum()
    es.backward()
    # broadcast count = 2*5 = 10
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape.of(1, 1, 3, 4), 10.0)))


fn test_expand_4d_last_dim_only() raises:
    print("test_expand_4d_last_dim_only")
    comptime dtype = DType.float32
    # (2,3,4,1) → (2,3,4,7)
    var a = Tensor[dtype].randn(2, 3, 4, 1)
    a.requires_grad_(True)
    var e = a.expand(2, 3, 4, 7)
    assert_true(e.shape() == Shape.of(2, 3, 4, 7))
    es = e.sum()
    es.backward()
    # each element broadcast 7 times
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape.of(2, 3, 4, 1), 7.0)))


# ------------------------------------------------------------
# Expand matches manual repeat behaviour (cross-validate)
# ------------------------------------------------------------

fn test_expand_matches_manual_tile() raises:
    print("test_expand_matches_manual_tile")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)  # (3,)
    var e = a.expand(4, 3)                                           # (4,3)
    # Manual construction of expected
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0]]
    )
    assert_true(e.all_close(expected))
    # Also verify elementwise: e[i,j] == a[j] for all i
    var col_means = e.mean(axes=[0])   # (3,) should equal a
    assert_true(col_means.all_close(a))


# ------------------------------------------------------------
# Expand is a view — no data copy (stride-zero check via value)
# ------------------------------------------------------------

fn test_expand_is_zero_stride_view() raises:
    print("test_expand_is_zero_stride_view")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[10.0, 20.0]], requires_grad=True)  # (1,2)
    var e = a.expand(100, 2)                                        # (100,2)
    # All 100 rows must have identical values — confirms stride-0 view
    var row0 = e.sum(axes=[0])   # (2,) = 100*[10,20]
    assert_true(row0.all_close(Tensor[dtype].d1([1000.0, 2000.0])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[100.0, 100.0]])))


# ============================================================
# MAIN
# ============================================================

fn main() raises:
    _ = """
    # 1D → nD
    test_expand_1d_to_2d_new_batch_dim()
    test_expand_1d_to_3d()

    # 2D expansions
    test_expand_2d_row_vector_to_matrix()
    test_expand_2d_col_vector_to_matrix()
    test_expand_2d_both_dims_size1()
    test_expand_2d_no_op_same_shape()

    # 3D expansions
    test_expand_3d_first_dim()
    test_expand_3d_last_dim()
    test_expand_3d_middle_dim()
    test_expand_3d_two_dims_broadcast()
    test_expand_3d_all_dims_size1()

    # API overload
    test_expand_shape_api_overload()

    # Grad correctness
    test_expand_grad_non_uniform_values()
    test_expand_grad_weighted_loss()

    # Round-trip reduce
    test_expand_then_sum_axis_round_trip()
    test_expand_then_mean_grad()

    # Common patterns
    test_expand_bias_broadcast_pattern()
    test_expand_grad_accumulation_two_expands()

    # track_grad=False
    test_expand_no_grad_tracking()

    # 4D
    test_expand_4d_first_two_dims()
    test_expand_4d_last_dim_only()

    # Cross-validates
    test_expand_matches_manual_tile()
    test_expand_is_zero_stride_view()

    print("All expand tests passed.")
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
