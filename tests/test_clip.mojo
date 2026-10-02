from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

# ===----------------------------------------------------------------------=== #
# Clip tests — prefix: clip_
#
# Added alongside the STE work because tenmo/fakequant.mojo exposed a latent
# bug in clip's non-contiguous BACKWARD branch: it called `storage_get` on a
# Gradbox, which has no such method (it has `buffer()`), so that branch had
# never been compiled. No test reached it -- clip had none at all.
#
# The branch under test: non-contiguous parent + contiguous grad output, which
# takes the IndexIterator fast path in ClipBackward. The STE only uses
# clip[track_grad=False], so without this file the backward path stays unproven.
# ===----------------------------------------------------------------------=== #


comptime F32 = DType.float32


# ===----------------------------------------------------------------------=== #
# Forward
# ===----------------------------------------------------------------------=== #


def test_clip_cpu_forward_basic() raises:
    var x = Tensor[F32].d1([-2.0, 0.5, 3.0, 7.0])
    var y = x.clip[track_grad=False](0.0, 5.0)
    assert_true(y.all_close(Tensor[F32].d1([0.0, 0.5, 3.0, 5.0])))


def test_clip_cpu_forward_2d() raises:
    var x = Tensor[F32].d2([[-1.0, 2.0], [9.0, 4.0]])
    var y = x.clip[track_grad=False](0.0, 5.0)
    var expected = Tensor[F32].d2([[0.0, 2.0], [5.0, 4.0]])
    assert_true(y.all_close(expected))


# ===----------------------------------------------------------------------=== #
# Backward — contiguous (the branch that did work)
# ===----------------------------------------------------------------------=== #


def test_clip_grad_contiguous() raises:
    # d clip/dx = 1 where min <= x <= max, else 0.
    var x = Tensor[F32].d1([-2.0, 2.0, 7.0], requires_grad=True)
    var y = x.clip(0.0, 5.0).sum()
    y.backward()
    var expected = Tensor[F32].d1([0.0, 1.0, 0.0])
    assert_true(x.grad().all_close(expected))


# ===----------------------------------------------------------------------=== #
# Backward — NON-CONTIGUOUS parent (the branch that was broken)
# ===----------------------------------------------------------------------=== #


def test_clip_grad_transposed_view() raises:
    # x = [[1,5],[2,6]]; transposed -> [[1,2],[5,6]] logically, stored
    # strided, so ClipBackward takes the IndexIterator branch.
    # clip(2,5): forward -> [[2,2],[5,5]]
    # gradient gate on the ORIGINAL values: 1 is below min -> 0,
    # 2 and 5 are in range -> 1, 6 is above max -> 0.
    var x = Tensor[F32].d2([[1.0, 5.0], [2.0, 6.0]], requires_grad=True)
    var t = x.transpose(1, 0)
    var y = t.clip(2.0, 5.0)
    var expected_fwd = Tensor[F32].d2([[2.0, 2.0], [5.0, 5.0]])
    assert_true(y.all_close(expected_fwd))

    var loss = y.sum()
    loss.backward()
    # Logically (after transpose) the gate is [0,1,1,0] -> [[0,1],[1,0]].
    var expected_grad = Tensor[F32].d2([[0.0, 1.0], [1.0, 0.0]])
    assert_true(x.grad().all_close(expected_grad))


def test_clip_grad_strided_slice() raises:
    var x = Tensor[F32].d1([1.0, 2.0, 3.0, 9.0], requires_grad=True)
    var s = x[0:4:2]  # [1.0, 3.0], a strided view
    var loss = s.clip(1.5, 5.0).sum()
    loss.backward()
    # 1.0 is below min -> 0; 3.0 is in range -> 1. The untouched elements of x
    # get nothing, since the view only feeds back its own positions.
    var expected = Tensor[F32].d1([0.0, 0.0, 1.0, 0.0])
    assert_true(x.grad().all_close(expected))


def test_clip_grad_transposed_view_weighted() raises:
    # Weighted upstream, so the IndexIterator branch must respect a
    # non-uniform grad rather than assuming ones.
    var x = Tensor[F32].d2([[1.0, 5.0], [2.0, 6.0]], requires_grad=True)
    var t = x.transpose(1, 0)
    var w = Tensor[F32].d2([[1.0, 2.0], [3.0, 4.0]])
    var loss = (t.clip(2.0, 5.0) * w).sum()
    loss.backward()
    # Logical t = x^T = [[1,2],[5,6]]; gate = [[0,1],[1,0]]; so
    # dL/dt = w * gate = [[0,2],[3,0]]. Then x = t^T, so dL/dx is the
    # TRANSPOSE of dL/dt = [[0,3],[2,0]]. Deliberately asymmetric: a
    # symmetric expectation would not catch a missing transpose.
    var expected = Tensor[F32].d2([[0.0, 3.0], [2.0, 0.0]])
    assert_true(x.grad().all_close(expected))


def test_clip_eval_erases_graph() raises:
    var x = Tensor[F32].d1([1.0, 2.0], requires_grad=True)
    var y = x.clip[track_grad=False](0.0, 5.0)
    assert_true(not y.has_ancestry())


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
