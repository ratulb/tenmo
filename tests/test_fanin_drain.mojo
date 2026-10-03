# Fanin-drain regression suite.
#
# Engine contract: `parent_ids` is the fanin-completion signal — the
# appended set must equal the ancestry set. Handlers used to skip the
# append for non-requiring parents, which corrupts scheduling: a skipped
# non-requiring INTERIOR parent never executes, so any requiring node
# that shares a child with it (diamond) never drains either — silent
# zero grads far from the actual defect.
#
# Semantics note (torch-cut): an untracked node still BLOCKS flow —
# update_grad no-ops and its grad is never computed, so the cut edge
# contributes zeros (torch treats non-requiring inputs as constants the
# same way). The append fix restores the OTHER paths, not the cut one.
#
# Each test is therefore a DIAMOND: w (leaf) -> x (tracked interior) ->
# b (untracked interior, via requires_grad_(False) flip) -> c1 (fixed op
# + requiring sibling) and x -> c2 (x*10 second path). Pre-fix w starves
# (0) and x keeps its unconsumed c2 grad (10s); post-fix w carries the
# c2 path (20s) while x reads 0 — x is interior, so its grad is consumed
# (cleared) when x drains into w. Sibling grads lock the consumer's own
# math in the mixed setting.
#
# Covers every handler fixed for item 16: matmul (both legs), mul,
# div, add-broadcast (shared BroadcastBackward struct), concat (all 3
# paths), stack (both legs), pad, maxpool, conv2d.

from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.cnn import Conv2dFused
from tenmo.pooling import MaxPool2d


def test_fanin_matmul_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior [[2,4],[6,8]]
    var eye = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var b = x.matmul(eye)
    b.requires_grad_(False)  # untracked interior, ancestry [x, eye]
    var v = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    var c1 = Tensor[dtype].matmul(b, v)
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut: b edge contributes 0, so x == c2 path only (10s);
    # w == 10 * dx/dw = 20s (0 pre-fix).
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))
    # Sibling: dL/dv[j,k] = col sums of b = [8, 12].
    assert_true(v.grad() == Tensor[dtype].d2([[8.0, 8.0], [12.0, 12.0]]))


def test_fanin_mul_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior
    var eye = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var b = x.matmul(eye)
    b.requires_grad_(False)  # untracked interior
    var v = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var c1 = b * v
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))
    # Sibling: dL/dv = b = [[2,4],[6,8]].
    assert_true(v.grad() == Tensor[dtype].d2([[2.0, 4.0], [6.0, 8.0]]))


def test_fanin_div_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior
    var eye = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var b = x.matmul(eye)
    b.requires_grad_(False)  # untracked interior
    var v = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    var c1 = b / v  # == b
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))
    # Sibling: dL/dv = -b/v^2 = -[[2,4],[6,8]].
    assert_true(v.grad() == Tensor[dtype].d2([[-2.0, -4.0], [-6.0, -8.0]]))


def test_fanin_add_broadcast_drains_diamond() raises:
    comptime dtype = DType.float32
    # Covers the shared BroadcastBackward struct (add/sub/mul/div
    # broadcast paths all route through it).
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior
    var eye = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var b = x.matmul(eye)
    b.requires_grad_(False)  # untracked interior
    var row = Tensor[dtype].d1([100.0, 200.0], requires_grad=True)
    var c1 = b + row  # broadcast (2,2)+(2,)
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))
    # Sibling: dL/drow = [2, 2].
    assert_true(row.grad() == Tensor[dtype].d1([2.0, 2.0]))


def test_fanin_concat_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior
    var b = x.flatten()  # (4,), tracked
    b.requires_grad_(False)  # untracked interior, ancestry [x]
    var leaf = Tensor[dtype].d1([10.0, 20.0], requires_grad=True)
    var tensors = List[Tensor[dtype]]()
    tensors.append(b)
    tensors.append(leaf)
    var c1 = Tensor[dtype].concat(tensors, axis=0)  # (6,)
    var c2 = (x * 10.0).flatten()  # (4,) second path, same scale
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # dL/dx = c2 path only: ones(2,2)*10; dw = 2*that.
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))
    # Sibling: dL/dleaf = [1, 1].
    assert_true(leaf.grad() == Tensor[dtype].d1([1.0, 1.0]))


def test_fanin_stack_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior
    var b = x.flatten()  # (4,), tracked
    b.requires_grad_(False)  # untracked interior, ancestry [x]
    var leaf = Tensor[dtype].d1([7.0, 8.0, 9.0, 10.0], requires_grad=True)
    var tensors = List[Tensor[dtype]]()
    tensors.append(b)
    tensors.append(leaf)
    var c1 = Tensor[dtype].stack(tensors, axis=0)  # (2,4)
    var c2 = (x * 10.0).flatten()
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))
    # Sibling: dL/dleaf = ones(4).
    assert_true(leaf.grad() == Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0]))


def test_fanin_pad_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var two = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
    var x = w * two  # tracked interior
    var eye = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var b = x.matmul(eye)
    b.requires_grad_(False)  # untracked interior
    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 1))
    pad.append((0, 0))
    # Override: lone untracked input would otherwise leave c1 untracked.
    var c1 = Tensor[dtype].pad(
        b, pad, mode="constant", value=0.0, requires_grad=True
    )  # (3,2)
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut: the b edge contributes 0, so x carries the c2 path only — and
    # x is interior, so its grad is consumed (cleared) when x drains into w
    # (clearing semantics; pre-fix x never drains and the 10s linger).
    assert_true(x.grad() == Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]]))
    # w is a leaf: 0 pre-fix (x stuck), 20s post-fix (10 * dx/dw).
    assert_true(w.grad() == Tensor[dtype].d2([[20.0, 20.0], [20.0, 20.0]]))


def test_fanin_pool_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].zeros(1, 1, 4, 4, requires_grad=True)
    var v: Scalar[dtype] = 1
    for h in range(4):
        for ww in range(4):
            w[0, 0, h, ww] = v
            v += Scalar[dtype](1)
    var two = Tensor[dtype].ones(1, 1, 4, 4) * 2.0
    var x = w * two  # tracked interior, values 2..32
    var b = x * two
    b.requires_grad_(False)  # untracked interior, ancestry [x, two]
    var pool = MaxPool2d[dtype](kernel_size=2)
    var c1 = pool(b)  # (1,1,2,2), tracked (module forces it in training)
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    # Cut + interior clearing: x drains into w (consumed → 0 post-fix;
    # the 10s linger pre-fix because x never executes).
    var tens = Tensor[dtype].ones(1, 1, 4, 4) * 10.0
    var twenties = Tensor[dtype].ones(1, 1, 4, 4) * 20.0
    assert_true(x.grad() == Tensor[dtype].zeros_like(tens))
    assert_true(w.grad() == twenties)


def test_fanin_conv_drains_diamond() raises:
    comptime dtype = DType.float32
    var w = Tensor[dtype].zeros(1, 1, 3, 3, requires_grad=True)
    var v: Scalar[dtype] = 1
    for h in range(3):
        for ww in range(3):
            w[0, 0, h, ww] = v
            v += Scalar[dtype](1)
    var two = Tensor[dtype].ones(1, 1, 3, 3) * 2.0
    var x = w * two  # tracked interior
    var b = x * two
    b.requires_grad_(False)  # untracked interior, ancestry [x, two]
    var kernel = Tensor[dtype].zeros(1, 1, 2, 2, requires_grad=True)
    kernel[0, 0, 0, 0] = 1.0
    kernel[0, 0, 1, 1] = 1.0
    var c1 = Conv2dFused[dtype].forward(b, kernel, stride=1, padding="valid")
    var c2 = x * 10.0
    var loss = c1.sum() + c2.sum()
    loss.backward()
    var tens = Tensor[dtype].ones(1, 1, 3, 3) * 10.0
    var twenties = Tensor[dtype].ones(1, 1, 3, 3) * 20.0
    assert_true(x.grad() == Tensor[dtype].zeros_like(tens))
    assert_true(w.grad() == twenties)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
