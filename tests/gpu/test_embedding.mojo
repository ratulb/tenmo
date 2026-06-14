from std.testing import assert_true, assert_false, TestSuite
from std.sys import has_accelerator
from std.math import abs, sqrt
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.embedding import Embedding
from tenmo.common_utils import i, s
from tenmo.shared import Reduction


# =============================================================================
# Exhaustive tests for Embedding layer
# Prefix: test_emb_ on all test names to avoid collision with existing tests
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · Initialisation
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Forward · basic lookup
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · gradient flows into looked-up rows
# ─────────────────────────────────────────────────────────────────────────────


# ── CPU / MEAN ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · Freeze / unfreeze / from_pretrained
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CPU · max_norm renormalisation
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Forward
# ─────────────────────────────────────────────────────────────────────────────


def test_emb_gpu_fwd_basic_lookup() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var emb = Embedding[dtype](
            num_embeddings=5, embedding_dim=3, init_method="zero"
        ).to_gpu()
        # Set rows on CPU first then move — or set via fill after to_gpu
        # Use from_pretrained to set known weights
        var weights = Tensor[dtype].d2(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        var emb_gpu = (
            Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        )
        var result = emb_gpu([0, 2, 4])
        assert_true(result.shape() == Shape(3, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0],
                    ]
                )
            )
        )


def test_emb_gpu_fwd_repeated_index() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        var emb = (
            Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        )
        var result = emb([0, 0, 1])
        assert_true(result.shape() == Shape(3, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ]
                )
            )
        )


def test_emb_gpu_fwd_fuse_sum() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(
                weights,
                freeze=False,
                reduction=Reduction(1),
            )
            .to_gpu()
        )
        var result = emb([0, 1])
        assert_true(result.shape() == Shape(3))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([5.0, 7.0, 9.0]))
        )


def test_emb_gpu_fwd_eval_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var emb = (
            Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        )
        emb.eval()
        var result = emb([0, 1])
        assert_false(result.requires_grad)


def test_emb_gpu_fwd_output_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var emb = Embedding[dtype](
            num_embeddings=100, embedding_dim=64
        ).to_gpu()
        var result = emb([0, 5, 10, 15, 20])
        assert_true(result.shape() == Shape(5, 64))


def test_emb_gpu_fwd_padding_idx_zeros() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [9.0, 9.0, 9.0],
                [1.0, 2.0, 3.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(weights, padding_idx=0, freeze=False)
            .to_gpu()
        )
        var result = emb([0])
        assert_true(result.to_cpu().all_close(Tensor[dtype].zeros(Shape(1, 3))))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — GPU · Backward
# ─────────────────────────────────────────────────────────────────────────────


def test_emb_gpu_bwd_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        var emb = (
            Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        )
        var result = emb([1, 3])
        var loss = result.sum()
        loss.backward()
        # Rows 1 and 3 get grad=[1,1,1]
        var shared_grad = emb.weight.grad()
        assert_true(
            shared_grad[i(1), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        assert_true(
            shared_grad[i(3), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        # Non-looked-up rows get zero grad
        assert_true(
            shared_grad[i(0), s()]
            .to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )


def test_emb_gpu_bwd_repeated_index_accumulates() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        var emb = (
            Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        )
        var result = emb([0, 0])
        var loss = result.sum()
        loss.backward()
        # Row 0 looked up twice → grad = [2,2,2] via atomic scatter-add
        var shared_grad = emb.weight.grad()
        assert_true(
            shared_grad[i(0), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([2.0, 2.0, 2.0]))
        )


def test_emb_gpu_bwd_fuse_sum() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(
                weights,
                freeze=False,
                reduction=Reduction(1),
            )
            .to_gpu()
        )
        var result = emb([0, 2])  # (3,) via SUM
        var loss = result.sum()
        loss.backward()
        # Both looked-up rows receive same grad = [1,1,1]
        var shared_grad = emb.weight.grad()
        assert_true(
            shared_grad[i(0), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        assert_true(
            shared_grad[i(2), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        assert_true(
            shared_grad[i(1), s()]
            .to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )


# ── GPU / MEAN ────────────────────────────────────────────────────────────────


def test_emb_gpu_fwd_fuse_mean() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(
                weights,
                freeze=False,
                reduction=Reduction(0),
            )
            .to_gpu()
        )
        var result = emb([0, 1])
        assert_true(result.shape() == Shape(3))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([2.5, 3.5, 4.5]))
        )


def test_emb_gpu_bwd_fuse_mean() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(
                weights,
                freeze=False,
                reduction=Reduction(0),
            )
            .to_gpu()
        )
        var result = emb([0, 2])  # (3,) via MEAN
        var loss = result.sum()
        loss.backward()
        var shared_grad = emb.weight.grad()
        assert_true(
            shared_grad[i(0), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([0.5, 0.5, 0.5]))
        )
        assert_true(
            shared_grad[i(2), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([0.5, 0.5, 0.5]))
        )
        assert_true(
            shared_grad[i(1), s()]
            .to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )


def test_emb_gpu_bwd_padding_idx_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(weights, padding_idx=0, freeze=False)
            .to_gpu()
        )
        var result = emb([0, 1])
        var loss = result.sum()
        loss.backward()
        var shared_grad = emb.weight.grad()
        # Padding row must have zero grad
        assert_true(
            shared_grad[i(0), s()]
            .to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )
        # Non-padding row has correct grad
        assert_true(
            shared_grad[i(1), s()]
            .to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )


def test_emb_gpu_bwd_chained() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        var emb = (
            Embedding[dtype]
            .from_pretrained(
                weights,
                freeze=False,
                reduction=Reduction(1),
            )
            .to_gpu()
        )
        var w2 = (
            Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0], requires_grad=True).to_gpu()
        )
        var result = emb([0, 1])  # (4,) via SUM
        var pred = result.dot(w2).sigmoid()
        var target = Tensor[dtype].scalar(1.0).to_gpu()
        var diff = pred - target
        var diff_sqrd = diff * diff
        loss = diff_sqrd.squeeze()
        loss.backward()
        var shared_grad = emb.weight.grad()
        # Grad flows back to rows 0 and 1
        assert_false(
            shared_grad[i(0), s()]
            .to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(4)))
        )
        assert_false(
            shared_grad[i(1), s()]
            .to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(4)))
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll Embedding tests passed!")
