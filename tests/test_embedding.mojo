from std.testing import assert_true, assert_false, TestSuite
from std.sys import has_accelerator
from std.math import abs, sqrt
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.embedding import Embedding
from tenmo.common_utils import i, s


# =============================================================================
# Exhaustive tests for Embedding layer
# Prefix: test_emb_ on all test names to avoid collision with existing tests
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · Initialisation
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_cpu_init_shape() raises:
    print("test_emb_cpu_init_shape")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=10, embedding_dim=4)
    assert_true(emb.weight.shape() == Shape(10, 4))
    assert_true(emb.num_embeddings == 10)
    assert_true(emb.embedding_dim  == 4)
    assert_true(emb.weight.requires_grad)
    assert_true(emb.training)


fn test_emb_cpu_init_normal() raises:
    print("test_emb_cpu_init_normal")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=100, embedding_dim=16,
        init_method="normal", init_seed=42
    )
    assert_true(emb.weight.shape() == Shape(100, 16))
    # Normal init — weights should not all be zero
    assert_false(emb.weight.all_close(Tensor[dtype].zeros(Shape(100, 16))))


fn test_emb_cpu_init_uniform() raises:
    print("test_emb_cpu_init_uniform")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=50, embedding_dim=8,
        init_method="uniform", init_seed=1
    )
    assert_true(emb.weight.shape() == Shape(50, 8))
    assert_false(emb.weight.all_close(Tensor[dtype].zeros(Shape(50, 8))))


fn test_emb_cpu_init_xavier() raises:
    print("test_emb_cpu_init_xavier")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=20, embedding_dim=8,
        init_method="xavier", init_seed=7
    )
    assert_true(emb.weight.shape() == Shape(20, 8))


fn test_emb_cpu_init_kaiming() raises:
    print("test_emb_cpu_init_kaiming")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=20, embedding_dim=8,
        init_method="kaiming", init_seed=7
    )
    assert_true(emb.weight.shape() == Shape(20, 8))


fn test_emb_cpu_init_zero() raises:
    print("test_emb_cpu_init_zero")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=10, embedding_dim=4,
        init_method="zero"
    )
    assert_true(emb.weight.all_close(Tensor[dtype].zeros(Shape(10, 4))))


fn test_emb_cpu_init_freeze() raises:
    print("test_emb_cpu_init_freeze")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=10, embedding_dim=4,
        freeze=True
    )
    assert_false(emb.weight.requires_grad)


fn test_emb_cpu_init_padding_idx_zeros() raises:
    print("test_emb_cpu_init_padding_idx_zeros")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=10, embedding_dim=4,
        padding_idx=0, init_method="normal", init_seed=1
    )
    # Row 0 must be all zeros regardless of init_method
    var row0 = emb.weight[i(0), s()]
    assert_true(row0.all_close(Tensor[dtype].zeros(Shape(4))))
    # Other rows must not all be zero
    var row1 = emb.weight[i(1), s()]
    assert_false(row1.all_close(Tensor[dtype].zeros(Shape(4))))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Forward · basic lookup
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_cpu_fwd_single_index() raises:
    print("test_emb_cpu_fwd_single_index")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    # Set row 2 to known values
    emb.weight.fill(Tensor[dtype].d1([1.0, 2.0, 3.0]), i(2), s())
    var result = emb([2])
    assert_true(result.shape() == Shape(1, 3))
    assert_true(result.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0]])))


fn test_emb_cpu_fwd_multiple_indices() raises:
    print("test_emb_cpu_fwd_multiple_indices")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    emb.weight.fill(Tensor[dtype].d1([1.0, 0.0, 0.0]), i(0), s())
    emb.weight.fill(Tensor[dtype].d1([0.0, 1.0, 0.0]), i(1), s())
    emb.weight.fill(Tensor[dtype].d1([0.0, 0.0, 1.0]), i(2), s())
    var result = emb([0, 1, 2])
    assert_true(result.shape() == Shape(3, 3))
    assert_true(result.all_close(Tensor[dtype].d2([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])))


fn test_emb_cpu_fwd_repeated_index() raises:
    print("test_emb_cpu_fwd_repeated_index")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    emb.weight.fill(Tensor[dtype].d1([4.0, 5.0, 6.0]), i(3), s())
    # Same index twice — both rows should be identical copies
    var result = emb([3, 3])
    assert_true(result.shape() == Shape(2, 3))
    assert_true(result.all_close(Tensor[dtype].d2([
        [4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0],
    ])))


fn test_emb_cpu_fwd_output_shape() raises:
    print("test_emb_cpu_fwd_output_shape")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=100, embedding_dim=64)
    var result = emb([0, 5, 10, 15, 20])
    assert_true(result.shape() == Shape(5, 64))


fn test_emb_cpu_fwd_fuse_sum() raises:
    print("test_emb_cpu_fwd_fuse_sum")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    emb.weight.fill(Tensor[dtype].d1([1.0, 2.0, 3.0]), i(0), s())
    emb.weight.fill(Tensor[dtype].d1([4.0, 5.0, 6.0]), i(1), s())
    # fuse_sum=True: gather rows 0,1 then sum → (3,)
    var result = emb([0, 1], fuse_sum=True)
    assert_true(result.shape() == Shape(3))
    assert_true(result.all_close(Tensor[dtype].d1([5.0, 7.0, 9.0])))


fn test_emb_cpu_fwd_padding_idx_returns_zeros() raises:
    print("test_emb_cpu_fwd_padding_idx_returns_zeros")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=5, embedding_dim=3,
        padding_idx=0, init_method="normal", init_seed=1
    )
    var result = emb([0])
    assert_true(result.shape() == Shape(1, 3))
    assert_true(result.all_close(Tensor[dtype].zeros(Shape(1, 3))))


fn test_emb_cpu_fwd_eval_mode_no_grad() raises:
    print("test_emb_cpu_fwd_eval_mode_no_grad")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3)
    emb.eval()
    var result = emb([0, 1, 2])
    assert_false(result.requires_grad)


fn test_emb_cpu_fwd_train_mode_has_grad() raises:
    print("test_emb_cpu_fwd_train_mode_has_grad")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3)
    emb.train()
    var result = emb([0, 1, 2])
    assert_true(result.requires_grad)

fn test_emb_cpu_fwd_int64_tensor_indices() raises:
    print("test_emb_cpu_fwd_int64_tensor_indices")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    emb.weight.fill(Tensor[dtype].d1([7.0, 8.0, 9.0]), i(2), s())
    var idx = Tensor[DType.int64].d1([2])
    var result = emb(idx)
    assert_true(result.shape() == Shape(1, 3))
    assert_true(result.all_close(Tensor[dtype].d2([[7.0, 8.0, 9.0]])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · gradient flows into looked-up rows
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_cpu_bwd_single_index() raises:
    print("test_emb_cpu_bwd_single_index")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    var result = emb([2])
    var loss   = result.sum()
    loss.backward()
    # Row 2 gets grad=1 for each element, all other rows get grad=0
    var grad = emb.weight.grad().detach(share=True)
    #var grad_row2 = emb.weight.grad()[i(2), s()]
    var grad_row2 = grad[i(2), s()]
    assert_true(grad_row2.all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))
    #var grad_row0 = emb.weight.grad()[i(0), s()]
    var grad_row0 = grad[i(0), s()]
    assert_true(grad_row0.all_close(Tensor[dtype].zeros(Shape(3))))


fn test_emb_cpu_bwd_multiple_indices() raises:
    print("test_emb_cpu_bwd_multiple_indices")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    var result = emb([0, 1, 2])
    var loss   = result.sum()
    loss.backward()
    # Rows 0,1,2 each get grad=[1,1,1], rows 3,4 get grad=[0,0,0]
    var shared_grad = emb.weight.grad().detach(share=True)
    for row in range(3):
        assert_true(
            shared_grad[i(row), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
    for row in range(3, 5):
        assert_true(
            shared_grad[i(row), s()].all_close(Tensor[dtype].zeros(Shape(3)))
        )


fn test_emb_cpu_bwd_repeated_index_accumulates() raises:
    print("test_emb_cpu_bwd_repeated_index_accumulates")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    # Row 2 looked up twice — grad should accumulate (scatter-add)
    var result = emb([2, 2])
    var loss   = result.sum()
    loss.backward()
    # Row 2 contributes 2 × [1,1,1] = [2,2,2]
    var shared_grad = emb.weight.grad().detach(share=True)
    assert_true(
        shared_grad[i(2), s()].all_close(Tensor[dtype].d1([2.0, 2.0, 2.0]))
    )


fn test_emb_cpu_bwd_fuse_sum() raises:
    print("test_emb_cpu_bwd_fuse_sum")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, init_method="zero")
    var result = emb([0, 1, 3], fuse_sum=True)   # output shape (3,)
    var loss   = result.sum()
    loss.backward()
    # Each looked-up row receives same grad = [1,1,1]
    # (sum backward broadcasts scalar grad to all rows)
    var shared_grad = emb.weight.grad().detach(share=True)
    assert_true(
        shared_grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
    )
    assert_true(
        shared_grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
    )
    assert_true(
        shared_grad[i(3), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
    )
    # Non-looked-up rows get zero grad
    assert_true(
        shared_grad[i(2), s()].all_close(Tensor[dtype].zeros(Shape(3)))
    )


fn test_emb_cpu_bwd_padding_idx_no_grad() raises:
    print("test_emb_cpu_bwd_padding_idx_no_grad")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=5, embedding_dim=3,
        padding_idx=0, init_method="normal", init_seed=1
    )
    # Look up padding index — should produce zeros AND receive no grad
    var result = emb([0, 1])
    var loss   = result.sum()
    loss.backward()
    # Row 0 (padding) must have zero grad
    var shared_grad = emb.weight.grad().detach(share=True)
    assert_true(
        shared_grad[i(0), s()].all_close(Tensor[dtype].zeros(Shape(3)))
    )
    # Row 1 must have non-zero grad
    assert_true(
        shared_grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
    )


fn test_emb_cpu_bwd_frozen_no_grad() raises:
    print("test_emb_cpu_bwd_frozen_no_grad")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=5, embedding_dim=3,
        freeze=True, init_method="normal", init_seed=1
    )
    var result = emb([0, 1, 2])
    assert_false(result.requires_grad)
    # No backward needed — weight has no gradbox when frozen


fn test_emb_cpu_bwd_chained_linear() raises:
    print("test_emb_cpu_bwd_chained_linear")
    comptime dtype = DType.float32
    # Embedding → sum → dot with output weights → sigmoid → MSE
    var emb = Embedding[dtype](num_embeddings=10, embedding_dim=4, init_method="zero")
    var w2  = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    var result  = emb([2, 3], fuse_sum=True)   # (4,)
    var pred    = result.dot(w2).sigmoid()
    var target  = Tensor[dtype].scalar(1.0)
    var diff    = pred - target
    var diff_sqrd    = (diff * diff)
    loss = diff_sqrd.squeeze()
    loss.backward()
    # Grad must flow back to embedding rows 2 and 3
    var shared_grad = emb.weight.grad().detach(share=True)
    var grad2 = shared_grad[i(2), s()]
    var grad3 = shared_grad[i(3), s()]
    # Both should be non-zero (exact values depend on sigmoid derivative)
    assert_false(grad2.all_close(Tensor[dtype].zeros(Shape(4))))
    assert_false(grad3.all_close(Tensor[dtype].zeros(Shape(4))))
    # Rows not looked up should have zero grad
    assert_true(shared_grad[i(0), s()].all_close(Tensor[dtype].zeros(Shape(4))))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · Freeze / unfreeze / from_pretrained
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_cpu_freeze_unfreeze() raises:
    print("test_emb_cpu_freeze_unfreeze")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3)
    assert_true(emb.weight.requires_grad)
    emb.freeze()
    assert_false(emb.weight.requires_grad)
    emb.unfreeze()
    assert_true(emb.weight.requires_grad)


fn test_emb_cpu_from_pretrained_values() raises:
    print("test_emb_cpu_from_pretrained_values")
    comptime dtype = DType.float32
    var pretrained = Tensor[dtype].d2([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    var emb = Embedding[dtype].from_pretrained(pretrained)
    assert_true(emb.weight.shape() == Shape(3, 3))
    assert_true(emb.weight.all_close(pretrained))


fn test_emb_cpu_from_pretrained_frozen_by_default() raises:
    print("test_emb_cpu_from_pretrained_frozen_by_default")
    comptime dtype = DType.float32
    var pretrained = Tensor[dtype].d2([[1.0, 2.0],[3.0, 4.0]])
    var emb = Embedding[dtype].from_pretrained(pretrained)
    assert_false(emb.weight.requires_grad)


fn test_emb_cpu_from_pretrained_unfrozen() raises:
    print("test_emb_cpu_from_pretrained_unfrozen")
    comptime dtype = DType.float32
    var pretrained = Tensor[dtype].d2([[1.0, 2.0],[3.0, 4.0]])
    var emb = Embedding[dtype].from_pretrained(pretrained, freeze=False)
    assert_true(emb.weight.requires_grad)


fn test_emb_cpu_from_pretrained_with_padding() raises:
    print("test_emb_cpu_from_pretrained_with_padding")
    comptime dtype = DType.float32
    var pretrained = Tensor[dtype].d2([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    var emb = Embedding[dtype].from_pretrained(pretrained, padding_idx=0)
    # Row 0 must be zeroed even though pretrained had non-zero values
    assert_true(emb.weight[i(0), s()].all_close(Tensor[dtype].zeros(Shape(3))))
    # Other rows unchanged
    assert_true(emb.weight[i(1), s()].all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


fn test_emb_cpu_parameters_not_empty_when_grad() raises:
    print("test_emb_cpu_parameters_not_empty_when_grad")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3)
    var params = emb.parameters()
    assert_true(len(params) == 1)


fn test_emb_cpu_parameters_empty_when_frozen() raises:
    print("test_emb_cpu_parameters_empty_when_frozen")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=5, embedding_dim=3, freeze=True)
    var params = emb.parameters()
    assert_true(len(params) == 0)


fn test_emb_cpu_num_parameters() raises:
    print("test_emb_cpu_num_parameters")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](num_embeddings=100, embedding_dim=16)
    assert_true(emb.num_parameters() == 1600)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CPU · max_norm renormalisation
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_cpu_max_norm_clips_large_rows() raises:
    print("test_emb_cpu_max_norm_clips_large_rows")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=3, embedding_dim=2,
        max_norm=1.0, init_method="zero"
    )
    # Set row 0 to a large vector with norm > 1
    emb.weight.fill(Tensor[dtype].d1([3.0, 4.0]), i(0), s())   # norm=5
    var result = emb([0])
    # After max_norm=1.0 clipping, norm of result row should be <= 1.0
    var norm = result.squeeze().norm().item()
    assert_true(abs(norm - 1.0) < 1e-5)


fn test_emb_cpu_max_norm_preserves_small_rows() raises:
    print("test_emb_cpu_max_norm_preserves_small_rows")
    comptime dtype = DType.float32
    var emb = Embedding[dtype](
        num_embeddings=3, embedding_dim=2,
        max_norm=10.0, init_method="zero"
    )
    # Set row 0 to a small vector with norm < max_norm
    emb.weight.fill(Tensor[dtype].d1([0.3, 0.4]), i(0), s())   # norm=0.5
    var result = emb([0])
    # Small norm — should be unchanged
    assert_true(result.all_close(Tensor[dtype].d2([[0.3, 0.4]])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Forward
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_gpu_fwd_basic_lookup() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_fwd_basic_lookup")
        comptime dtype = DType.float32
        var emb = Embedding[dtype](
            num_embeddings=5, embedding_dim=3, init_method="zero"
        ).to_gpu()
        # Set rows on CPU first then move — or set via fill after to_gpu
        # Use from_pretrained to set known weights
        var weights = Tensor[dtype].d2([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
        var emb_gpu = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var result  = emb_gpu([0, 2, 4])
        assert_true(result.shape() == Shape(3, 3))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])))


fn test_emb_gpu_fwd_repeated_index() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_fwd_repeated_index")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var result = emb([0, 0, 1])
        assert_true(result.shape() == Shape(3, 3))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])))


fn test_emb_gpu_fwd_fuse_sum() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_fwd_fuse_sum")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var result = emb([0, 1], fuse_sum=True)
        assert_true(result.shape() == Shape(3))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([5.0, 7.0, 9.0])))


fn test_emb_gpu_fwd_eval_no_grad() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_fwd_eval_no_grad")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([[1.0, 2.0],[3.0, 4.0]])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        emb.eval()
        var result = emb([0, 1])
        assert_false(result.requires_grad)


fn test_emb_gpu_fwd_output_shape() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_fwd_output_shape")
        comptime dtype = DType.float32
        var emb = Embedding[dtype](
            num_embeddings=100, embedding_dim=64
        ).to_gpu()
        var result = emb([0, 5, 10, 15, 20])
        assert_true(result.shape() == Shape(5, 64))


fn test_emb_gpu_fwd_padding_idx_zeros() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_fwd_padding_idx_zeros")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [9.0, 9.0, 9.0],
            [1.0, 2.0, 3.0],
        ])
        var emb = Embedding[dtype].from_pretrained(
            weights, padding_idx=0, freeze=False
        ).to_gpu()
        var result = emb([0])
        assert_true(result.to_cpu().all_close(Tensor[dtype].zeros(Shape(1, 3))))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — GPU · Backward
# ─────────────────────────────────────────────────────────────────────────────

fn test_emb_gpu_bwd_basic() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_bwd_basic")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var result = emb([1, 3])
        var loss   = result.sum()
        loss.backward()
        # Rows 1 and 3 get grad=[1,1,1]
        var shared_grad = emb.weight.grad().detach(share=True)
        assert_true(
            shared_grad[i(1), s()].to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        assert_true(
            shared_grad[i(3), s()].to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        # Non-looked-up rows get zero grad
        assert_true(
            shared_grad[i(0), s()].to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )


fn test_emb_gpu_bwd_repeated_index_accumulates() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_bwd_repeated_index_accumulates")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var result = emb([0, 0])
        var loss   = result.sum()
        loss.backward()
        # Row 0 looked up twice → grad = [2,2,2] via atomic scatter-add
        var shared_grad = emb.weight.grad().detach(share=True)
        assert_true(
            shared_grad[i(0), s()].to_cpu()
            .all_close(Tensor[dtype].d1([2.0, 2.0, 2.0]))
        )
fn test_emb_gpu_bwd_fuse_sum() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_bwd_fuse_sum")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var result = emb([0, 2], fuse_sum=True)   # (3,)
        var loss   = result.sum()
        loss.backward()
        # Both looked-up rows receive same grad = [1,1,1]
        var shared_grad = emb.weight.grad().detach(share=True)
        assert_true(
            emb.weight.grad()[i(0), s()].to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        assert_true(
            shared_grad[i(2), s()].to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )
        assert_true(
            shared_grad[i(1), s()].to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )


fn test_emb_gpu_bwd_padding_idx_no_grad() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_bwd_padding_idx_no_grad")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        var emb = Embedding[dtype].from_pretrained(
            weights, padding_idx=0, freeze=False
        ).to_gpu()
        var result = emb([0, 1])
        var loss   = result.sum()
        loss.backward()
        var shared_grad = emb.weight.grad().detach(share=True)
        # Padding row must have zero grad
        assert_true(
            shared_grad[i(0), s()].to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(3)))
        )
        # Non-padding row has correct grad
        assert_true(
            shared_grad[i(1), s()].to_cpu()
            .all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
        )


fn test_emb_gpu_bwd_chained() raises:
    comptime if has_accelerator():
        print("test_emb_gpu_bwd_chained")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].d2([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        var emb = Embedding[dtype].from_pretrained(weights, freeze=False).to_gpu()
        var w2  = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0], requires_grad=True).to_gpu()
        var result = emb([0, 1], fuse_sum=True)   # (4,)
        var pred   = result.dot(w2).sigmoid()
        var target = Tensor[dtype].scalar(1.0).to_gpu()
        var diff   = pred - target
        var diff_sqrd   = (diff * diff)
        loss = diff_sqrd.squeeze()
        loss.backward()
        var shared_grad = emb.weight.grad().detach(share=True)
        # Grad flows back to rows 0 and 1
        assert_false(
            shared_grad[i(0), s()].to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(4)))
        )
        assert_false(
            shared_grad[i(1), s()].to_cpu()
            .all_close(Tensor[dtype].zeros(Shape(4)))
        )

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll Embedding tests passed!")



