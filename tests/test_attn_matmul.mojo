from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.common_utils import i, s

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — manual 2D matmul reference for small matrices
# ─────────────────────────────────────────────────────────────────────────────

fn matmul_ref[dtype: DType](
    A: Tensor[dtype], B: Tensor[dtype]
) -> Tensor[dtype]:
    """Naive O(n^3) 2D matmul for reference comparison."""
    var M = A.shape()[0]
    var K = A.shape()[1]
    var N = B.shape()[1]
    var out = Tensor[dtype].zeros(Shape(M, N))
    for i in range(M):
        for k in range(K):
            for j in range(N):
                out[i, j] = out[i, j] + A[i, k] * B[k, j]
    return out^


# ═════════════════════════════════════════════════════════════════════════════
# CPU TESTS
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — 2D mm
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_2d_basic() raises:
    print("test_batchmm_2d_basic")
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(2, 2))
    assert_true(C.all_close(Tensor[dtype].d2([[19.0, 22.0], [43.0, 50.0]])))


fn test_batchmm_2d_non_square() raises:
    print("test_batchmm_2d_non_square")
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
    var B = Tensor[dtype].arange(1.0, 9.0).reshape(4, 2)
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(3, 2))
    var reference = matmul_ref(A.reshape(3, 4), B.reshape(4, 2))
    assert_true(C.all_close[atol=1e-4](reference))


# ─────────────────────────────────────────────────────────────────────────────
# 2. 3D BATCHED mm
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_3d_shape() raises:
    print("test_batchmm_3d_shape")
    comptime dtype = DType.float32
    var A = Tensor[dtype].ones(Shape(4, 3, 5))
    var B = Tensor[dtype].ones(Shape(4, 5, 2))
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(4, 3, 2))


fn test_batchmm_3d_values() raises:
    print("test_batchmm_3d_values")
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(1.0, 13.0).reshape(2, 2, 3)
    var B = Tensor[dtype].arange(1.0, 13.0).reshape(2, 3, 2)
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(2, 2, 2))
    var A0  = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B0  = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var ref0 = matmul_ref(A0, B0)
    var A1  = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    var B1  = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var ref1 = matmul_ref(A1, B1)
    assert_true(C[i(0), s(), s()].all_close[atol=1e-3](ref0))
    assert_true(C[i(1), s(), s()].all_close[atol=1e-3](ref1))


fn test_batchmm_3d_identity() raises:
    print("test_batchmm_3d_identity")
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(1.0, 25.0).reshape(3, 4, 2)
    var I = Tensor[dtype].zeros(Shape(3, 2, 2))
    for b in range(3):
        I[b, 0, 0] = 1.0
        I[b, 1, 1] = 1.0
    var C = A.matmul(I)
    assert_true(C.shape() == Shape(3, 4, 2))
    assert_true(C.all_close[atol=1e-5](A))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 4D BATCHED mm — attention shapes
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_4d_shape_scores() raises:
    print("test_batchmm_4d_shape_scores")
    comptime dtype = DType.float32
    var Q = Tensor[dtype].ones(Shape(2, 4, 8, 16))
    var K = Tensor[dtype].ones(Shape(2, 4, 16, 8))
    var scores = Q.matmul(K)
    assert_true(scores.shape() == Shape(2, 4, 8, 8))


fn test_batchmm_4d_shape_context() raises:
    print("test_batchmm_4d_shape_context")
    comptime dtype = DType.float32
    var weights = Tensor[dtype].ones(Shape(2, 4, 8, 8))
    var V       = Tensor[dtype].ones(Shape(2, 4, 8, 16))
    var ctx     = weights.matmul(V)
    assert_true(ctx.shape() == Shape(2, 4, 8, 16))


fn test_batchmm_4d_scores_values() raises:
    print("test_batchmm_4d_scores_values")
    comptime dtype = DType.float32
    var Q  = Tensor[dtype].ones(Shape(1, 1, 2, 3))
    var Kt = Tensor[dtype].ones(Shape(1, 1, 3, 2))
    var scores = Q.matmul(Kt)
    assert_true(scores.shape() == Shape(1, 1, 2, 2))
    assert_true(scores.all_close[atol=1e-5](
        Tensor[dtype].full(Shape(1, 1, 2, 2), 3.0)
    ))


fn test_batchmm_4d_context_values() raises:
    print("test_batchmm_4d_context_values")
    comptime dtype = DType.float32
    var B = 1; var H = 1; var T = 4; var D = 2
    var weights = Tensor[dtype].full(Shape(B, H, T, T), 1.0 / Float32(T))
    var V       = Tensor[dtype].arange(0.0, 8.0).reshape(B, H, T, D)
    var ctx     = weights.matmul(V)
    assert_true(ctx.shape() == Shape(B, H, T, D))
    var expected = Tensor[dtype].zeros(Shape(B, H, T, D))
    for t in range(T):
        expected[0, 0, t, 0] = 3.0
        expected[0, 0, t, 1] = 4.0
    assert_true(ctx.all_close[atol=1e-5](expected))


fn test_batchmm_4d_multi_head_independent() raises:
    print("test_batchmm_4d_multi_head_independent")
    comptime dtype = DType.float32
    var B = 1; var H = 2; var T = 3; var D = 4
    var weights = Tensor[dtype].ones(Shape(B, H, T, T))
    var V = Tensor[dtype].zeros(Shape(B, H, T, D))
    for t in range(T):
        for d in range(D):
            V[0, 1, t, d] = 1.0
    var ctx = weights.matmul(V)
    assert_true(ctx.shape() == Shape(B, H, T, D))
    for t in range(T):
        for d in range(D):
            assert_true(abs(ctx[0, 0, t, d]) < 1e-5)
            assert_true(abs(ctx[0, 1, t, d] - Float32(T)) < 1e-4)


fn test_batchmm_4d_realistic_attention_shape() raises:
    print("test_batchmm_4d_realistic_attention_shape")
    comptime dtype = DType.float32
    var Q  = Tensor[dtype].randn(Shape(2, 4, 16, 8),  mean=0.0, std=0.02)
    var Kt = Tensor[dtype].randn(Shape(2, 4, 8,  16), mean=0.0, std=0.02)
    var V  = Tensor[dtype].randn(Shape(2, 4, 16, 8),  mean=0.0, std=0.02)
    var scores  = Q.matmul(Kt)
    var context = scores.matmul(V)
    assert_true(scores.shape()  == Shape(2, 4, 16, 16))
    assert_true(context.shape() == Shape(2, 4, 16, 8))


fn test_batchmm_4d_broadcast_batch_dim() raises:
    print("test_batchmm_4d_broadcast_batch_dim")
    comptime dtype = DType.float32
    var Q  = Tensor[dtype].ones(Shape(1, 2, 4, 8))
    var Kt = Tensor[dtype].ones(Shape(3, 2, 8, 4))
    var scores = Q.matmul(Kt)
    assert_true(scores.shape() == Shape(3, 2, 4, 4))


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKWARD — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_4d_backward_scores() raises:
    print("test_batchmm_4d_backward_scores")
    comptime dtype = DType.float32
    var B = 1; var H = 2; var T = 3; var D = 4
    var Q  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
    var Kt = Tensor[dtype].ones(Shape(B, H, D, T), requires_grad=True)
    var scores = Q.matmul(Kt)
    var loss = scores.sum()
    loss.backward()
    assert_true(Q.grad().all_close[atol=1e-4](
        Tensor[dtype].full(Shape(B, H, T, D), Float32(T))
    ))
    assert_true(Kt.grad().all_close[atol=1e-4](
        Tensor[dtype].full(Shape(B, H, D, T), Float32(T))
    ))


fn test_batchmm_4d_backward_context() raises:
    print("test_batchmm_4d_backward_context")
    comptime dtype = DType.float32
    var B = 1; var H = 2; var T = 3; var D = 4
    var W = Tensor[dtype].ones(Shape(B, H, T, T), requires_grad=True)
    var V = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
    var ctx = W.matmul(V)
    var loss = ctx.sum()
    loss.backward()
    assert_true(W.grad().all_close[atol=1e-4](
        Tensor[dtype].full(Shape(B, H, T, T), Float32(D))
    ))
    assert_true(V.grad().all_close[atol=1e-4](
        Tensor[dtype].full(Shape(B, H, T, D), Float32(T))
    ))


fn test_batchmm_4d_backward_full_attention_chain() raises:
    print("test_batchmm_4d_backward_full_attention_chain")
    comptime dtype = DType.float32
    var B = 1; var H = 1; var T = 4; var D = 4
    var Q  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
    var Kt = Tensor[dtype].ones(Shape(B, H, D, T), requires_grad=True)
    var V  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
    var scores  = Q.matmul(Kt)
    var weights = scores.softmax[track_grad=True]([-1])
    var ctx     = weights.matmul(V)
    var loss    = ctx.sum()
    loss.backward()
    assert_true(Q.grad().shape()  == Shape(B, H, T, D))
    assert_true(Kt.grad().shape() == Shape(B, H, D, T))
    assert_true(V.grad().shape()  == Shape(B, H, T, D))
    assert_true(Q.grad().sum().item()  == Q.grad().sum().item())
    assert_true(Kt.grad().sum().item() == Kt.grad().sum().item())
    assert_true(V.grad().sum().item()  == V.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# GPU TESTS
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — 2D mm GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_gpu_2d_basic() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_2d_basic")
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]]).to_gpu()
        var C = A.matmul(B)
        assert_true(C.shape() == Shape(2, 2))
        assert_true(C.to_cpu().all_close(
            Tensor[dtype].d2([[19.0, 22.0], [43.0, 50.0]])
        ))


fn test_batchmm_gpu_2d_non_square() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_2d_non_square")
        comptime dtype = DType.float32
        var A_cpu = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
        var B_cpu = Tensor[dtype].arange(1.0, 9.0).reshape(4, 2)
        var reference   = matmul_ref(A_cpu, B_cpu)
        var C = A_cpu.to_gpu().matmul(B_cpu.to_gpu())
        assert_true(C.shape() == Shape(3, 2))
        assert_true(C.to_cpu().all_close[atol=1e-4](reference))


# ─────────────────────────────────────────────────────────────────────────────
# 2. 3D BATCHED mm GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_gpu_3d_shape() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_3d_shape")
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(Shape(4, 3, 5)).to_gpu()
        var B = Tensor[dtype].ones(Shape(4, 5, 2)).to_gpu()
        var C = A.matmul(B)
        assert_true(C.shape() == Shape(4, 3, 2))


fn test_batchmm_gpu_3d_values() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_3d_values")
        comptime dtype = DType.float32
        var A_cpu = Tensor[dtype].arange(1.0, 13.0).reshape(2, 2, 3)
        var B_cpu = Tensor[dtype].arange(1.0, 13.0).reshape(2, 3, 2)
        var C = A_cpu.to_gpu().matmul(B_cpu.to_gpu()).to_cpu()
        assert_true(C.shape() == Shape(2, 2, 2))
        var A0   = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var B0   = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var ref0 = matmul_ref(A0, B0)
        var A1   = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        var B1   = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        var ref1 = matmul_ref(A1, B1)
        assert_true(C[i(0), s(), s()].all_close[atol=1e-3](ref0))
        assert_true(C[i(1), s(), s()].all_close[atol=1e-3](ref1))


fn test_batchmm_gpu_3d_identity() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_3d_identity")
        comptime dtype = DType.float32
        var A_cpu = Tensor[dtype].arange(1.0, 25.0).reshape(3, 4, 2)
        var I_cpu = Tensor[dtype].zeros(Shape(3, 2, 2))
        for b in range(3):
            I_cpu[b, 0, 0] = 1.0
            I_cpu[b, 1, 1] = 1.0
        var C = A_cpu.to_gpu().matmul(I_cpu.to_gpu()).to_cpu()
        assert_true(C.shape() == Shape(3, 4, 2))
        assert_true(C.all_close[atol=1e-5](A_cpu))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 4D BATCHED mm GPU — attention shapes
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_gpu_4d_shape_scores() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_shape_scores")
        comptime dtype = DType.float32
        var Q = Tensor[dtype].ones(Shape(2, 4, 8, 16)).to_gpu()
        var K = Tensor[dtype].ones(Shape(2, 4, 16, 8)).to_gpu()
        var scores = Q.matmul(K)
        assert_true(scores.shape() == Shape(2, 4, 8, 8))


fn test_batchmm_gpu_4d_shape_context() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_shape_context")
        comptime dtype = DType.float32
        var weights = Tensor[dtype].ones(Shape(2, 4, 8, 8)).to_gpu()
        var V       = Tensor[dtype].ones(Shape(2, 4, 8, 16)).to_gpu()
        var ctx     = weights.matmul(V)
        assert_true(ctx.shape() == Shape(2, 4, 8, 16))


fn test_batchmm_gpu_4d_scores_values() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_scores_values")
        comptime dtype = DType.float32
        var Q  = Tensor[dtype].ones(Shape(1, 1, 2, 3)).to_gpu()
        var Kt = Tensor[dtype].ones(Shape(1, 1, 3, 2)).to_gpu()
        var scores = Q.matmul(Kt).to_cpu()
        assert_true(scores.shape() == Shape(1, 1, 2, 2))
        assert_true(scores.all_close[atol=1e-5](
            Tensor[dtype].full(Shape(1, 1, 2, 2), 3.0)
        ))


fn test_batchmm_gpu_4d_context_values() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_context_values")
        comptime dtype = DType.float32
        var B = 1; var H = 1; var T = 4; var D = 2
        var weights = Tensor[dtype].full(
            Shape(B, H, T, T), 1.0 / Float32(T)
        ).to_gpu()
        var V = Tensor[dtype].arange(0.0, 8.0).reshape(B, H, T, D).to_gpu()
        var ctx = weights.matmul(V).to_cpu()
        assert_true(ctx.shape() == Shape(B, H, T, D))
        var expected = Tensor[dtype].zeros(Shape(B, H, T, D))
        for t in range(T):
            expected[0, 0, t, 0] = 3.0
            expected[0, 0, t, 1] = 4.0
        assert_true(ctx.all_close[atol=1e-5](expected))


fn test_batchmm_gpu_4d_multi_head_independent() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_multi_head_independent")
        comptime dtype = DType.float32
        var B = 1; var H = 2; var T = 3; var D = 4
        var weights = Tensor[dtype].ones(Shape(B, H, T, T)).to_gpu()
        var V_cpu = Tensor[dtype].zeros(Shape(B, H, T, D))
        for t in range(T):
            for d in range(D):
                V_cpu[0, 1, t, d] = 1.0
        var ctx = weights.matmul(V_cpu.to_gpu()).to_cpu()
        assert_true(ctx.shape() == Shape(B, H, T, D))
        for t in range(T):
            for d in range(D):
                assert_true(abs(ctx[0, 0, t, d]) < 1e-5)
                assert_true(abs(ctx[0, 1, t, d] - Float32(T)) < 1e-4)


fn test_batchmm_gpu_4d_realistic_attention_shape() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_realistic_attention_shape")
        comptime dtype = DType.float32
        var Q  = Tensor[dtype].randn(Shape(2, 4, 16, 8),  mean=0.0, std=0.02).to_gpu()
        var Kt = Tensor[dtype].randn(Shape(2, 4, 8,  16), mean=0.0, std=0.02).to_gpu()
        var V  = Tensor[dtype].randn(Shape(2, 4, 16, 8),  mean=0.0, std=0.02).to_gpu()
        var scores  = Q.matmul(Kt)
        var context = scores.matmul(V)
        assert_true(scores.shape()  == Shape(2, 4, 16, 16))
        assert_true(context.shape() == Shape(2, 4, 16, 8))


fn test_batchmm_gpu_4d_broadcast_batch_dim() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_broadcast_batch_dim")
        comptime dtype = DType.float32
        var Q  = Tensor[dtype].ones(Shape(1, 2, 4, 8)).to_gpu()
        var Kt = Tensor[dtype].ones(Shape(3, 2, 8, 4)).to_gpu()
        var scores = Q.matmul(Kt)
        assert_true(scores.shape() == Shape(3, 2, 4, 4))


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKWARD — GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_gpu_4d_backward_scores() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_backward_scores")
        comptime dtype = DType.float32
        var B = 1; var H = 2; var T = 3; var D = 4
        var Q_cpu  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
        var Kt_cpu = Tensor[dtype].ones(Shape(B, H, D, T), requires_grad=True)
        var scores = Q_cpu.to_gpu().matmul(Kt_cpu.to_gpu())
        var loss   = scores.sum()
        loss.backward()
        assert_true(Q_cpu.grad().all_close[atol=1e-4](
            Tensor[dtype].full(Shape(B, H, T, D), Float32(T))
        ))
        assert_true(Kt_cpu.grad().all_close[atol=1e-4](
            Tensor[dtype].full(Shape(B, H, D, T), Float32(T))
        ))


fn test_batchmm_gpu_4d_backward_context() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_backward_context")
        comptime dtype = DType.float32
        var B = 1; var H = 2; var T = 3; var D = 4
        var W_cpu = Tensor[dtype].ones(Shape(B, H, T, T), requires_grad=True)
        var V_cpu = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
        var ctx  = W_cpu.to_gpu().matmul(V_cpu.to_gpu())
        var loss = ctx.sum()
        loss.backward()
        assert_true(W_cpu.grad().all_close[atol=1e-4](
            Tensor[dtype].full(Shape(B, H, T, T), Float32(D))
        ))
        assert_true(V_cpu.grad().all_close[atol=1e-4](
            Tensor[dtype].full(Shape(B, H, T, D), Float32(T))
        ))


fn test_batchmm_gpu_4d_backward_full_attention_chain() raises:
    comptime if has_accelerator():
        print("test_batchmm_gpu_4d_backward_full_attention_chain")
        comptime dtype = DType.float32
        var B = 1; var H = 1; var T = 4; var D = 4
        var Q_cpu  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
        var Kt_cpu = Tensor[dtype].ones(Shape(B, H, D, T), requires_grad=True)
        var V_cpu  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
        var scores  = Q_cpu.to_gpu().matmul(Kt_cpu.to_gpu())
        var weights = scores.softmax[track_grad=True]([-1])
        var ctx     = weights.matmul(V_cpu.to_gpu())
        var loss    = ctx.sum()
        loss.backward()
        assert_true(Q_cpu.grad().shape()  == Shape(B, H, T, D))
        assert_true(Kt_cpu.grad().shape() == Shape(B, H, D, T))
        assert_true(V_cpu.grad().shape()  == Shape(B, H, T, D))
        assert_true(Q_cpu.grad().sum().item()  == Q_cpu.grad().sum().item())
        assert_true(Kt_cpu.grad().sum().item() == Kt_cpu.grad().sum().item())
        assert_true(V_cpu.grad().sum().item()  == V_cpu.grad().sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# 5. CPU / GPU PARITY
# ─────────────────────────────────────────────────────────────────────────────

fn test_batchmm_parity_2d() raises:
    comptime if has_accelerator():
        print("test_batchmm_parity_2d")
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
        var B = Tensor[dtype].arange(1.0, 9.0).reshape(4, 2)
        var cpu_out = A.matmul(B)
        var gpu_out = A.to_gpu().matmul(B.to_gpu()).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_batchmm_parity_3d() raises:
    comptime if has_accelerator():
        print("test_batchmm_parity_3d")
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(1.0, 25.0).reshape(2, 3, 4)
        var B = Tensor[dtype].arange(1.0, 25.0).reshape(2, 4, 3)
        var cpu_out = A.matmul(B)
        var gpu_out = A.to_gpu().matmul(B.to_gpu()).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-3](gpu_out))


fn test_batchmm_parity_4d_scores() raises:
    comptime if has_accelerator():
        print("test_batchmm_parity_4d_scores")
        comptime dtype = DType.float32
        var Q  = Tensor[dtype].randn(Shape(2, 4, 8, 16), mean=0.0, std=0.02)
        var Kt = Tensor[dtype].randn(Shape(2, 4, 16, 8), mean=0.0, std=0.02)
        var cpu_out = Q.matmul(Kt)
        var gpu_out = Q.to_gpu().matmul(Kt.to_gpu()).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_batchmm_parity_4d_context() raises:
    comptime if has_accelerator():
        print("test_batchmm_parity_4d_context")
        comptime dtype = DType.float32
        var W = Tensor[dtype].randn(Shape(2, 4, 8, 8),  mean=0.0, std=0.02)
        var V = Tensor[dtype].randn(Shape(2, 4, 8, 16), mean=0.0, std=0.02)
        var cpu_out = W.matmul(V)
        var gpu_out = W.to_gpu().matmul(V.to_gpu()).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_batchmm_parity_4d_backward() raises:
    comptime if has_accelerator():
        print("test_batchmm_parity_4d_backward")
        comptime dtype = DType.float32
        var B = 1; var H = 2; var T = 4; var D = 4

        var Q_cpu_leaf  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
        var Kt_cpu_leaf = Tensor[dtype].ones(Shape(B, H, D, T), requires_grad=True)
        var loss_cpu = Q_cpu_leaf.matmul(Kt_cpu_leaf).sum()
        loss_cpu.backward()

        var Q_gpu_leaf  = Tensor[dtype].ones(Shape(B, H, T, D), requires_grad=True)
        var Kt_gpu_leaf = Tensor[dtype].ones(Shape(B, H, D, T), requires_grad=True)
        var loss_gpu = Q_gpu_leaf.to_gpu().matmul(Kt_gpu_leaf.to_gpu()).sum()
        loss_gpu.backward()

        assert_true(Q_cpu_leaf.grad().all_close[atol=1e-4](Q_gpu_leaf.grad()))
        assert_true(Kt_cpu_leaf.grad().all_close[atol=1e-4](Kt_gpu_leaf.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll attn matmul tests passed!")
