from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from mnemonics import mv

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

fn close_enough[dtype: DType](
    mut a: Tensor[dtype], mut b: Tensor[dtype]
) raises -> Bool:
    var a_gpu = a.to_gpu()
    return a_gpu.all_close(b.to_gpu())


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════

fn test_mvnd_2d_M_1d_v() raises:
    """M[m, k] @ v[k] → out[m]. Simplest case, no batch dims."""
    print("test_mvnd_2d_M_1d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # M = [[1,2,3],[4,5,6]], v = [1,1,1]
        # out = [1+2+3, 4+5+6] = [6, 15]
        var M = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var v = Tensor[dtype].d1([1, 1, 1])
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_2d_M_1d_v passed")


fn test_mvnd_known_values() raises:
    """Hand-computed result verified directly against GPU output."""
    print("test_mvnd_known_values")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # M = [[1,0],[0,1],[1,1]], v = [3, 4]
        # out = [1*3+0*4, 0*3+1*4, 1*3+1*4] = [3, 4, 7]
        var M = Tensor[dtype].d2([[1, 0], [0, 1], [1, 1]])
        var v = Tensor[dtype].d1([3, 4])
        var expected = Tensor[dtype].d1([3, 4, 7])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_known_values passed")


fn test_mvnd_identity_matrix() raises:
    """I @ v = v."""
    print("test_mvnd_identity_matrix")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].eye(4)
        var v = Tensor[dtype].d1([2, 5, 1, 8])
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_identity_matrix passed")


fn test_mvnd_zero_vector() raises:
    """M @ zero_vector = zero output."""
    print("test_mvnd_zero_vector")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(3, 4)
        var v = Tensor[dtype].zeros(4)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_zero_vector passed")


fn test_mvnd_ones_vector() raises:
    """M @ ones = row sums of M."""
    print("test_mvnd_ones_vector")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
        var M = Tensor[dtype].d2([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        var v = Tensor[dtype].ones(4)
        var expected = Tensor[dtype].d1([10, 26, 42])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_ones_vector passed")


fn test_mvnd_single_row_matrix() raises:
    """M[1, k] @ v[k] → out[1]. Single row edge case."""
    print("test_mvnd_single_row_matrix")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2, 3, 4]])   # (1, 3)
        var v = Tensor[dtype].d1([1, 2, 3])
        # out = [2*1 + 3*2 + 4*3] = [20]
        var expected = Tensor[dtype].d1([20])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_single_row_matrix passed")


fn test_mvnd_single_col_matrix() raises:
    """M[m, 1] @ v[1] → out[m]. k=1 edge case."""
    print("test_mvnd_single_col_matrix")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2], [3], [5]])   # (3, 1)
        var v = Tensor[dtype].d1([4])
        # out = [8, 12, 20]
        var expected = Tensor[dtype].d1([8, 12, 20])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_single_col_matrix passed")


fn test_mvnd_large_k() raises:
    """Large k to stress the dot product loop."""
    print("test_mvnd_large_k")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var k = 512
        var m = 32
        var M = Tensor[dtype].ones(m, k)
        var v = Tensor[dtype].ones(k)
        # each output element = k = 512
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_large_k passed")


fn test_mvnd_large_m() raises:
    """M > block_size to exercise multi-block coverage."""
    print("test_mvnd_large_m")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var m = 1024   # larger than default block_size=256
        var k = 8
        var M = Tensor[dtype].ones(m, k)
        var v = Tensor[dtype].ones(k)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_large_m passed")


fn test_mvnd_negative_values() raises:
    """Negative values in both M and v."""
    print("test_mvnd_negative_values")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[-1, 2], [3, -4]])
        var v = Tensor[dtype].d1([-1, 2])
        # out = [(-1*-1 + 2*2), (3*-1 + -4*2)] = [5, -11]
        var expected = Tensor[dtype].d1([5, -11])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_negative_values passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — M and v same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════

fn test_mvnd_batched_3d_M_2d_v() raises:
    """M[b, m, k] @ v[b, k] → out[b, m]."""
    print("test_mvnd_batched_3d_M_2d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)   # (2, 3, 3)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)      # (2, 3)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_3d_M_2d_v passed")


fn test_mvnd_batched_4d_M_3d_v() raises:
    """M[a, b, m, k] @ v[a, b, k] → out[a, b, m]."""
    print("test_mvnd_batched_4d_M_3d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)   # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(2, 3, 5)       # (2, 3, 5)
        # each output element = k = 5
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_4d_M_3d_v passed")


fn test_mvnd_batched_arange_values() raises:
    """Batched with non-trivial arange values to catch index mapping errors."""
    print("test_mvnd_batched_arange_values")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # M: (3, 4, 2),  v: (3, 2)
        var M = Tensor[dtype].arange(24)
        M = M.reshape(3, 4, 2)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(3, 2)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_batched_arange_values passed")


fn test_mvnd_known_values_batched() raises:
    """Hand-computed batched result verified directly against GPU."""
    print("test_mvnd_known_values_batched")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: [[1,0],[0,1]] @ [1,2] = [1, 2]
        # batch 1: [[2,0],[0,2]] @ [3,4] = [6, 8]
        var M = Tensor[dtype].d3([[[1,0],[0,1]], [[2,0],[0,2]]])
        var v = Tensor[dtype].d2([[1, 2], [3, 4]])
        var expected = Tensor[dtype].d2([[1, 2], [6, 8]])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_known_values_batched passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — M and v have different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════

fn test_mvnd_broadcast_3d_M_1d_v() raises:
    """M[b, m, k] broadcast against v[k] → out[b, m]."""
    print("test_mvnd_broadcast_3d_M_1d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)           # (2, 3, 3)
        var v = Tensor[dtype].d1([1, 0, 1])  # (3,) broadcasts over batch
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_3d_M_1d_v passed")


fn test_mvnd_broadcast_2d_M_3d_v() raises:
    """M[m, k] broadcast against v[b, k] → out[b, m]."""
    print("test_mvnd_broadcast_2d_M_3d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)              # (4, 3) — no batch
        var v = Tensor[dtype].arange(9)
        v = v.reshape(3, 3)              # (3, 3) — batch of 3
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_2d_M_3d_v passed")


fn test_mvnd_broadcast_4d_M_2d_v() raises:
    """M[a, b, m, k] broadcast against v[b, k] → out[a, b, m]."""
    print("test_mvnd_broadcast_4d_M_2d_v")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)   # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(3, 5)          # (3, 5) — broadcasts over dim 0
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_4d_M_2d_v passed")


fn test_mvnd_broadcast_size1_batch() raises:
    """V with size-1 batch dim that broadcasts across M's batch."""
    print("test_mvnd_broadcast_size1_batch")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(4, 3, 5)   # (4, 3, 5)
        var v = Tensor[dtype].ones(1, 5)       # (1, 5) → broadcasts to (4, 5)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_size1_batch passed")


fn test_mvnd_broadcast_known_values() raises:
    """Hand-computed broadcast result verified directly against GPU."""
    print("test_mvnd_broadcast_known_values")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # M[0] = [[1,2],[3,4]], M[1] = [[5,6],[7,8]]
        # v = [1, 1]  (no batch — broadcasts across both)
        # out[0] = [3, 7],  out[1] = [11, 15]
        var M = Tensor[dtype].d3([[[1,2],[3,4]], [[5,6],[7,8]]])
        var v = Tensor[dtype].d1([1, 1])
        var expected = Tensor[dtype].d2([[3, 7], [11, 15]])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))
    print("test_mvnd_broadcast_known_values passed")


fn test_mvnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""
    print("test_mvnd_broadcast_large_batch")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var m = 64
        var k = 32
        var M = Tensor[dtype].ones(128, m, k)   # large batch on M side
        var v = Tensor[dtype].ones(k)            # no batch — broadcasts
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_mvnd_broadcast_large_batch passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    # Basic correctness
    test_mvnd_2d_M_1d_v()
    test_mvnd_known_values()
    test_mvnd_identity_matrix()
    test_mvnd_zero_vector()
    test_mvnd_ones_vector()
    test_mvnd_single_row_matrix()
    test_mvnd_single_col_matrix()
    test_mvnd_large_k()
    test_mvnd_large_m()
    test_mvnd_negative_values()

    # Batched — same batch shape
    test_mvnd_batched_3d_M_2d_v()
    test_mvnd_batched_4d_M_3d_v()
    test_mvnd_batched_arange_values()
    test_mvnd_known_values_batched()

    # Broadcast — different batch ranks
    test_mvnd_broadcast_3d_M_1d_v()
    test_mvnd_broadcast_2d_M_3d_v()
    test_mvnd_broadcast_4d_M_2d_v()
    test_mvnd_broadcast_size1_batch()
    test_mvnd_broadcast_known_values()
    test_mvnd_broadcast_large_batch()
