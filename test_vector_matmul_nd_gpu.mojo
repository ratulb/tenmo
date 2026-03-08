from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from mnemonics import vm

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

fn test_vmnd_1d_v_2d_M() raises:
    """V[k] @ M[k, n] → out[n]. Simplest case, no batch dims."""
    print("test_vmnd_1d_v_2d_M")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 2, 3],  M = [[1,0],[0,1],[1,1]]
        # out = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        var v = Tensor[dtype].d1([1, 2, 3])
        var M = Tensor[dtype].d2([[1, 0], [0, 1], [1, 1]])
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_1d_v_2d_M passed")


fn test_vmnd_identity_matrix() raises:
    """V @ I = v."""
    print("test_vmnd_identity_matrix")
    _="""@parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([3, 1, 4, 1, 5])
        var I = Tensor[dtype].eye(5)
        var cpu_result = v.matmul[mode=vm](I)
        var v_gpu = v.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](I.to_gpu())
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_identity_matrix passed")"""


fn test_vmnd_zero_vector() raises:
    """Zero vector gives zero output."""
    print("test_vmnd_zero_vector")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].zeros(4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_zero_vector passed")


fn test_vmnd_ones_vector() raises:
    """Ones vector sums columns of M."""
    print("test_vmnd_ones_vector")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(3)
        var M = Tensor[dtype].d2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        # out = [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_ones_vector passed")


fn test_vmnd_single_output_element() raises:
    """N=1: output is a scalar-like vector."""
    print("test_vmnd_single_output_element")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([2, 3, 4])
        var M = Tensor[dtype].d2([[1], [2], [3]])
        # out = [2*1 + 3*2 + 4*3] = [20]
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_single_output_element passed")


fn test_vmnd_large_k() raises:
    """Large k to stress the dot product loop."""
    print("test_vmnd_large_k")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var k = 512
        var n = 64
        var v = Tensor[dtype].ones(k)
        var M = Tensor[dtype].ones(k, n)
        # out[j] = k for all j
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_large_k passed")


fn test_vmnd_large_n() raises:
    """N > block_size to exercise multi-block coverage."""
    print("test_vmnd_large_n")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var k = 8
        var n = 1024   # larger than default block_size=256
        var v = Tensor[dtype].ones(k)
        var M = Tensor[dtype].ones(k, n)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_large_n passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — v and M same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════

fn test_vmnd_batched_2d_v_3d_M() raises:
    """V[b, k] @ M[b, k, n] → out[b, n]."""
    print("test_vmnd_batched_2d_v_3d_M")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)          # (2, 3)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)      # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_2d_v_3d_M passed")


fn test_vmnd_batched_3d_v_4d_M() raises:
    """V[a, b, k] @ M[a, b, k, n] → out[a, b, n]."""
    print("test_vmnd_batched_3d_v_4d_M")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(2, 3, 4)      # (2, 3, 4)
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        # out[a,b,j] = 4.0 for all (a,b,j)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_3d_v_4d_M passed")


fn test_vmnd_batched_arange_values() raises:
    """Batched with non-trivial values to catch index mapping errors."""
    print("test_vmnd_batched_arange_values")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # v: (3, 4),  M: (3, 4, 2)
        var v = Tensor[dtype].arange(12)
        v = v.reshape(3, 4)
        var M = Tensor[dtype].arange(24)
        M = M.reshape(3, 4, 2)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_batched_arange_values passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — v and M have different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════

fn test_vmnd_broadcast_v1d_M3d() raises:
    """V[k] broadcast against M[b, k, n] → out[b, n]."""
    print("test_vmnd_broadcast_v1d_M3d")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([1, 0, 1])          # (3,)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)                        # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v1d_M3d passed")


fn test_vmnd_broadcast_v2d_M3d() raises:
    """V[1, k] broadcast against M[b, k, n] → out[b, n]."""
    print("test_vmnd_broadcast_v2d_M3d")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)             # (1, 4)
        var M = Tensor[dtype].arange(48)
        M = M.reshape(3, 4, 4)                        # (3, 4, 4)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v2d_M3d passed")


fn test_vmnd_broadcast_v3d_M2d() raises:
    """V[a, b, k] broadcast against M[k, n] → out[a, b, n]."""
    print("test_vmnd_broadcast_v3d_M2d")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(24)
        v = v.reshape(2, 3, 4)                        # (2, 3, 4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)                            # (4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_v3d_M2d passed")


fn test_vmnd_broadcast_both_size1() raises:
    """Both v and M have a size-1 batch dim that broadcasts."""
    print("test_vmnd_broadcast_both_size1")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)             # (1, 4) → broadcasts to (3, 4)
        var M = Tensor[dtype].ones(3, 4, 5)          # (3, 4, 5)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_both_size1 passed")


fn test_vmnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""
    print("test_vmnd_broadcast_large_batch")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # v: (32, k), M: (k, n) — M broadcast across 32 batch elements
        var k = 64
        var n = 128
        var v = Tensor[dtype].ones(32, k)
        var M = Tensor[dtype].ones(k, n)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))
    print("test_vmnd_broadcast_large_batch passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical correctness — spot-check known values
# ═══════════════════════════════════════════════════════════════════════════════

fn test_vmnd_known_values_no_batch() raises:
    """Hand-computed result verified against GPU."""
    print("test_vmnd_known_values_no_batch")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 2],  M = [[3, 4, 5], [6, 7, 8]]
        # out = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        var v = Tensor[dtype].d1([1, 2])
        var M = Tensor[dtype].d2([[3, 4, 5], [6, 7, 8]])
        var expected = Tensor[dtype].d1([15, 18, 21])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))
    print("test_vmnd_known_values_no_batch passed")


fn test_vmnd_known_values_batched() raises:
    """Hand-computed batched result verified against GPU."""
    print("test_vmnd_known_values_batched")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: v=[1,0] @ M=[[1,2],[3,4]] = [1, 2]
        # batch 1: v=[0,1] @ M=[[5,6],[7,8]] = [7, 8]
        var v = Tensor[dtype].d2([[1, 0], [0, 1]])
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var expected = Tensor[dtype].d2([[1, 2], [7, 8]])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))
    print("test_vmnd_known_values_batched passed")


fn test_vmnd_known_values_broadcast() raises:
    """Hand-computed broadcast result verified against GPU."""
    print("test_vmnd_known_values_broadcast")
    @parameter
    if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 1]  (no batch)
        # M[0] = [[1,2],[3,4]] → out[0] = [4, 6]
        # M[1] = [[5,6],[7,8]] → out[1] = [12, 14]
        var v = Tensor[dtype].d1([1, 1])
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var expected = Tensor[dtype].d2([[4, 6], [12, 14]])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))
    print("test_vmnd_known_values_broadcast passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    # Basic correctness
    test_vmnd_1d_v_2d_M()
    test_vmnd_identity_matrix()
    test_vmnd_zero_vector()
    test_vmnd_ones_vector()
    test_vmnd_single_output_element()
    test_vmnd_large_k()
    test_vmnd_large_n()

    # Batched — same batch shape
    test_vmnd_batched_2d_v_3d_M()
    test_vmnd_batched_3d_v_4d_M()
    test_vmnd_batched_arange_values()

    # Broadcast — different batch ranks
    test_vmnd_broadcast_v1d_M3d()
    test_vmnd_broadcast_v2d_M3d()
    test_vmnd_broadcast_v3d_M2d()
    test_vmnd_broadcast_both_size1()
    test_vmnd_broadcast_large_batch()

    # Known values — spot checks
    test_vmnd_known_values_no_batch()
    test_vmnd_known_values_batched()
    test_vmnd_known_values_broadcast()
