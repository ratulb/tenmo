from tenmo import Tensor
from shapes import Shape
from testing import assert_true
from strides import Strides
from operators import vm

# alias vm = 1


fn main() raises:
    test_vector_matrix_no_batch()
    test_vector_matrix_batch_M_only()
    test_vector_matrix_batch_v_only()
    test_vector_matrix_both_batched()
    test_vector_matrix_broadcast_batch()
    test_vector_matrix_asymmetric_shapes()
    test_vector_matrix_single_element()
    print("=" * 50)
    print("ALL VECTOR-MATRIX TESTS PASSED! ✓")
    print("=" * 50)

    # ===== BASIC VECTOR-MATRIX TESTS =====
    test_vector_matrix_1d_2d_basic()
    test_vector_matrix_identity()
    test_vector_matrix_zeros()
    test_vector_matrix_single_element_orig()

    # ===== BATCHED VECTOR-MATRIX TESTS =====
    test_vector_matrix_2d_2d_batched()
    test_vector_matrix_2d_3d_batched()
    test_vector_matrix_broadcast_vector()
    test_vector_matrix_broadcast_matrix()
    test_vector_matrix_3d_3d_high_batch()
    test_vector_matrix_4d_batch()

    # ===== VIEW TESTS WITH GRADIENTS =====
    test_vector_matrix_with_vector_view()
    test_vector_matrix_with_matrix_view()

    # ===== EDGE CASE TESTS =====
    test_vector_matrix_large_dimensions()
    test_vector_matrix_non_contiguous_batch()

    # ===== COMPREHENSIVE TEST FUNCTIONS =====
    test_vector_matrix_basic_forward_backward()
    test_matrix_vector_basic_forward_backward()
    test_vector_matrix_batched()
    test_vector_matrix_broadcasting()
    test_vector_matrix_with_views()
    test_vector_matrix_singleton_batch()
    test_vector_matrix_high_dimensional_batch()
    test_matmul_nd_with_view_offset_grad()

    print("All vector-matrix tests passed! ✓")


# ===== BASIC VECTOR-MATRIX TESTS =====


fn test_vector_matrix_no_batch() raises:
    """Test: v[k] @ M[k,n] -> result[n]."""
    print("test_vector_matrix_no_batch")
    alias dtype = DType.float32

    # v = [1, 2, 3]         (3,)
    # M = [[1, 2],
    #      [3, 4],
    #      [5, 6]]          (3x2)
    # result = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]

    var v = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var M = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 22 + 28 = 50
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(r.all_close(Tensor[dtype].d1([22.0, 28.0])))

    # grad_out = [1, 1]
    # grad_v[i] = sum_j(M[i,j] * grad_out[j])
    # grad_v = [1*1+2*1, 3*1+4*1, 5*1+6*1] = [3, 7, 11]
    print("grad_v:")
    v.grad().print()
    assert_true(v.grad().all_close(Tensor[dtype].d1([3.0, 7.0, 11.0])))

    # grad_M[i,j] = v[i] * grad_out[j]
    # grad_M = [[1*1, 1*1],
    #           [2*1, 2*1],
    #           [3*1, 3*1]] = [[1,1], [2,2], [3,3]]
    print("grad_M:")
    M.grad().print()
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )
    )
    print("✓ PASSED\n")


fn test_vector_matrix_batch_M_only() raises:
    """Test: v[k] @ M[batch,k,n] -> result[batch,n]."""
    print("test_vector_matrix_batch_M_only")
    alias dtype = DType.float32

    # v = [1, 2]            (2,)
    # M[0] = [[1, 2, 3],
    #         [4, 5, 6]]    batch=0
    # M[1] = [[7, 8, 9],
    #         [10, 11, 12]] batch=1
    #
    # result[0] = v @ M[0] = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    # result[1] = v @ M[1] = [1*7+2*10, 1*8+2*11, 1*9+2*12] = [27, 30, 33]

    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var M = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 9+12+15+27+30+33 = 126
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(
        r.all_close(Tensor[dtype].d2([[9.0, 12.0, 15.0], [27.0, 30.0, 33.0]]))
    )

    # grad_out = [[1, 1, 1],
    #             [1, 1, 1]]
    # grad_v[i] = sum over batches and j of (M[b,i,j] * grad_out[b,j])
    # grad_v[0] = (1+2+3) + (7+8+9) = 30
    # grad_v[1] = (4+5+6) + (10+11+12) = 48
    print("grad_v:")
    v.grad().print()
    assert_true(v.grad().all_close(Tensor[dtype].d1([30.0, 48.0])))

    # grad_M[b,i,j] = v[i] * grad_out[b,j]
    # For batch 0: grad_M[0] = [[1*1, 1*1, 1*1],
    #                           [2*1, 2*1, 2*1]] = [[1,1,1], [2,2,2]]
    # For batch 1: grad_M[1] = [[1*1, 1*1, 1*1],
    #                           [2*1, 2*1, 2*1]] = [[1,1,1], [2,2,2]]
    print("grad_M:")
    M.grad().print()
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ]
            )
        )
    )
    print("✓ PASSED\n")


fn test_vector_matrix_batch_v_only() raises:
    """Test: v[batch,k] @ M[k,n] -> result[batch,n]."""
    print("test_vector_matrix_batch_v_only")
    alias dtype = DType.float32

    # v[0] = [1, 0]         batch=0
    # v[1] = [0, 1]         batch=1
    # M = [[1, 2],
    #      [3, 4]]          (2x2)
    #
    # result[0] = v[0] @ M = [1*1+0*3, 1*2+0*4] = [1, 2]
    # result[1] = v[1] @ M = [0*1+1*3, 0*2+1*4] = [3, 4]

    var v = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var M = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 1+2+3+4 = 10
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(r.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))

    # grad_out = [[1, 1],
    #             [1, 1]]
    # grad_v[b,i] = sum_j(M[i,j] * grad_out[b,j])
    # For batch 0: grad_v[0] = [1*1+2*1, 3*1+4*1] = [3, 7]
    # For batch 1: grad_v[1] = [1*1+2*1, 3*1+4*1] = [3, 7]
    print("grad_v:")
    v.grad().print()
    assert_true(v.grad().all_close(Tensor[dtype].d2([[3.0, 7.0], [3.0, 7.0]])))

    # grad_M[i,j] = sum over batches of (v[b,i] * grad_out[b,j])
    # For batch 0: contribution = [[1*1, 1*1], [0*1, 0*1]] = [[1,1], [0,0]]
    # For batch 1: contribution = [[0*1, 0*1], [1*1, 1*1]] = [[0,0], [1,1]]
    # grad_M = [[1,1], [0,0]] + [[0,0], [1,1]] = [[1,1], [1,1]]
    print("grad_M:")
    M.grad().print()
    assert_true(M.grad().all_close(Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])))
    print("✓ PASSED\n")


fn test_vector_matrix_both_batched() raises:
    """Test: v[batch,k] @ M[batch,k,n] -> result[batch,n]."""
    print("test_vector_matrix_both_batched")
    alias dtype = DType.float32

    # v[0] = [1, 2]         batch=0
    # v[1] = [3, 4]         batch=1
    # M[0] = [[1, 2, 3],
    #         [4, 5, 6]]    batch=0
    # M[1] = [[7, 8, 9],
    #         [10, 11, 12]] batch=1
    #
    # result[0] = v[0] @ M[0] = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    # result[1] = v[1] @ M[1] = [3*7+4*10, 3*8+4*11, 3*9+4*12] = [61, 68, 75]

    var v = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var M = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 9+12+15+61+68+75 = 240
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(
        r.all_close(Tensor[dtype].d2([[9.0, 12.0, 15.0], [61.0, 68.0, 75.0]]))
    )

    # grad_out = [[1, 1, 1],
    #             [1, 1, 1]]
    # grad_v[b,i] = sum_j(M[b,i,j] * grad_out[b,j])
    # For batch 0: grad_v[0] = [1*1+2*1+3*1, 4*1+5*1+6*1] = [6, 15]
    # For batch 1: grad_v[1] = [7*1+8*1+9*1, 10*1+11*1+12*1] = [24, 33]
    print("grad_v:")
    v.grad().print()
    assert_true(
        v.grad().all_close(Tensor[dtype].d2([[6.0, 15.0], [24.0, 33.0]]))
    )

    # grad_M[b,i,j] = v[b,i] * grad_out[b,j]
    # For batch 0: grad_M[0] = [[1*1, 1*1, 1*1],
    #                           [2*1, 2*1, 2*1]] = [[1,1,1], [2,2,2]]
    # For batch 1: grad_M[1] = [[3*1, 3*1, 3*1],
    #                           [4*1, 4*1, 4*1]] = [[3,3,3], [4,4,4]]
    print("grad_M:")
    M.grad().print()
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                ]
            )
        )
    )
    print("✓ PASSED\n")


fn test_vector_matrix_broadcast_batch() raises:
    """Test: v[3,k] @ M[2,3,k,n] -> result[2,3,n] (broadcasting)."""
    print("test_vector_matrix_broadcast_batch")
    alias dtype = DType.float32

    # v has batch dims [3]
    # M has batch dims [2, 3]
    # v broadcasts to [2, 3] by replicating across first dimension
    #
    # v[0] = [1, 0]
    # v[1] = [0, 1]
    # v[2] = [1, 1]
    #
    # M[0,0] = [[1, 2], [3, 4]]
    # M[0,1] = [[2, 3], [4, 5]]
    # M[0,2] = [[3, 4], [5, 6]]
    # M[1,0] = [[4, 5], [6, 7]]
    # M[1,1] = [[5, 6], [7, 8]]
    # M[1,2] = [[6, 7], [8, 9]]
    #
    # For batch (0,0): v[0] @ M[0,0] = [1,0] @ [[1,2],[3,4]] = [1, 2]
    # For batch (0,1): v[1] @ M[0,1] = [0,1] @ [[2,3],[4,5]] = [4, 5]
    # For batch (0,2): v[2] @ M[0,2] = [1,1] @ [[3,4],[5,6]] = [8, 10]
    # For batch (1,0): v[0] @ M[1,0] = [1,0] @ [[4,5],[6,7]] = [4, 5]
    # For batch (1,1): v[1] @ M[1,1] = [0,1] @ [[5,6],[7,8]] = [7, 8]
    # For batch (1,2): v[2] @ M[1,2] = [1,1] @ [[6,7],[8,9]] = [14, 16]

    var v = Tensor[dtype].d2(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True
    )

    var M = Tensor[dtype].d4(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
                [[3.0, 4.0], [5.0, 6.0]],
            ],
            [
                [[4.0, 5.0], [6.0, 7.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[6.0, 7.0], [8.0, 9.0]],
            ],
        ],
        requires_grad=True,
    )

    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 1+2+4+5+8+10+4+5+7+8+14+16 = 84
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(
        r.all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 2.0], [4.0, 5.0], [8.0, 10.0]],
                    [[4.0, 5.0], [7.0, 8.0], [14.0, 16.0]],
                ]
            )
        )
    )

    # grad_out[b1,b2,j] = 1 for all elements
    # grad_v[b2,i] = sum over b1 and j of (M[b1,b2,i,j] * grad_out[b1,b2,j])
    # Since v is broadcast across b1, it gets contributions from both b1=0 and b1=1
    #
    # For v[0]: sum from M[0,0] and M[1,0]
    #   grad_v[0,0] = (1+2) + (4+5) = 12
    #   grad_v[0,1] = (3+4) + (6+7) = 20
    # For v[1]: sum from M[0,1] and M[1,1]
    #   grad_v[1,0] = (2+3) + (5+6) = 16
    #   grad_v[1,1] = (4+5) + (7+8) = 24
    # For v[2]: sum from M[0,2] and M[1,2]
    #   grad_v[2,0] = (3+4) + (6+7) = 20
    #   grad_v[2,1] = (5+6) + (8+9) = 28
    print("grad_v:")
    v.grad().print()
    assert_true(
        v.grad().all_close(
            Tensor[dtype].d2([[12.0, 20.0], [16.0, 24.0], [20.0, 28.0]])
        )
    )

    # grad_M[b1,b2,i,j] = v[b2,i] * grad_out[b1,b2,j]
    # For M[0,0]: uses v[0]=[1,0], grad_out all 1s
    #   grad_M[0,0] = [[1*1, 1*1], [0*1, 0*1]] = [[1,1], [0,0]]
    # For M[0,1]: uses v[1]=[0,1]
    #   grad_M[0,1] = [[0*1, 0*1], [1*1, 1*1]] = [[0,0], [1,1]]
    # For M[0,2]: uses v[2]=[1,1]
    #   grad_M[0,2] = [[1*1, 1*1], [1*1, 1*1]] = [[1,1], [1,1]]
    # Same pattern for M[1,*]
    print("grad_M:")
    M.grad().print()
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d4(
                [
                    [
                        [[1.0, 1.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                        [[1.0, 1.0], [1.0, 1.0]],
                    ],
                    [
                        [[1.0, 1.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                        [[1.0, 1.0], [1.0, 1.0]],
                    ],
                ]
            )
        )
    )
    print("✓ PASSED\n")


fn test_vector_matrix_asymmetric_shapes() raises:
    """Test: v[batch,k] @ M[batch,k,n] with k≠n to ensure no accidental transposes.
    """
    print("test_vector_matrix_asymmetric_shapes")
    alias dtype = DType.float32

    # v[0] = [1, 2, 3]      batch=0, k=3
    # v[1] = [4, 5, 6]      batch=1, k=3
    # M[0] = [[1, 2],
    #         [3, 4],
    #         [5, 6]]       batch=0, k=3, n=2
    # M[1] = [[7, 8],
    #         [9, 10],
    #         [11, 12]]     batch=1, k=3, n=2
    #
    # result[0] = v[0] @ M[0] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    # result[1] = v[1] @ M[1] = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]

    var v = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var M = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 22+28+139+154 = 343
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(r.all_close(Tensor[dtype].d2([[22.0, 28.0], [139.0, 154.0]])))

    # grad_out = [[1, 1],
    #             [1, 1]]
    # grad_v[b,i] = sum_j(M[b,i,j] * grad_out[b,j])
    # For batch 0: grad_v[0] = [1*1+2*1, 3*1+4*1, 5*1+6*1] = [3, 7, 11]
    # For batch 1: grad_v[1] = [7*1+8*1, 9*1+10*1, 11*1+12*1] = [15, 19, 23]
    print("grad_v:")
    v.grad().print()
    assert_true(
        v.grad().all_close(
            Tensor[dtype].d2([[3.0, 7.0, 11.0], [15.0, 19.0, 23.0]])
        )
    )

    # grad_M[b,i,j] = v[b,i] * grad_out[b,j]
    # For batch 0: grad_M[0] = [[1*1, 1*1],
    #                           [2*1, 2*1],
    #                           [3*1, 3*1]] = [[1,1], [2,2], [3,3]]
    # For batch 1: grad_M[1] = [[4*1, 4*1],
    #                           [5*1, 5*1],
    #                           [6*1, 6*1]] = [[4,4], [5,5], [6,6]]
    print("grad_M:")
    M.grad().print()
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                    [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
                ]
            )
        )
    )
    print("✓ PASSED\n")


fn test_vector_matrix_single_element() raises:
    """Test: v[k] @ M[k,1] -> result[1] (edge case: n=1)."""
    print("test_vector_matrix_single_element")
    alias dtype = DType.float32

    # v = [2, 3, 4]         (3,)
    # M = [[1],
    #      [2],
    #      [3]]             (3x1)
    # result = [2*1+3*2+4*3] = [20]

    var v = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var M = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()  # loss = 20
    loss.backward()

    print("Forward:")
    r.print()
    assert_true(r.all_close(Tensor[dtype].d1([20.0])))

    # grad_out = [1]
    # grad_v = [1*1, 2*1, 3*1] = [1, 2, 3]
    print("grad_v:")
    v.grad().print()
    assert_true(v.grad().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))

    # grad_M = [[2*1], [3*1], [4*1]] = [[2], [3], [4]]
    print("grad_M:")
    M.grad().print()
    assert_true(M.grad().all_close(Tensor[dtype].d2([[2.0], [3.0], [4.0]])))
    print("✓ PASSED\n")


fn test_vector_matrix_1d_2d_basic() raises:
    print("test_vector_matrix_1d_2d_basic")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var M = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # Expected: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    assert_true(r.all_close(Tensor.d1([22.0, 28.0])))
    assert_true(
        v.grad().all_close(Tensor.d1([3.0, 7.0, 11.0]))
    )  # sum of columns of M
    assert_true(
        M.grad().all_close(Tensor.d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    )


fn test_vector_matrix_identity() raises:
    print("test_vector_matrix_identity")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var identity = Tensor.d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True
    )
    var r = v.matmul[mode=vm](identity)
    var loss = r.sum()
    loss.backward()

    assert_true(r.all_close(v))
    assert_true(v.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))
    assert_true(
        identity.grad().all_close(
            Tensor.d2([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        )
    )


fn test_vector_matrix_zeros() raises:
    print("test_vector_matrix_zeros")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var M = Tensor.d2([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    assert_true(r.all_close(Tensor.d1([0.0, 0.0])))
    assert_true(v.grad().all_close(Tensor.d1([0.0, 0.0, 0.0])))
    assert_true(
        M.grad().all_close(Tensor.d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    )


# ===== BATCHED VECTOR-MATRIX TESTS =====


fn test_vector_matrix_2d_2d_batched() raises:
    print("test_vector_matrix_2d_2d_batched")
    alias dtype = DType.float32
    var v = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )  # batch=2, dim=2
    var M = Tensor.d2(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )  # 2x2 matrix
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # Each batch: v[i] @ identity = v[i]
    assert_true(r.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])))
    assert_true(v.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(M.grad().all_close(Tensor.d2([[4.0, 4.0], [6.0, 6.0]])))


fn test_vector_matrix_2d_3d_batched() raises:
    print("test_vector_matrix_2d_3d_batched")
    alias dtype = DType.float32
    var v = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )  # batch=2, dim=2
    var M = Tensor[dtype].d3(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )  # batch=2, 2x2
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # Batch 0: [1,2] @ [[1,0],[0,1]] = [1,2]
    # Batch 1: [3,4] @ [[2,0],[0,2]] = [6,8]

    assert_true(r.all_close(Tensor[dtype].d2([[1.0, 2.0], [6.0, 8.0]])))
    assert_true(v.grad().all_close(Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0]])))
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]
            )
        )
    )


fn test_vector_matrix_broadcast_vector() raises:
    print("test_vector_matrix_broadcast_vector")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0], requires_grad=True)  # single vector
    var M = Tensor.d3(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )  # batch=2, 2x2
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # Vector broadcasted to both batches
    # Batch 0: [1,2] @ [[1,0],[0,1]] = [1,2]
    # Batch 1: [1,2] @ [[2,0],[0,2]] = [2,4]
    assert_true(r.all_close(Tensor.d2([[1.0, 2.0], [2.0, 4.0]])))
    assert_true(
        v.grad().all_close(Tensor.d1([3.0, 3.0]))
    )  # summed over batches
    assert_true(
        M.grad().all_close(
            Tensor.d3([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]])
        )
    )


fn test_vector_matrix_broadcast_matrix() raises:
    print("test_vector_matrix_broadcast_matrix")
    alias dtype = DType.float32
    var v = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # batch=2
    var M = Tensor.d2(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )  # single matrix
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # Matrix broadcasted to both vector batches
    assert_true(r.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])))
    assert_true(v.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(M.grad().all_close(Tensor.d2([[4.0, 4.0], [6.0, 6.0]])))


# ===== 3D+ BATCH DIMENSION TESTS =====


fn test_vector_matrix_3d_3d_high_batch() raises:
    print("test_vector_matrix_3d_3d_high_batch")
    alias dtype = DType.float32
    var v = Tensor.d3(
        [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True
    )  # shape: [2, 1, 2]
    var M = Tensor.d3(
        [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
    )  # shape: [1, 2, 2]
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # Complex broadcasting: v[2,1,2] × M[1,2,2] → [2,1,2]
    assert_true(r.shape() == Shape(2, 1, 2))
    assert_true(v.grad().shape() == Shape(2, 1, 2))
    assert_true(M.grad().shape() == Shape(1, 2, 2))


fn test_vector_matrix_4d_batch() raises:
    print("test_vector_matrix_4d_batch")
    alias dtype = DType.float32
    var v = Tensor.d4(
        [[[[1.0, 2.0]]]], requires_grad=True
    )  # shape: [1, 1, 1, 2]
    var M = Tensor.d2(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )  # shape: [2, 2]
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    assert_true(r.shape() == Shape(1, 1, 1, 2))
    assert_true(r.all_close(Tensor.d4([[[[1.0, 2.0]]]])))
    assert_true(v.grad().shape() == Shape(1, 1, 1, 2))


# ===== VIEW TESTS WITH GRADIENTS =====


fn test_vector_matrix_with_vector_view() raises:
    print("test_vector_matrix_with_vector_view")
    alias dtype = DType.float32
    var base_v = Tensor.d1([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var v_view = base_v.view(
        shape=Shape(3), strides=Strides(1), offset=1
    )  # [1,2,3]
    var M = Tensor.d2([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], requires_grad=True)
    var r = v_view.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # [1,2,3] @ [[1,0],[0,1],[0,0]] = [1,2]
    assert_true(r.all_close(Tensor.d1([1.0, 2.0])))
    # Gradients should flow to viewed portion [1,2,3]
    assert_true(base_v.grad().all_close(Tensor.d1([0.0, 1.0, 1.0, 0.0, 0.0])))


fn test_vector_matrix_with_matrix_view() raises:
    print("test_vector_matrix_with_matrix_view")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0], requires_grad=True)
    var base_M = Tensor.d2(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True
    )
    var M_view = base_M.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=3
    )  # [[1,0],[0,1]]
    var r = v.matmul[mode=vm](M_view)
    var loss = r.sum()
    loss.backward()

    # [1,2] @ [[1,0],[0,1]] = [1,2]
    assert_true(r.all_close(Tensor.d1([1.0, 2.0])))
    # Gradients should flow to viewed portion
    assert_true(
        base_M.grad().all_close(
            Tensor.d2([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        )
    )


# ===== EDGE CASE TESTS =====


fn test_vector_matrix_single_element_orig() raises:
    print("test_vector_matrix_single_element_orig")
    alias dtype = DType.float32
    var v = Tensor.d1([2.0], requires_grad=True)  # 1D vector with 1 element
    var M = Tensor.d2([[3.0]], requires_grad=True)  # 1x1 matrix
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    assert_true(r.all_close(Tensor.d1([6.0])))
    assert_true(v.grad().all_close(Tensor.d1([3.0])))
    assert_true(M.grad().all_close(Tensor.d2([[2.0]])))


fn test_vector_matrix_large_dimensions() raises:
    print("test_vector_matrix_large_dimensions")
    alias dtype = DType.float32
    # Test with larger dimensions to catch indexing issues
    var v = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var M = Tensor.d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    var r = v.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # [1,2,3,4] @ 4x3 matrix = [1,2,3]
    assert_true(r.all_close(Tensor.d1([1.0, 2.0, 3.0])))
    assert_true(v.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 0.0])))


fn test_vector_matrix_non_contiguous_batch() raises:
    print("test_vector_matrix_non_contiguous_batch")
    alias dtype = DType.float32
    var base = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
    )
    # Take every other row as vectors
    var v_view = base.view(
        shape=Shape(2, 3), strides=Strides(6, 1), offset=0
    )  # [[1,2,3], [7,8,9]]
    var M = Tensor.d2([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], requires_grad=True)
    var r = v_view.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # [[1,2,3], [7,8,9]] @ 3x2 matrix = [[1,2], [7,8]]
    assert_true(r.all_close(Tensor.d2([[1.0, 2.0], [7.0, 8.0]])))
    # Gradients should flow to the selected rows
    assert_true(
        base.grad().all_close(
            Tensor.d2([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        )
    )


# ------------------- Vector-Matrix / Matrix-Vector Test Suite -------------------


fn test_vector_matrix_basic_forward_backward() raises:
    print("test_vector_matrix_basic_forward_backward")
    var v = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var M = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)  # [2]

    # Expected: [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    assert_true(r.all_close(Tensor.d1([22.0, 28.0])))

    var loss = r.sum()
    loss.backward()
    # grad_v = M @ 1 = [1+2, 3+4, 5+6] ? wait compute manually
    # grad_v = sum over rows of M = [1+2, 3+4, 5+6]? actually grad_v = dL/dr @ M^T = [1,1] @ M^T = sum along output dim
    ####assert_true(v.grad().all_close(Tensor.d1([3.0+5.0+7.0?])))  # we need correct calculation
    # let's compute precisely
    # grad_r = [1,1] (sum)
    # grad_v = grad_r @ M^T = [1,1] @ [[1,3,5],[2,4,6]] = [1*1+1*2, 1*3+1*4, 1*5+1*6] = [3,7,11]
    assert_true(v.grad().all_close(Tensor.d1([3.0, 7.0, 11.0])))
    # grad_M = v^T @ grad_r = [[1],[2],[3]] @ [1,1] = [[1,1],[2,2],[3,3]]
    assert_true(
        M.grad().all_close(Tensor.d2([[1, 1], [2, 2], [3, 3]]).float64())
    )


fn test_matrix_vector_basic_forward_backward() raises:
    print("test_matrix_vector_basic_forward_backward")
    var M = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var v = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    # Matrix @ Vector
    var r = M.matmul[mode=vm](v.unsqueeze([-1])).squeeze([-1])  # [2]

    # Expected: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    assert_true(r.all_close(Tensor.d1([14.0, 32.0])))

    var loss = r.sum()
    loss.backward()
    # grad_v = M^T @ grad_r = [[1,4],[2,5],[3,6]] @ [1,1] = [5,7,9]
    assert_true(v.grad().all_close(Tensor.d1([5.0, 7.0, 9.0])))
    # grad_M = grad_r[:,None] @ v[None,:] = [[1*1,1*2,1*3],[1*1,1*2,1*3]] = [[1,2,3],[1,2,3]]
    assert_true(M.grad().all_close(Tensor.d2([[1, 2, 3], [1, 2, 3]]).float64()))


fn test_vector_matrix_batched() raises:
    print("test_vector_matrix_batched")
    var v = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )  # batch=2, k=2
    var M = Tensor.d3(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [0.0, 1.0]]], requires_grad=True
    )  # batch=2, 2x2
    var r = v.matmul[mode=vm](M)  # [2,2]

    var expected = Tensor.d2(
        [[1 * 1 + 2 * 0, 1 * 0 + 2 * 1], [3 * 2 + 4 * 0, 3 * 1 + 4 * 1]]
    )  # [[1,2],[6,7]]
    assert_true(r.all_close(expected.float64()))

    var loss = r.sum()
    loss.backward()
    # Check grad shapes
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_vector_matrix_broadcasting() raises:
    print("test_vector_matrix_broadcasting")
    var v = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # batch=2
    var M = Tensor.d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)  # no batch
    var r = v.matmul[mode=vm](M)  # [2,2] batch broadcasting

    var expected = Tensor.d2([[1, 2], [3, 4]])
    assert_true(r.all_close(expected.float64()))
    var loss = r.sum()
    loss.backward()
    assert_true(v.grad().all_close(Tensor.d2([[1, 1], [1, 1]]).float64()))
    assert_true(
        M.grad().all_close(Tensor.d2([[4, 4], [6, 6]]).float64())
    )  # sum over batch, double check sums


fn test_vector_matrix_with_views() raises:
    print("test_vector_matrix_with_views")
    var base = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var v = base.view(
        shape=Shape(2, 3), strides=Strides(3, 1), offset=0
    )  # [[1,2,3],[4,5,6]]
    var M = Tensor.d2([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)  # [2,2]
    var loss = r.sum()
    loss.backward()
    # Check that grads flow correctly through the view to the base tensor
    assert_true(base.grad().shape() == base.shape())


fn test_vector_matrix_singleton_batch() raises:
    print("test_vector_matrix_singleton_batch")
    var v = Tensor.d2([[1.0, 2.0]], requires_grad=True)  # batch=1
    var M = Tensor.d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var r = v.matmul[mode=vm](M)  # [1,2]
    var loss = r.sum()
    loss.backward()
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_vector_matrix_high_dimensional_batch() raises:
    print("test_vector_matrix_high_dimensional_batch")
    var v = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # [1,2,2]
    var M = Tensor.d3([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)  # [1,2,2]
    var r = v.matmul[mode=vm](M)  # [1,2,2]
    var loss = r.sum()
    loss.backward()
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_matmul_nd_with_view_offset_grad() raises:
    print("test_matmul_nd_with_view_offset_grad")
    alias dtype = DType.float32
    var base_A = Tensor[dtype].d3(
        [
            [[0.0, 0.0], [0.0, 0.0]],  # Padding
            [[1.0, 2.0], [3.0, 4.0]],  # Actual data
            [[5.0, 6.0], [7.0, 8.0]],  # More data
        ],
        requires_grad=True,
    )

    # Create view skipping first batch, taking next 2 batches
    var A_view = base_A.view(
        shape=Shape(2, 2, 2),
        strides=Strides(2, 2, 1),
        offset=4,  # Skip first 2x2 matrix (4 elements)
    )
    # var A_view = base_A[il(1), s(), s()]
    # var A_view = base_A[s(3), s(), s()]

    var B = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var C = A_view.matmul[mode=vm](B)
    var loss = C.sum()
    loss.backward()

    # Gradients should only flow to the viewed portion (batches 1 and 2)
    var expected_base_grad = Tensor[dtype].d3(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [2.0, 2.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )
    assert_true(base_A.grad().all_close(expected_base_grad))
