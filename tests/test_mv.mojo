from tenmo import Tensor
from shapes import Shape
from operators import mv

# alias mv = 2 # matrix vector


fn main() raises:
    run_all_matrix_vector_tests()
    test_matrix_vector_no_batch()
    test_matrix_vector_batch_v_only()
    test_matrix_vector_batch_M_only()
    test_matrix_vector_both_batched()
    test_matrix_vector_broadcast_batch()
    print("=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)


# ===== BASIC MATRIX-VECTOR TESTS =====

from testing import assert_true
from strides import Strides


fn test_matrix_vector_no_batch() raises:
    """Test: M[m,k] @ v[k] -> result[m]."""
    print("test_matrix_vector_no_batch")
    alias dtype = DType.float32

    # M = [[1, 2, 3],
    #      [4, 5, 6]]  (2x3)
    # v = [1, 2, 3]    (3,)
    # result = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]

    var M = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()  # loss = 14 + 32 = 46
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d1([14.0, 32.0])))

    # grad_out = [1, 1]
    # grad_M[i,j] = grad_out[i] * v[j]
    # grad_M = [[1*1, 1*2, 1*3],
    #           [1*1, 1*2, 1*3]] = [[1,2,3], [1,2,3]]
    assert_true(
        M.grad().all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    )

    # grad_v[j] = sum_i(M[i,j] * grad_out[i])
    # grad_v = [1*1+4*1, 2*1+5*1, 3*1+6*1] = [5, 7, 9]
    assert_true(v.grad().all_close(Tensor[dtype].d1([5.0, 7.0, 9.0])))
    print("✓ PASSED\n")


fn test_matrix_vector_batch_v_only() raises:
    """Test: M[m,k] @ v[batch,k] -> result[batch,m]."""
    print("test_matrix_vector_batch_v_only")
    alias dtype = DType.float32

    # M = [[1, 2],
    #      [3, 4]]  (2x2)
    # v = [[1, 0],   batch=0
    #      [0, 1]]   batch=1
    #
    # result[0] = M @ v[0] = [1*1+2*0, 3*1+4*0] = [1, 3]
    # result[1] = M @ v[1] = [1*0+2*1, 3*0+4*1] = [2, 4]

    var M = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()  # loss = 1+3+2+4 = 10
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))

    # grad_out = [[1, 1],
    #             [1, 1]]
    # grad_M = sum over batches of (grad_out[b] ⊗ v[b])
    # For batch 0: grad_out[0] ⊗ v[0] = [1,1]^T ⊗ [1,0] = [[1,0], [1,0]]
    # For batch 1: grad_out[1] ⊗ v[1] = [1,1]^T ⊗ [0,1] = [[0,1], [0,1]]
    # grad_M = [[1,0], [1,0]] + [[0,1], [0,1]] = [[1,1], [1,1]]
    assert_true(M.grad().all_close(Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])))

    # grad_v[b,j] = sum_i(M[i,j] * grad_out[b,i])
    # For batch 0: grad_v[0] = M^T @ grad_out[0] = [[1,3],[2,4]] @ [1,1] = [4, 6]
    # For batch 1: grad_v[1] = M^T @ grad_out[1] = [[1,3],[2,4]] @ [1,1] = [4, 6]
    assert_true(v.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 6.0]])))
    print("✓ PASSED\n")


fn test_matrix_vector_batch_M_only() raises:
    """Test: M[batch,m,k] @ v[k] -> result[batch,m]."""
    print("test_matrix_vector_batch_M_only")
    alias dtype = DType.float32

    # M[0] = [[1, 2],   batch=0
    #         [3, 4]]
    # M[1] = [[5, 6],   batch=1
    #         [7, 8]]
    # v = [1, 1]
    #
    # result[0] = M[0] @ v = [1+2, 3+4] = [3, 7]
    # result[1] = M[1] @ v = [5+6, 7+8] = [11, 15]

    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 1.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()  # loss = 3+7+11+15 = 36
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d2([[3.0, 7.0], [11.0, 15.0]])))

    # grad_out = [[1, 1],
    #             [1, 1]]
    # grad_M[b,i,j] = grad_out[b,i] * v[j]
    # For batch 0: grad_M[0] = [1,1]^T ⊗ [1,1] = [[1,1], [1,1]]
    # For batch 1: grad_M[1] = [1,1]^T ⊗ [1,1] = [[1,1], [1,1]]
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
            )
        )
    )

    # grad_v[j] = sum over batches and i of (M[b,i,j] * grad_out[b,i])
    # grad_v[0] = 1*1 + 3*1 + 5*1 + 7*1 = 16
    # grad_v[1] = 2*1 + 4*1 + 6*1 + 8*1 = 20
    assert_true(v.grad().all_close(Tensor[dtype].d1([16.0, 20.0])))
    print("✓ PASSED\n")


fn test_matrix_vector_both_batched() raises:
    """Test: M[batch,m,k] @ v[batch,k] -> result[batch,m]."""
    print("test_matrix_vector_both_batched")
    alias dtype = DType.float32

    # M[0] = [[1, 2],   batch=0
    #         [3, 4]]
    # M[1] = [[5, 6],   batch=1
    #         [7, 8]]
    # v[0] = [2, 1]     batch=0
    # v[1] = [1, 2]     batch=1
    #
    # result[0] = M[0] @ v[0] = [1*2+2*1, 3*2+4*1] = [4, 10]
    # result[1] = M[1] @ v[1] = [5*1+6*2, 7*1+8*2] = [17, 23]

    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()  # loss = 4+10+17+23 = 54
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d2([[4.0, 10.0], [17.0, 23.0]])))

    # grad_out = [[1, 1],
    #             [1, 1]]
    # grad_M[b,i,j] = grad_out[b,i] * v[b,j]
    # For batch 0: grad_M[0] = [1,1]^T ⊗ [2,1] = [[2,1], [2,1]]
    # For batch 1: grad_M[1] = [1,1]^T ⊗ [1,2] = [[1,2], [1,2]]
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [[[2.0, 1.0], [2.0, 1.0]], [[1.0, 2.0], [1.0, 2.0]]]
            )
        )
    )

    # grad_v[b,j] = sum_i(M[b,i,j] * grad_out[b,i])
    # For batch 0: grad_v[0] = [[1,3],[2,4]] @ [1,1] = [4, 6]
    # For batch 1: grad_v[1] = [[5,7],[6,8]] @ [1,1] = [12, 14]
    assert_true(
        v.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [12.0, 14.0]]))
    )
    print("✓ PASSED\n")


fn test_matrix_vector_broadcast_batch() raises:
    """Test: M[2,3,m,k] @ v[3,k] -> result[2,3,m] (broadcasting)."""
    print("test_matrix_vector_broadcast_batch")
    alias dtype = DType.float32

    # M has batch dims [2, 3]
    # v has batch dims [3] (broadcasts to [2, 3])
    # Simple case: 2x2 matrices and 2-element vectors
    #
    # M[0,0] = [[1, 2], [3, 4]]
    # M[0,1] = [[2, 3], [4, 5]]
    # M[0,2] = [[3, 4], [5, 6]]
    # M[1,0] = [[4, 5], [6, 7]]
    # M[1,1] = [[5, 6], [7, 8]]
    # M[1,2] = [[6, 7], [8, 9]]
    #
    # v[0] = [1, 0]
    # v[1] = [0, 1]
    # v[2] = [1, 1]
    #
    # For batch (0,0): M[0,0] @ v[0] = [[1,2],[3,4]] @ [1,0] = [1, 3]
    # For batch (0,1): M[0,1] @ v[1] = [[2,3],[4,5]] @ [0,1] = [3, 5]
    # For batch (0,2): M[0,2] @ v[2] = [[3,4],[5,6]] @ [1,1] = [7, 11]
    # For batch (1,0): M[1,0] @ v[0] = [[4,5],[6,7]] @ [1,0] = [4, 6]
    # For batch (1,1): M[1,1] @ v[1] = [[5,6],[7,8]] @ [0,1] = [6, 8]
    # For batch (1,2): M[1,2] @ v[2] = [[6,7],[8,9]] @ [1,1] = [13, 17]

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

    var v = Tensor[dtype].d2(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True
    )

    var r = M.matmul[mode=mv](v)
    var loss = r.sum()  # loss = 1+3+3+5+7+11+4+6+6+8+13+17 = 84
    loss.backward()

    assert_true(
        r.all_close(
            Tensor[dtype].d3(
                [
                    [[1.0, 3.0], [3.0, 5.0], [7.0, 11.0]],
                    [[4.0, 6.0], [6.0, 8.0], [13.0, 17.0]],
                ]
            )
        )
    )

    # grad_out[b1,b2,i] = 1 for all elements
    # grad_M[b1,b2,i,j] = grad_out[b1,b2,i] * v[b2,j]
    # Since v is broadcast across b1, M gets contributions from both b1=0 and b1=1
    #
    # For M[0,0]: uses v[0]=[1,0], grad_out from (0,0) and (1,0)
    #   grad = [1,1]^T ⊗ [1,0] (from 0,0) = [[1,0],[1,0]]
    # For M[0,1]: uses v[1]=[0,1]
    #   grad = [1,1]^T ⊗ [0,1] = [[0,1],[0,1]]
    # For M[0,2]: uses v[2]=[1,1]
    #   grad = [1,1]^T ⊗ [1,1] = [[1,1],[1,1]]
    # Same for M[1,*]
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d4(
                [
                    [
                        [[1.0, 0.0], [1.0, 0.0]],
                        [[0.0, 1.0], [0.0, 1.0]],
                        [[1.0, 1.0], [1.0, 1.0]],
                    ],
                    [
                        [[1.0, 0.0], [1.0, 0.0]],
                        [[0.0, 1.0], [0.0, 1.0]],
                        [[1.0, 1.0], [1.0, 1.0]],
                    ],
                ]
            )
        )
    )

    # grad_v[b2,j] = sum over b1 and i of (M[b1,b2,i,j] * grad_out[b1,b2,i])
    # For v[0]: sum from M[0,0] and M[1,0]
    #   grad_v[0,0] = (1+3) + (4+6) = 14
    #   grad_v[0,1] = (2+4) + (5+7) = 18
    # For v[1]: sum from M[0,1] and M[1,1]
    #   grad_v[1,0] = (2+4) + (5+7) = 18
    #   grad_v[1,1] = (3+5) + (6+8) = 22
    # For v[2]: sum from M[0,2] and M[1,2]
    #   grad_v[2,0] = (3+5) + (6+8) = 22
    #   grad_v[2,1] = (4+6) + (7+9) = 26
    assert_true(
        v.grad().all_close(
            Tensor[dtype].d2([[14.0, 18.0], [18.0, 22.0], [22.0, 26.0]])
        )
    )
    print("✓ PASSED\n")


fn test_matrix_vector_basic_forward_backward() raises:
    print("test_matrix_vector_basic_forward_backward")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # Expected: [[1,2,3],[4,5,6]] @ [1,2,3] = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    assert_true(r.all_close(Tensor[dtype].d1([14.0, 32.0])))
    # grad_v = M^T @ [1,1] = [[1,4],[2,5],[3,6]] @ [1,1] = [5,7,9]
    assert_true(v.grad().all_close(Tensor[dtype].d1([5.0, 7.0, 9.0])))
    # grad_M = [1,1] @ v^T = [[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3]]
    assert_true(
        M.grad().all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    )


fn test_matrix_vector_identity() raises:
    print("test_matrix_vector_identity")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # I @ v = v
    assert_true(r.all_close(v))
    # grad_v = I^T @ [1,1,1] = [1,1,1]
    assert_true(v.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))
    # grad_M = [1,1,1] @ v^T = [[1],[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3],[1,2,3]]
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d2(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
            )
        )
    )


fn test_matrix_vector_zeros_matrix() raises:
    print("test_matrix_vector_zeros_matrix")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d1([0.0, 0.0])))
    # grad_v = zeros_matrix^T @ [1,1] = [0,0,0]
    assert_true(v.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))
    # grad_M = [1,1] @ v^T = [[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3]]
    assert_true(
        M.grad().all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    )


fn test_matrix_vector_zeros_vector() raises:
    print("test_matrix_vector_zeros_vector")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([0.0, 0.0, 0.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d1([0.0, 0.0])))
    # grad_v = M^T @ [1,1] = [[1,4],[2,5],[3,6]] @ [1,1] = [5,7,9]
    assert_true(v.grad().all_close(Tensor[dtype].d1([5.0, 7.0, 9.0])))
    # grad_M = [1,1] @ zeros_vector^T = [[0,0,0],[0,0,0]]
    assert_true(
        M.grad().all_close(Tensor[dtype].d2([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    )


fn test_matrix_vector_single_element() raises:
    print("test_matrix_vector_single_element")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2([[3.0]], requires_grad=True)
    var v = Tensor[dtype].d1([2.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    assert_true(r.all_close(Tensor[dtype].d1([6.0])))
    assert_true(v.grad().all_close(Tensor[dtype].d1([3.0])))
    assert_true(M.grad().all_close(Tensor[dtype].d2([[2.0]])))


# ===== BATCHED MATRIX-VECTOR TESTS =====


fn test_matrix_vector_3d_2d_batched_matrix() raises:
    print("test_matrix_vector_3d_2d_batched_matrix")
    alias dtype = DType.float32
    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # Batch 0: [[1,2],[3,4]] @ [1,2] = [5,11]
    # Batch 1: [[5,6],[7,8]] @ [1,2] = [17,23]
    assert_true(r.all_close(Tensor[dtype].d2([[5.0, 11.0], [17.0, 23.0]])))
    # grad_v = sum over batches: M0^T @ [1,1] + M1^T @ [1,1] = [[1,3],[2,4]]@[1,1] + [[5,7],[6,8]]@[1,1] = [4,6] + [12,14] = [16,20]
    assert_true(v.grad().all_close(Tensor[dtype].d1([16.0, 20.0])))
    # grad_M = batch outer products: [[[1],[1]]@[[1,2]], [[1],[1]]@[[1,2]]] = [[[1,2],[1,2]], [[1,2],[1,2]]]
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d3(
                [[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]
            )
        )
    )


fn test_matrix_vector_2d_2d_batched_vector() raises:
    print("test_matrix_vector_2d_2d_batched_vector")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()
    assert_true(r.all_close(Tensor[dtype].d2([[5.0, 11.0], [11.0, 25.0]])))
    assert_true(M.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 6.0]])))
    assert_true(v.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 6.0]])))


fn test_matrix_vector_3d_3d_batched_both() raises:
    print("test_matrix_vector_3d_3d_batched_both")
    alias dtype = DType.float32
    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # Batch 0: [[1,2],[3,4]] @ [1,2] = [5,11]
    # Batch 1: [[5,6],[7,8]] @ [3,4] = [39,53]
    assert_true(r.all_close(Tensor[dtype].d2([[5.0, 11.0], [39.0, 53.0]])))
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_matrix_vector_broadcast_matrix() raises:
    print("test_matrix_vector_broadcast_matrix")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # Matrix broadcast to 3 batches
    # All batches use same M: [[1,2],[3,4]]
    # Results: [5,11], [11,25], [17,39]
    assert_true(
        r.all_close(Tensor[dtype].d2([[5.0, 11.0], [11.0, 25.0], [17.0, 39.0]]))
    )
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_matrix_vector_broadcast_vector() raises:
    print("test_matrix_vector_broadcast_vector")
    alias dtype = DType.float32
    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # Vector broadcast to 2 matrix batches
    assert_true(r.all_close(Tensor[dtype].d2([[5.0, 11.0], [17.0, 23.0]])))
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


# ===== HIGHER DIMENSIONAL BATCHING TESTS =====


fn test_matrix_vector_4d_batch() raises:
    print("test_matrix_vector_4d_batch")
    alias dtype = DType.float32
    var M = Tensor[dtype].d4([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    var v = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    assert_true(r.shape() == Shape(1, 1, 2))
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_matrix_vector_3d_4d_broadcast() raises:
    print("test_matrix_vector_3d_4d_broadcast")
    alias dtype = DType.float32
    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d4([[[[1.0, 2.0]]]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # CORRECTED ASSERTIONS:
    assert_true(r.shape() == Shape(1, 1, 2, 2))  # Fixed shape expectation
    assert_true(v.grad().shape() == v.shape())  # This should now pass ✓
    assert_true(M.grad().shape() == M.shape())  # This should now pass ✓

    # Optional: also check gradient values match PyTorch
    var expected_M_grad = Tensor[dtype].d3(
        [[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]
    )
    var expected_v_grad = Tensor[dtype].d4([[[[16.0, 20.0]]]])
    assert_true(M.grad().all_close(expected_M_grad))
    assert_true(v.grad().all_close(expected_v_grad))


# ===== VIEW TESTS WITH GRADIENTS =====


fn test_matrix_vector_with_matrix_view() raises:
    print("test_matrix_vector_with_matrix_view")
    alias dtype = DType.float32
    var base_M = Tensor[dtype].d1(
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0], requires_grad=True
    )
    # var M_view = base_M.view(shape=Shape(2, 2), strides=Strides(3, 1), offset=2)
    var M_view = base_M.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var r = M_view.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # M_view = [[1,2],[3,4]] @ [1,2] = [5,11]
    assert_true(r.all_close(Tensor[dtype].d1([5.0, 11.0])))
    # Gradients should flow only to viewed portion [1,2,3,4]
    assert_true(
        base_M.grad().all_close(
            Tensor[dtype].d1([0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0])
        )
    )


fn test_matrix_vector_with_vector_view() raises:
    print("test_matrix_vector_with_vector_view")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var base_v = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 0.0], requires_grad=True)
    var v_view = base_v.view(shape=Shape(3), strides=Strides(1), offset=1)
    var r = M.matmul[mode=mv](v_view)
    var loss = r.sum()
    loss.backward()

    # v_view = [1,2,3], M @ v_view = [14,32]
    assert_true(r.all_close(Tensor[dtype].d1([14.0, 32.0])))
    # Gradients should flow only to viewed portion [1,2,3]
    assert_true(
        base_v.grad().all_close(Tensor[dtype].d1([0.0, 5.0, 7.0, 9.0, 0.0]))
    )


fn test_matrix_vector_double_view() raises:
    print("test_matrix_vector_double_view")
    alias dtype = DType.float32
    var base_M = Tensor[dtype].d2(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    var M_view = base_M.view(shape=Shape(2, 3), strides=Strides(3, 1), offset=3)
    var base_v = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 0.0], requires_grad=True)
    var v_view = base_v.view(shape=Shape(3), strides=Strides(1), offset=1)
    var r = M_view.matmul[mode=mv](v_view)
    var loss = r.sum()
    loss.backward()

    # M_view = [[1,2,3],[4,5,6]], v_view = [1,2,3]
    # Result = [14,32]
    assert_true(r.all_close(Tensor[dtype].d1([14.0, 32.0])))
    assert_true(base_M.grad().shape() == base_M.shape())
    assert_true(base_v.grad().shape() == base_v.shape())


# ===== EDGE CASE TESTS =====


fn test_matrix_vector_large_dimensions() raises:
    print("test_matrix_vector_large_dimensions")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # [1,2,3,4] @ [1,2,3,4] = [30, 70]
    assert_true(r.all_close(Tensor[dtype].d1([30.0, 70.0])))
    assert_true(v.grad().all_close(Tensor[dtype].d1([6.0, 8.0, 10.0, 12.0])))
    assert_true(
        M.grad().all_close(
            Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        )
    )


fn test_matrix_vector_non_contiguous_batch() raises:
    print("test_matrix_vector_non_contiguous_batch")
    alias dtype = DType.float32
    var base = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    # var M_view = base.view(shape=Shape(2, 2, 2), strides=Strides(4, 2, 1), offset=0)
    var M_view = base.view(
        shape=Shape(2, 2, 2), strides=Strides(8, 2, 1), offset=0
    )
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var r = M_view.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # M_view takes batches 0 and 2: [[[1,2],[3,4]], [[5,6],[7,8]]]
    # Results: [5,11], [17,23]
    assert_true(r.all_close(Tensor[dtype].d2([[5.0, 11.0], [17.0, 23.0]])))
    assert_true(base.grad().shape() == base.shape())


fn test_matrix_vector_singleton_batch() raises:
    print("test_matrix_vector_singleton_batch")
    alias dtype = DType.float32
    var M = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var v = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    assert_true(r.shape() == Shape(1, 2))
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_matrix_vector_complex_broadcasting() raises:
    print("test_matrix_vector_complex_broadcasting")
    alias dtype = DType.float32
    var M = Tensor[dtype].d4([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    var v = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
    var r = M.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    assert_true(r.shape() == Shape(1, 1, 2))
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


# ===== GRADIENT ACCUMULATION TESTS =====


fn test_matrix_vector_multiple_backward_calls() raises:
    print("test_matrix_vector_multiple_backward_calls")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)

    # First backward pass
    var r1 = M.matmul[mode=mv](v)
    var loss1 = r1.sum()
    loss1.backward()

    var v_grad_after_first = v.grad()
    var M_grad_after_first = M.grad()
    var r2 = M.matmul[mode=mv](v)
    var loss2 = r2.sum()
    loss2.backward()

    # Gradients should be doubled
    assert_true(v.grad().all_close(v_grad_after_first * 2.0))
    assert_true(M.grad().all_close(M_grad_after_first * 2.0))


fn test_matrix_vector_no_grad() raises:
    print("test_matrix_vector_no_grad")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=False)
    var r = M.matmul[mode=mv](v)

    # Should work without gradients
    assert_true(r.all_close(Tensor[dtype].d1([5.0, 11.0])))


# ===== COMPREHENSIVE TEST FUNCTION =====


fn run_all_matrix_vector_tests() raises:
    # Basic functionality
    test_matrix_vector_basic_forward_backward()
    test_matrix_vector_identity()
    test_matrix_vector_zeros_matrix()
    test_matrix_vector_zeros_vector()
    test_matrix_vector_single_element()

    # Batched operations
    test_matrix_vector_3d_2d_batched_matrix()
    test_matrix_vector_2d_2d_batched_vector()
    test_matrix_vector_3d_3d_batched_both()
    test_matrix_vector_broadcast_matrix()
    test_matrix_vector_broadcast_vector()

    # Higher dimensional batching
    test_matrix_vector_4d_batch()
    test_matrix_vector_3d_4d_broadcast()

    # View operations
    test_matrix_vector_with_matrix_view()
    test_matrix_vector_with_vector_view()
    test_matrix_vector_double_view()

    # Edge cases
    test_matrix_vector_large_dimensions()
    test_matrix_vector_non_contiguous_batch()
    test_matrix_vector_singleton_batch()
    test_matrix_vector_complex_broadcasting()

    # Gradient behavior
    test_matrix_vector_multiple_backward_calls()
    test_matrix_vector_no_grad()

    print("All matrix-vector multiplication tests passed! ✓")
