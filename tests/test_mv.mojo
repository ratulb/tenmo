from tenmo import Tensor
from shapes import Shape
from matrixvector import MatrixVectorMulNd


fn main() raises:
    run_all_matrix_vector_tests()


# ===== BASIC MATRIX-VECTOR TESTS =====

from testing import assert_true
from strides import Strides


fn test_matrix_vector_basic_forward_backward() raises:
    print("test_matrix_vector_basic_forward_backward")
    alias dtype = DType.float32
    var M = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
    var loss = r.sum()
    loss.backward()

    # Batch 0: [[1,2],[3,4]] @ [1,2] = [5,11]
    # Batch 1: [[1,2],[3,4]] @ [3,4] = [11,25]
    assert_true(r.all_close(Tensor[dtype].d2([[5.0, 11.0], [11.0, 25.0]])))
    # grad_v = M^T @ [[1,1],[1,1]] = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    assert_true(v.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 6.0]])))
    # grad_M = [[1,1],[1,1]] @ v^T = [[1,1],[1,1]] @ [[1,2],[3,4]] = [[4,6],[4,6]]
    assert_true(M.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 6.0]])))


fn test_matrix_vector_3d_3d_batched_both() raises:
    print("test_matrix_vector_3d_3d_batched_both")
    alias dtype = DType.float32
    var M = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var v = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M_view, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v_view)
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
    var r = MatrixVectorMulNd[dtype].forward(M_view, v_view)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M_view, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r1 = MatrixVectorMulNd[dtype].forward(M, v)
    var loss1 = r1.sum()
    loss1.backward()

    var v_grad_after_first = v.grad().contiguous()
    var M_grad_after_first = M.grad().contiguous()
    # Second backward pass (should accumulate)
    var r2 = MatrixVectorMulNd[dtype].forward(M, v)
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
    var r = MatrixVectorMulNd[dtype].forward(M, v)

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
