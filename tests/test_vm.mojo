from tenmo import Tensor
from shapes import Shape
from testing import assert_true
from strides import Strides
from matmul import MatmulNd
from vectormatrix import VectorMatmulNd


fn main() raises:
    # ===== BASIC VECTOR-MATRIX TESTS =====
    test_vector_matrix_1d_2d_basic()
    test_vector_matrix_identity()
    test_vector_matrix_zeros()
    test_vector_matrix_single_element()

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

    pass


# ===== BASIC VECTOR-MATRIX TESTS =====


fn test_vector_matrix_1d_2d_basic() raises:
    print("test_vector_matrix_1d_2d_basic")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var M = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v, identity)
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
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v, M)
    var loss = r.sum()
    loss.backward()

    # Each batch: v[i] @ identity = v[i]
    assert_true(r.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])))
    assert_true(v.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(M.grad().all_close(Tensor.d2([[4.0, 4.0], [6.0, 6.0]])))


fn test_vector_matrix_2d_3d_batched() raises:
    print("test_vector_matrix_2d_3d_batched")
    alias dtype = DType.float32
    var v = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )  # batch=2, dim=2
    var M = Tensor.d3(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )  # batch=2, 2x2
    var r = VectorMatmulNd.forward(v, M)
    var loss = r.sum()
    loss.backward()

    # Batch 0: [1,2] @ [[1,0],[0,1]] = [1,2]
    # Batch 1: [3,4] @ [[2,0],[0,2]] = [6,8]
    assert_true(r.all_close(Tensor.d2([[1.0, 2.0], [6.0, 8.0]])))
    assert_true(v.grad().all_close(Tensor.d2([[1.0, 1.0], [2.0, 2.0]])))
    assert_true(
        M.grad().all_close(
            Tensor.d3([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        )
    )


fn test_vector_matrix_broadcast_vector() raises:
    print("test_vector_matrix_broadcast_vector")
    alias dtype = DType.float32
    var v = Tensor.d1([1.0, 2.0], requires_grad=True)  # single vector
    var M = Tensor.d3(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )  # batch=2, 2x2
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v_view, M)
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
    var r = VectorMatmulNd.forward(v, M_view)
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


fn test_vector_matrix_single_element() raises:
    print("test_vector_matrix_single_element")
    alias dtype = DType.float32
    var v = Tensor.d1([2.0], requires_grad=True)  # 1D vector with 1 element
    var M = Tensor.d2([[3.0]], requires_grad=True)  # 1x1 matrix
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v, M)
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
    var r = VectorMatmulNd.forward(v_view, M)
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
    var r = VectorMatmulNd.forward(v, M)  # [2]

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
    var r = MatmulNd.forward(M, v.unsqueeze([-1])).squeeze([-1])  # [2]

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
    var r = VectorMatmulNd.forward(v, M)  # [2,2]

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
    var r = VectorMatmulNd.forward(v, M)  # [2,2] batch broadcasting

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
    var r = VectorMatmulNd.forward(v, M)  # [2,2]
    var loss = r.sum()
    loss.backward()
    # Check that grads flow correctly through the view to the base tensor
    assert_true(base.grad().shape() == base.shape())


fn test_vector_matrix_singleton_batch() raises:
    print("test_vector_matrix_singleton_batch")
    var v = Tensor.d2([[1.0, 2.0]], requires_grad=True)  # batch=1
    var M = Tensor.d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var r = VectorMatmulNd.forward(v, M)  # [1,2]
    var loss = r.sum()
    loss.backward()
    assert_true(v.grad().shape() == v.shape())
    assert_true(M.grad().shape() == M.shape())


fn test_vector_matrix_high_dimensional_batch() raises:
    print("test_vector_matrix_high_dimensional_batch")
    var v = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # [1,2,2]
    var M = Tensor.d3([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)  # [1,2,2]
    var r = VectorMatmulNd.forward(v, M)  # [1,2,2]
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
    var C = A_view.matmul_nd(B)
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
