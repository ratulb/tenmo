from tenmo import Tensor
from shapes import Shape
from testing import assert_true
from strides import Strides
from common_utils import s, il


fn main() raises:
    test_matmul_nd_comprehensive()
    test_matmul_nd_complete()


fn test_matmul_nd_3d_basic_with_grad() raises:
    print("test_matmul_nd_3d_basic_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var B = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # Verify gradients numerically
    var expected_A_grad = Tensor[dtype].d3(
        [
            [[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]],
            [[15.0, 19.0, 23.0], [15.0, 19.0, 23.0]],
        ]
    )
    var expected_B_grad = Tensor[dtype].d3(
        [
            [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]],
            [[17.0, 17.0], [19.0, 19.0], [21.0, 21.0]],
        ]
    )
    assert_true(A.grad().all_close(expected_A_grad))
    assert_true(B.grad().all_close(expected_B_grad))


fn test_matmul_nd_3d_broadcast_A_with_grad() raises:
    print("test_matmul_nd_3d_broadcast_A_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # Shape(1,2,2)
    var B = Tensor[dtype].d3(
        [[[2.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 3.0]]], requires_grad=True
    )  # Shape(2,2,2)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # A was broadcasted from (1,2,2) to (2,2,2), so gradients should be summed
    var expected_A_grad = Tensor[dtype].d3([[[5.0, 5.0], [5.0, 5.0]]])
    var expected_B_grad = Tensor[dtype].d3(
        [[[4.0, 4.0], [6.0, 6.0]], [[4.0, 4.0], [6.0, 6.0]]]
    )
    assert_true(A.grad().all_close(expected_A_grad))
    assert_true(B.grad().all_close(expected_B_grad))


fn test_matmul_nd_3d_broadcast_B_with_grad() raises:
    print("test_matmul_nd_3d_broadcast_B_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # Shape(2,2,2)
    var B = Tensor[dtype].d3(
        [[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
    )  # Shape(1,2,2)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # B was broadcasted from (1,2,2) to (2,2,2), so gradients should be summed
    var expected_A_grad = Tensor[dtype].d3(
        [[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]]
    )
    var expected_B_grad = Tensor[dtype].d3([[[16.0, 16.0], [20.0, 20.0]]])
    assert_true(A.grad().all_close(expected_A_grad))
    assert_true(B.grad().all_close(expected_B_grad))


# ===== 4D BATCH MATMUL TESTS WITH GRADIENTS =====


fn test_matmul_nd_4d_basic_with_grad() raises:
    print("test_matmul_nd_4d_basic_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ],
        requires_grad=True,
    )
    var B = Tensor[dtype].d4(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
            [[[3.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 4.0]]],
        ],
        requires_grad=True,
    )
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # Check first batch element gradients
    var A_grad = A.grad().as_tensor()
    var first_batch_grad = A_grad[il(0, 0), s(), s()]
    var expected_first = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])
    assert_true(first_batch_grad.all_close(expected_first))


fn test_matmul_nd_4d_complex_broadcast_with_grad() raises:
    print("test_matmul_nd_4d_complex_broadcast_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d4(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        requires_grad=True,
    )  # Shape(1,2,2,2)
    var B = Tensor[dtype].d4(
        [[[[2.0, 0.0], [0.0, 2.0]]], [[[3.0, 0.0], [0.0, 3.0]]]],
        requires_grad=True,
    )  # Shape(2,1,2,2)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # A: (1,2,2,2) broadcast to (2,2,2,2), B: (2,1,2,2) broadcast to (2,2,2,2)
    assert_true(A.grad().shape() == Shape(1, 2, 2, 2))
    assert_true(B.grad().shape() == Shape(2, 1, 2, 2))


# ===== VIEW TESTS WITH GRADIENTS =====


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

    var B = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var C = A_view.matmul(B)
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


fn test_matmul_nd_with_strided_view_grad() raises:
    print("test_matmul_nd_with_strided_view_grad")
    alias dtype = DType.float32
    var base_A = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        requires_grad=True,
    )

    # Create view taking every other column
    var A_view = base_A.view(
        shape=Shape(3, 2),
        strides=Strides(4, 2),  # Skip every other column
        offset=1,  # Start from second element
    )

    var B = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var C = A_view.matmul(B)
    var loss = C.sum()
    loss.backward()

    # Gradients should only flow to the strided elements
    var expected_base_grad = Tensor[dtype].d2(
        [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    )
    assert_true(base_A.grad().all_close(expected_base_grad))


# ===== EDGE CASE TESTS WITH GRADIENTS =====


fn test_matmul_nd_single_batch_with_grad() raises:
    print("test_matmul_nd_single_batch_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var B = Tensor[dtype].d3([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()
    var expected_A_grad = Tensor[dtype].d3([[[2.0, 2.0], [2.0, 2.0]]])
    var expected_B_grad = Tensor[dtype].d3([[[4.0, 4.0], [6.0, 6.0]]])
    assert_true(A.grad().all_close(expected_A_grad))
    assert_true(B.grad().all_close(expected_B_grad))


fn test_matmul_nd_large_batch_small_matrices_with_grad() raises:
    print("test_matmul_nd_large_batch_small_matrices_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[2.0]], [[3.0]], [[4.0]]], requires_grad=True)
    var B = Tensor[dtype].d3([[[3.0]], [[4.0]], [[5.0]]], requires_grad=True)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    var expected_A_grad = Tensor[dtype].d3([[[3.0]], [[4.0]], [[5.0]]])
    var expected_B_grad = Tensor[dtype].d3([[[2.0]], [[3.0]], [[4.0]]])
    assert_true(A.grad().all_close(expected_A_grad))
    assert_true(B.grad().all_close(expected_B_grad))


fn test_matmul_nd_identity_batch_with_grad() raises:
    print("test_matmul_nd_identity_batch_with_grad")
    _ = """alias dtype = DType.float32
    var A = Tensor[dtype].d3([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ], requires_grad=True)
    var identity = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var B = Tensor.stack([identity, identity], axis=0)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # A × I = A, so gradient should be ones
    var expected_A_grad = Tensor[dtype].d3([
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]]
    ])
    assert_true(A.grad().all_close(expected_A_grad))"""


fn test_matmul_nd_zeros_with_grad() raises:
    print("test_matmul_nd_zeros_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var B = Tensor[dtype].zeros(Shape(2, 2, 2), requires_grad=True)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    # C is zeros, so gradient through sum should give ones for C, but A and B gradients depend on the operation
    var expected_A_grad = Tensor[dtype].zeros(Shape(2, 2, 2))
    var expected_B_grad = Tensor[dtype].d3(
        [[[4.0, 4.0], [6.0, 6.0]], [[12.0, 12.0], [14.0, 14.0]]]
    )
    assert_true(A.grad().all_close(expected_A_grad))
    assert_true(B.grad().all_close(expected_B_grad))


# ===== MIXED DIMENSION TESTS WITH GRADIENTS =====


fn test_matmul_nd_mixed_batch_dims_with_grad() raises:
    print("test_matmul_nd_mixed_batch_dims_with_grad")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # Shape(2,2,2)
    var B = Tensor[dtype].d4(
        [[[[2.0, 0.0], [0.0, 2.0]]], [[[3.0, 0.0], [0.0, 3.0]]]],
        requires_grad=True,
    )  # Shape(2,1,2,2)
    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    assert_true(C.shape() == Shape(2, 2, 2, 2))
    assert_true(A.grad().shape() == Shape(2, 2, 2))
    assert_true(B.grad().shape() == Shape(2, 1, 2, 2))


# ===== COMPREHENSIVE TEST FUNCTIONS =====


fn test_matmul_nd_forward_comprehensive() raises:
    print("Running comprehensive matmul_nd forward tests...")
    test_matmul_nd_3d_basic_with_grad()
    test_matmul_nd_3d_broadcast_A_with_grad()
    test_matmul_nd_3d_broadcast_B_with_grad()
    test_matmul_nd_4d_basic_with_grad()
    test_matmul_nd_4d_complex_broadcast_with_grad()
    test_matmul_nd_single_batch_with_grad()
    test_matmul_nd_large_batch_small_matrices_with_grad()
    test_matmul_nd_identity_batch_with_grad()
    test_matmul_nd_zeros_with_grad()
    test_matmul_nd_mixed_batch_dims_with_grad()
    print("All matmul_nd forward tests passed! ✓")


fn test_matmul_nd_view_gradients() raises:
    print("Testing matmul_nd with views and gradients...")
    test_matmul_nd_with_view_offset_grad()
    test_matmul_nd_with_strided_view_grad()
    print("All matmul_nd view gradient tests passed! ✓")


fn test_matmul_nd_complete() raises:
    print("Running complete matmul_nd test suite...")
    test_matmul_nd_forward_comprehensive()
    test_matmul_nd_view_gradients()
    print("All matmul_nd tests passed! ✓")


fn test_matmul_nd_3d_basic() raises:
    print("test_matmul_nd_3d_basic")
    alias dtype = DType.float32
    # Batch of 2 matrices: 2x(2x3) × 2x(3x2) → 2x(2x2)
    var A = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    var B = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    var C = A.matmul[track_grad=False](B)
    var expected = Tensor[dtype].d3(
        [[[22.0, 28.0], [49.0, 64.0]], [[220.0, 244.0], [301.0, 334.0]]]
    )

    assert_true(C.all_close(expected))


fn test_matmul_nd_3d_broadcast_A() raises:
    print("test_matmul_nd_3d_broadcast_A")
    alias dtype = DType.float32
    # A: 1x(2x3) broadcast to match B: 2x(3x2)
    var A = Tensor[dtype].d3([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    var B = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    var C = A.matmul[track_grad=False](B)
    var expected = Tensor[dtype].d3(
        [[[22.0, 28.0], [49.0, 64.0]], [[58.0, 64.0], [139.0, 154.0]]]
    )
    assert_true(C.all_close(expected))


fn test_matmul_nd_3d_broadcast_B() raises:
    print("test_matmul_nd_3d_broadcast_B")
    alias dtype = DType.float32
    # B: 1x(3x2) broadcast to match A: 2x(2x3)
    var A = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    var B = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    var C = A.matmul[track_grad=False](B)
    var expected = Tensor[dtype].d3(
        [[[22.0, 28.0], [49.0, 64.0]], [[76.0, 100.0], [103.0, 136.0]]]
    )
    assert_true(C.all_close(expected))


# ===== 4D BATCH MATMUL TESTS =====


fn test_matmul_nd_4d_basic() raises:
    print("test_matmul_nd_4d_basic")
    alias dtype = DType.float32
    # 2x2 batch of 2x3 matrices × 2x2 batch of 3x2 matrices → 2x2 batch of 2x2 matrices
    var A = Tensor[dtype].d4(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            [
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
            ],
        ]
    )
    var B = Tensor[dtype].d4(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            [
                [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
                [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
            ],
        ]
    )
    var C = A.matmul[track_grad=False](B)
    # Only check first batch element to keep test manageable
    var first_batch = C[il(0, 0), s(), s()]
    var expected_first = Tensor[dtype].d2([[22.0, 28.0], [49.0, 64.0]])
    assert_true(first_batch.all_close(expected_first))


fn test_matmul_nd_4d_complex_broadcast() raises:
    print("test_matmul_nd_4d_complex_broadcast")
    alias dtype = DType.float32
    # A: 1x2 batch, B: 2x1 batch → broadcast to 2x2 batch
    var A = Tensor[dtype].d4(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
    )  # Shape(1, 2, 2, 2)

    var B = Tensor[dtype].d4(
        [[[[2.0, 0.0], [0.0, 2.0]]], [[[3.0, 0.0], [0.0, 3.0]]]]
    )  # Shape(2, 1, 2, 2)

    var C = A.matmul[track_grad=False](B)
    # Result should be Shape(2, 2, 2, 2)
    assert_true(C.shape() == Shape(2, 2, 2, 2))

    # Check specific elements
    _ = """var c0000 = C.load[simdwidth=1, validated=True](0, 0, 0, 0)
    assert_almost_equal(c0000, 2.0)  # 1*2 + 2*0 = 2"""


# ===== EDGE CASE TESTS =====


fn test_matmul_nd_single_batch_element() raises:
    print("test_matmul_nd_single_batch_element")
    alias dtype = DType.float32
    # Single batch element: 1x(2x3) × 1x(3x2) → 1x(2x2)
    var A = Tensor[dtype].d3([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    var B = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    var C = A.matmul[track_grad=False](B)
    var expected = Tensor[dtype].d3([[[22.0, 28.0], [49.0, 64.0]]])
    assert_true(C.all_close(expected))


fn test_matmul_nd_large_batch_small_matrices() raises:
    print("test_matmul_nd_large_batch_small_matrices")
    alias dtype = DType.float32
    # Many batches of 1x1 matrices
    var A = Tensor[dtype].d3(
        [[[2.0]], [[3.0]], [[4.0]], [[5.0]]]
    )  # Shape(4, 1, 1)
    var B = Tensor[dtype].d3(
        [[[3.0]], [[4.0]], [[5.0]], [[6.0]]]
    )  # Shape(4, 1, 1)
    var C = A.matmul[track_grad=False](B)
    var expected = Tensor[dtype].d3([[[6.0]], [[12.0]], [[20.0]], [[30.0]]])
    assert_true(C.all_close(expected))


fn test_matmul_nd_identity_batch() raises:
    print("test_matmul_nd_identity_batch")
    _ = """alias dtype = DType.float32
    # Batch of identity matrices
    var identity = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    # Create batch of identity matrices
    var B = Tensor[dtype].stack([identity, identity], axis=0)
    var C = A.matmul[track_grad=False](B)
    # A × I should equal A
    assert_true(C.all_close(A))"""


fn test_matmul_nd_zeros_batch() raises:
    print("test_matmul_nd_zeros_batch")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var B = Tensor[dtype].zeros(Shape(2, 2, 2))
    var C = A.matmul[track_grad=False](B)
    var expected = Tensor[dtype].zeros(Shape(2, 2, 2))
    assert_true(C.all_close(expected))


fn test_matmul_nd_ones_batch() raises:
    print("test_matmul_nd_ones_batch")
    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var B = Tensor[dtype].ones(Shape(2, 2, 2))
    var C = A.matmul[track_grad=False](B)
    # Each output element is sum of row in A
    var expected = Tensor[dtype].d3(
        [[[3.0, 3.0], [7.0, 7.0]], [[11.0, 11.0], [15.0, 15.0]]]
    )
    assert_true(C.all_close(expected))


# ===== MIXED DIMENSION TESTS =====


fn test_matmul_nd_mixed_batch_dims() raises:
    print("test_matmul_nd_mixed_batch_dims")
    alias dtype = DType.float32
    # A: 3D, B: 4D with broadcasting
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )  # Shape(2, 2, 2)

    var B = Tensor[dtype].d4(
        [[[[2.0, 0.0], [0.0, 2.0]]], [[[3.0, 0.0], [0.0, 3.0]]]]
    )  # Shape(2, 1, 2, 2)

    var C = A.matmul[track_grad=False](B)
    assert_true(C.shape() == Shape(2, 2, 2, 2))


# ===== CONSOLIDATED TEST FUNCTION =====


fn test_matmul_nd_comprehensive() raises:
    print("Running comprehensive matmul_nd tests...")
    test_matmul_batched_non_contiguous_view_case()
    test_matmul_nd_3d_basic()
    test_matmul_nd_3d_broadcast_A()
    test_matmul_nd_3d_broadcast_B()
    test_matmul_nd_4d_basic()
    test_matmul_nd_4d_complex_broadcast()
    test_matmul_nd_single_batch_element()
    test_matmul_nd_large_batch_small_matrices()
    test_matmul_nd_identity_batch()
    test_matmul_nd_zeros_batch()
    test_matmul_nd_ones_batch()
    test_matmul_nd_mixed_batch_dims()
    test_matmul2d_basic_case()
    test_matmul2d_rectangular_case()
    test_matmul2d_non_square_large_case()
    test_matmul_batched_basic()
    test_matmul_batched_broadcast_B()
    test_matmul_batched_broadcast_A()
    test_matmul_batched_3d_input_case()
    test_matmul_batched_result_shape()
    test_matmul_nd_with_higher_dim_batch()
    print("All matmul_nd forward tests passed! ✓")


# ================================================================
#  Batch MatMul Forward Tests
# ================================================================


fn test_matmul2d_basic_case() raises:
    print("test_matmul2d_basic_case")
    var A = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor.d2([[5.0, 6.0], [7.0, 8.0]])
    var C = A.matmul(B)
    assert_true(C.all_close(Tensor.d2([[19.0, 22.0], [43.0, 50.0]])))


fn test_matmul2d_rectangular_case() raises:
    print("test_matmul2d_rectangular_case")
    var A = Tensor.d2([[1.0, 2.0, 3.0]])
    var B = Tensor.d2([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    var C = A.matmul(B)
    assert_true(C.all_close(Tensor.d2([[40.0, 46.0]])))


fn test_matmul2d_non_square_large_case() raises:
    print("test_matmul2d_non_square_large_case")
    var A = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var B = Tensor.d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    var C = A.matmul(B)
    assert_true(
        C.all_close(
            Tensor.d2(
                [[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]
            )
        )
    )


fn test_matmul_batched_basic() raises:
    print("test_matmul_batched_basic")
    var A = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    var B = Tensor.d3([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    var C = A.matmul(B)
    assert_true(
        C.all_close(
            Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 12.0], [14.0, 16.0]]])
        )
    )


fn test_matmul_batched_broadcast_B() raises:
    print("test_matmul_batched_broadcast_B")
    var A = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    var B = Tensor.d2([[2.0, 0.0], [0.0, 2.0]])
    var C = A.matmul(B)
    assert_true(
        C.all_close(
            Tensor.d3([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]])
        )
    )


fn test_matmul_batched_broadcast_A() raises:
    print("test_matmul_batched_broadcast_A")
    var A = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor.d3([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    var C = A.matmul(B)
    assert_true(
        C.all_close(
            Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]])
        )
    )


fn test_matmul_batched_3d_input_case() raises:
    print("test_matmul_batched_3d_input_case")
    var A = Tensor.d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    var B = Tensor.d3(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ]
    )
    var C = A.matmul(B)
    assert_true(
        C.all_close(
            Tensor.d3(
                [[[4.0, 5.0], [10.0, 11.0]], [[76.0, 100.0], [103.0, 136.0]]]
            )
        )
    )


fn test_matmul_batched_non_contiguous_view_case() raises:
    print("test_matmul_batched_non_contiguous_view_case")
    var A = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    # Make a view skipping the first batch
    var A_view = A.view(
        shape=Shape(1, 2, 2), strides=Strides(4, 2, 1), offset=4
    )
    var B = Tensor.d2([[1.0, 0.0], [0.0, 1.0]])

    var C = A_view.matmul(B)
    assert_true(C.all_close(Tensor.d3([[[5.0, 6.0], [7.0, 8.0]]])))


fn test_matmul_batched_result_shape() raises:
    print("test_matmul_batched_result_shape")
    var A = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    var B = Tensor.d2([[1.0, 0.0], [0.0, 1.0]])
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(2, 2, 2))


fn test_matmul_nd_with_higher_dim_batch() raises:
    print("test_matmul_nd_with_higher_dim_batch")
    var A = Tensor.d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]
    )
    var B = Tensor.d2([[1.0, 0.0], [0.0, 1.0]])
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(2, 2, 2, 2))
    assert_true(C.all_close(A))  # Identity matrix case
