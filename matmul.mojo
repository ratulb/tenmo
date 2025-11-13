from tenmo import Tensor
from algorithm import vectorize
from sys import simd_width_of
from matrixshapevalidator import MatrixShapeValidator
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from gradbox import Gradbox
from ancestry import Ancestor
from shapes import Shape
from broadcasthelper import ShapeBroadcaster
from common_utils import il, s


@fieldwise_init
@register_passable
struct Matmul2dBackward[dtype: DType](ImplicitlyCopyable):
    fn backward[
        simdwidth: Int = simd_width_of[dtype]()
    ](self, output: Tensor[dtype]) -> List[
        Tuple[Ancestor[dtype], Gradbox[dtype], Int]
    ]:
        var grad_out = output.grad().copy()  # Always contiguous
        var A = output.ancestry().get(0)  # First input
        var B = output.ancestry().get(1)  # Second input

        var A_shape = A.shape()
        var B_shape = B.shape()
        var m = A_shape[0]
        var n = A_shape[1]  # == B_shape[0]
        var p = B_shape[1]

        var result = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]()

        # ===== GRADIENT FOR A: dL/dA = dL/dC × B^T =====
        if A.requires_grad():
            var grad_A = Gradbox[dtype].zeros(Shape([m, n]))

            # Only compute B^T if we need gradients for A. We male B^T contiguous
            var B_T = B.tensor().transpose[track_grad=False](1, 0).contiguous()

            for i in range(m):
                for k in range(p):
                    var grad_ik = grad_out.load[simdwidth=1, validated=True](
                        i, k
                    )

                    @parameter
                    fn process_columns_a[simd_width: Int](j: Int):
                        var b_vec = B_T.load[
                            simdwidth=simd_width, validated=True
                        ](k, j)
                        var grad_a_vec = grad_A.load[
                            simdwidth=simd_width, validated=True
                        ](i, j)
                        var result = grad_ik * b_vec + grad_a_vec
                        grad_A.store[simdwidth=simd_width, validated=True](
                            i, j, result
                        )

                    vectorize[process_columns_a, simdwidth](n)

            result.append((A.copy(), grad_A^, AddTensor))

        # ===== GRADIENT FOR B: dL/dB = A^T × dL/dC =====
        if B.requires_grad():
            var grad_B = Gradbox[dtype].zeros(Shape([n, p]))

            # Only compute A^T if we need gradients for B
            # var A_T = A.transpose().contiguous()
            var A_T = A.tensor().transpose[track_grad=False](1, 0)

            for j in range(n):
                for i in range(m):
                    var a_ji = A_T.load[simdwidth=1, validated=True](j, i)

                    @parameter
                    fn process_columns_b[simd_width: Int](k: Int):
                        var grad_vec = grad_out.load[
                            simdwidth=simd_width, validated=True
                        ](i, k)
                        var grad_b_vec = grad_B.load[
                            simdwidth=simd_width, validated=True
                        ](j, k)
                        var result = a_ji * grad_vec + grad_b_vec
                        grad_B.store[simdwidth=simd_width, validated=True](
                            j, k, result
                        )

                    vectorize[process_columns_b, simdwidth](p)

            result.append((B^, grad_B^, AddTensor))

        return result^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct Matmul2d[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    @always_inline
    fn forward[
        track_grad: Bool = True, simdwidth: Int = simd_width_of[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        var A_shape = A.shape()
        var B_shape = B.shape()
        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]
        # Validate shapes
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)
        var C = Tensor[dtype].zeros(Shape([m, p]))

        # Check if B is SIMD-friendly (contiguous in its row-major layout)
        var contiguous = B.is_contiguous()

        if contiguous:
            # FAST PATH: SIMD vectorization over columns
            for i in range(m):
                for k in range(n):
                    var a_ik = A.load[simdwidth=1, validated=True](
                        i, k
                    )  # Single scalar load

                    @parameter
                    fn process_columns[simd_width: Int](j: Int):
                        # Load contiguous SIMD vector from B's row
                        var b_vec = B.load[
                            simdwidth=simd_width, validated=True
                        ](k, j)
                        var c_vec = C.load[
                            simdwidth=simd_width, validated=True
                        ](i, j)
                        var result = a_ik * b_vec + c_vec
                        C.store[simdwidth=simd_width, validated=True](
                            i, j, result
                        )

                    # Vectorize across all columns of B and C
                    vectorize[process_columns, simdwidth](p)

        else:
            # SLOW PATH: Scalar fallback for non-contiguous B
            for i in range(m):
                for k in range(n):
                    var a_ik = A.load[simdwidth=1, validated=True](i, k)
                    for j in range(p):
                        # Scalar access - works for any stride pattern
                        var current = C.load[simdwidth=1, validated=True](i, j)
                        var result = current + (
                            a_ik * B.load[simdwidth=1, validated=True](k, j)
                        )
                        C.store[simdwidth=1, validated=True](
                            i,
                            j,
                            result,
                        )

        # Only attach backward handler if gradients are needed
        @parameter
        if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backward_fn = Matmul2dBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(backward_fn^)
                C.add_ancestry(A)
                C.add_ancestry(B)

        return C^

    # Tensor and Gradbox matmul_2d - No backward fn
    @staticmethod
    @always_inline
    fn forward[
        simdwidth: Int = simd_width_of[dtype]()
    ](A: Tensor[dtype], B: Gradbox[dtype]) -> Gradbox[dtype]:
        var A_shape = A.shape()
        var B_shape = B.shape()
        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]
        # Validate shapes
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var C = Gradbox[dtype].zeros(Shape([m, p]))

        # FAST PATH: SIMD vectorization over columns - Gradbox is always contiguous
        for i in range(m):
            for k in range(n):
                var a_ik = A.load[simdwidth=1, validated=True](
                    i, k
                )  # Single scalar load

                @parameter
                fn process_columns[simd_width: Int](j: Int):
                    # Load contiguous SIMD vector from B's row
                    var b_vec = B.load[simdwidth=simd_width, validated=True](
                        k, j
                    )
                    var c_vec = C.load[simdwidth=simd_width, validated=True](
                        i, j
                    )
                    var result = a_ik * b_vec + c_vec
                    C.store[simdwidth=simd_width, validated=True](i, j, result)

                # Vectorize across all columns of B and C
                vectorize[process_columns, simdwidth](p)

        return C^

    # Gradbox and Tensor matmul_2d - No backward fn
    @staticmethod
    @always_inline
    fn forward[
        simdwidth: Int = simd_width_of[dtype]()
    ](A: Gradbox[dtype], B: Tensor[dtype]) -> Gradbox[dtype]:
        var A_shape = A.shape()
        var B_shape = B.shape()
        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]
        # Validate shapes
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var C = Gradbox[dtype].zeros(Shape([m, p]))
        var contiguous = B.is_contiguous()

        if contiguous:
            # FAST PATH: SIMD vectorization over columns - Gradbox is always contiguous
            # B_contiguous = B.copy() if B.is_contiguous() else B.contiguous()
            for i in range(m):
                for k in range(n):
                    var a_ik = A.load[simdwidth=1, validated=True](
                        i, k
                    )  # Single scalar load

                    @parameter
                    fn process_columns[simd_width: Int](j: Int):
                        # Load contiguous SIMD vector from B's row
                        var b_vec = B.load[
                            simdwidth=simd_width, validated=True
                        ](k, j)
                        var c_vec = C.load[
                            simdwidth=simd_width, validated=True
                        ](i, j)
                        var result = a_ik * b_vec + c_vec
                        C.store[simdwidth=simd_width, validated=True](
                            i, j, result
                        )

                    # Vectorize across all columns of B and C
                    vectorize[process_columns, simdwidth](p)
        else:
            # SLOW PATH: Scalar fallback for non-contiguous B
            for i in range(m):
                for k in range(n):
                    var a_ik = A.load[simdwidth=1, validated=True](i, k)
                    for j in range(p):
                        # Scalar access - works for any stride pattern
                        var current = C.load[simdwidth=1, validated=True](i, j)
                        var result = current + (
                            a_ik * B.load[simdwidth=1, validated=True](k, j)
                        )
                        C.store[simdwidth=1, validated=True](
                            i,
                            j,
                            result,
                        )

        return C^


@fieldwise_init
@register_passable
struct MatmulNdBackward1[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var grad_out = output.grad().copy()
        print("grad_out is shared: ", grad_out.shared())
        print("grad_out is shared: ", grad_out.shared())
        print("grad_out is shared: ", grad_out.shared())
        print("grad_out is shared: ", grad_out.shared())
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        var results = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]()

        if A.requires_grad():
            var grad_A = MatmulNd.forward[track_grad=False](
                grad_out.as_tensor(),
                B.tensor().transpose[track_grad=False](axes=[-1, -2]),
            ).as_gradbox()
            grad_A = grad_A.sum_over_broadcasted_axes(A.shape())
            results.append((A.copy(), grad_A^, AddTensor))

        if B.requires_grad():
            var grad_B = MatmulNd.forward[track_grad=False](
                A.tensor().transpose[track_grad=False](axes=[-1, -2]),
                grad_out.as_tensor(),
            ).as_gradbox()
            grad_B = grad_B.sum_over_broadcasted_axes(B.shape())
            results.append((B^, grad_B^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct MatmulNdBackward2[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var grad_out = output.grad().copy()  # shape: batch_shape + [m, n]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)
        var A_shape = A.shape()
        var B_shape = B.shape()

        var m = A_shape[-2]
        var k = A_shape[-1]
        var n = B_shape[-1]

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )

        var batch_dims_a = A_shape[:-2]
        var batch_dims_b = B_shape[:-2]

        var results = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]()

        # Gradient wrt A: dA = grad_out @ B^T
        if A.requires_grad():
            var grad_A = Gradbox[dtype].zeros(A_shape, share=True)
            for indices in batch_shape:
                var A_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, batch_dims_a
                )
                var B_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, batch_dims_b
                )

                var grad_out_slice = grad_out[il(indices), s(), s()]
                var B_slice = B.tensor()[il(B_indices), s(), s()]
                var B_T = B_slice.transpose[track_grad=False](1, 0)

                var grad_A_slice = Matmul2d[dtype].forward(grad_out_slice, B_T)
                grad_A[il(A_indices), s(), s()].buffer.fill_equal_shape(
                    grad_A_slice.buffer
                )

            # Reduce over broadcasted axes if A was broadcasted
            grad_A = grad_A.sum_over_broadcasted_axes(A_shape)
            results.append((A.copy(), grad_A^, AddTensor))

        # Gradient wrt B: dB = A^T @ grad_out
        if B.requires_grad():
            var grad_B = Gradbox[dtype].zeros(B_shape, share=True)
            for indices in batch_shape:
                var A_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, batch_dims_a
                )
                var B_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, batch_dims_b
                )

                var A_slice = A.tensor()[il(A_indices), s(), s()]
                var grad_out_slice = grad_out[il(indices), s(), s()]
                var A_T = A_slice.transpose[track_grad=False](1, 0)

                var grad_B_slice = Matmul2d[dtype].forward(A_T, grad_out_slice)
                grad_B[il(B_indices), s(), s()].buffer.fill_equal_shape(
                    grad_B_slice.buffer
                )

            grad_B = grad_B.sum_over_broadcasted_axes(B_shape)
            results.append((B^, grad_B^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct MatmulNdBackward[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var grad_out = output.grad().copy()  # GradBox: batch_shape + [m, n]
        var A = output.ancestry().get(0)  # Tensor ancestor
        var B = output.ancestry().get(1)  # Tensor ancestor
        var A_shape = A.shape()
        var B_shape = B.shape()

        var results = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]()

        # Gradient for A: dA = dC @ B^T
        if A.requires_grad():
            # Transpose B: batch_shape + [k, n] → batch_shape + [n, k]
            var B_transposed = B.tensor().transpose[track_grad=False](
                axes=[-1, -2]
            )

            # Batched matmul: GradBox @ Tensor → GradBox (no backward needed)
            var A_batch_grad = MatmulNd.forward(grad_out, B_transposed)

            # Sum over broadcasted dimensions
            var final_grad_A = A_batch_grad.sum_over_broadcasted_axes(A_shape)
            results.append((A.copy(), final_grad_A^, AddTensor))

        # Gradient for B: dB = A^T @ dC
        if B.requires_grad():
            # Transpose A: batch_shape + [m, k] → batch_shape + [k, m]
            var A_transposed = A.tensor().transpose[track_grad=False](
                axes=[-1, -2]
            )

            # Batched matmul: Tensor @ GradBox → GradBox (no backward needed)
            var B_batch_grad = MatmulNd.forward(A_transposed, grad_out)

            # Sum over broadcasted dimensions
            var final_grad_B = B_batch_grad.sum_over_broadcasted_axes(B_shape)
            results.append((B^, final_grad_B^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct MatmulNd[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        A_shape = A.shape()
        B_shape = B.shape()
        MatrixShapeValidator.validate_matrix_shapes_nd(A_shape, B_shape)
        # shapes: batch + [m, k], batch + [k, n]
        batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )  # all dims except last 2

        m = A_shape[-2]
        n = B_shape[-1]

        batch_dims_a = A_shape[:-2]
        batch_dims_b = B_shape[:-2]

        out_shape = batch_shape + [m, n]
        C = Tensor[dtype].zeros(out_shape)

        for indices in batch_shape:
            # select batch slices
            A_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            B_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )
            A_slice = A[il(A_indices), s(), s()]
            B_slice = B[il(B_indices), s(), s()]
            C_slice = C[il(indices), s(), s()]

            result = Matmul2d.forward[track_grad=False](
                A_slice,
                B_slice,
            )
            C_slice.buffer.fill_equal_shape(result.buffer)

        # Only attach backward handler if gradients are needed
        @parameter
        if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backward_fn = MatmulNdBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(backward_fn^)
                C.add_ancestry(A)
                C.add_ancestry(B)

        return C^

    @staticmethod
    fn forward(A: Tensor[dtype], B: Gradbox[dtype]) -> Gradbox[dtype]:
        A_shape = A.shape()
        B_shape = B.shape()
        MatrixShapeValidator.validate_matrix_shapes_nd(A_shape, B_shape)

        # shapes: batch + [m, k], batch + [k, n]
        batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )
        m = A_shape[-2]
        n = B_shape[-1]
        batch_dims_a = A_shape[:-2]
        batch_dims_b = B_shape[:-2]

        out_shape = batch_shape + [m, n]
        var C = Gradbox[dtype].zeros(out_shape, share=True)

        for indices in batch_shape:
            var A_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            var B_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )

            var A_slice = A[il(A_indices), s(), s()]
            var B_slice = B[il(B_indices), s(), s()]
            var C_slice = C[il(indices), s(), s()]

            # Use 2D matmul for GradBox (no backward needed)
            var result = Matmul2d.forward(A_slice, B_slice)
            C_slice.buffer.fill_equal_shape[overwrite=False](result.buffer)

        return C^

    # Gradbox and Tensor matmul_nd - No backward fn needed
    @staticmethod
    fn forward(A: Gradbox[dtype], B: Tensor[dtype]) -> Gradbox[dtype]:
        A_shape = A.shape()
        B_shape = B.shape()
        MatrixShapeValidator.validate_matrix_shapes_nd(A_shape, B_shape)

        batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )
        m = A_shape[-2]
        n = B_shape[-1]
        batch_dims_a = A_shape[:-2]
        batch_dims_b = B_shape[:-2]

        out_shape = batch_shape + [m, n]
        var C = Gradbox[dtype].zeros(out_shape, share=True)

        for indices in batch_shape:
            var A_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            var B_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )

            var A_slice = A[il(A_indices), s(), s()]
            var B_slice = B[il(B_indices), s(), s()]
            var C_slice = C[il(indices), s(), s()]

            # Use 2D matmul for GradBox (no backward needed)
            var result = Matmul2d.forward(A_slice, B_slice)
            C_slice.buffer.fill_equal_shape[overwrite=False](result.buffer)

        return C^


fn main() raises:
    test_matmul_nd_with_view_offset_grad()


from testing import assert_true
from strides import Strides


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

    A_view.print()

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
    base_A.grad().print()
    assert_true(base_A.grad().all_close(expected_base_grad))
