from tenmo import Tensor
from algorithm import vectorize
from sys import simd_width_of
from matrixshapevalidator import MatrixShapeValidator
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from gradbox import Gradbox
from ancestry import Ancestor
from shapes import Shape


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

        # FAST PATH: SIMD vectorization over columns - Gradbox is always contiguous
        B_contiguous = B.copy() if B.is_contiguous() else B.contiguous()
        for i in range(m):
            for k in range(n):
                var a_ik = A.load[simdwidth=1, validated=True](
                    i, k
                )  # Single scalar load

                @parameter
                fn process_columns[simd_width: Int](j: Int):
                    # Load contiguous SIMD vector from B's row
                    var b_vec = B_contiguous.load[
                        simdwidth=simd_width, validated=True
                    ](k, j)
                    var c_vec = C.load[simdwidth=simd_width, validated=True](
                        i, j
                    )
                    var result = a_ik * b_vec + c_vec
                    C.store[simdwidth=simd_width, validated=True](i, j, result)

                # Vectorize across all columns of B and C
                vectorize[process_columns, simdwidth](p)

        return C^


from broadcasthelper import ShapeBroadcaster
from common_utils import il, s


@fieldwise_init
@register_passable
struct MatmulNdBackward[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var grad_out = output.grad().copy()
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


fn main() raises:
    pass
