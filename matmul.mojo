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
from common_utils import il, s, panic


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
            var A_batch_grad = MatmulNd[dtype].forward(grad_out, B_transposed)

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
            var B_batch_grad = MatmulNd[dtype].forward(A_transposed, grad_out)

            # Sum over broadcasted dimensions
            var final_grad_B = B_batch_grad.sum_over_broadcasted_axes(B_shape)
            results.append((B^, final_grad_B^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct MatmulNd[dtype: DType](ImplicitlyCopyable):
    @always_inline
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

            result = Matmul2d[dtype].forward[track_grad=False](
                A_slice,
                B_slice,
            )
            C_slice.buffer.fill_equal_shape[overwrite=True](result.buffer)

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

    @always_inline
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
            var result = Matmul2d[dtype].forward(A_slice, B_slice)
            C_slice.buffer.fill_equal_shape[overwrite=False](result.buffer)

        return C^

    # Gradbox and Tensor matmul_nd - No backward fn needed
    @always_inline
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
            var result = Matmul2d[dtype].forward(A_slice, B_slice)
            C_slice.buffer.fill_equal_shape[overwrite=False](result.buffer)

        return C^


from vectormatrix import VectorMatmulNd
from matrixvector import MatrixVectorMulNd


@fieldwise_init
@register_passable
struct Matmul[dtype: DType](ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True, mode: Int = mm
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        @parameter
        if mode == mm:
            # Step 1: Pure analysis - get the opcode
            var opcode = classify_matmul(A.shape(), B.shape())

            # Step 2: Simple dispatch based on opcode
            if dot == opcode:
                return A.dot[track_grad](B)

            if vm == opcode:
                return VectorMatmulNd[dtype].forward[track_grad](A, B)

            if mv == opcode:
                return MatrixVectorMulNd[dtype].forward[track_grad](A, B)

            if mm == opcode:
                return MatmulNd[dtype].forward[track_grad](A, B)

            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[dtype].scalar(0)

        elif mode == dot:
            return A.dot[track_grad](B)

        elif mode == vm:
            return VectorMatmulNd[dtype].forward[track_grad](A, B)

        elif mode == mv:
            return MatrixVectorMulNd[dtype].forward[track_grad](A, B)
        else:
            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[dtype].scalar(0)

    @always_inline
    @staticmethod
    fn forward(A: Tensor[dtype], B: Gradbox[dtype]) -> Gradbox[dtype]:
        return MatmulNd[dtype].forward(A, B)

    @always_inline
    @staticmethod
    fn forward(A: Gradbox[dtype], B: Tensor[dtype]) -> Gradbox[dtype]:
        return MatmulNd[dtype].forward(A, B)


alias dot = 0  # dot product
alias vm = 1  # vector & tensor matmul
alias mv = 2  # tensor & vector matmul
alias mm = 3  # tensor & tensor matmul
alias invalid = 4  # Invalid case


fn classify_matmul(a: Shape, b: Shape) -> Int:
    var rank_a = a.rank()
    var rank_b = b.rank()

    if rank_a <= 1 and rank_b <= 1:
        return dot
    elif rank_a == 1 and rank_b >= 2:
        return vm
    elif rank_a >= 2 and rank_b == 1:
        return mv
    else:  # rank_a >= 2 and rank_b >= 2
        if a[-1] == b[-2]:
            return mm
        else:
            return invalid


fn main() raises:
    pass
