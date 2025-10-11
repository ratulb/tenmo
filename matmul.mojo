from tensors import Tensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from sys import simdwidthof
from common_utils import il, s
from dotproduct import DotBackward
from vectormatrixmm import VectorMatrixMMBackward
from matrixvectormm import MatrixVectorMMBackward


@fieldwise_init
@register_passable
struct MatmulBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        ancestor_1 = output.ancestry().get(0)
        ancestor_2 = output.ancestry().get(1)

        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        dA, dB = Self.matmul_backward(
            output.gradients(),
            ancestor_1.inner_address(),
            ancestor_2.inner_address(),
            False,
            False,
        )

        if dA:
            outgoing_grads.append(
                (
                    ancestor_1,
                    dA.take(),
                    AddTensor,
                )
            )
        if dB:
            outgoing_grads.append(
                (
                    ancestor_2,
                    dB.take(),
                    AddTensor,
                )
            )

        return outgoing_grads

    @staticmethod
    fn matmul_backward(
        gradients_ptr: UnsafePointer[Tensor[dtype]],
        A_ptr: UnsafePointer[Tensor[dtype]],
        B_ptr: UnsafePointer[Tensor[dtype]],
        trans_a: Bool = False,
        trans_b: Bool = False,
    ) -> (Optional[Tensor[dtype]], Optional[Tensor[dtype]]):
        var dA: Optional[Tensor[dtype]] = None
        var dB: Optional[Tensor[dtype]] = None

        A = A_ptr[]
        B = B_ptr[]
        gradients = gradients_ptr[]

        if not trans_a and not trans_b:
            # Forward: C = A @ B
            if A.requires_grad:
                B_transposed = B.transpose[track_grad=False](
                    requires_grad=False
                )

                dA = Optional(gradients.matmul(B_transposed))
                B_transposed.free()
            if B.requires_grad:
                A_transposed = A.transpose[track_grad=False](
                    requires_grad=False
                )
                dB = Optional(A_transposed.matmul(gradients))
                A_transposed.free()

        elif trans_a and not trans_b:
            # Forward: C = A^T @ B
            if A.requires_grad:
                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )

                dA = Optional(B.matmul(grad_transposed))
                grad_transposed.free()

            if B.requires_grad:
                dB = Optional(A.matmul(gradients))

        elif not trans_a and trans_b:
            # Forward: C = A @ B^T
            if A.requires_grad:
                dA = Optional(gradients.matmul(B))

            if B.requires_grad:
                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )

                dB = Optional(grad_transposed.matmul(A))
                grad_transposed.free()

        else:
            # trans_a and trans_b
            # Forward: C = A^T @ B^T
            if A.requires_grad:
                B_transposed = B.transpose[track_grad=False](
                    requires_grad=False
                )

                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )

                dA = Optional(B_transposed.matmul(grad_transposed))
                grad_transposed.free()
                B_transposed.free()
            if B.requires_grad:
                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )

                A_transposed = A.transpose[track_grad=False](
                    requires_grad=False
                )

                dB = Optional(grad_transposed.matmul(A_transposed))
                A_transposed.free()
                grad_transposed.free()
            gradients.free()
        return dA, dB


@fieldwise_init
@register_passable
struct BatchedMatmulBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        ancestor_1 = output.ancestry().get(0)
        ancestor_2 = output.ancestry().get(1)
        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        dA, dB = Self.batched_matmul_backward(
            output.gradients(),
            ancestor_1.inner_address(),
            ancestor_2.inner_address(),
            False,
            False,
        )

        if dA:
            outgoing_grads.append(
                (
                    ancestor_1,
                    dA.take(),
                    AddTensor,
                )
            )
        if dB:
            outgoing_grads.append(
                (
                    ancestor_2,
                    dB.take(),
                    AddTensor,
                )
            )

        return outgoing_grads

    @staticmethod
    fn batched_matmul_backward(
        gradients_ptr: UnsafePointer[Tensor[dtype]],
        A_ptr: UnsafePointer[Tensor[dtype]],
        B_ptr: UnsafePointer[Tensor[dtype]],
        trans_a: Bool = False,
        trans_b: Bool = False,
    ) -> (Optional[Tensor[dtype]], Optional[Tensor[dtype]]):
        var dA: Optional[Tensor[dtype]] = None
        var dB: Optional[Tensor[dtype]] = None

        A = A_ptr[]
        B = B_ptr[]
        gradients = gradients_ptr[]
        if not trans_a and not trans_b:
            # Forward: C = A @ B
            if A.requires_grad:
                B_transposed = B.transpose[track_grad=False](
                    axes=[-1, -2], requires_grad=False
                )
                A_batch_grad = gradients.matmul_nd(B_transposed)
                dA = Optional(
                    Tensor[dtype].sum_over_broadcasted_axes(
                        A_batch_grad, A.shape
                    )
                )
                B_transposed.free()
                A_batch_grad.free()
            if B.requires_grad:
                A_transposed = A.transpose[track_grad=False](
                    axes=[-1, -2], requires_grad=False
                )
                B_batch_grad = A_transposed.matmul_nd(gradients)
                dB = Optional(
                    Tensor[dtype].sum_over_broadcasted_axes(
                        B_batch_grad, B.shape
                    )
                )

                A_transposed.free()
                B_batch_grad.free()

        elif trans_a and not trans_b:
            # Forward: C = A^T @ B
            if A.requires_grad:
                axes = gradients.shape.intlist()
                axes.swap(-2, -1)
                grad_transposed = gradients.transpose[track_grad=False](
                    axes=axes, requires_grad=False
                )
                A_batch_grad = B.matmul_nd(grad_transposed)
                dA = Optional(
                    Tensor[dtype].sum_over_broadcasted_axes(
                        A_batch_grad, A.shape
                    )
                )

                grad_transposed.free()
                A_batch_grad.free()

            if B.requires_grad:
                B_batch_grad = A.matmul_nd(gradients)
                dB = Optional(
                    Tensor[dtype].sum_over_broadcasted_axes(
                        B_batch_grad, B.shape
                    )
                )
                B_batch_grad.free()

        elif not trans_a and trans_b:
            # Forward: C = A @ B^T
            if A.requires_grad:
                dA = Optional(gradients.matmul(B))
            if B.requires_grad:
                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )
                dB = Optional(grad_transposed.matmul(A))
                grad_transposed.free()

        else:
            # trans_a and trans_b
            # Forward: C = A^T @ B^T
            if A.requires_grad:
                B_transposed = B.transpose[track_grad=False](
                    requires_grad=False
                )
                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )

                dA = Optional(B_transposed.matmul(grad_transposed))
                B_transposed.free()
                grad_transposed.free()

            if B.requires_grad:
                grad_transposed = gradients.transpose[track_grad=False](
                    requires_grad=False
                )
                A_transposed = A.transpose[track_grad=False](
                    requires_grad=False
                )

                dB = Optional(grad_transposed.matmul(A_transposed))
                grad_transposed.free()
                A_transposed.free()
            gradients.free()

        return dA, dB


@fieldwise_init
@register_passable
struct Matmul[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True, simd_width: Int = simdwidthof[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        rank_a = A.rank()
        rank_b = B.rank()
        requires_grad = A.requires_grad or B.requires_grad

        if rank_a <= 1 and rank_b <= 1:
            C = A.dot[track_grad=False](B, requires_grad=False)

            @parameter
            if track_grad:
                if requires_grad:
                    C.requires_grad_()
                    backward_fn = DotBackward[dtype]().into_backward_fn()
                    C.backwardFn = Optional(backward_fn)
                    C.add_ancestry(A, B)
            return C

        elif rank_a == 1 and rank_b >= 2:
            C = A.vector_matrix_mm[track_grad=False](B, requires_grad=False)

            @parameter
            if track_grad:
                if requires_grad:
                    C.requires_grad_()
                    backward_fn = VectorMatrixMMBackward[
                        dtype
                    ]().into_backward_fn()
                    C.backwardFn = Optional(backward_fn)
                    C.add_ancestry(A, B)
            return C

        elif rank_a >= 2 and rank_b == 1:
            C = A.matrix_vector_mm[track_grad=False](B, requires_grad=False)

            @parameter
            if track_grad:
                if requires_grad:
                    C.requires_grad_()
                    backward_fn = MatrixVectorMMBackward[
                        dtype
                    ]().into_backward_fn()
                    C.backwardFn = Optional(backward_fn)
                    C.add_ancestry(A, B)
            return C

        else:
            C = A.matmul_nd[track_grad=False](B, requires_grad=False)

            @parameter
            if track_grad:
                if requires_grad:
                    C.requires_grad_()
                    if rank_a == 2 and rank_b == 2:
                        mbfn = MatmulBackward[dtype]().into_backward_fn()
                        C.backwardFn = Optional(mbfn)
                    else:
                        bmbfn = BatchedMatmulBackward[
                            dtype
                        ]().into_backward_fn()
                        C.backwardFn = Optional(bmbfn)

                    C.add_ancestry(A, B)

            return C


@fieldwise_init
@register_passable
struct Matmul_2d[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True, simd_width: Int = simdwidthof[dtype]()
    ](
        A_ptr: UnsafePointer[Tensor[dtype]],
        B_ptr: UnsafePointer[Tensor[dtype]],
        C_ptr: UnsafePointer[Tensor[dtype]] = UnsafePointer[Tensor[dtype]](),
        requires_grad: Bool = True,
    ) -> Tensor[dtype]:
        A = A_ptr[]
        B = B_ptr[]

        Shape.validate_matrix_shapes_2d(A.shape, B.shape)

        rows_a = A.shape[0]
        cols_a = A.shape[1]
        cols_b = B.shape[1]
        packed = B.is_contiguous()

        C = C_ptr[] if C_ptr.__as_bool__() else Tensor[dtype].zeros(
            rows_a, cols_b
        )
        for i in range(0, rows_a):
            for j in range(0, cols_b, simd_width):
                mbatch = min(simd_width, cols_b - j)
                var accum = SIMD[dtype, simd_width](0)

                for k in range(0, cols_a):
                    scalar_a = A.load(i, k)
                    if packed and mbatch == simd_width:
                        simd_vector = B.load[simd_width](k, j)
                        accum += simd_vector * scalar_a
                    else:
                        # mbatch < simd_width or scattered B cols
                        for step in range(0, mbatch):
                            scalar_b = B.load(k, j + step)
                            accum[step] += scalar_a * scalar_b

                if mbatch == simd_width:
                    C.store[simd_width](i, j, accum)
                else:
                    for step in range(0, mbatch):
                        C.store(i, j + step, accum[step])

        @parameter
        if track_grad:
            grad_required = (
                A.requires_grad or B.requires_grad
            ) and requires_grad

            if grad_required:
                C.requires_grad_(True)
                backward_fn = MatmulBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(backward_fn)
                C.add_ancestry(A, B)

        return C


@fieldwise_init
@register_passable
struct Matmul_nd[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        A: Tensor[dtype], B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        Shape.validate_matrix_shapes_nd(A.shape, B.shape)
        # shapes: batch + [m, k], batch + [k, n]
        batch_shape = Shape.broadcast_shape(
            A.shape[0:-2], B.shape[0:-2]
        )  # all dims except last 2

        m = A.shape[-2]
        n = B.shape[-1]

        batch_dims_a = A.shape[:-2]
        batch_dims_b = B.shape[:-2]

        out_shape = batch_shape + [m, n]
        C = Tensor[dtype].zeros(out_shape)

        for indices in batch_shape:
            # select batch slices
            A_indices = Tensor[dtype].broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            B_indices = Tensor[dtype].broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )
            A_slice = A[il(A_indices), s(), s()]
            B_slice = B[il(B_indices), s(), s()]
            C_slice = C[il(indices), s(), s()]

            _result = Matmul_2d.forward[track_grad=False](
                UnsafePointer(to=A_slice),
                UnsafePointer(to=B_slice),
                UnsafePointer(to=C_slice),
                requires_grad=False,
            )
            A_slice.free()
            B_slice.free()
            C_slice.free()
            _result.free()

        @parameter
        if track_grad:
            grad_required = (
                A.requires_grad or B.requires_grad
            ) and requires_grad

            if grad_required:
                C.requires_grad_()

                two_dim = A.rank() == 2 and B.rank() == 2

                if two_dim:
                    mbfn = MatmulBackward[dtype]().into_backward_fn()
                    C.backwardFn = Optional(mbfn)
                else:
                    bmbfn = BatchedMatmulBackward[dtype]().into_backward_fn()
                    C.backwardFn = Optional(bmbfn)

                C.add_ancestry(A, B)

        return C


fn main():
    pass
