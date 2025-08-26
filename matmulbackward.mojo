from tensors import Tensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
@register_passable
struct MatmulBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        dA, dB = Self.matmul_backward[dtype](
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
                    dA.value(),
                    AddTensor,
                )
            )
        if dB:
            outgoing_grads.append(
                (
                    ancestor_2,
                    dB.value(),
                    AddTensor,
                )
            )

        return outgoing_grads

    @staticmethod
    fn matmul_backward[
        dtype: DType
    ](
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
                dA = Optional(
                    gradients.matmul(B.transpose(requires_grad=False))
                )
            if B.requires_grad:
                dB = Optional(
                    A.transpose(requires_grad=False).matmul(gradients)
                )

        elif trans_a and not trans_b:
            # Forward: C = A^T @ B
            if A.requires_grad:
                dA = Optional(
                    B.matmul(gradients.transpose(requires_grad=False))
                )
            if B.requires_grad:
                dB = Optional(A.matmul(gradients))

        elif not trans_a and trans_b:
            # Forward: C = A @ B^T
            if A.requires_grad:
                dA = Optional(gradients.matmul(B))
            if B.requires_grad:
                dB = Optional(
                    gradients.transpose(requires_grad=False).matmul(A)
                )

        else:
            # trans_a and trans_b
            # Forward: C = A^T @ B^T
            if A.requires_grad:
                dA = Optional(
                    B.transpose(requires_grad=False).matmul(
                        gradients.transpose(requires_grad=False)
                    )
                )
            if B.requires_grad:
                dB = Optional(
                    gradients.transpose(requires_grad=False).matmul(
                        A.transpose(requires_grad=False)
                    )
                )

        return dA, dB


@fieldwise_init
@register_passable
struct BatchedMatmulBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]
        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        dA, dB = Self.batched_matmul_backward[dtype](
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
                    dA.value(),
                    AddTensor,
                )
            )
        if dB:
            outgoing_grads.append(
                (
                    ancestor_2,
                    dB.value(),
                    AddTensor,
                )
            )

        return outgoing_grads

    @staticmethod
    fn batched_matmul_backward[
        dtype: DType
    ](
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
                A_batch_grad = gradients.matmul_nd(
                    B.transpose(axes=[-1, -2], requires_grad=False)
                )
                dA = Optional(
                    Self.sum_over_broadcasted_axes[dtype](A_batch_grad, A.shape)
                )
            if B.requires_grad:
                B_batch_grad = A.transpose(
                    axes=[-1, -2], requires_grad=False
                ).matmul_nd(gradients)
                dB = Optional(
                    Self.sum_over_broadcasted_axes[dtype](B_batch_grad, B.shape)
                )

        elif trans_a and not trans_b:
            # Forward: C = A^T @ B
            if A.requires_grad:
                axes = gradients.shape.intlist()
                axes.swap(-2, -1)
                A_batch_grad = B.matmul_nd(
                    gradients.transpose(axes=axes, requires_grad=False)
                )
                dA = Optional(
                    Self.sum_over_broadcasted_axes[dtype](A_batch_grad, A.shape)
                )

            if B.requires_grad:
                B_batch_grad = A.matmul_nd(gradients)
                dB = Optional(
                    Self.sum_over_broadcasted_axes[dtype](B_batch_grad, B.shape)
                )

        elif not trans_a and trans_b:
            # Forward: C = A @ B^T
            if A.requires_grad:
                dA = Optional(gradients.matmul(B))
            if B.requires_grad:
                dB = Optional(
                    gradients.transpose(requires_grad=False).matmul(A)
                )

        else:
            # trans_a and trans_b
            # Forward: C = A^T @ B^T
            if A.requires_grad:
                dA = Optional(
                    B.transpose(requires_grad=False).matmul(
                        gradients.transpose(requires_grad=False)
                    )
                )
            if B.requires_grad:
                dB = Optional(
                    gradients.transpose(requires_grad=False).matmul(
                        A.transpose(requires_grad=False)
                    )
                )

        return dA, dB

    @staticmethod
    fn sum_over_broadcasted_axes[
        dtype: DType
    ](batch_grad: Tensor[dtype], recipient_shape: Shape) -> Tensor[dtype]:
        """Sum over dimensions that were broadcasted in the forward pass."""
        result = batch_grad
        current_shape = batch_grad.shape

        # Sum over extra leading dimensions
        while len(current_shape) > len(recipient_shape):
            result = result.sum(axes=[0], keepdims=False)
            current_shape = result.shape

        # Sum over mismatched dimensions
        for i in range(len(recipient_shape)):
            if current_shape[i] != recipient_shape[i] and current_shape[i] > 1:
                result = result.sum(axes=[i], keepdims=True)
                current_shape = result.shape
        return result


fn main():
    pass
