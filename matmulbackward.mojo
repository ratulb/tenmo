from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
struct MatmulBackward[dtype: DType](Copyable & Movable & Stringable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        var gradientsputs: List[
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
            gradientsputs.append(
                (
                    ancestor_1,
                    dA.value(),
                    AddTensor,
                )
            )
        if dB:
            gradientsputs.append(
                (
                    ancestor_2,
                    dB.value(),
                    AddTensor,
                )
            )

        return gradientsputs

    fn __str__(self) -> String:
        return "MatmulBackward"

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

    @staticmethod
    fn sum_over_broadcasted_axes(
        tensor: Tensor[dtype], target_shape: List[Int]
    ) -> Tensor[dtype]:
        """Sum over dimensions that were broadcasted in the forward pass."""
        result = tensor
        current_shape = tensor.shape

        # Sum over extra leading dimensions
        while len(current_shape) > len(target_shape):
            result = result.sum(axes=[0], keepdims=False)
            current_shape = result.shape

        # Sum over mismatched dimensions
        for i in range(len(target_shape)):
            if current_shape[i] != target_shape[i] and current_shape[i] > 1:
                result = result.sum(axes=[i], keepdims=True)
                current_shape = result.shape
        return result


fn main():
    pass
