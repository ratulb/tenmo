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
        gradients = output.gradients()[]
        var gradientsputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]
        if ancestor_1.requires_grad():
            ancestor_2_tensor = ancestor_2.tensor()
            ancestor_2_transposed = ancestor_2_tensor.transpose(
                requires_grad=False
            ).contiguous()
            ancestor_1_grad_share = gradients.matmul(ancestor_2_transposed)
            gradientsputs.append(
                (
                    ancestor_1,
                    ancestor_1_grad_share,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            ancestor_1_tensor = ancestor_1.tensor()
            ancestor_1_transposed = ancestor_1_tensor.transpose(
                requires_grad=False
            ).contiguous()
            ancestor_2_grad_share = ancestor_1_transposed.matmul(gradients)
            gradientsputs.append(
                (
                    ancestor_2,
                    ancestor_2_grad_share,
                    AddTensor,
                )
            )

        return gradientsputs

    fn __str__(self) -> String:
        return "MatmulBackward"

    @staticmethod
    fn matmul_backward(
        gradients_ptr: UnsafePointer[Tensor[dtype]],
        A_ptr: UnsafePointer[Tensor[dtype]],
        B_ptr: UnsafePointer[Tensor[dtype]],
        trans_a: Bool = False,
        trans_b: Bool = False,
    ) -> (Optional[Tensor[dtype]], Optional[Tensor[dtype]]):
        """
        Computes gradients for A and B in the matrix multiplication C = f(A, B).

        Args:
            gradients_ptr: Gradient pointer of the loss with respect to the output C.
            A_ptr: First input tensor pointer from the forward pass.
            B_ptr: Second input tensor pointer from the forward pass.
            trans_a: Whether A was transposed in the forward pass.
            trans_b: Whether B was transposed in the forward pass.

        Returns:
            A tuple (dA, dB) representing gradients with respect to A and B.
            If a tensor requires no gradient, None is returned.
        """
        var dA: Optional[Tensor[dtype]] = None
        var dB: Optional[Tensor[dtype]] = None
        gradients = gradients_ptr[]
        A = A_ptr[]
        B = B_ptr[]

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


fn main():
    pass
