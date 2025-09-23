from tensors import Tensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
@register_passable
struct VectorMatrixMMBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        # Ancestors (original inputs)
        var ancestor_1 = output.ancestry().get(0)[]  # vector A
        var ancestor_2 = output.ancestry().get(1)[]  # tensor B

        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        # Upstream gradient: batch_shape + [m_out]
        var gradients = output.gradients()[]

        # Tensor B and its shapes
        var tensor_b = ancestor_2.tensor()
        var b_shape = tensor_b.shape
        var batch_shape = b_shape[0:-2]  # may be empty
        var m_out = b_shape[-1]

        # --- Lift gradients to ensure shape batch_shape + [1, m_out]
        var grad_lifted: Tensor[dtype]
        if len(batch_shape) > 0:
            grad_lifted = gradients.reshape(
                batch_shape + [1, m_out], requires_grad=False
            )
        else:
            grad_lifted = gradients.reshape([1, m_out], requires_grad=False)

        # -----------------------
        # Gradient w.r.t. vector A
        # -----------------------
        if ancestor_1.requires_grad():
            var B_t = tensor_b.transpose(
                axes=[-1, -2], requires_grad=False
            )  # batch_shape + [m_out, n]
            var A_batch_grad = grad_lifted.matmul_nd(
                B_t
            )  # batch_shape + [1, n]
            var dA = Tensor[dtype].sum_over_broadcasted_axes(
                A_batch_grad, ancestor_1.shape()
            )
            outgoing_grads.append((ancestor_1, dA, AddTensor))

        # -----------------------
        # Gradient w.r.t. tensor B
        # -----------------------
        if ancestor_2.requires_grad():
            # Prepare A as [1, n] and expand to batch_shape + [1, n] using reshape + expand
            var A_tensor = ancestor_1.tensor()  # shape [n]
            var A_lifted = A_tensor.reshape(
                [1, A_tensor.shape[0]], requires_grad=False
            )

            var A_expanded_shape = batch_shape + [1, A_tensor.shape[0]]
            var A_padded = A_lifted.reshape(
                Shape(1) * len(batch_shape) + [1, A_tensor.shape[0]],
                requires_grad=False,
            )
            A_expanded = A_padded.expand(A_expanded_shape, requires_grad=False)

            # Compute per-batch (n,1) @ (1, m_out)
            var A_expanded_T = A_expanded.transpose(
                axes=[-1, -2], requires_grad=False
            ).contiguous()  # batch_shape + [n,1]

            var B_batch_grad = A_expanded_T.matmul_nd(
                grad_lifted
            )  # batch_shape + [n, m_out]

            # Reduce broadcasted axes to match original B shape
            var dB = Tensor[dtype].sum_over_broadcasted_axes(
                B_batch_grad, tensor_b.shape
            )
            outgoing_grads.append((ancestor_2, dB, AddTensor))

        return outgoing_grads


fn main():
    pass
