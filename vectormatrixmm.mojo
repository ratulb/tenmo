from tensors import Tensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from common_utils import panic


@fieldwise_init
@register_passable
struct VectorMatrixMMBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        # Ancestors (original inputs)
        var ancestor_1 = output.ancestry().get(0)  # vector A
        var ancestor_2 = output.ancestry().get(1)  # tensor B

        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        # Upstream gradient: batch_shape + [m_out]
        var gradients = output.grad()

        # Tensor B and its shapes
        var tensor_b = ancestor_2.tensor()
        var b_shape = tensor_b.shape
        var batch_shape = b_shape[0:-2]  # may be empty
        var m_out = b_shape[-1]

        # --- Lift gradients to ensure shape batch_shape + [1, m_out]
        var grad_lifted: Tensor[dtype]
        if len(batch_shape) > 0:
            grad_lifted = gradients.reshape[track_grad=False](
                batch_shape + [1, m_out], requires_grad=False
            )
        else:
            grad_lifted = gradients.reshape[track_grad=False](
                [1, m_out], requires_grad=False
            )

        # -----------------------
        # Gradient w.r.t. vector A
        # -----------------------
        if ancestor_1.requires_grad():
            var B_t = tensor_b.transpose[track_grad=False](
                axes=[-1, -2], requires_grad=False
            )  # batch_shape + [m_out, n]
            var A_batch_grad = grad_lifted.matmul_nd[track_grad=False](
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
            var A_lifted = A_tensor.reshape[track_grad=False](
                [1, A_tensor.shape[0]], requires_grad=False
            )

            var A_expanded_shape = batch_shape + [1, A_tensor.shape[0]]
            var A_padded = A_lifted.reshape[track_grad=False](
                Shape(1) * len(batch_shape) + [1, A_tensor.shape[0]],
                requires_grad=False,
            )
            A_expanded = A_padded.expand[track_grad=False](
                A_expanded_shape, requires_grad=False
            )

            # Compute per-batch (n,1) @ (1, m_out)
            var A_expanded_T = A_expanded.transpose[track_grad=False](
                axes=[-1, -2], requires_grad=False
            ).contiguous[
                track_grad=False
            ]()  # batch_shape + [n,1]

            var B_batch_grad = A_expanded_T.matmul_nd[track_grad=False](
                grad_lifted
            )  # batch_shape + [n, m_out]

            # Reduce broadcasted axes to match original B shape
            var dB = Tensor[dtype].sum_over_broadcasted_axes(
                B_batch_grad, tensor_b.shape
            )
            outgoing_grads.append((ancestor_2, dB, AddTensor))

        return outgoing_grads


@fieldwise_init
@register_passable
struct VectorMatrixMM[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        A: Tensor[dtype], mut B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        # A: (n,)(or batched: batch_A..., n)
        # B: (..., n, m)  (rank >= 2)
        if A.rank() != 1:
            panic("vector_matrix_mm: A must be rank-1 (vector)")
        if B.rank() < 2:
            panic("vector_matrix_mm: B must be rank>=2 (matrix or higher)")

        # n = contraction dim
        n = A.shape[0]
        if B.shape[-2] != n:
            panic(
                "vector_matrix_mm: incompatible shapes (A.shape[0] !="
                " B.shape[-2])"
            )

        # --- Lift A to (..., 1, n) so it matches matmul_nd's A shape of (..., m, k)
        # Start as (1, n)
        A_lifted = A.reshape[track_grad=False](
            1, -1, requires_grad=False
        )  # shape (1, n)

        # Determine target batch_shape from B (all dims except last two)
        batch_shape = B.shape[0:-2]  # can be empty
        var A_expanded: Tensor[dtype]
        # Expand A_lifted to broadcast over B's batch dims:
        # - If batch_shape is empty, A_expanded stays (1, n)
        # - Otherwise we want shape batch_shape + [1, n]
        if len(batch_shape) > 0:
            # prepend required number of leading dims = len(batch_shape)
            A_padded = A_lifted
            intermediates = [A_padded]
            for _ in range(len(batch_shape)):
                current = intermediates[-1]
                unsqueezed = current.unsqueeze[track_grad=False](
                    0, requires_grad=False
                )
                intermediates.append(unsqueezed)
                # A_expanded = A_expanded.unsqueeze(
                #   0, requires_grad = False
                # )  # add leading dims to the front
            # Now A_expanded.shape == (1,...,1, n) ; expand to batch_shape + [1, n]
            A_last_padded = intermediates[-1].contiguous[track_grad=False]()
            A_expanded = A_last_padded.expand[track_grad=False](
                batch_shape + [1, n], requires_grad=False
            )
        else:
            A_expanded = A_lifted  # shape (1,n)

        # --- Call matmul_nd (handles batching/broadcasting across batch_shape)
        # Note: matmul_nd expects A.shape = batch + [m, k], B.shape = batch + [k, n_out]
        # For us m == 1 and k == n
        C = Tensor[dtype].matmul_nd[track_grad=False](
            A_expanded, B, requires_grad=False
        )  # shape: batch_shape + [1, m_out]

        # --- Squeeze out the intermediary m==1 dimension to match PyTorch-style (batch, m_out) -> if no batch, just (m_out,)
        # m_out == B.shape[-1]
        if len(batch_shape) == 0:
            out = C.reshape[track_grad=False](
                [C.shape[1]], requires_grad=False
            )  # (m_out,)
        else:
            # remove the singular second-last dim (axis = -2)
            # we can reshape: batch_shape + [B.shape[-1]]
            out = C.reshape[track_grad=False](
                batch_shape + [B.shape[-1]], requires_grad=False
            )

        @parameter
        if track_grad:
            # --- Attach autograd wrapper that routes backward to VectorMatrixMMBackward
            grad_required = (
                A.requires_grad or B.requires_grad
            ) and requires_grad
            if grad_required:
                out.requires_grad_(True)
                out.backwardFn = Optional(
                    VectorMatrixMMBackward[dtype]().into_backward_fn()
                )
            out.add_ancestry(A, B)
        return out


fn main():
    pass
