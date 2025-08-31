from tensors import Tensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from common_utils import panic


@fieldwise_init
@register_passable
struct MatrixVectorMMBackward[dtype: DType](Copyable):
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

        # Upstream gradient (may be rank = batch_shape + [n] or batch_shape + [n,1])
        var gradients = output.gradients()[]
        var tensor_a = ancestor_1.tensor()
        var tensor_b = ancestor_2.tensor()
        var a_shape = tensor_a.shape

        # rows (n) and cols (m) of A: A is (..., n, m)
        var n = a_shape[-2]
        var m = a_shape[-1]
        var batch_shape = a_shape[0:-2] if len(a_shape) > 2 else Shape.Void

        # -----------------------
        # Normalize upstream gradient to shape: batch_shape + [n, 1]
        # -----------------------
        # var grad_lifted: Tensor[dtype]
        var grad_lifted = gradients
        # common case: gradients shape == batch_shape + [n]
        if gradients.rank() == len(batch_shape) + 1:
            # gradients = batch_shape + [n]  -> lift to batch_shape + [n,1]
            if gradients.shape[-1] != n:
                panic(
                    "MatrixVectorMMBackward: unexpected last dim for gradients"
                )
            grad_lifted = gradients.reshape(
                batch_shape + [n, 1], requires_grad=False
            )

        # other common case: gradients = batch_shape + [n,1]
        elif gradients.rank() == len(batch_shape) + 2:
            if gradients.shape[-2] == n and gradients.shape[-1] == 1:
                grad_lifted = gradients.reshape(
                    batch_shape + [n, 1], requires_grad=False
                )
            elif gradients.shape[-1] == n and gradients.shape[-2] == 1:
                # rare orientation (1,n) — transpose to get (n,1)
                grad_lifted = gradients.transpose(
                    axes=[-2, -1], requires_grad=False
                ).reshape(batch_shape + [n, 1], requires_grad=False)
            else:
                panic(
                    "MatrixVectorMMBackward: unsupported gradient shape"
                    " (len=batch+2)"
                )
        # unbatched simple case: gradients is [n]
        elif (
            gradients.rank() == 1
            and len(batch_shape) == 0
            and gradients.shape[0] == n
        ):
            grad_lifted = gradients.reshape([n, 1], requires_grad=False)
        else:
            panic(
                "MatrixVectorMMBackward: unsupported upstream gradient"
                " rank/shape"
            )

        # -----------------------
        # Gradient w.r.t. A: dA = grad_lifted @ B_row  -> shape batch_shape + [n, m]
        # -----------------------
        if ancestor_1.requires_grad():
            var B_lifted: Tensor[dtype]
            if tensor_b.rank() == 1:
                # B is (m,) -> row (1,m) -> pad & expand to batch_shape + [1,m]
                var b_row = tensor_b.reshape([1, m], requires_grad=False)
                if len(batch_shape) > 0:
                    var b_padded = b_row.reshape(
                        [1] * len(batch_shape) + [1, m], requires_grad=False
                    )
                    B_lifted = b_padded.expand(
                        batch_shape + [1, m], requires_grad=False
                    )
                else:
                    B_lifted = b_row
            else:
                # B already has batch dims -> ensure shape batch_shape + [1,m]
                B_lifted = tensor_b.reshape(
                    batch_shape + [1, m], requires_grad=False
                )

            var dA = grad_lifted.matmul_nd(B_lifted)  # -> batch_shape + [n, m]
            dA = Tensor[dtype].sum_over_broadcasted_axes(dA, tensor_a.shape)
            outgoing_grads.append((ancestor_1, dA, AddTensor))

        # -----------------------
        # Gradient w.r.t. B: dB = A^T @ grad_lifted  -> produces batch_shape + [m,1] -> squeeze to [m]
        # -----------------------
        if ancestor_2.requires_grad():
            # ensure A has shape batch_shape + [n, m]
            var A_expanded = tensor_a
            if len(batch_shape) > 0:
                A_expanded = A_expanded.reshape(
                    batch_shape + [n, m], requires_grad=False
                )

            var A_t = A_expanded.transpose(
                axes=[-1, -2], requires_grad=False
            ).contiguous()  # batch_shape + [m, n]
            var dB_full = A_t.matmul_nd(grad_lifted)  # -> batch_shape + [m, 1]

            # reshape to batch_shape + [m]
            var dB: Tensor[dtype]
            if len(batch_shape) > 0:
                dB = dB_full.reshape(batch_shape + [m], requires_grad=False)
            else:
                dB = dB_full.reshape([m], requires_grad=False)

            dB = Tensor[dtype].sum_over_broadcasted_axes(dB, tensor_b.shape)
            outgoing_grads.append((ancestor_2, dB, AddTensor))

        return outgoing_grads


fn main() raises:
    pass


