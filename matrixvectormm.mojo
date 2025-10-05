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
        var gradients = output.grad()
        var tensor_a = ancestor_1.tensor()
        var tensor_b = ancestor_2.tensor()
        var a_shape = tensor_a.shape

        # rows (n) and cols (m) of A: A is (..., n, m)
        var n = a_shape[-2]
        var m = a_shape[-1]
        # var batch_shape = a_shape[0:-2] if len(a_shape) > 2 else Shape(True)
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
            grad_lifted = gradients.reshape[track_grad=False](
                batch_shape + [n, 1], requires_grad=False
            )

        # other common case: gradients = batch_shape + [n,1]
        elif gradients.rank() == len(batch_shape) + 2:
            if gradients.shape[-2] == n and gradients.shape[-1] == 1:
                grad_lifted = gradients.reshape[track_grad=False](
                    batch_shape + [n, 1], requires_grad=False
                )
            elif gradients.shape[-1] == n and gradients.shape[-2] == 1:
                # rare orientation (1,n) — transpose to get (n,1)
                grad_lifted = gradients.transpose[track_grad=False](
                    axes=[-2, -1], requires_grad=False
                ).reshape[track_grad=False](
                    batch_shape + [n, 1], requires_grad=False
                )
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
            grad_lifted = gradients.reshape[track_grad=False](
                [n, 1], requires_grad=False
            )
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
                var b_row = tensor_b.reshape[track_grad=False](
                    [1, m], requires_grad=False
                )
                if len(batch_shape) > 0:
                    var b_padded = b_row.reshape[track_grad=False](
                        [1] * len(batch_shape) + [1, m], requires_grad=False
                    )
                    B_lifted = b_padded.expand[track_grad=False](
                        batch_shape + [1, m], requires_grad=False
                    )
                else:
                    B_lifted = b_row
            else:
                # B already has batch dims -> ensure shape batch_shape + [1,m]
                B_lifted = tensor_b.reshape[track_grad=False](
                    batch_shape + [1, m], requires_grad=False
                )

            var dA = grad_lifted.matmul_nd[track_grad=False](
                B_lifted
            )  # -> batch_shape + [n, m]
            dA = Tensor[dtype].sum_over_broadcasted_axes(dA, tensor_a.shape)
            outgoing_grads.append((ancestor_1, dA, AddTensor))

        # -----------------------
        # Gradient w.r.t. B: dB = A^T @ grad_lifted  -> produces batch_shape + [m,1] -> squeeze to [m]
        # -----------------------
        if ancestor_2.requires_grad():
            # ensure A has shape batch_shape + [n, m]
            var A_expanded = tensor_a
            if len(batch_shape) > 0:
                A_expanded = A_expanded.reshape[track_grad=False](
                    batch_shape + [n, m], requires_grad=False
                )

            var A_t = A_expanded.transpose[track_grad=False](
                axes=[-1, -2], requires_grad=False
            ).contiguous[
                track_grad=False
            ]()  # batch_shape + [m, n]
            var dB_full = A_t.matmul_nd[track_grad=False](
                grad_lifted
            )  # -> batch_shape + [m, 1]

            # reshape to batch_shape + [m]
            var dB: Tensor[dtype]
            if len(batch_shape) > 0:
                dB = dB_full.reshape[track_grad=False](
                    batch_shape + [m], requires_grad=False
                )
            else:
                dB = dB_full.reshape[track_grad=False]([m], requires_grad=False)

            dB = Tensor[dtype].sum_over_broadcasted_axes(dB, tensor_b.shape)
            outgoing_grads.append((ancestor_2, dB, AddTensor))

        return outgoing_grads


@fieldwise_init
@register_passable
struct MatrixVectorMM[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut A: Tensor[dtype], B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        # --------------------------
        # Shapes
        # --------------------------
        # A: batch_shape + [n, m]
        # B: shape [m] or batch_shape + [m]
        # result: batch_shape + [n]
        var a_shape = A.shape
        var b_shape = B.shape

        var batch_shape = a_shape[0:-2]  # may be empty
        var n = a_shape[-2]
        var m = a_shape[-1]

        # --------------------------
        # Lift B to batch shape if needed
        # --------------------------
        var B_lifted: Tensor[dtype]
        if len(b_shape) == 1:
            # B is 1D vector -> reshape to [1, m] then expand to batch_shape + [1, m]
            var B_reshaped = B.reshape[track_grad=False](
                [1, m], requires_grad=False
            )
            if len(batch_shape) > 0:
                B_lifted = B_reshaped.expand[track_grad=False](
                    batch_shape + [1, m], requires_grad=False
                )
            else:
                B_lifted = B_reshaped
        else:
            # B already has batch dimensions
            B_lifted = B

        # --------------------------
        # Compute result: batch matrix-vector multiplication
        # result = A @ B_lifted^T ? Actually, B_lifted is (batch_shape + [1, m]),
        # A: (batch_shape + [n, m]), so matmul_nd(A, B_lifted.T)
        # --------------------------
        B_lifted_contiguous = B_lifted.contiguous[track_grad=False]()

        B_T = B_lifted_contiguous.transpose[track_grad=False](
            axes=[-1, -2], requires_grad=False
        ).contiguous[
            track_grad=False
        ]()  # shape: batch_shape + [m,1]

        out = A.matmul_nd[track_grad=False](B_T)  # shape: batch_shape + [n,1]
        out = out.reshape[track_grad=False](
            batch_shape + [n], requires_grad=False
        )

        @parameter
        if track_grad:
            grad_required = (
                A.requires_grad or B.requires_grad
            ) and requires_grad
            if grad_required:
                out.requires_grad_()

                backward_fn = MatrixVectorMMBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(
                    TensorLite[dtype].of(A), TensorLite[dtype].of(B)
                )

        return out


fn main() raises:
    pass
