from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from gradbox import Gradbox
from ancestry import Ancestor
from shapes import Shape
from broadcasthelper import ShapeBroadcaster
from common_utils import il, s, panic
from matmul import Matmul2d


@fieldwise_init
@register_passable
struct VectorMatmulNd[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](v: Tensor[dtype], M: Tensor[dtype]) -> Tensor[dtype]:
        var v_shape = v.shape()
        var M_shape = M.shape()

        # Validate: v[..., k] × M[..., k, n] → out[..., n]
        if v_shape.rank() < 1:
            panic("VectorMatmulNd: vector must have at least 1 dimension")
        if M_shape.rank() < 2:
            panic("VectorMatmulNd: matrix must have at least 2 dimensions")

        var k_v = v_shape[-1]
        var k_M = M_shape[-2]
        var n = M_shape[-1]

        if k_v != k_M:
            panic("VectorMatmulNd: inner dimensions must match")

        # Broadcast batch dimensions
        var v_batch_dims = v_shape[:-1]
        var M_batch_dims = M_shape[:-2]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_dims, M_batch_dims
        )

        # Output shape: batch + [n]
        var out_shape = batch_shape + [n]
        var result = Tensor[dtype].zeros(out_shape)

        # Process each batch element
        for indices in batch_shape:
            var v_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, v_batch_dims
            )
            var M_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, M_batch_dims
            )

            var v_slice = v[il(v_indices), s()]  # Tensor[k]
            var M_slice = M[il(M_indices), s(), s()]  # Tensor[k, n]
            var result_slice = result[il(indices), s()]  # Tensor[n]

            # Use existing 2D matmul by lifting vector to matrix
            var v_lifted = v_slice.unsqueeze(-2)  # Tensor[1, k]
            var result_lifted = Matmul2d[dtype].forward[track_grad=False](
                v_lifted, M_slice
            )  # Tensor[1, n]
            var result_final = result_lifted.squeeze([-2])  # Tensor[n]

            # Copy result
            result_slice.buffer.fill_equal_shape[overwrite=True](
                result_final.buffer
            )

        # Setup backward
        @parameter
        if track_grad:
            var requires_grad = v.requires_grad or M.requires_grad
            if requires_grad:
                result.requires_grad_(True)
                var backward_fn = VectorMatmulNdBackward[
                    dtype
                ]().into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(v)
                result.add_ancestry(M)

        return result^


@fieldwise_init
@register_passable
struct VectorMatmulNdBackward[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var grad_out = output.grad().copy()  # GradBox: batch + [n]
        var v = output.ancestry().get(0)  # Tensor: batch_v + [k]
        var M = output.ancestry().get(1)  # Tensor: batch_M + [k, n]

        var v_shape = v.shape()
        var M_shape = M.shape()
        var grad_out_shape = grad_out.shape()

        # Recompute broadcasting
        var v_batch_dims = v_shape[:-1]
        var M_batch_dims = M_shape[:-2]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_dims, M_batch_dims
        )

        var results = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]()

        # Gradient for v: dv = grad_out @ M^T
        if v.requires_grad():
            var grad_v = Gradbox[dtype].zeros(v_shape, share=True)

            for indices in batch_shape:
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )
                var grad_out_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, grad_out_shape[:-1]
                )

                var M_slice = M.tensor()[
                    il(M_indices), s(), s()
                ]  # Tensor[k, n]
                var grad_out_slice = grad_out[
                    il(grad_out_indices), s()
                ]  # GradBox[n]
                var grad_v_slice = grad_v[il(v_indices), s()]  # GradBox[k]

                # Compute: grad_v = grad_out @ M^T
                var grad_out_lifted = grad_out_slice.unsqueeze(
                    [-2]
                )  # GradBox[1, n]
                var M_T = M_slice.transpose[track_grad=False](
                    1, 0
                )  # Tensor[n, k]
                var grad_v_lifted = Matmul2d[dtype].forward(
                    grad_out_lifted, M_T
                )  # GradBox[1, k]
                var grad_v_final = grad_v_lifted.squeeze([-2])  # GradBox[k]

                grad_v_slice.buffer.fill_equal_shape[overwrite=False](
                    grad_v_final.buffer
                )

            results.append((v.copy(), grad_v^, AddTensor))

        # Gradient for M: dM = v^T @ grad_out (outer product)
        if M.requires_grad():
            var grad_M = Gradbox[dtype].zeros(M_shape, share=True)

            for indices in batch_shape:
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )
                var grad_out_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, grad_out_shape[:-1]
                )

                var v_slice = v.tensor()[il(v_indices), s()]  # Tensor[k]
                var grad_out_slice = grad_out[
                    il(grad_out_indices), s()
                ]  # GradBox[n]
                var grad_M_slice = grad_M[
                    il(M_indices), s(), s()
                ]  # GradBox[k, n]

                # Compute outer product: grad_M = v^T @ grad_out
                # More efficient: directly create column vector without transpose
                var v_col = v_slice.unsqueeze(-1)  # Tensor[k, 1]
                var grad_out_row = grad_out_slice.unsqueeze(
                    [-2]
                )  # GradBox[1, n]
                var grad_M_slice_result = v_col * grad_out_row  # GradBox[k, n]

                grad_M_slice.buffer.fill_equal_shape[overwrite=False](
                    grad_M_slice_result.buffer
                )

            results.append((M^, grad_M^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


fn main() raises:
    pass
