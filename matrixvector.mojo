from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from gradbox import Gradbox
from ancestry import Ancestor
from shapes import Shape
from broadcasthelper import ShapeBroadcaster
from common_utils import il, s, panic
from matmul import Matmul2d, MatmulNd


# Key Differences from VectorMatmulNdBackward:
# 1. Gradient for M (dM)
# MatrixVectorMulNd: dM = grad_out @ v^T
# Shapes: [..., m] @ [..., k]^T → [..., m, k]
# var grad_out_col = grad_out_slice.unsqueeze(-1)  # [m, 1]
# var v_row = v_slice.unsqueeze(-2)               # [1, k]
# var grad_M_slice_result = Matmul2d.forward(grad_out_col, v_row)  # [m, k]

# VectorMatmulNd: dM = v^T @ grad_out
# var v_col = v_slice.unsqueeze(-1)           # [k, 1]
# var grad_out_row = grad_out_slice.unsqueeze(-2)  # [1, n]
# var grad_M_slice_result = Matmul2d.forward(v_col, grad_out_row)  # [k, n]
# 2. Gradient for v (dv)

# MatrixVectorMulNd: dv = M^T @ grad_out
# Shapes: [..., k, m] @ [..., m] → [..., k]
# var M_T = M_slice.transpose[track_grad=False](1, 0)  # [k, m]
# var grad_out_col = grad_out_slice.unsqueeze(-1)      # [m, 1]
# var grad_v_lifted = Matmul2d.forward(M_T, grad_out_col)  # [k, 1]
# var grad_v_final = grad_v_lifted.squeeze(-1)         # [k]

# VectorMatmulNd: dv = grad_out @ M^T
# var grad_out_lifted = grad_out_slice.unsqueeze(-2)  # [1, n]
# var M_T = M_slice.transpose[track_grad=False](1, 0)  # [n, k]
# var grad_v_lifted = Matmul2d.forward(grad_out_lifted, M_T)  # [1, k]
# var grad_v_final = grad_v_lifted.squeeze(-2)         # [k]
# 3. Ancestry Order

# MatrixVectorMulNd: M first, then v
# var M = output.ancestry().get(0)  # Tensor: batch_M + [m, k]
# var v = output.ancestry().get(1)  # Tensor: batch_v + [k]

# VectorMatmulNd: v first, then M
# var v = output.ancestry().get(0)  # Tensor: batch_v + [k]
# var M = output.ancestry().get(1)  # Tensor: batch_M + [k, n]
# 4. Dimension Shapes

# MatrixVectorMulNd
# var grad_out_shape = grad_out.shape()  # batch + [m]
# var M_shape = M.shape()                # batch_M + [m, k]
# var v_shape = v.shape()                # batch_v + [k]

# VectorMatmulNd
# var grad_out_shape = grad_out.shape()  # batch + [n]
# var v_shape = v.shape()                # batch_v + [k]
# var M_shape = M.shape()                # batch_M + [k, n]
# Mathematical Verification:
# For the operation: r = M @ v where:

# M shape: [..., m, k]

# v shape: [..., k]

# r shape: [..., m]

# The gradients are:

# dM = grad_out @ v^T → shape [..., m, k]

# dv = M^T @ grad_out → shape [..., k]


@fieldwise_init
@register_passable
struct MatrixVectorMulNdBackward[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref grad_out = output.grad()  # GradBox: batch + [m]
        var M = output.ancestry().get(0)  # Tensor: batch_M + [m, k]
        var v = output.ancestry().get(1)  # Tensor: batch_v + [k]

        var M_shape = M.shape()
        var v_shape = v.shape()
        var grad_out_shape = grad_out.shape()

        # Recompute broadcasting
        var M_batch_dims = M_shape[:-2]
        var v_batch_dims = v_shape[:-1]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_dims, v_batch_dims
        )

        var results = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]()

        # Gradient for M: dM = grad_out @ v^T
        if M.requires_grad():
            var grad_M = Gradbox[dtype].zeros(M_shape, share=True)

            for indices in batch_shape:
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )
                var grad_out_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, grad_out_shape[:-1]
                )

                var v_slice = v.tensor()[il(v_indices), s()]  # Tensor[k]
                var grad_out_slice = grad_out[
                    il(grad_out_indices), s()
                ]  # GradBox[m]
                var grad_M_slice = grad_M[
                    il(M_indices), s(), s()
                ]  # GradBox[m, k]

                # Compute: grad_M = grad_out @ v^T
                var grad_out_col = grad_out_slice.unsqueeze(
                    [-1]
                )  # GradBox[m, 1]
                var v_row = v_slice.unsqueeze([-2])  # Tensor[1, k]
                # Element-wise multiplication for outer product
                var grad_M_slice_result = grad_out_col * v_row  # GradBox[m, k]

                grad_M_slice.buffer.fill_equal_shape[overwrite=False](
                    grad_M_slice_result.buffer
                )

            results.append((M.copy(), grad_M^, AddTensor))

        # Gradient for v: dv = M^T @ grad_out
        if v.requires_grad():
            var grad_v = Gradbox[dtype].zeros(v_shape, share=True)

            for indices in batch_shape:
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )
                var grad_out_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, grad_out_shape[:-1]
                )

                var M_slice = M.tensor()[
                    il(M_indices), s(), s()
                ]  # Tensor[m, k]
                var grad_out_slice = grad_out[
                    il(grad_out_indices), s()
                ]  # GradBox[m]
                var grad_v_slice = grad_v[il(v_indices), s()]  # GradBox[k]

                # Compute: grad_v = M^T @ grad_out
                var M_T = M_slice.transpose[track_grad=False](
                    1, 0
                )  # Tensor[k, m]
                var grad_out_col = grad_out_slice.unsqueeze(
                    [-1]
                )  # GradBox[m, 1]
                var grad_v_lifted = Matmul2d[dtype].forward(
                    M_T, grad_out_col
                )  # GradBox[k, 1]
                var grad_v_final = grad_v_lifted.squeeze([-1])  # GradBox[k]

                grad_v_slice.buffer.fill_equal_shape[overwrite=False](
                    grad_v_final.buffer
                )

            results.append((v^, grad_v^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


# Key Differences from VectorMatmulNd:
# 1. Dimension Validation

# MatrixVectorMulNd: M[..., m, k] @ v[..., k] → out[..., m]
# var k_M = M_shape[-1]  # Last dimension of matrix (inner)
# var k_v = v_shape[-1]  # Last dimension of vector
# var m = M_shape[-2]    # Second last dimension of matrix (output size)

# VectorMatmulNd: v[..., k] @ M[..., k, n] → out[..., n]
# var k_v = v_shape[-1]
# var k_M = M_shape[-2]
# var n = M_shape[-1]
# 2. Batch Dimension Extraction

# MatrixVectorMulNd
# var M_batch_dims = M_shape[:-2]  # All but last 2 dims
# var v_batch_dims = v_shape[:-1]  # All but last dim

# VectorMatmulNd
# var v_batch_dims = v_shape[:-1]
# var M_batch_dims = M_shape[:-2]
# 3. Lifting and Matmul

# MatrixVectorMulNd: M[m,k] @ v[k,1] → [m,1] → [m]
# var v_lifted = v_slice.unsqueeze(-1)  # [k, 1]
# var result_lifted = Matmul2d.forward(M_slice, v_lifted)  # [m, 1]
# var result_final = result_lifted.squeeze([-1])  # [m]

# VectorMatmulNd: v[1,k] @ M[k,n] → [1,n] → [n]
# var v_lifted = v_slice.unsqueeze(-2)  # [1, k]
# var result_lifted = Matmul2d.forward(v_lifted, M_slice)  # [1, n]
# var result_final = result_lifted.squeeze([-2])  # [n]
# 4. Output Shape

# MatrixVectorMulNd: batch + [m]  (m from matrix rows)
# var out_shape = batch_shape + [m]

# VectorMatmulNd: batch + [n]  (n from matrix columns)
# var out_shape = batch_shape + [n]
# 5. Ancestry Order

# MatrixVectorMulNd: matrix first, then vector
# result.add_ancestry(M)
# result.add_ancestry(v)

# VectorMatmulNd: vector first, then matrix
# result.add_ancestry(v)
# result.add_ancestry(M)


@fieldwise_init
@register_passable
struct MatrixVectorMulNd[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](M: Tensor[dtype], v: Tensor[dtype]) -> Tensor[dtype]:
        var M_shape = M.shape()
        var v_shape = v.shape()

        # Validate: M[..., m, k] × v[..., k] → out[..., m]
        if M_shape.rank() < 2:
            panic("MatrixVectorMulNd: matrix must have at least 2 dimensions")
        if v_shape.rank() < 1:
            panic("MatrixVectorMulNd: vector must have at least 1 dimension")

        var k_M = M_shape[-1]  # Last dimension of matrix
        var k_v = v_shape[-1]  # Last dimension of vector
        var m = M_shape[-2]  # Second last dimension of matrix

        if k_M != k_v:
            panic("MatrixVectorMulNd: inner dimensions must match")

        # Broadcast batch dimensions
        var M_batch_dims = M_shape[:-2]  # All but last 2 dims of matrix
        var v_batch_dims = v_shape[:-1]  # All but last dim of vector
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_dims, v_batch_dims
        )

        # Output shape: batch + [m]
        var out_shape = batch_shape + [m]
        var result = Tensor[dtype].zeros(out_shape)

        # Process each batch element
        for indices in batch_shape:
            var M_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, M_batch_dims
            )
            var v_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, v_batch_dims
            )

            var M_slice = M[il(M_indices), s(), s()]  # Tensor[m, k]
            var v_slice = v[il(v_indices), s()]  # Tensor[k]
            var result_slice = result[il(indices), s()]  # Tensor[m]

            # Use existing 2D matmul by lifting vector to matrix
            var v_lifted = v_slice.unsqueeze(-1)  # Tensor[k, 1]
            var result_lifted = Matmul2d[dtype].forward[track_grad=False](
                M_slice, v_lifted
            )  # Tensor[m, 1]
            var result_final = result_lifted.squeeze([-1])  # Tensor[m]

            # Copy result
            result_slice.buffer.fill_equal_shape[overwrite=True](
                result_final.buffer
            )

        # Setup backward
        @parameter
        if track_grad:
            var requires_grad = M.requires_grad or v.requires_grad
            if requires_grad:
                result.requires_grad_(True)
                var backward_fn = MatrixVectorMulNdBackward[
                    dtype
                ]().into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(M)
                result.add_ancestry(v)

        return result^


fn main() raises:
    pass
