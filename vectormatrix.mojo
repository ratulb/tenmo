from tenmo import Tensor
from backpropagation import Delegate, BackwardFn, BACKWARD_VECTOR_MATMUL
from operators import AddTensor
from gradbox import Gradbox
from shapes import Shape
from broadcasthelper import ShapeBroadcaster
from common_utils import il, s, panic
from matmul import Matmul2d
from sys import simd_width_of


@fieldwise_init
@register_passable
struct VectorMatmulNd[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True, simdwidth: Int = simd_width_of[Self.dtype]()
    ](v: Tensor[Self.dtype], M: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var v_shape = v.shape()
        var M_shape = M.shape()

        # Validate: v[..., k] × M[..., k, n] → out[..., n]
        if v_shape.rank() < 1:
            panic("VectorMatmulNd: vector must have at least 1 dimension")
        if M_shape.rank() < 2:
            panic("VectorMatmulNd: matrix must have at least 2 dimensions")

        var k = v_shape[-1]
        var k_M = M_shape[-2]
        var n = M_shape[-1]

        if k != k_M:
            panic("VectorMatmulNd: inner dimensions must match")

        # Broadcast batch dimensions
        var v_batch_dims = v_shape[:-1]
        var M_batch_dims = M_shape[:-2]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_dims, M_batch_dims
        )

        var out_shape = batch_shape + [n]
        var result = Tensor[Self.dtype].zeros(out_shape)

        # Hoist metadata for vector and matrix
        var v_stride = v.buffer.strides[-1]
        var v_offset = v.buffer.offset
        var v_data = v.buffer.buffer.data

        var M_stride0 = M.buffer.strides[-2]
        var M_stride1 = M.buffer.strides[-1]
        var M_offset = M.buffer.offset
        var M_data = M.buffer.buffer.data
        var M_contiguous = M.is_contiguous()

        var result_stride = result.buffer.strides[-1]
        var result_offset = result.buffer.offset
        var result_data = result.buffer.buffer.data

        # Process each batch element
        for indices in batch_shape:
            var v_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, v_batch_dims
            )
            var M_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, M_batch_dims
            )

            # Calculate base offsets for this batch
            var v_base = v_offset
            for i in range(v_indices.size()):
                v_base += v_indices[i] * v.buffer.strides[i]

            var M_base = M_offset
            for i in range(M_indices.size()):
                M_base += M_indices[i] * M.buffer.strides[i]

            var result_base = result_offset
            for i in range(indices.size()):
                result_base += indices[i] * result.buffer.strides[i]

            # Optimized vector-matrix multiply: result[n] = v[k] @ M[k, n]
            if M_contiguous:
                # Fast path: M has contiguous rows - MANUAL VECTORIZATION
                alias simd_width = simdwidth
                var num_full_vectors = n // simd_width
                var remainder = n % simd_width

                # Process full SIMD vectors
                for vec_idx in range(num_full_vectors):
                    var j = vec_idx * simd_width
                    var accumulator = SIMD[Self.dtype, simd_width](0)

                    # Dot product: sum over k
                    for i in range(k):
                        var v_val = v_data[v_base + i * v_stride]
                        var m_addr = M_base + i * M_stride0 + j * M_stride1
                        var m_vec = M_data.load[width=simd_width](m_addr)
                        accumulator += v_val * m_vec

                    var result_addr = result_base + j * result_stride
                    result_data.store[width=simd_width](
                        result_addr, accumulator
                    )

                # Process remaining elements
                if remainder > 0:
                    var j = num_full_vectors * simd_width
                    for offset in range(remainder):
                        var accumulator: Scalar[Self.dtype] = 0

                        for i in range(k):
                            var v_val = v_data[v_base + i * v_stride]
                            var m_val = M_data[
                                M_base
                                + i * M_stride0
                                + (j + offset) * M_stride1
                            ]
                            accumulator += v_val * m_val

                        result_data[
                            result_base + (j + offset) * result_stride
                        ] = accumulator
            else:
                # Slow path: non-contiguous M
                for j in range(n):
                    var accumulator: Scalar[Self.dtype] = 0

                    for i in range(k):
                        var v_val = v_data[v_base + i * v_stride]
                        var m_val = M_data[
                            M_base + i * M_stride0 + j * M_stride1
                        ]
                        accumulator += v_val * m_val

                    result_data[result_base + j * result_stride] = accumulator

        # Setup backward
        @parameter
        if track_grad:
            var requires_grad = v.requires_grad or M.requires_grad
            if requires_grad:
                result.requires_grad_(True)
                var backward_fn = VectorMatmulNdBackward[
                    Self.dtype
                ]().into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(v)
                result.add_ancestry(M)

        return result^


@fieldwise_init
@register_passable
struct VectorMatmulNdBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_VECTOR_MATMUL

    fn backward[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](self, read output: Tensor[Self.dtype]) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        ref grad_out = output.gradients()[]
        var v = output.ancestry().get(0)
        var M = output.ancestry().get(1)

        var v_shape = v.shape()
        var M_shape = M.shape()

        var k = v_shape[-1]
        var n = M_shape[-1]

        var v_batch_dims = v_shape[:-1]
        var M_batch_dims = M_shape[:-2]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_dims, M_batch_dims
        )

        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        # Gradient for v: dv[k] = grad_out[n] @ M^T[n, k]
        # This is equivalent to: dv[k] = sum_j(grad_out[j] * M[k, j])
        if v.requires_grad:
            var grad_v = Gradbox[Self.dtype].zeros(v_shape, share=True)

            # Hoist metadata
            var grad_out_stride = grad_out.strides()[-1]
            var grad_out_data = grad_out.buffer.buffer.data

            var M_stride0 = M.buffer.strides[-2]
            var M_stride1 = M.buffer.strides[-1]
            var M_offset = M.buffer.offset
            var M_data = M.buffer.buffer.data

            var grad_v_stride = grad_v.strides()[-1]
            var grad_v_offset = grad_v.offset()
            var grad_v_data = grad_v.buffer.buffer.data

            for indices in batch_shape:
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )

                # Calculate bases
                var grad_out_base = 0
                for i in range(indices.size()):
                    grad_out_base += indices[i] * grad_out.strides()[i]

                var M_base = M_offset
                for i in range(M_indices.size()):
                    M_base += M_indices[i] * M.buffer.strides[i]

                var grad_v_base = grad_v_offset
                for i in range(v_indices.size()):
                    grad_v_base += v_indices[i] * grad_v.strides()[i]

                # Compute: grad_v[k] = sum_j(grad_out[j] * M[k, j])
                for i in range(k):
                    var accumulator: Scalar[Self.dtype] = 0

                    for j in range(n):
                        var grad_val = grad_out_data[
                            grad_out_base + j * grad_out_stride
                        ]
                        var m_val = M_data[
                            M_base + i * M_stride0 + j * M_stride1
                        ]
                        accumulator += grad_val * m_val

                    var grad_v_addr = grad_v_base + i * grad_v_stride
                    grad_v_data[grad_v_addr] += accumulator

            results.append((v, grad_v^, AddTensor))

        # Gradient for M: dM[k, n] = v[k] ⊗ grad_out[n] (outer product)
        if M.requires_grad:
            var grad_M = Gradbox[Self.dtype].zeros(M_shape, share=True)

            # Hoist metadata
            var v_stride = v.buffer.strides[-1]
            var v_offset = v.buffer.offset
            var v_data = v.buffer.buffer.data

            var grad_out_stride = grad_out.strides()[-1]
            var grad_out_data = grad_out.buffer.buffer.data

            var grad_M_stride0 = grad_M.strides()[-2]
            var grad_M_stride1 = grad_M.strides()[-1]
            var grad_M_offset = grad_M.offset()
            var grad_M_data = grad_M.buffer.buffer.data
            var grad_M_contiguous = (
                True  # Gradbox is always contiguous and has zero offset
            )

            for indices in batch_shape:
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )

                var v_base = v_offset
                for i in range(v_indices.size()):
                    v_base += v_indices[i] * v.buffer.strides[i]

                var grad_out_base = 0
                for i in range(indices.size()):
                    grad_out_base += indices[i] * grad_out.strides()[i]

                var grad_M_base = grad_M_offset
                for i in range(M_indices.size()):
                    grad_M_base += M_indices[i] * grad_M.strides()[i]

                # Outer product: grad_M[k, n] = v[k] * grad_out[n]
                if grad_M_contiguous:
                    # MANUAL VECTORIZATION for outer product
                    alias simd_width = simdwidth
                    var num_full_vectors = n // simd_width
                    var remainder = n % simd_width

                    for i in range(k):
                        var v_val = v_data[v_base + i * v_stride]
                        var grad_M_row_base = grad_M_base + i * grad_M_stride0

                        # Process full SIMD vectors
                        for vec_idx in range(num_full_vectors):
                            var j = vec_idx * simd_width
                            var grad_out_addr = (
                                grad_out_base + j * grad_out_stride
                            )
                            var grad_out_vec = grad_out_data.load[
                                width=simd_width
                            ](grad_out_addr)

                            var grad_M_addr = (
                                grad_M_row_base + j * grad_M_stride1
                            )
                            var current = grad_M_data.load[width=simd_width](
                                grad_M_addr
                            )
                            grad_M_data.store[width=simd_width](
                                grad_M_addr, current + v_val * grad_out_vec
                            )

                        # Process remaining elements
                        if remainder > 0:
                            var j = num_full_vectors * simd_width
                            for offset in range(remainder):
                                var grad_out_val = grad_out_data[
                                    grad_out_base
                                    + (j + offset) * grad_out_stride
                                ]
                                var grad_M_addr = (
                                    grad_M_row_base
                                    + (j + offset) * grad_M_stride1
                                )
                                grad_M_data[grad_M_addr] += v_val * grad_out_val
                else:
                    for i in range(k):
                        var v_val = v_data[v_base + i * v_stride]
                        for j in range(n):
                            var grad_out_val = grad_out_data[
                                grad_out_base + j * grad_out_stride
                            ]
                            var grad_M_addr = (
                                grad_M_base
                                + i * grad_M_stride0
                                + j * grad_M_stride1
                            )
                            grad_M_data[grad_M_addr] += v_val * grad_out_val

            results.append((M, grad_M^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


fn main():
    pass
