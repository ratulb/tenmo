from tenmo import Tensor
from backpropagation import BackwardFnArg, BACKWARD_VECTOR_MATMUL
from mnemonics import AddTensor
from gradbox import Gradbox
from broadcasthelper import ShapeBroadcaster
from common_utils import panic
from std.sys import simd_width_of, has_accelerator
from ndbuffer import NDBuffer
from vectormatrix_kernel import VectorMatmulNdGpu
from ancestry import Ancestor


@fieldwise_init
struct VectorMatmulNdBackward[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](output: Ancestor[Self.dtype]) -> List[
        Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        ref grad_out = output.gradients()[]
        ref v_ref = output.ancestry().get(0)
        var v = Tensor[Self.dtype](
            v_ref.buffer(), requires_grad=v_ref.requires_grad
        )
        var M_ref = output.ancestry().get(1)
        var M = Tensor[Self.dtype](
            M_ref.buffer(), requires_grad=M_ref.requires_grad
        )

        var v_shape = v.shape()
        var M_shape = M.shape()

        var k = v_shape[-1]
        var n = M_shape[-1]

        var v_batch_shape = v_shape[:-1]
        var M_batch_shape = M_shape[:-2]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_shape, M_batch_shape
        )

        var results = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        # Gradient for v: dv[k] = grad_out[n] @ M^T[n, k]
        # This is equivalent to: dv[k] = sum_j(grad_out[j] * M[k, j])
        if v.requires_grad:
            var grad_v = Gradbox[Self.dtype].zeros(v_shape, share=True)

            # Hoist metadata
            var grad_out_stride = grad_out.strides()[-1]
            var grad_out_data = grad_out.data_ptr()

            var M_stride0 = M.buffer.strides[-2]
            var M_stride1 = M.buffer.strides[-1]
            var M_offset = M.buffer.offset
            var M_data = M.data_ptr()

            var grad_v_stride = grad_v.strides()[-1]
            var grad_v_offset = grad_v.offset()
            var grad_v_data = grad_v.data_ptr()

            for indices in batch_shape:
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_shape
                )
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_shape
                )

                # Calculate bases
                var grad_out_base = 0
                for i in range(len(indices)):
                    grad_out_base += indices[i] * grad_out.strides()[i]

                var M_base = M_offset
                for i in range(len(M_indices)):
                    M_base += M_indices[i] * M.buffer.strides[i]

                var grad_v_base = grad_v_offset
                for i in range(len(v_indices)):
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

            results.append((v_ref, grad_v^, AddTensor))

        # Gradient for M: dM[k, n] = v[k] ⊗ grad_out[n] (outer product)
        if M.requires_grad:
            var grad_M = Gradbox[Self.dtype].zeros(M_shape, share=True)

            # Hoist metadata
            var v_stride = v.buffer.strides[-1]
            var v_offset = v.buffer.offset
            var v_data = v.data_ptr()

            var grad_out_stride = grad_out.strides()[-1]
            var grad_out_data = grad_out.data_ptr()

            var grad_M_stride0 = grad_M.strides()[-2]
            var grad_M_stride1 = grad_M.strides()[-1]
            var grad_M_offset = grad_M.offset()
            var grad_M_data = grad_M.data_ptr()
            var grad_M_contiguous = (
                True  # Gradbox is always contiguous and has zero offset
            )

            for indices in batch_shape:
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_shape
                )
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_shape
                )

                var v_base = v_offset
                for i in range(len(v_indices)):
                    v_base += v_indices[i] * v.buffer.strides[i]

                var grad_out_base = 0
                for i in range(len(indices)):
                    grad_out_base += indices[i] * grad_out.strides()[i]

                var grad_M_base = grad_M_offset
                for i in range(len(M_indices)):
                    grad_M_base += M_indices[i] * grad_M.strides()[i]

                # Outer product: grad_M[k, n] = v[k] * grad_out[n]
                if grad_M_contiguous:
                    # MANUAL VECTORIZATION for outer product
                    comptime simd_width = simdwidth
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

            results.append((M_ref, grad_M^, AddTensor))

        return results^


@fieldwise_init
struct VectorMatmulNd[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](v: NDBuffer[Self.dtype], M: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        var v_shape = v.shape
        var M_shape = M.shape

        # Validate: v[..., k] × M[..., k, n] → out[..., n]
        if v_shape.rank() < 1:
            panic("VectorMatmulNd: vector must have rank >= 1")
        if M_shape.rank() < 2:
            panic("VectorMatmulNd: matrix must have rank >= 2")

        var k = v_shape[-1]
        var k_M = M_shape[-2]
        var n = M_shape[-1]

        if k != k_M:
            panic("VectorMatmulNd: inner dimensions must match")

        # Broadcast batch dimensions
        var v_batch_shape = v_shape[:-1]
        var M_batch_shape = M_shape[:-2]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_shape, M_batch_shape
        )

        var out_shape = batch_shape + [n]
        var result = NDBuffer[Self.dtype].zeros(out_shape)

        # Hoist metadata for vector and matrix
        var v_stride = v.strides[-1]
        var v_offset = v.offset
        var v_data = v.data_ptr()

        var M_stride0 = M.strides[-2]
        var M_stride1 = M.strides[-1]
        var M_offset = M.offset
        var M_data = M.data_ptr()
        var M_contiguous = M.is_contiguous()

        var result_stride = result.strides[-1]
        var result_offset = result.offset
        var result_data = (
            result.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Process each batch element
        for indices in batch_shape:
            var v_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, v_batch_shape
            )
            var M_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, M_batch_shape
            )

            # Calculate base offsets for this batch
            var v_base = v_offset
            for i in range(len(v_indices)):
                v_base += v_indices[i] * v.strides[i]

            var M_base = M_offset
            for i in range(len(M_indices)):
                M_base += M_indices[i] * M.strides[i]

            var result_base = result_offset
            for i in range(len(indices)):
                result_base += indices[i] * result.strides[i]

            # Optimized vector-matrix multiply: result[n] = v[k] @ M[k, n]
            if M_contiguous:
                # Fast path: M has contiguous rows - MANUAL VECTORIZATION
                comptime simd_width = simdwidth
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

        return result

    @staticmethod
    fn forward[
        track_grad: Bool = True, simdwidth: Int = simd_width_of[Self.dtype]()
    ](v: Tensor[Self.dtype], M: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if v.is_on_gpu() and M.is_on_gpu():
                try:
                    out = VectorMatmulNdGpu[Self.dtype].launch[block_size=256](
                        v.buffer, M.buffer
                    )
                except e:
                    print(e)
                    print(
                        "VectorMatmulNd - GPU vector matrix multiplication",
                    )
                    out = Self.forward[simdwidth=simdwidth](v.buffer, M.buffer)
            else:
                out = Self.forward[simdwidth=simdwidth](v.buffer, M.buffer)
        else:
            out = Self.forward[simdwidth=simdwidth](v.buffer, M.buffer)

        var result = Tensor[Self.dtype](out^, requires_grad=False)

        comptime if track_grad:
            var requires_grad = v.requires_grad or M.requires_grad
            if requires_grad:
                result.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_VECTOR_MATMUL
                )
                result.add_ancestry(backwardFnArg^, v, M)

        return result^
