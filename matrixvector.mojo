from tenmo import Tensor
from backpropagation import BackwardFnArg, BACKWARD_MATRIX_VECTOR_MUL
from mnemonics import AddTensor
from gradbox import Gradbox
from broadcasthelper import ShapeBroadcaster
from common_utils import panic
from matmul import Matmul2d, MatmulNd
from std.sys import simd_width_of, has_accelerator
from ndbuffer import NDBuffer
from matrixvector_kernel import MatrixVectorNdGpu

@fieldwise_init
struct MatrixVectorMulNdBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        comptime simdwidth = simd_width_of[Self.dtype]()

        ref grad_out = output.gradients()[]
        var M = output.ancestry().get(0)
        var v = output.ancestry().get(1)

        var M_shape = M.shape()
        var v_shape = v.shape()

        var m = M_shape[-2]
        var k = M_shape[-1]

        var M_batch_dims = M_shape[:-2]
        var v_batch_dims = v_shape[:-1]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_dims, v_batch_dims
        )

        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        # Gradient for M: dM[m, k] = grad_out[m] ⊗ v[k] (outer product)
        if M.requires_grad:
            var grad_M = Gradbox[Self.dtype].zeros(M_shape, share=True)

            # Hoist metadata
            var grad_out_stride = grad_out.strides()[-1]
            var grad_out_data = grad_out.data_ptr()

            var v_stride = v.buffer.strides[-1]
            var v_offset = v.buffer.offset
            var v_data = v.data_ptr()

            var grad_M_stride0 = grad_M.strides()[-2]
            var grad_M_stride1 = grad_M.strides()[-1]
            var grad_M_offset = grad_M.offset()
            var grad_M_data = grad_M.data_ptr()
            var grad_M_contiguous = grad_M.is_contiguous()

            for indices in batch_shape:
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )

                var grad_out_base = 0
                for i in range(len(indices)):
                    grad_out_base += indices[i] * grad_out.strides()[i]

                var v_base = v_offset
                for i in range(len(v_indices)):
                    v_base += v_indices[i] * v.buffer.strides[i]

                var grad_M_base = grad_M_offset
                for i in range(len(M_indices)):
                    grad_M_base += M_indices[i] * grad_M.strides()[i]

                # Outer product: grad_M[m, k] = grad_out[m] * v[k]
                if grad_M_contiguous:
                    # MANUAL VECTORIZATION for outer product
                    comptime simd_width = simdwidth
                    var num_full_vectors = k // simd_width
                    var remainder = k % simd_width

                    for i in range(m):
                        var grad_out_val = grad_out_data[
                            grad_out_base + i * grad_out_stride
                        ]
                        var grad_M_row_base = grad_M_base + i * grad_M_stride0

                        # Process full SIMD vectors
                        for vec_idx in range(num_full_vectors):
                            var j = vec_idx * simd_width
                            var v_addr = v_base + j * v_stride
                            var v_vec = v_data.load[width=simd_width](v_addr)

                            var grad_M_addr = (
                                grad_M_row_base + j * grad_M_stride1
                            )
                            var current = grad_M_data.load[width=simd_width](
                                grad_M_addr
                            )
                            grad_M_data.store[width=simd_width](
                                grad_M_addr, current + grad_out_val * v_vec
                            )

                        # Process remaining elements
                        if remainder > 0:
                            var j = num_full_vectors * simd_width
                            for offset in range(remainder):
                                var v_val = v_data[
                                    v_base + (j + offset) * v_stride
                                ]
                                var grad_M_addr = (
                                    grad_M_row_base
                                    + (j + offset) * grad_M_stride1
                                )
                                grad_M_data[grad_M_addr] += grad_out_val * v_val
                else:
                    for i in range(m):
                        var grad_out_val = grad_out_data[
                            grad_out_base + i * grad_out_stride
                        ]
                        for j in range(k):
                            var v_val = v_data[v_base + j * v_stride]
                            var grad_M_addr = (
                                grad_M_base
                                + i * grad_M_stride0
                                + j * grad_M_stride1
                            )
                            grad_M_data[grad_M_addr] += grad_out_val * v_val

            results.append((M, grad_M^, AddTensor))

        # Gradient for v: dv[k] = M^T[k, m] @ grad_out[m]
        # This is: dv[k] = sum_i(M[i, k] * grad_out[i])
        if v.requires_grad:
            var grad_v = Gradbox[Self.dtype].zeros(v_shape, share=True)

            # Hoist metadata
            var M_stride0 = M.buffer.strides[-2]
            var M_stride1 = M.buffer.strides[-1]
            var M_offset = M.buffer.offset
            var M_data = M.data_ptr()

            var grad_out_stride = grad_out.strides()[-1]
            var grad_out_data = grad_out.data_ptr()

            var grad_v_stride = grad_v.strides()[-1]
            var grad_v_offset = grad_v.offset()
            var grad_v_data = grad_v.data_ptr()

            for indices in batch_shape:
                var M_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, M_batch_dims
                )
                var v_indices = ShapeBroadcaster.broadcasted_indices(
                    indices, batch_shape, v_batch_dims
                )

                var M_base = M_offset
                for i in range(len(M_indices)):
                    M_base += M_indices[i] * M.buffer.strides[i]

                var grad_out_base = 0
                for i in range(len(indices)):
                    grad_out_base += indices[i] * grad_out.strides()[i]

                var grad_v_base = grad_v_offset
                for i in range(len(v_indices)):
                    grad_v_base += v_indices[i] * grad_v.strides()[i]

                # Compute: grad_v[k] = sum_i(M[i, k] * grad_out[i])
                for j in range(k):
                    var accumulator: Scalar[Self.dtype] = 0

                    for i in range(m):
                        var m_val = M_data[
                            M_base + i * M_stride0 + j * M_stride1
                        ]
                        var grad_val = grad_out_data[
                            grad_out_base + i * grad_out_stride
                        ]
                        accumulator += m_val * grad_val

                    var grad_v_addr = grad_v_base + j * grad_v_stride
                    grad_v_data[grad_v_addr] += accumulator

            results.append((v, grad_v^, AddTensor))

        return results^


@fieldwise_init
struct MatrixVectorMulNd[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn forward[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](M: NDBuffer[Self.dtype], v: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        var M_shape = M.shape
        var v_shape = v.shape

        # Validate: M[..., m, k] × v[..., k] → out[..., m]
        if M_shape.rank() < 2:
            panic("MatrixVectorMulNd: matrix must have at least 2 dimensions")
        if v_shape.rank() < 1:
            panic("MatrixVectorMulNd: vector must have at least 1 dimension")

        var k = M_shape[-1]
        var k_v = v_shape[-1]
        var m = M_shape[-2]

        if k != k_v:
            panic("MatrixVectorMulNd: inner dimensions must match")

        # Broadcast batch dimensions
        var M_batch_dims = M_shape[:-2]
        var v_batch_dims = v_shape[:-1]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_dims, v_batch_dims
        )

        var out_shape = batch_shape + [m]
        var result = NDBuffer[Self.dtype].zeros(out_shape)

        # Hoist metadata
        var M_stride0 = M.strides[-2]
        var M_stride1 = M.strides[-1]
        var M_offset = M.offset
        var M_data = M.data_ptr()

        var v_stride = v.strides[-1]
        var v_offset = v.offset
        var v_data = v.data_ptr()

        var result_stride = result.strides[-1]
        var result_offset = result.offset
        var result_data = result.data_ptr()

        # Process each batch element
        for indices in batch_shape:
            var M_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, M_batch_dims
            )
            var v_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, v_batch_dims
            )

            # Calculate base offsets
            var M_base = M_offset
            for i in range(len(M_indices)):
                M_base += M_indices[i] * M.strides[i]

            var v_base = v_offset
            for i in range(len(v_indices)):
                v_base += v_indices[i] * v.strides[i]

            var result_base = result_offset
            for i in range(len(indices)):
                result_base += indices[i] * result.strides[i]

            # Optimized matrix-vector multiply: result[m] = M[m, k] @ v[k]
            # For each output element: result[i] = sum_j(M[i,j] * v[j])
            for i in range(m):
                var accumulator: Scalar[Self.dtype] = 0
                var M_row_base = M_base + i * M_stride0

                # Dot product over k dimension
                for j in range(k):
                    var m_val = M_data[M_row_base + j * M_stride1]
                    var v_val = v_data[v_base + j * v_stride]
                    accumulator += m_val * v_val

                result_data[result_base + i * result_stride] = accumulator

        return result

    @staticmethod
    fn forward[
        track_grad: Bool = True, simdwidth: Int = simd_width_of[Self.dtype]()
    ](M: Tensor[Self.dtype], v: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        _="""var M_shape = M.shape()
        var v_shape = v.shape()

        # Validate: M[..., m, k] × v[..., k] → out[..., m]
        if M_shape.rank() < 2:
            panic("MatrixVectorMulNd: matrix must have at least 2 dimensions")
        if v_shape.rank() < 1:
            panic("MatrixVectorMulNd: vector must have at least 1 dimension")

        var k = M_shape[-1]
        var k_v = v_shape[-1]
        var m = M_shape[-2]

        if k != k_v:
            panic("MatrixVectorMulNd: inner dimensions must match")

        # Broadcast batch dimensions
        var M_batch_dims = M_shape[:-2]
        var v_batch_dims = v_shape[:-1]
        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_dims, v_batch_dims
        )

        var out_shape = batch_shape + [m]
        var result = Tensor[Self.dtype].zeros(out_shape)

        # Hoist metadata
        var M_stride0 = M.buffer.strides[-2]
        var M_stride1 = M.buffer.strides[-1]
        var M_offset = M.offset()
        var M_data = M.data_ptr()

        var v_stride = v.buffer.strides[-1]
        var v_offset = v.offset()
        var v_data = v.data_ptr()

        var result_stride = result.buffer.strides[-1]
        var result_offset = result.buffer.offset
        var result_data = result.data_ptr()

        # Process each batch element
        for indices in batch_shape:
            var M_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, M_batch_dims
            )
            var v_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, v_batch_dims
            )

            # Calculate base offsets
            var M_base = M_offset
            for i in range(M_indices.size()):
                M_base += M_indices[i] * M.buffer.strides[i]

            var v_base = v_offset
            for i in range(v_indices.size()):
                v_base += v_indices[i] * v.buffer.strides[i]

            var result_base = result_offset
            for i in range(indices.size()):
                result_base += indices[i] * result.buffer.strides[i]

            # Optimized matrix-vector multiply: result[m] = M[m, k] @ v[k]
            # For each output element: result[i] = sum_j(M[i,j] * v[j])
            for i in range(m):
                var accumulator: Scalar[Self.dtype] = 0
                var M_row_base = M_base + i * M_stride0

                # Dot product over k dimension
                for j in range(k):
                    var m_val = M_data[M_row_base + j * M_stride1]
                    var v_val = v_data[v_base + j * v_stride]
                    accumulator += m_val * v_val

                result_data[result_base + i * result_stride] = accumulator"""

        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if M.is_on_gpu() and v.is_on_gpu():
                try:
                    out = MatrixVectorNdGpu[Self.dtype].launch[block_size=256](
                        M.buffer, v.buffer
                    )
                except e:
                    print(e)
                    print(
                        "MatrixVectorMulNd - GPU vector matrix multiplication failed",
                    )
                    out = Self.forward[simdwidth=simdwidth](M.buffer, v.buffer)
            else:
                out = Self.forward[simdwidth=simdwidth](M.buffer, v.buffer)
        else:
            out = Self.forward[simdwidth=simdwidth](M.buffer, v.buffer)

        var result = Tensor[Self.dtype](out^, requires_grad=False)

        #var ndb = Self.forward[simdwidth=simdwidth](M.buffer, v.buffer)
        #var result = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var requires_grad = M.requires_grad or v.requires_grad
            if requires_grad:
                result.requires_grad_(True)
                result.bwdFnArg = Optional(BackwardFnArg[Self.dtype].null_arg(BACKWARD_MATRIX_VECTOR_MUL))
                result.add_ancestry(M, v)

        return result^
