from tenmo import Tensor
from sys import simd_width_of
from matrixshapevalidator import MatrixShapeValidator
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_MATMUL_ND,
    BACKWARD_MATMUL_2D,
)
from operators import AddTensor, mm, vm, mv, dot, invalid
from gradbox import Gradbox
from shapes import Shape
from broadcasthelper import ShapeBroadcaster
from common_utils import il, s, panic, log_debug
from vectormatrix import VectorMatmulNd
from matrixvector import MatrixVectorMulNd


alias TILE_SIZE = 64


@fieldwise_init
@register_passable
struct Matmul2dBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_MATMUL_2D

    @always_inline
    fn backward[
        simdwidth: Int = simd_width_of[Self.dtype](), tile_size: Int = TILE_SIZE
    ](self, output: Tensor[Self.dtype]) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        ref grad_out = output.gradients()[]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        var m = A.shape()[0]
        var n = A.shape()[1]
        var p = B.shape()[1]

        var result = List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]()

        # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
        # grad_A[i,j] = sum_k(grad_out[i,k] * B[j,k])  ← Reading B transposed!
        if A.requires_grad:
            var grad_A = Gradbox[Self.dtype].zeros(Shape([m, n]))

            # Hoist all metadata
            var grad_out_stride0 = grad_out.strides()[0]
            var grad_out_stride1 = grad_out.strides()[1]
            var grad_out_data = grad_out.buffer.buffer.data

            var B_stride0 = B.buffer.strides[0]
            var B_stride1 = B.buffer.strides[1]
            var B_offset = B.buffer.offset
            var B_data = B.buffer.buffer.data

            var grad_A_stride0 = grad_A.strides()[0]
            var grad_A_stride1 = grad_A.strides()[1]
            var grad_A_data = grad_A.buffer.buffer.data

            # TILED computation - accessing B in transposed order
            for i_tile in range(0, m, tile_size):
                for j_tile in range(0, n, tile_size):
                    for k_tile in range(0, p, tile_size):
                        var i_end = min(i_tile + tile_size, m)
                        var j_end = min(j_tile + tile_size, n)
                        var k_end = min(k_tile + tile_size, p)

                        for i in range(i_tile, i_end):
                            var grad_out_row_base = i * grad_out_stride0
                            var grad_A_row_base = i * grad_A_stride0

                            # Manual SIMD vectorization
                            var j = j_tile

                            # Main vectorized loop
                            while j + simdwidth <= j_end:
                                var grad_A_addr = (
                                    grad_A_row_base + j * grad_A_stride1
                                )
                                var accumulator = grad_A_data.load[
                                    width=simdwidth
                                ](grad_A_addr)

                                for k in range(k_tile, k_end):
                                    var grad_addr = (
                                        grad_out_row_base + k * grad_out_stride1
                                    )
                                    var grad_ik = grad_out_data[grad_addr]

                                    # B[j,k] - reading B transposed
                                    var b_addr = (
                                        j * B_stride0 + B_offset + k * B_stride1
                                    )
                                    var b_vec = B_data.load[width=simdwidth](
                                        b_addr
                                    )

                                    accumulator += grad_ik * b_vec

                                grad_A_data.store[width=simdwidth](
                                    grad_A_addr, accumulator
                                )
                                j += simdwidth

                            # Tail handling
                            while j < j_end:
                                var grad_A_addr = (
                                    grad_A_row_base + j * grad_A_stride1
                                )
                                var accumulator = grad_A_data[grad_A_addr]

                                for k in range(k_tile, k_end):
                                    var grad_addr = (
                                        grad_out_row_base + k * grad_out_stride1
                                    )
                                    var b_addr = (
                                        j * B_stride0 + B_offset + k * B_stride1
                                    )
                                    accumulator += (
                                        grad_out_data[grad_addr]
                                        * B_data[b_addr]
                                    )

                                grad_A_data[grad_A_addr] = accumulator
                                j += 1

            result.append((A, grad_A^, AddTensor))

        # ===== GRADIENT FOR B: dL/dB = A^T × grad_out =====
        # grad_B[j,k] = sum_i(A[i,j] * grad_out[i,k])  ← Reading A transposed!
        if B.requires_grad:
            var grad_B = Gradbox[Self.dtype].zeros(Shape([n, p]))

            var A_stride0 = A.buffer.strides[0]
            var A_stride1 = A.buffer.strides[1]
            var A_offset = A.buffer.offset
            var A_data = A.buffer.buffer.data

            var grad_out_stride0 = grad_out.strides()[0]
            var grad_out_stride1 = grad_out.strides()[1]
            var grad_out_data = grad_out.buffer.buffer.data

            var grad_B_stride0 = grad_B.strides()[0]
            var grad_B_stride1 = grad_B.strides()[1]
            var grad_B_data = grad_B.buffer.buffer.data

            # TILED computation
            for j_tile in range(0, n, tile_size):
                for k_tile in range(0, p, tile_size):
                    for i_tile in range(0, m, tile_size):
                        var j_end = min(j_tile + tile_size, n)
                        var k_end = min(k_tile + tile_size, p)
                        var i_end = min(i_tile + tile_size, m)

                        for j in range(j_tile, j_end):
                            var grad_B_row_base = j * grad_B_stride0

                            # Manual SIMD vectorization
                            var k = k_tile

                            # Main vectorized loop
                            while k + simdwidth <= k_end:
                                var grad_B_addr = (
                                    grad_B_row_base + k * grad_B_stride1
                                )
                                var accumulator = grad_B_data.load[
                                    width=simdwidth
                                ](grad_B_addr)

                                for i in range(i_tile, i_end):
                                    # A[i,j] - reading A transposed
                                    var a_addr = (
                                        i * A_stride0 + A_offset + j * A_stride1
                                    )
                                    var a_ij = A_data[a_addr]

                                    # grad_out[i,k] - contiguous access
                                    var grad_addr = (
                                        i * grad_out_stride0
                                        + k * grad_out_stride1
                                    )
                                    var grad_vec = grad_out_data.load[
                                        width=simdwidth
                                    ](grad_addr)

                                    accumulator += a_ij * grad_vec

                                grad_B_data.store[width=simdwidth](
                                    grad_B_addr, accumulator
                                )
                                k += simdwidth

                            # Tail handling
                            while k < k_end:
                                var grad_B_addr = (
                                    grad_B_row_base + k * grad_B_stride1
                                )
                                var accumulator = grad_B_data[grad_B_addr]

                                for i in range(i_tile, i_end):
                                    var a_addr = (
                                        i * A_stride0 + A_offset + j * A_stride1
                                    )
                                    var grad_addr = (
                                        i * grad_out_stride0
                                        + k * grad_out_stride1
                                    )
                                    accumulator += (
                                        A_data[a_addr]
                                        * grad_out_data[grad_addr]
                                    )

                                grad_B_data[grad_B_addr] = accumulator
                                k += 1

            result.append((B^, grad_B^, AddTensor))

        return result^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Matmul2d[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    @always_inline
    fn forward[
        track_grad: Bool = True,
        simdwidth: Int = simd_width_of[Self.dtype](),
        tile_size: Int = TILE_SIZE,  # Tune this: 32, 64, or 128
    ](A: Tensor[Self.dtype], B: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()
        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]

        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)
        var C = Tensor[Self.dtype].zeros(Shape([m, p]))

        # Hoist metadata
        var A_stride0 = A.buffer.strides[0]
        var A_stride1 = A.buffer.strides[1]
        var A_offset = A.buffer.offset
        var A_data = A.buffer.buffer.data

        var B_stride0 = B.buffer.strides[0]
        var B_stride1 = B.buffer.strides[1]
        var B_offset = B.buffer.offset
        var B_data = B.buffer.buffer.data
        var B_contiguous = B.is_contiguous()

        var C_stride0 = C.buffer.strides[0]
        var C_stride1 = C.buffer.strides[1]
        var C_offset = C.buffer.offset
        var C_data = C.buffer.buffer.data

        if B_contiguous:
            # ========================================
            # TILED/BLOCKED MATMUL with Manual SIMD
            # ========================================
            for i_tile in range(0, m, tile_size):
                for j_tile in range(0, p, tile_size):
                    for k_tile in range(0, n, tile_size):
                        # Compute tile boundaries
                        var i_end = min(i_tile + tile_size, m)
                        var j_end = min(j_tile + tile_size, p)
                        var k_end = min(k_tile + tile_size, n)

                        # Process this tile
                        for i in range(i_tile, i_end):
                            var a_row_base = i * A_stride0 + A_offset
                            var c_row_base = i * C_stride0 + C_offset

                            # Manual SIMD vectorization for columns
                            var j = j_tile
                            var cols_remaining = j_end - j_tile

                            # Main vectorized loop (process simdwidth columns at a time)
                            while j + simdwidth <= j_end:
                                var c_addr = c_row_base + j * C_stride1
                                var accumulator = C_data.load[width=simdwidth](
                                    c_addr
                                )

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    var a_ik = A_data[a_addr]

                                    var b_addr = (
                                        k * B_stride0 + B_offset + j * B_stride1
                                    )
                                    var b_vec = B_data.load[width=simdwidth](
                                        b_addr
                                    )

                                    accumulator += a_ik * b_vec

                                C_data.store[width=simdwidth](
                                    c_addr, accumulator
                                )
                                j += simdwidth

                            # Tail handling: process remaining columns one at a time
                            while j < j_end:
                                var c_addr = c_row_base + j * C_stride1
                                var accumulator = C_data[c_addr]

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    var b_addr = (
                                        k * B_stride0 + B_offset + j * B_stride1
                                    )
                                    accumulator += (
                                        A_data[a_addr] * B_data[b_addr]
                                    )

                                C_data[c_addr] = accumulator
                                j += 1
        else:
            # Non-contiguous path (scalar)
            for i in range(m):
                var a_row_base = i * A_stride0 + A_offset
                var c_row_base = i * C_stride0 + C_offset

                for j in range(p):
                    var accumulator: Scalar[dtype] = 0

                    for k in range(n):
                        var a_addr = a_row_base + k * A_stride1
                        var b_addr = k * B_stride0 + B_offset + j * B_stride1
                        accumulator += A_data[a_addr] * B_data[b_addr]

                    var c_addr = c_row_base + j * C_stride1
                    C_data[c_addr] = accumulator

        @parameter
        if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backward_fn = Matmul2dBackward[
                    Self.dtype
                ]().into_backward_fn()
                C.backwardFn = Optional(backward_fn^)
                C.add_ancestry(A)
                C.add_ancestry(B)

        return C^

    @staticmethod
    @always_inline
    fn forward[
        simdwidth: Int = simd_width_of[Self.dtype](), tile_size: Int = TILE_SIZE
    ](A: Tensor[Self.dtype], B: Gradbox[Self.dtype]) -> Gradbox[Self.dtype]:
        var m = A.shape()[0]
        var n = A.shape()[1]
        var p = B.shape()[1]

        var C = Gradbox[Self.dtype].zeros(Shape([m, p]))

        # Hoist metadata
        ref A_strides = A.strides()
        var A_stride0 = A_strides[0]
        var A_stride1 = A_strides[1]
        var A_offset = A.offset()
        var A_data = A.buffer.buffer.data

        ref B_strides = B.strides()
        var B_stride0 = B_strides[0]
        var B_stride1 = B_strides[1]
        var B_data = B.buffer.buffer.data

        ref C_strides = C.strides()
        var C_stride0 = C_strides[0]
        var C_stride1 = C_strides[1]
        var C_offset = C.offset()
        var C_data = C.buffer.buffer.data

        # ========================================
        # TILED MATMUL with Manual SIMD
        # ========================================
        for i_tile in range(0, m, tile_size):
            for j_tile in range(0, p, tile_size):
                for k_tile in range(0, n, tile_size):
                    var i_end = min(i_tile + tile_size, m)
                    var j_end = min(j_tile + tile_size, p)
                    var k_end = min(k_tile + tile_size, n)

                    for i in range(i_tile, i_end):
                        var a_row_base = i * A_stride0 + A_offset
                        var c_row_base = i * C_stride0 + C_offset

                        # Manual SIMD vectorization
                        var j = j_tile

                        # Main vectorized loop
                        while j + simdwidth <= j_end:
                            var c_addr = c_row_base + j * C_stride1
                            var accumulator = C_data.load[width=simdwidth](
                                c_addr
                            )

                            for k in range(k_tile, k_end):
                                var a_addr = a_row_base + k * A_stride1
                                var a_ik = A_data[a_addr]

                                var b_addr = k * B_stride0 + j * B_stride1
                                var b_vec = B_data.load[width=simdwidth](b_addr)

                                accumulator += a_ik * b_vec

                            C_data.store[width=simdwidth](c_addr, accumulator)
                            j += simdwidth

                        # Tail handling: remaining columns
                        while j < j_end:
                            var c_addr = c_row_base + j * C_stride1
                            var accumulator = C_data[c_addr]

                            for k in range(k_tile, k_end):
                                var a_addr = a_row_base + k * A_stride1
                                var b_addr = k * B_stride0 + j * B_stride1
                                accumulator += A_data[a_addr] * B_data[b_addr]

                            C_data[c_addr] = accumulator
                            j += 1

        return C^

    @staticmethod
    @always_inline
    fn forward[
        simdwidth: Int = simd_width_of[Self.dtype](), tile_size: Int = TILE_SIZE
    ](A: Gradbox[Self.dtype], B: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        var m = A.shape()[0]
        var n = A.shape()[1]
        var p = B.shape()[1]

        var C = Gradbox[Self.dtype].zeros(Shape([m, p]))
        var contiguous = B.is_contiguous()

        if contiguous:
            # Hoist metadata
            ref A_strides = A.strides()
            var A_stride0 = A_strides[0]
            var A_stride1 = A_strides[1]
            var A_data = A.buffer.buffer.data

            ref B_strides = B.strides()
            var B_stride0 = B_strides[0]
            var B_stride1 = B_strides[1]
            var B_offset = B.offset()
            var B_data = B.buffer.buffer.data

            ref C_strides = C.strides()
            var C_stride0 = C_strides[0]
            var C_stride1 = C_strides[1]
            var C_data = C.buffer.buffer.data

            # ========================================
            # TILED MATMUL with Manual SIMD
            # ========================================
            for i_tile in range(0, m, tile_size):
                for j_tile in range(0, p, tile_size):
                    for k_tile in range(0, n, tile_size):
                        var i_end = min(i_tile + tile_size, m)
                        var j_end = min(j_tile + tile_size, p)
                        var k_end = min(k_tile + tile_size, n)

                        # Process tile
                        for i in range(i_tile, i_end):
                            var a_row_base = i * A_stride0
                            var c_row_base = i * C_stride0

                            # Manual SIMD vectorization
                            var j = j_tile

                            # Main vectorized loop
                            while j + simdwidth <= j_end:
                                var c_addr = c_row_base + j * C_stride1
                                var accumulator = C_data.load[width=simdwidth](
                                    c_addr
                                )

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    var a_ik = A_data[a_addr]

                                    var b_addr = (
                                        k * B_stride0 + B_offset + j * B_stride1
                                    )
                                    var b_vec = B_data.load[width=simdwidth](
                                        b_addr
                                    )

                                    accumulator += a_ik * b_vec

                                C_data.store[width=simdwidth](
                                    c_addr, accumulator
                                )
                                j += simdwidth

                            # Tail handling: remaining columns
                            while j < j_end:
                                var c_addr = c_row_base + j * C_stride1
                                var accumulator = C_data[c_addr]

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    var b_addr = (
                                        k * B_stride0 + B_offset + j * B_stride1
                                    )
                                    accumulator += (
                                        A_data[a_addr] * B_data[b_addr]
                                    )

                                C_data[c_addr] = accumulator
                                j += 1
        else:
            # Non-contiguous fallback (scalar path)
            ref A_strides = A.strides()
            var A_stride0 = A_strides[0]
            var A_stride1 = A_strides[1]
            var A_data = A.buffer.buffer.data

            ref B_strides = B.strides()
            var B_stride0 = B_strides[0]
            var B_stride1 = B_strides[1]
            var B_offset = B.offset()
            var B_data = B.buffer.buffer.data

            ref C_strides = C.strides()
            var C_stride0 = C_strides[0]
            var C_stride1 = C_strides[1]
            var C_data = C.buffer.buffer.data

            for i in range(m):
                var a_row_base = i * A_stride0
                var c_row_base = i * C_stride0

                for j in range(p):
                    var accumulator: Scalar[dtype] = 0

                    for k in range(n):
                        var a_addr = a_row_base + k * A_stride1
                        var b_addr = k * B_stride0 + B_offset + j * B_stride1
                        accumulator += A_data[a_addr] * B_data[b_addr]

                    var c_addr = c_row_base + j * C_stride1
                    C_data[c_addr] = accumulator

        return C^


@fieldwise_init
@register_passable
struct MatmulNdBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_MATMUL_ND

    fn backward[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](self, output: Tensor[Self.dtype]) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        ref grad_out = output.gradients()[]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        ref A_shape = A.shape()
        ref B_shape = B.shape()
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        if A.requires_grad:
            var B_transposed = B.transpose[track_grad=False](
                axes=[-1, -2]
            ).contiguous[track_grad=False]()

            var A_batch_grad = MatmulNd[Self.dtype].forward(
                grad_out, B_transposed
            )
            var final_grad_A = A_batch_grad^.sum_over_broadcasted_axes(A_shape)

            results.append((A, final_grad_A^, AddTensor))

        if B.requires_grad:
            var A_transposed = A.transpose[track_grad=False](axes=[-1, -2])
            var B_batch_grad = MatmulNd[Self.dtype].forward(
                A_transposed, grad_out
            )

            var final_grad_B = B_batch_grad^.sum_over_broadcasted_axes(B_shape)

            results.append((B^, final_grad_B^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct MatmulNd[dtype: DType](ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True, simdwidth: Int = simd_width_of[Self.dtype]()
    ](mut A: Tensor[Self.dtype], mut B: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()

        # Short-circuit for pure 2D case
        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward[track_grad](A, B)

        MatrixShapeValidator.validate_matrix_shapes_nd(A_shape, B_shape)
        # shapes: batch + [m, k], batch + [k, n]
        batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )  # all dims except last 2

        m = A_shape[-2]
        n = B_shape[-1]

        batch_dims_a = A_shape[:-2]
        batch_dims_b = B_shape[:-2]

        out_shape = batch_shape + [m, n]
        C = Tensor[dtype].zeros(out_shape)

        for indices in batch_shape:
            # select batch slices
            A_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            B_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )
            A_slice = A.__getitem__[track_grad=False](il(A_indices), s(), s())
            B_slice = B.__getitem__[track_grad=False](il(B_indices), s(), s())
            C_slice = C.__getitem__[track_grad=False](il(indices), s(), s())

            result = Matmul2d[Self.dtype].forward[track_grad=False](
                A_slice,
                B_slice,
            )
            C_slice.buffer.copy_from_alike[overwrite=True](result.buffer)

        # Only attach backward handler if gradients are needed
        @parameter
        if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backward_fn = MatmulNdBackward[
                    Self.dtype
                ]().into_backward_fn()
                C.backwardFn = Optional(backward_fn^)
                C.add_ancestry(A)
                C.add_ancestry(B)

        return C^

    @always_inline
    @staticmethod
    fn forward(
        mut A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()
        # Short-circuit for 2D
        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward(A, B)

        MatrixShapeValidator.validate_matrix_shapes_nd(A_shape, B_shape)

        # shapes: batch + [m, k], batch + [k, n]
        batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )
        m = A_shape[-2]
        n = B_shape[-1]
        batch_dims_a = A_shape[:-2]
        batch_dims_b = B_shape[:-2]

        out_shape = batch_shape + [m, n]
        var C = Gradbox[dtype].zeros(
            out_shape, share=True
        )  # Views are not allowed from unshared gradbox

        for indices in batch_shape:
            var A_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            var B_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )

            var A_slice = A.__getitem__[track_grad=False](
                il(A_indices), s(), s()
            )
            var B_slice = B[il(B_indices), s(), s()]
            var C_slice = C[il(indices), s(), s()]

            # Use 2D matmul for GradBox (no backward needed)
            var result = Matmul2d[Self.dtype].forward(
                A_slice, B_slice
            )  # Backward fn would not be attached
            C_slice.buffer.copy_from_alike[overwrite=False](result.buffer^)

        return C^

    # Gradbox and Tensor matmul_nd - No backward fn needed
    @always_inline
    @staticmethod
    fn forward(
        A: Gradbox[Self.dtype], mut B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()

        # Short-circuit for 2D
        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward(A, B)

        MatrixShapeValidator.validate_matrix_shapes_nd(A_shape, B_shape)

        batch_shape = ShapeBroadcaster.broadcast_shape(
            A_shape[0:-2], B_shape[0:-2]
        )
        m = A_shape[-2]
        n = B_shape[-1]
        batch_dims_a = A_shape[:-2]
        batch_dims_b = B_shape[:-2]

        out_shape = batch_shape + [m, n]
        var C = Gradbox[dtype].zeros(out_shape, share=True)

        for indices in batch_shape:
            var A_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            var B_indices = ShapeBroadcaster.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )

            var A_slice = A[il(A_indices), s(), s()]
            var B_slice = B.__getitem__[track_grad=False](
                il(B_indices), s(), s()
            )
            var C_slice = C[il(indices), s(), s()]

            # Use 2D matmul for GradBox (no backward needed)
            var result = Matmul2d[Self.dtype].forward(A_slice, B_slice)
            C_slice.buffer.copy_from_alike[overwrite=False](result.buffer^)

        return C^


@fieldwise_init
@register_passable
struct Matmul[dtype: DType](ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True, mode: Int = mm
    ](mut A: Tensor[Self.dtype], mut B: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        @parameter
        if mode == mm:
            # Step 1: Pure analysis - get the opcode
            var opcode = classify_matmul(A.shape(), B.shape())

            # Step 2: Simple dispatch based on opcode

            if dot == opcode:
                return A.dot[track_grad](B)

            if vm == opcode:
                return VectorMatmulNd[Self.dtype].forward[track_grad](A, B)

            if mv == opcode:
                return MatrixVectorMulNd[Self.dtype].forward[track_grad](A, B)

            if mm == opcode:
                return MatmulNd[Self.dtype].forward[track_grad](A, B)

            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[Self.dtype].scalar(0)

        elif mode == dot:
            return A.dot[track_grad](B)

        elif mode == vm:
            return VectorMatmulNd[Self.dtype].forward[track_grad](A, B)

        elif mode == mv:
            return MatrixVectorMulNd[Self.dtype].forward[track_grad](A, B)
        else:
            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[Self.dtype].scalar(0)

    @always_inline
    @staticmethod
    fn forward(
        mut A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return MatmulNd[Self.dtype].forward(A, B)

    @always_inline
    @staticmethod
    fn forward(
        A: Gradbox[Self.dtype], mut B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return MatmulNd[Self.dtype].forward(A, B)


# alias dot = 0  # dot product
# alias vm = 1  # vector & tensor matmul
# alias mv = 2  # tensor & vector matmul
# alias mm = 3  # tensor & tensor matmul
# alias invalid = 4  # Invalid case


fn classify_matmul(a: Shape, b: Shape) -> Int:
    var rank_a = a.rank()
    var rank_b = b.rank()

    if rank_a <= 1 and rank_b <= 1:
        return dot
    elif rank_a == 1 and rank_b >= 2:
        return vm
    elif rank_a >= 2 and rank_b == 1:
        return mv
    else:  # rank_a >= 2 and rank_b >= 2
        if a[-1] == b[-2]:
            return mm
        else:
            return invalid


fn main() raises:
    pass
