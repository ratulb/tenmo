from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import BackwardFn, Integer, BackwardFnType

from .gradbox import Gradbox
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .shared.panic import panic
from std.sys import has_accelerator, simd_width_of
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .kernels.cumsum_kernel import CumsumKernel


@fieldwise_init
struct CumsumBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var axis = output.ancestry().backward_fn().get[Integer]().value
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)

        var grad_ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if gradbox.is_on_gpu():
                var shape = gradbox.shape()
                var rank = shape.rank()
                var outer: Int = 1
                for i in range(axis):
                    outer *= shape[i]
                var axis_size = shape[axis]
                var inner: Int = 1
                for i in range(axis + 1, rank):
                    inner *= shape[i]
                try:
                    var result = CumsumKernel[Self.dtype].launch_backward(
                        gradbox.buffer().layout(),
                        gradbox.buffer().device_state.value(),
                        axis,
                        outer,
                        axis_size,
                        inner,
                    )
                    grad_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0], result[1]
                    )
                except e:
                    print(e)
                    panic("CumsumBackward → GPU backward launch failed")
                    grad_ndb = NDBuffer[Self.dtype].Empty()
            else:
                grad_ndb = cumsum_backward_cpu[Self.dtype](
                    gradbox.buffer(), axis
                )
        else:
            grad_ndb = cumsum_backward_cpu[Self.dtype](gradbox.buffer(), axis)
        var gradbox_ancestor = Gradbox[Self.dtype](grad_ndb^)

        if parent.requires_grad:
            parent.update_grad(gradbox_ancestor^, AddTensor, None)
        parent_ids.append(parent._id)

        gradbox.zero_grad()


@fieldwise_init
struct Cumsum[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        var norm_axis = axis
        if norm_axis < 0:
            norm_axis = rank + norm_axis
        if norm_axis < 0 or norm_axis >= rank:
            panic(
                "cumsum: invalid axis "
                + String(axis)
                + " for tensor of rank "
                + String(rank)
            )

        var ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if self.buffer.is_on_gpu():
                var outer: Int = 1
                for i in range(norm_axis):
                    outer *= shape[i]
                var axis_size = shape[norm_axis]
                var inner: Int = 1
                for i in range(norm_axis + 1, rank):
                    inner *= shape[i]
                try:
                    var result = CumsumKernel[Self.dtype].launch(
                        self.buffer.layout(),
                        self.buffer.device_state.value(),
                        norm_axis,
                        outer,
                        axis_size,
                        inner,
                        sync,
                    )
                    ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0], result[1]
                    )
                except e:
                    panic("cumsum GPU forward failed: " + String(e))
                    ndb = NDBuffer[Self.dtype].Empty()
            else:
                ndb = cumsum_cpu[Self.dtype](self.buffer, norm_axis)
        else:
            ndb = cumsum_cpu[Self.dtype](self.buffer, norm_axis)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.integer_arg[Self.dtype](
                    norm_axis, CumsumBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = False
                out.add_ancestry(backwardFn^, self)

        return out^


def cumsum_cpu[
    dtype: DType,
](inp: NDBuffer[dtype], axis: Int,) -> NDBuffer[dtype]:
    var shape = inp.shape
    var rank = shape.rank()
    var axis_size = shape[axis]
    var out = NDBuffer[dtype].zeros(shape)

    if axis_size <= 1:
        if axis_size == 1:
            out.copy_from_alike(inp)
        return out^

    var inner: Int = 1
    for i in range(axis + 1, rank):
        inner *= shape[i]

    var outer: Int = 1
    for i in range(axis):
        outer *= shape[i]

    if inp.is_contiguous():
        var in_ptr = inp.data_ptr()
        var in_offset = inp.offset
        var out_ptr = out.data_ptr()
        var out_offset = out.offset
        var seg_stride = axis_size * inner
        comptime simdwidth = (
            1 if dtype == DType.bool else simd_width_of[dtype]()
        )
        var full_chunks = inner // simdwidth
        var remainder = inner % simdwidth
        var n_threads = num_physical_cores()

        def cumsum_seg(o: Int) {imm}:
            var in_base = in_offset + o * seg_stride
            var out_base = out_offset + o * seg_stride

            # k == 0: out[o, 0, :] = in[o, 0, :]
            var off = 0
            for chunk in range(full_chunks):
                var v = in_ptr.unsafe_load[width=simdwidth](in_base + off)
                out_ptr.unsafe_store[width=simdwidth](out_base + off, v)
                off += simdwidth
            if remainder > 0:
                for j in range(remainder):
                    out_ptr[unsafe_offset=out_base + off + j] = in_ptr[
                        unsafe_offset=in_base + off + j
                    ]

            # k > 0: out[k] = out[k-1] + in[k]; lanes are independent
            for k in range(1, axis_size):
                var prev = out_base + (k - 1) * inner
                var cur = out_base + k * inner
                var sel = in_base + k * inner
                var off2 = 0
                for chunk in range(full_chunks):
                    var v = out_ptr.unsafe_load[width=simdwidth](
                        prev + off2
                    ) + in_ptr.unsafe_load[width=simdwidth](sel + off2)
                    out_ptr.unsafe_store[width=simdwidth](cur + off2, v)
                    off2 += simdwidth
                if remainder > 0:
                    for j in range(remainder):
                        var ii = off2 + j
                        out_ptr[unsafe_offset=cur + ii] = (
                            out_ptr[unsafe_offset=prev + ii]
                            + in_ptr[unsafe_offset=sel + ii]
                        )

        if outer >= n_threads and shape.numels() >= n_threads * 32768:
            parallelize(cumsum_seg, outer, n_threads)
        else:
            for o in range(outer):
                cumsum_seg(o)
    else:
        # Strided input: independent (outer, inner) lanes, each a serial
        # scan of length axis_size. Lane setup does the div/mod ONCE per
        # lane (amortized over axis_size elements); lanes parallelize.
        var in_ptr = inp.data_ptr()
        var in_base_off = inp.offset
        var ax_stride = inp.strides[axis]
        var out_ptr = out.data_ptr()
        var out_off = out.offset
        var lanes = outer * inner
        var n_threads = num_physical_cores()

        def cumsum_lane(l: Int) {imm}:
            var o = l // inner
            var ii = l % inner
            var in_base = in_base_off
            var rem_o = o
            for d in range(axis - 1, -1, -1):
                var sz = shape[d]
                var c = rem_o % sz
                rem_o //= sz
                in_base += c * inp.strides[d]
            var rem_i = ii
            for d in range(rank - 1, axis, -1):
                var sz = shape[d]
                var c = rem_i % sz
                rem_i //= sz
                in_base += c * inp.strides[d]
            var out_lane = out_off + o * axis_size * inner + ii
            var acc = in_ptr[unsafe_offset=in_base]
            out_ptr[unsafe_offset=out_lane] = acc
            for k in range(1, axis_size):
                acc += in_ptr[unsafe_offset=in_base + k * ax_stride]
                out_ptr[unsafe_offset=out_lane + k * inner] = acc

        if lanes >= n_threads and shape.numels() >= n_threads * 32768:
            parallelize(cumsum_lane, lanes, n_threads)
        else:
            for l in range(lanes):
                cumsum_lane(l)

    return out^


def cumsum_backward_cpu[
    dtype: DType,
](grad: NDBuffer[dtype], axis: Int,) -> NDBuffer[dtype]:
    var shape = grad.shape
    var rank = shape.rank()
    var axis_size = shape[axis]
    var out = NDBuffer[dtype].zeros(shape)

    if axis_size <= 1:
        if axis_size == 1:
            out.copy_from_alike(grad)
        return out^

    var inner: Int = 1
    for i in range(axis + 1, rank):
        inner *= shape[i]

    var outer: Int = 1
    for i in range(axis):
        outer *= shape[i]

    if grad.is_contiguous():
        var in_ptr = grad.data_ptr()
        var in_offset = grad.offset
        var out_ptr = out.data_ptr()
        var out_offset = out.offset
        var seg_stride = axis_size * inner
        comptime simdwidth = (
            1 if dtype == DType.bool else simd_width_of[dtype]()
        )
        var full_chunks = inner // simdwidth
        var remainder = inner % simdwidth
        var n_threads = num_physical_cores()

        def cumsum_backward_seg(o: Int) {imm}:
            var in_base = in_offset + o * seg_stride
            var out_base = out_offset + o * seg_stride

            # k == axis_size - 1: out[o, last, :] = in[o, last, :]
            var last = (axis_size - 1) * inner
            var off = 0
            for chunk in range(full_chunks):
                var v = in_ptr.unsafe_load[width=simdwidth](
                    in_base + last + off
                )
                out_ptr.unsafe_store[width=simdwidth](out_base + last + off, v)
                off += simdwidth
            if remainder > 0:
                for j in range(remainder):
                    out_ptr[unsafe_offset=out_base + last + off + j] = in_ptr[
                        unsafe_offset=in_base + last + off + j
                    ]

            # k < last: out[k] = in[k] + out[k+1]; lanes are independent
            for k in range(axis_size - 2, -1, -1):
                var cur = out_base + k * inner
                var nxt = out_base + (k + 1) * inner
                var sel = in_base + k * inner
                var off2 = 0
                for chunk in range(full_chunks):
                    var v = out_ptr.unsafe_load[width=simdwidth](
                        nxt + off2
                    ) + in_ptr.unsafe_load[width=simdwidth](sel + off2)
                    out_ptr.unsafe_store[width=simdwidth](cur + off2, v)
                    off2 += simdwidth
                if remainder > 0:
                    for j in range(remainder):
                        var ii = off2 + j
                        out_ptr[unsafe_offset=cur + ii] = (
                            out_ptr[unsafe_offset=nxt + ii]
                            + in_ptr[unsafe_offset=sel + ii]
                        )

        if outer >= n_threads and shape.numels() >= n_threads * 32768:
            parallelize(cumsum_backward_seg, outer, n_threads)
        else:
            for o in range(outer):
                cumsum_backward_seg(o)
    else:
        # Strided grad: reverse lane scans (out[k] = grad[k] + out[k+1]
        # runs as a reverse accumulation, so no successor reads and no
        # collected index list are needed). Lanes parallelize.
        var in_ptr = grad.data_ptr()
        var in_base_off = grad.offset
        var ax_stride = grad.strides[axis]
        var out_ptr = out.data_ptr()
        var out_off = out.offset
        var lanes = outer * inner
        var n_threads = num_physical_cores()

        def cumsum_backward_lane(l: Int) {imm}:
            var o = l // inner
            var ii = l % inner
            var in_base = in_base_off
            var rem_o = o
            for d in range(axis - 1, -1, -1):
                var sz = shape[d]
                var c = rem_o % sz
                rem_o //= sz
                in_base += c * grad.strides[d]
            var rem_i = ii
            for d in range(rank - 1, axis, -1):
                var sz = shape[d]
                var c = rem_i % sz
                rem_i //= sz
                in_base += c * grad.strides[d]
            var out_lane = out_off + o * axis_size * inner + ii
            var acc = Scalar[dtype](0)
            for k in range(axis_size - 1, -1, -1):
                acc += in_ptr[unsafe_offset=in_base + k * ax_stride]
                out_ptr[unsafe_offset=out_lane + k * inner] = acc

        if lanes >= n_threads and shape.numels() >= n_threads * 32768:
            parallelize(cumsum_backward_lane, lanes, n_threads)
        else:
            for l in range(lanes):
                cumsum_backward_lane(l)

    return out^
