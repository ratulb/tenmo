from tenmo import Tensor
from shapes import Shape
from strides import Strides
from backpropagation import BackwardFnArg, ArgumentType, BACKWARD_VIEW
from mnemonics import AddTensor, ZeroGrad
from validators import Validator
from gradbox import Gradbox
from intarray import IntArray
from common_utils import panic, log_warning
from std.sys import simd_width_of
from device import DeviceState
from std.sys import has_accelerator
from ndbuffer import NDBuffer


@fieldwise_init
struct ViewBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(output: Tensor[Self.dtype]) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:

        comptime if has_accelerator():
            if output.is_on_gpu():
                return Self.backward_gpu(output)
        return Self.backward_cpu(output)

    @staticmethod
    fn backward_cpu(output: Tensor[Self.dtype]) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        var (shape, strides, offset) = output.bwd_fn_arg().arg[Tuple[Shape, Strides, Int]]
        var parent = output.ancestry().get(0)
        ref gradbox = output.gradients()[]

        var parent_shape = parent.shape()
        ref parent_strides = parent.strides()
        var parent_offset = parent.offset()
        var parent_max_index = parent.max_index()
        var parent_gradbox = Gradbox[Self.dtype].zeros(parent_shape)

        # Special case: scalar parent
        if parent_shape.rank() == 0:
            parent_gradbox[IntArray()] = gradbox.item()
        else:
            var view_rank = shape.rank()
            var parent_rank = parent_shape.rank()

            var view_data = gradbox.data_ptr()
            var view_offset = gradbox.offset()
            ref view_strides = gradbox.strides()

            var parent_grad_data = parent_gradbox.data_ptr()
            var parent_grad_offset = parent_gradbox.offset()
            ref parent_grad_strides = parent_gradbox.strides()

            var use_fast_path = shape == parent_shape

            if use_fast_path:
                var numel = shape.num_elements()
                for i in range(numel):
                    parent_grad_data[parent_grad_offset + i] += view_data[
                        view_offset + i
                    ]
            else:
                var position = IntArray.with_capacity(parent_rank)

                for child_coord in shape:
                    position.clear()
                    var abs_index = offset
                    for i in range(view_rank):
                        abs_index += strides[i] * child_coord[i]

                        if (
                            abs_index < parent_offset
                            or abs_index > parent_max_index
                        ):
                            continue

                    var parent_rel_index = abs_index - parent_offset
                    var remaining = parent_rel_index
                    var valid = True

                    for i in range(parent_rank):
                        var stride = parent_strides[i]
                        if stride == 0:
                            position.append(0)
                        else:
                            position.append(remaining // stride)
                            if (
                                position[i] < 0
                                or position[i] >= parent_shape[i]
                            ):
                                valid = False
                                break
                            remaining = remaining % stride

                    if valid and remaining == 0:
                        var parent_addr = parent_grad_offset
                        for i in range(parent_rank):
                            parent_addr += position[i] * parent_grad_strides[i]

                        var view_addr = view_offset
                        for i in range(view_rank):
                            view_addr += child_coord[i] * view_strides[i]

                        parent_grad_data[parent_addr] += view_data[view_addr]

        output.zero_grad()
        return [(parent^, parent_gradbox^, AddTensor)]

    @staticmethod
    fn backward_gpu(output: Tensor[Self.dtype]) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        var (shape, strides, offset) = output.bwd_fn_arg().arg[Tuple[Shape, Strides, Int]]
        var parent = output.ancestry().get(0)
        ref gradbox = output.gradients()[]  # GPU gradbox

        var parent_shape = parent.shape()
        var parent_offset = parent.offset()
        var parent_max_index = parent.max_index()

        # ── Materialise view gradbox from GPU to CPU ──────────────────────────
        # DeviceState.into() maps GPU buffer to host contiguously.
        # But we need the raw layout (shape + strides + offset) preserved
        # so the CPU backward logic can do correct coordinate mapping.
        # We achieve this by constructing a CPU NDBuffer that shares the
        # materialised data with the correct shape/strides/offset metadata.
        var cpu_gradbox: Gradbox[Self.dtype]
        try:
            # Materialise entire GPU DeviceBuffer to CPU — raw flat copy
            var ds = gradbox.buffer.device_state.value()
            var cpu_ndb_flat = ds.into(Shape(len(ds)))
            # Re-attach the view's logical shape/strides/offset
            # so backward_cpu coordinate mapping works correctly
            var cpu_ndb = cpu_ndb_flat.share(
                shape, Strides.default(shape), offset
            )
            cpu_gradbox = Gradbox[Self.dtype](cpu_ndb^, share=False)
        except e:
            panic(
                "ViewBackward backward_gpu: failed to materialise GPU gradbox:"
                + String(e)
            )
            # Unreachable — satisfies compiler
            cpu_gradbox = Gradbox[Self.dtype].zeros(shape)

        # ── Run CPU backward logic on materialised gradbox ────────────────────

        var parent_gradbox = Gradbox[Self.dtype].zeros(parent_shape)

        if parent_shape.rank() == 0:
            parent_gradbox[IntArray()] = cpu_gradbox.item()
        else:
            var view_rank = shape.rank()
            var parent_rank = parent_shape.rank()

            ref parent_strides = parent.strides()
            var parent_grad_offset = parent_gradbox.offset()
            ref parent_grad_strides = parent_gradbox.strides()
            var parent_grad_data = parent_gradbox.data_ptr()

            var view_data = cpu_gradbox.data_ptr()
            var view_offset = cpu_gradbox.offset()
            ref view_strides = cpu_gradbox.strides()

            var use_fast_path = shape == parent_shape

            if use_fast_path:
                var numel = shape.num_elements()
                for i in range(numel):
                    parent_grad_data[parent_grad_offset + i] += view_data[
                        view_offset + i
                    ]
            else:
                var position = IntArray.with_capacity(parent_rank)

                for child_coord in shape:
                    position.clear()
                    var abs_index = offset
                    for i in range(view_rank):
                        abs_index += strides[i] * child_coord[i]

                        if (
                            abs_index < parent_offset
                            or abs_index > parent_max_index
                        ):
                            continue

                    var parent_rel_index = abs_index - parent_offset
                    var remaining = parent_rel_index
                    var valid = True

                    for i in range(parent_rank):
                        var stride = parent_strides[i]
                        if stride == 0:
                            position.append(0)
                        else:
                            position.append(remaining // stride)
                            if (
                                position[i] < 0
                                or position[i] >= parent_shape[i]
                            ):
                                valid = False
                                break
                            remaining = remaining % stride

                    if valid and remaining == 0:
                        var parent_addr = parent_grad_offset
                        for i in range(parent_rank):
                            parent_addr += position[i] * parent_grad_strides[i]

                        var view_addr = view_offset
                        for i in range(view_rank):
                            view_addr += child_coord[i] * view_strides[i]

                        parent_grad_data[parent_addr] += view_data[view_addr]

        # ── Move parent_gradbox to GPU if parent is on GPU ────────────────────
        var final_gradbox: Gradbox[Self.dtype]
        if parent.is_on_gpu():
            try:
                var gpu = parent.buffer.device_state.value().get_gpu()
                var ds = DeviceState[Self.dtype](
                    parent_gradbox.buffer.numels(), gpu
                )
                ds.fill(parent_gradbox.buffer)
                var gpu_ndb = NDBuffer[Self.dtype].with_device_state(
                    ds^, parent_shape
                )
                final_gradbox = Gradbox[Self.dtype](gpu_ndb^, share=False)
            except e:
                panic(
                    "ViewBackward backward_gpu: failed to move parent_gradbox"
                    " to GPU: "
                    + String(e)
                )
                final_gradbox = parent_gradbox  # unreachable
        else:
            # Parent is CPU — use CPU gradbox directly
            final_gradbox = parent_gradbox^

        output.zero_grad()
        return [(parent^, final_gradbox^, AddTensor)]

@fieldwise_init
struct View[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[Self.dtype]:
        var abs_offset: Int
        var abs_strides: Strides

        if not validated:
            (abs_offset, abs_strides) = Validator.validate_view_params(
                tensor.buffer.size(), shape, strides, offset
            )
        else:
            abs_offset = offset
            abs_strides = strides
        # At this point, NDBuffer -> Buffer would be shared, ref counted if not already so
        var shared_ndb = tensor.buffer.share(shape, abs_strides, abs_offset)
        var out = Tensor[Self.dtype](shared_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var bwd_fn_arg = BackwardFnArg[Self.dtype](BACKWARD_VIEW,
                    ArgumentType[Self.dtype]((shape, abs_strides, abs_offset)
                ))
                out.bwdFnArg = Optional(bwd_fn_arg^)
                out.add_ancestry(tensor)

        return out^


fn main() raises:
    comptime dtype = DType.float32
    a = Tensor[dtype].arange(10, requires_grad=True)
    b = a.view(Shape(2, 5))
    c = b * 42
    c.backward()
    a.grad().print()
    pass
