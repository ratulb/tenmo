from .tensor import Tensor
from .shared.shapes import Shape
from .shared.strides import Strides
from .backpropagation import BackwardFn, BackwardFnType, ArgumentType

from .shared.mnemonics import AddTensor
from .validators import Validator
from .gradbox import Gradbox
from .shared.intarray import IntArray
from .shared.indexhelper import Idx
from .shared.panic import panic
from .gpu.device import (
    DeviceState,
)
from std.sys import has_accelerator
from .ndbuffer import NDBuffer
from .ancestry import Ancestor


@fieldwise_init
struct ViewArg(ArgumentType):
    """Forward-view descriptor stored in the graph for ViewBackward.

    `shape`/`strides` are the view's logical geometry; `offset` is the
    view's first logical element as an ABSOLUTE index into the shared
    storage buffer — never relative to another view. All offsets in the
    view machinery (ViewArg, NDBuffer, Gradbox) follow this convention,
    so offsets from different nodes are directly comparable.
    """

    var shape: Shape
    var strides: Strides
    var offset: Int


@fieldwise_init
struct ViewBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        comptime if has_accelerator():
            if output.gradients().is_on_gpu():
                Self.backward_gpu(output, parent_ids)
                ref gradbox = output.gradients()
                # View conduit: always cleared.
                gradbox.zero_grad()
                return
        Self.backward_cpu(output, parent_ids)
        ref gradbox = output.gradients()
        # View conduit: always cleared.
        gradbox.zero_grad()

    @staticmethod
    def backward_cpu(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var parent_ref = output.ancestry().get(0)

        if parent_ref.requires_grad:
            var parent_gradbox = Self.parent_gradbox_cpu(output)
            parent_ref.update_grad(parent_gradbox^, AddTensor, None)
        parent_ids.append(parent_ref._id)

    @staticmethod
    def backward_gpu(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref shape = output.ancestry().backward_fn().get[ViewArg]().shape
        var parent_ref = output.ancestry().get(0)
        ref gradbox = output.gradients()  # GPU gradbox

        # The re-attach below (share with default strides, offset 0) is
        # valid only because view gradboxes are always contiguous with
        # zero offset — fail loudly instead of silently mis-scattering if
        # that invariant ever changes. Offsets are absolute storage-buffer
        # indices (see ViewArg), so offset=0 here means "the materialised
        # copy starts at the view's first logical element".
        debug_assert(
            gradbox.is_contiguous() and gradbox.offset() == 0,
            "ViewBackward → backward_gpu: view gradbox must be contiguous,"
            " offset 0",
        )

        var parent_shape = parent_ref.shape()

        # Materialise view gradbox from GPU to CPU
        # DeviceState.into() maps GPU buffer to host contiguously.
        # But we need the raw layout (shape + strides + offset) preserved
        # so the CPU backward logic can do correct coordinate mapping.
        # We achieve this by constructing a CPU NDBuffer that shares the
        # materialised data with the correct shape/strides/offset metadata.
        var cpu_gradbox: Gradbox[Self.dtype]
        try:
            # Materialise entire GPU DeviceBuffer to CPU — raw flat copy
            var ds = gradbox.buffer().device_state.value()
            var cpu_ndb_flat = NDBuffer[Self.dtype].from_device_state(
                ds, Shape(len(ds))
            )
            # Re-attach the view's logical shape/strides
            # so backward_cpu coordinate mapping works correctly.
            # Use offset=0 because the materialized data starts at the view's
            # first element — the view's Gradbox is a fresh contiguous allocation,
            # not a zero-copy slice of the parent's buffer.
            var cpu_ndb = cpu_ndb_flat.share(shape, Strides.default(shape), 0)
            cpu_gradbox = Gradbox[Self.dtype](cpu_ndb^)
        except e:
            panic(
                "ViewBackward backward_gpu: failed to materialise GPU gradbox:"
                + String(e)
            )
            # Unreachable — satisfies compiler
            cpu_gradbox = Gradbox[Self.dtype].zeros(shape)

        # Run CPU backward logic on materialised gradbox

        # var parent_gradbox = Gradbox[Self.dtype].zeros(parent_shape)
        var parent_gradbox = Self.parent_gradbox_cpu(output, cpu_gradbox^)

        # Move parent_gradbox to GPU if parent is on GPU
        var final_gradbox: Gradbox[Self.dtype]
        if parent_ref.is_on_gpu():
            try:
                var gpu = parent_ref.ndb.value().device_state.value().get_gpu()
                var ds = DeviceState[Self.dtype](
                    parent_gradbox.buffer().numels(), gpu
                )
                parent_gradbox.buffer().fill_device_state(ds)
                var gpu_ndb = NDBuffer[Self.dtype].with_device_state(
                    ds^, parent_shape
                )
                final_gradbox = Gradbox[Self.dtype](gpu_ndb^)
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

        if parent_ref.requires_grad:
            parent_ref.update_grad(final_gradbox^, AddTensor, None)
        parent_ids.append(parent_ref._id)

    @staticmethod
    @always_inline
    def parent_gradbox_cpu(
        output: Ancestor[Self.dtype],
        materialized_gradbox: Optional[Gradbox[Self.dtype]] = None,
    ) -> Gradbox[Self.dtype]:
        var parent_ref = output.ancestry().get(0)
        ref bwd_arg = output.ancestry().backward_fn().get[ViewArg]()
        var (shape, strides, offset) = (
            bwd_arg.shape,
            bwd_arg.strides,
            bwd_arg.offset,
        )
        ref gradbox = materialized_gradbox.or_else(output.gradients())

        ref parent_shape = parent_ref.shape()
        ref parent_strides = parent_ref.strides()
        var parent_offset = parent_ref.offset()
        var parent_gradbox = Gradbox[Self.dtype].zeros(parent_shape)

        var parent_grad_data = parent_gradbox.data_ptr()

        # Special case: scalar parent
        if parent_shape.rank() == 0:
            parent_grad_data[unsafe_offset=0] = gradbox.item()
        else:
            var view_rank = shape.rank()
            var parent_rank = parent_shape.rank()

            var view_data = gradbox.data_ptr()
            var view_offset = gradbox.offset()
            var view_strides = gradbox.strides()

            var parent_grad_offset = parent_gradbox.offset()
            var parent_grad_strides = parent_gradbox.strides()

            # Fast path is valid ONLY for identical layouts. Same shape is
            # not enough: a same-shape view with different strides (e.g. a
            # transposed square) or a different offset maps logical
            # positions to different storage slots, so a flat elementwise
            # add would scatter grads onto the wrong parents. Layouts are
            # compared directly — sound because offsets on both sides are
            # absolute storage-buffer indices (see ViewArg).
            var use_fast_path = (
                shape == parent_shape
                and strides == parent_ref.strides()
                and offset == parent_ref.offset()
            )

            if use_fast_path:
                var numel = shape.num_elements()
                for i in range(numel):
                    parent_grad_data[
                        unsafe_offset=parent_grad_offset + i
                    ] += view_data[unsafe_offset=view_offset + i]
            else:
                # General scatter: for each logical coordinate of the view,
                # map view-logical -> absolute storage index (view offset +
                # strides, all absolute w.r.t. the shared storage buffer) ->
                # parent-relative index -> parent logical position (decomposed
                # with the parent's strides/offset) -> accumulate the view's
                # grad into the fresh parent gradbox. Stride-0 dims
                # (broadcast/expand) collapse to position 0. A coordinate is
                # skipped unless it decomposes exactly (valid and no
                # remainder), which is also what filters out-of-parent
                # indices.
                var position = IntArray.with_capacity(parent_rank)

                for child_coord in shape:
                    position.clear()
                    var abs_index = offset
                    for i in range(view_rank):
                        abs_index += strides[i] * child_coord[i]

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

                        parent_grad_data[
                            unsafe_offset=parent_addr
                        ] += view_data[unsafe_offset=view_addr]

        return parent_gradbox^


@fieldwise_init
struct View[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var abs_offset: Int
        var abs_strides: Strides

        if not validated:
            var storage_size: Int
            comptime if has_accelerator():
                if tensor.is_on_gpu():
                    storage_size = len(tensor.buffer.device_state.value())
                else:
                    storage_size = tensor.buffer.size()
            else:
                storage_size = tensor.buffer.size()
            (abs_offset, abs_strides) = Validator.validate_view_params(
                storage_size, shape, strides, offset
            )
        else:
            # validated=True contract: offset/strides are ALREADY absolute
            # storage-frame (e.g. Tensor.offset()/strides()). A relative
            # offset here silently mispoints the view — do not pass
            # validated=True with anything else.
            abs_offset = offset
            abs_strides = strides
        # NDBuffer.share builds a shared buffer view (read-only op)
        var shared_ndb = tensor.buffer.share(shape, abs_strides, abs_offset)
        var out = Tensor[Self.dtype](shared_ndb^, requires_grad=False)

        var view_arg = ViewArg(shape, abs_strides^, abs_offset)
        var grad_required = requires_grad.or_else(tensor.requires_grad)
        return Self.attach_bwd_hook_if_reqd_and_ret[track_grad](
            tensor, out^, view_arg^, grad_required
        )

    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        *slices: Slice,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var ndb = tensor.buffer.__getitem__(*slices)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        var view_arg = ViewArg(out.shape(), out.strides(), out.offset())
        var grad_required = requires_grad.or_else(tensor.requires_grad)
        return Self.attach_bwd_hook_if_reqd_and_ret[track_grad](
            tensor, out^, view_arg^, grad_required
        )

    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        tensor: Tensor[Self.dtype],
        *indices: Idx,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var ndb = tensor.buffer.__getitem__(*indices)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        var view_arg = ViewArg(out.shape(), out.strides(), out.offset())
        var grad_required = requires_grad.or_else(tensor.requires_grad)
        return Self.attach_bwd_hook_if_reqd_and_ret[track_grad](
            tensor, out^, view_arg^, grad_required
        )

    @always_inline
    @staticmethod
    def forward_list[
        track_grad: Bool = True,
    ](
        tensor: Tensor[Self.dtype],
        indices: List[Idx],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """List[Idx]-based forward (non-variadic) — used by the Python bindings.
        Identical semantics to the *indices: Idx overload above.
        """
        var ndb = tensor.buffer.view(indices)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        var view_arg = ViewArg(out.shape(), out.strides(), out.offset())
        var grad_required = requires_grad.or_else(tensor.requires_grad)
        return Self.attach_bwd_hook_if_reqd_and_ret[track_grad](
            tensor, out^, view_arg^, grad_required
        )

    @always_inline
    @staticmethod
    def attach_bwd_hook_if_reqd_and_ret[
        track_grad: Bool = True,
    ](
        parent: Tensor[Self.dtype],
        var out: Tensor[Self.dtype],
        var view_arg: ViewArg,
        requires_grad: Bool,
    ) -> Tensor[Self.dtype]:
        comptime if track_grad:
            if requires_grad:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    view_arg^,
                    ViewBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, parent)

        return out^
