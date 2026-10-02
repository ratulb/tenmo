from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import BackwardFn, BackwardFnType

from std.sys import has_accelerator
from .shared.shapes import Shape
from .validators import Validator
from .ndbuffer import NDBuffer
from .shared.strides import Strides
from .shared.indexhelper import IndexCalculator
from .ancestry import Ancestor


@fieldwise_init
struct ReshapeBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var reshaped = gradbox.reshape(ancestor.shape())
        if ancestor.requires_grad:
            ancestor.update_grad(reshaped^, AddTensor, None)
        parent_ids.append(ancestor._id)
        # View conduit: grad already forwarded to base above; always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Reshape[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            tensor.shape(), new_shape.intarray()
        )
        var strides = Strides.default(shape)
        # Use view (share) if the source is contiguous and the new shape
        # (including the source offset) fits in the underlying buffer,
        # otherwise materialize (contiguous). The contiguity check is
        # required: a non-contiguous source (transpose, negative stride,
        # stride-0 expand/tile) cannot be re-read with default contiguous
        # strides without changing the logical order. The source offset is
        # preserved so a dense-offset slice keeps pointing at its data.
        var src_offset = tensor.offset()
        var buffer_size: Int
        comptime if has_accelerator():
            if tensor.buffer.is_on_gpu():
                buffer_size = len(tensor.buffer.device_state.value())
            else:
                buffer_size = len(tensor.buffer.buffer)
        else:
            buffer_size = len(tensor.buffer.buffer)
        var ndb: NDBuffer[Self.dtype]
        if (
            tensor.buffer.is_contiguous()
            and IndexCalculator.max_storage_index(shape, strides, src_offset) < buffer_size
        ):
            ndb = tensor.buffer.share(shape, strides, offset=src_offset)
        else:
            ndb = tensor.buffer.contiguous(shape)

        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    ReshapeBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, tensor)

        return out^
