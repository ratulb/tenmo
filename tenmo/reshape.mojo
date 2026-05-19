from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, BACKWARD_RESHAPE
from .shapes import Shape
from .validators import Validator
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .strides import Strides
from .ancestry import Ancestor


@fieldwise_init
struct ReshapeBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var reshaped = gradbox.reshape(ancestor.buffer().shape)
        if ancestor.requires_grad:
            ancestor.update_grad(reshaped^, AddTensor, None)
        parent_ids.append(ancestor._id)
        gradbox.zero_grad()


@fieldwise_init
struct Reshape[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[Self.dtype]:
        var shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            tensor.shape(), new_shape.intarray()
        )
        var strides = Strides.default(shape)
        # Use view (share) if the new shape fits in the underlying buffer,
        # otherwise materialize (contiguous). This handles cases like expand+tile
        # where stride=0 creates more logical elements than physical storage.
        var buffer_size: Int
        comptime if has_accelerator():
            if tensor.buffer.is_on_gpu():
                buffer_size = len(tensor.buffer.device_state.value())
            else:
                buffer_size = len(tensor.buffer.buffer)
        else:
            buffer_size = len(tensor.buffer.buffer)
        var ndb: NDBuffer[Self.dtype]
        if IndexCalculator.max_index(shape, strides, 0) < buffer_size:
            ndb = tensor.buffer.share(shape, strides, offset=0)
        else:
            ndb = tensor.buffer.contiguous(shape)

        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_RESHAPE
                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^
