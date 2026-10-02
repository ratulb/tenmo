from .tensor import Tensor
from .backpropagation import BackwardFn, BackwardFnType

from .shared.mnemonics import AddTensor
from .ancestry import Ancestor


@fieldwise_init
struct FlattenBackward[dtype: DType](
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
        var ancestor_shape = ancestor.shape()
        var reshaped_grad = gradbox.reshape(ancestor_shape)
        if ancestor.requires_grad:
            ancestor.update_grad(reshaped_grad^, AddTensor, None)
        parent_ids.append(ancestor._id)
        gradbox.zero_grad()


struct FlattenForward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        var out = Tensor[Self.dtype](flattened_buffer^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    FlattenBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^
