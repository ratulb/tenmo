from .tensor import Tensor
from .shared.mnemonics import AddTensor, LOG_BACKWARD
from .backpropagation import BackwardFn, ScalarArg, BackwardFnType

from .gradbox import Gradbox
from .shared.constants import Epsilon
from .ancestry import Ancestor


@fieldwise_init
struct LogBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var epsilon = (
            output.ancestry().backward_fn().get[ScalarArg[Self.dtype]]().value
        )
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var result_ndb = parent.buffer().arithmetic_ops[LOG_BACKWARD](
            gradbox.buffer(), epsilon
        )
        var parent_gradbox = Gradbox[Self.dtype](result_ndb^)

        parent.update_grad(parent_gradbox^, AddTensor, None)

        parent_ids.append(parent._id)
        gradbox.zero_grad()


@fieldwise_init
struct Logarithm[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var result_ndb = self.buffer.log[epsilon]()
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.scalar_arg[Self.dtype](
                    epsilon, LogBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True

                out.add_ancestry(backwardFn^, self)

        return out^
