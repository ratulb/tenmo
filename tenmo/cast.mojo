"""Cast — `Tensor.to_dtype` as a differentiable graph op.

Registers `ToDtypeBackward` so gradients survive dtype conversions:
the cast is the mixed-dtype boundary of the erased graph.
Forward conversion itself lives in
NDBuffer.to_dtype; this module only supplies the backward handler and is
wired up from tensor.mojo when `NewType != Self.dtype`, both dtypes are
floating point, and grad tracking is on.
"""

from .backpropagation import BackwardFnType
from .gradbox import Gradbox
from .ancestry import Ancestor
from .shared.panic import panic
from .shared.mnemonics import AddTensor


@fieldwise_init
struct ToDtypeBackward[
    src: DType, dst: DType
](BackwardFnType, ImplicitlyCopyable, RegisterPassable):
    """Backward for a dtype cast: re-type the incoming seed into src dtype.

    The output node's gradbox holds dst-dtype gradients; the single parent is
    the pre-cast tensor in src dtype. d cast(x)/dx = 1 elementwise, so the
    seed's dtype conversion IS the whole gradient computation. The parent is
    fetched through `get_erased().as[src]()` because it is the one edge where
    parent dtype differs from the output dtype.
    """

    comptime datatype = Self.dst

    @staticmethod
    def backward(
        var output: Ancestor[Self.dst],
        mut parent_ids: List[UInt],
    ):
        comptime if Self.dst.is_floating_point():
            ref gradbox = output.gradients()
            var parent = output.ancestry().get_erased(0).as[Self.src]()
            var ndb = gradbox.buffer().to_dtype[Self.src]()
            var gradbox_parent = Gradbox[Self.src](ndb^)

            parent.update_grad(gradbox_parent^, AddTensor, None)

            parent_ids.append(parent._id)

            gradbox.zero_grad()
        else:
            panic(
                "ToDtypeBackward: destination dtype must be floating point"
            )
