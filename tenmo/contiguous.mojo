from .tensor import Tensor
from .backpropagation import BackwardFn, BackwardFnType

from .shared.mnemonics import AddTensor
from .shared.panic import panic
from .gradbox import Gradbox
from .shared.shapes import Shape
from .ancestry import Ancestor


@fieldwise_init
struct ContiguousBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        ref parent_shape = parent.shape()

        if parent.requires_grad:
            if gradbox.shape() == Shape():
                var parent_gradbox = Gradbox[Self.dtype].full(
                    parent_shape,
                    gradbox.item(),
                    device=gradbox.device(),
                )
                parent.update_grad(parent_gradbox^, AddTensor, None)
            elif gradbox.shape() == parent_shape:
                # Same logical contents, both contiguous: single bulk
                # clone instead of per-coordinate get/set.
                var parent_gradbox = gradbox.clone()
                parent.update_grad(parent_gradbox^, AddTensor, None)
            else:
                # Unreachable: forward preserves logical shape exactly, so
                # the output gradbox always matches the parent shape (scalar
                # outputs take the first arm). Fail loud on violation —
                # a silent wrong-shaped accumulate would corrupt training.
                panic(
                    "ContiguousBackward: gradbox shape",
                    String(gradbox.shape()),
                    "!= parent shape",
                    String(parent_shape),
                )

        parent_ids.append(parent._id)
        gradbox.zero_grad()


struct Contiguous[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        owned: Bool = True,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        # owned=True: Tensor.contiguous() promises an ALWAYS-owned copy
        # (see Tensor.contiguous docstring) — never the owned=False alias
        # fast path, even when the source is already contiguous+shared.
        var ndb = self.buffer.contiguous(owned=owned)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    ContiguousBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^
