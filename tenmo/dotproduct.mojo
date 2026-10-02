from .tensor import Tensor
from .ndbuffer import NDBuffer
from .shared.panic import panic
from .backpropagation import BackwardFn, BackwardFnType

from .shared.mnemonics import AddTensor
from std.sys import has_accelerator
from .ancestry import Ancestor
from .broadcast import Broadcast
from .kernels.dotproduct_kernel import DotProductKernel


@fieldwise_init
struct DotBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()
        var scalar_grad_value = gradbox.item()  # Scalar
        var tensor_lhs_ref = output.ancestry().get(0)
        var tensor_rhs_ref = output.ancestry().get(1)

        var tensor_lhs = Tensor[Self.dtype](
            tensor_lhs_ref.buffer(), requires_grad=tensor_lhs_ref.requires_grad
        )
        var tensor_rhs = Tensor[Self.dtype](
            tensor_rhs_ref.buffer(), requires_grad=tensor_rhs_ref.requires_grad
        )

        if tensor_lhs.requires_grad:
            var grad_tensor = tensor_rhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_lhs = grad_tensor^.as_gradbox(contiguous=False)
            tensor_lhs_ref.update_grad(gradbox_lhs^, AddTensor, None)
        # Always appended: parent_ids is the engine's fanin-completion
        # signal (appended set must equal ancestry set).
        parent_ids.append(tensor_lhs_ref._id)

        if tensor_rhs.requires_grad:
            var grad_tensor = tensor_lhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_rhs = grad_tensor^.as_gradbox(contiguous=False)
            tensor_rhs_ref.update_grad(gradbox_rhs^, AddTensor, None)
        parent_ids.append(tensor_rhs_ref._id)
        gradbox.zero_grad()


@fieldwise_init
struct Dot[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        lhs: Tensor[Self.dtype], rhs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        # Broadcast scalar → vector if needed
        # A scalar tensor (rank=0 or numels=1) is broadcast to match the other.
        # broadcast_to wires grad correctly so chain rule is preserved.
        # Each path constructs its pair exactly once (no unconditional
        # alias copies ahead of the conditional rebinding).
        var actual_lhs: Tensor[Self.dtype]
        var actual_rhs: Tensor[Self.dtype]

        if lhs.numels() == 1 and rhs.numels() > 1:
            actual_lhs = Broadcast[Self.dtype].forward[track_grad](
                lhs, rhs.shape()
            )
            actual_rhs = rhs.copy()
        elif rhs.numels() == 1 and lhs.numels() > 1:
            actual_lhs = lhs.copy()
            actual_rhs = Broadcast[Self.dtype].forward[track_grad](
                rhs, lhs.shape()
            )
        else:
            actual_lhs = lhs.copy()
            actual_rhs = rhs.copy()

        # Rank and size validation (on post-broadcast tensors)
        if actual_lhs.rank() > 1 or actual_rhs.rank() > 1:
            panic("Tensor → dot: not supported for rank > 1")
        if actual_lhs.numels() != actual_rhs.numels():
            panic(
                "Tensor → dot: size does not match ",
                String(actual_lhs.numels()),
                " vs ",
                String(actual_rhs.numels()),
            )

        var out: Tensor[Self.dtype]

        comptime if has_accelerator():
            if actual_lhs.is_on_gpu() and actual_rhs.is_on_gpu():
                try:
                    var (result_layout, result_storage) = DotProductKernel[
                        Self.dtype
                    ].launch[suppress_validation=True](
                        actual_lhs.buffer.layout(),
                        actual_lhs.buffer.device_state.value(),
                        actual_rhs.buffer.layout(),
                        actual_rhs.buffer.device_state.value(),
                        sync=sync,
                    )
                    var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result_layout, result_storage
                    )
                    out = Tensor[Self.dtype](out_ndb^, requires_grad=False)
                except e:
                    print(e)
                    panic("Dot - GPU operation failed")
                    out = Tensor[Self.dtype].scalar(0)
            else:
                out = Tensor[Self.dtype].scalar(
                    actual_lhs.buffer.contiguous_buffer().dot(
                        actual_rhs.buffer.contiguous_buffer()
                    ),
                    requires_grad=False,
                )
        else:
            out = Tensor[Self.dtype].scalar(
                actual_lhs.buffer.contiguous_buffer().dot(
                    actual_rhs.buffer.contiguous_buffer()
                ),
                requires_grad=False,
            )

        comptime if track_grad:
            if actual_lhs.requires_grad or actual_rhs.requires_grad:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    DotBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, actual_lhs, actual_rhs)

        return out^
