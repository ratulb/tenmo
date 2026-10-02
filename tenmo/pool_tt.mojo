"""GPU MaxPool2d via TileTensor — the CPU `MaxPool2d` is untouched.

`MaxPoolTT` mirrors the CPU `MaxPool2d` contract (`tenmo/pooling.mojo`):
rank-4 input, square kernel, stride (default = kernel), symmetric padding,
same output formula, first-max-wins ties, neg_inf-filled all-padded windows.
The argmax mask is device-resident int64 and rides in the backward arg, so
forward, backward, and the optimizer step all stay on-device — the CPU
`MaxPool2d` has no GPU kernel (its `data_ptr()` reads would silently misread
device memory), which is what this struct replaces on the GPU path.

GPU-only by design: CPU input panics fail-loud (no silent per-op transfers —
move the model once with `to_gpu()`). The backward likewise panics on CPU
grads. Use `MaxPool2d` for CPU.
"""

from .tensor import Tensor
from .shared.shapes import Shape
from .shared.layout import Layout
from .gradbox import Gradbox
from .backpropagation import BackwardFn, ArgumentType, BackwardFnType

from .layer_trait import LayerTrait
from .shared.mnemonics import AddTensor
from .ndbuffer import NDBuffer
from .shared.panic import panic
from .ancestry import Ancestor
from .kernels.pool_tt import PoolTt
from std.sys.info import has_accelerator


@fieldwise_init
struct MaxPoolTTBwdArg(ArgumentType):
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var input_shape: Shape
    var argmax_mask: NDBuffer[DType.int64]


@fieldwise_init
struct MaxPoolTTBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = output.ancestry().backward_fn().get[MaxPoolTTBwdArg]()
        ref grad_out = output.gradients()

        var ancestor_ref = output.ancestry().get(0)

        # Grad is computed only if needed, but the id is ALWAYS appended:
        # parent_ids is the engine's fanin-completion signal (appended set
        # must equal ancestry set).
        if ancestor_ref.requires_grad:
            comptime if has_accelerator():
                if grad_out.is_on_gpu():
                    try:
                        var result = PoolTt[Self.dtype].backward(
                            grad_out.buffer().layout(),
                            grad_out.buffer().device_state.value(),
                            bwd_arg.argmax_mask.device_state.value(),
                            bwd_arg.input_shape[2],
                            bwd_arg.input_shape[3],
                            sync=True,
                        )
                        var grad_ndb = NDBuffer[
                            Self.dtype
                        ].with_layout_device_state(result[0], result[1])
                        ancestor_ref.update_grad(
                            Gradbox[Self.dtype](grad_ndb^), AddTensor, None
                        )
                    except e:
                        panic(
                            "MaxPoolTTBackward GPU backward failed: "
                            + String(e)
                        )
                    parent_ids.append(ancestor_ref._id)
                    grad_out.zero_grad()
                    return
            panic(
                "MaxPoolTTBackward requires GPU grads (MaxPoolTT is"
                " GPU-only; CPU path is MaxPool2d)"
            )
        parent_ids.append(ancestor_ref._id)

        grad_out.zero_grad()


@fieldwise_init
struct MaxPoolTT[dtype: DType](LayerTrait & RegisterPassable):
    """
    Batched, multi-channel 2D Max Pooling on GPU.

    Same contract as the CPU `MaxPool2d`; forward, argmax mask, backward,
    and grad accumulation all stay on-device via `PoolTt`.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype
    var training: Bool
    var kernel_size: Int
    var stride: Int
    var padding: Int

    def __init__(
        out self,
        kernel_size: Int = 2,
        stride: Optional[Int] = None,
        padding: Int = 0,
    ):
        self.training = True
        self.kernel_size = kernel_size
        self.stride = stride.or_else(kernel_size)
        self.padding = padding

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return Self.forward[track_grad=True](
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                sync=sync,
            )
        else:
            return Self.forward[track_grad=False](
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                sync=sync,
            )

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        input_tensor: Tensor[Self.dtype],
        kernel_size: Int = 2,
        stride: Optional[Int] = None,
        padding: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        if not input_tensor.is_on_gpu():
            panic(
                "MaxPoolTT requires GPU input: move the tensor with to_gpu()"
                " (CPU path is MaxPool2d)"
            )
        ref input_shape = input_tensor.shape()
        if input_shape.rank() != 4:
            panic("MaxPoolTT expects 4D input: (N, C, H_in, W_in)")

        var s = stride.or_else(kernel_size)

        try:
            var result = PoolTt[Self.dtype].forward(
                input_tensor.buffer.layout(),
                input_tensor.buffer.device_state.value(),
                kernel_size,
                s,
                padding,
                sync=sync,
            )
            var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                result[0], result[1]
            )
            ref out_shape = out_ndb.shape
            var mask_ndb = NDBuffer[DType.int64].with_layout_device_state(
                Layout(
                    Shape(
                        out_shape[0], out_shape[1], out_shape[2], out_shape[3]
                    )
                ),
                result[2],
            )
            var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

            # Setup gradient tracking (mask lives in the arg — no
            # needs_parent_data, mirroring MaxPool2d).
            comptime if track_grad:
                var grad_required = requires_grad.or_else(
                    input_tensor.requires_grad
                )
                if grad_required:
                    out.requires_grad_(True)
                    var backwardFn = BackwardFn(
                        MaxPoolTTBwdArg(
                            kernel_size,
                            s,  # Stride
                            padding,
                            input_shape,
                            mask_ndb,
                        ),
                        MaxPoolTTBackward[Self.dtype](),
                    )
                    out.add_ancestry(backwardFn^, input_tensor)

            return out^
        except e:
            panic("MaxPoolTT GPU forward failed: " + String(e))
            # Unreachable — satisfies definite assignment.
            return Tensor[Self.dtype].zeros(Shape(1))

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False
