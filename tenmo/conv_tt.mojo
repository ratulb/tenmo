"""GPU Conv2D forward + backward via TileTensor.

`ConvTT.forward` mirrors the CPU `Conv.forward` contract (`tenmo/conv.mojo`):
rank-4 image, rank-4 kernel, rank-1 bias (required), scalar stride and
dilation, asymmetric padding, same dilated output formula. Image and kernel
may be strided views — the kernel consumes them natively, no
`materialize_contiguous` round-trip (the CPU core panics on views; this op
is strictly more general).

GPU-only by design: CPU input panics fail-loud (no silent per-op transfers —
move the model once with `to_gpu()`). The backward likewise panics on CPU
grads. Forward, backward, and grad accumulation all stay on-device.
Use `Conv` for CPU.
"""

from .tensor import Tensor
from .ndbuffer import NDBuffer
from .shared.shapes import Shape
from .shared.panic import panic
from .kernels.conv_tt import ConvTt
from .gradbox import Gradbox
from .backpropagation import BackwardFn, ArgumentType, BackwardFnType
from .ancestry import Ancestor
from .shared.mnemonics import AddTensor
from .layer_trait import LayerTrait
from .weight_init import Weights
from .named_parameter import NamedParameter
from .gpu.device import GPU
from std.sys.info import has_accelerator


@fieldwise_init
struct ConvTTBwdArg(ArgumentType):
    var N: Int
    var C_in: Int
    var H_in: Int
    var W_in: Int
    var C_out: Int
    var KH: Int
    var KW: Int
    var H_out: Int
    var W_out: Int
    var stride: Int
    var dilation: Int
    var pad_top: Int
    var pad_left: Int


@fieldwise_init
struct ConvTTBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = output.ancestry().backward_fn().get[ConvTTBwdArg]()
        # Parent order matches the forward add_ancestry call below:
        # image, kernel, bias (same order as the CPU ConvBackward).
        var image = output.ancestry().get(0)
        var kernel = output.ancestry().get(1)
        var bias = output.ancestry().get(2)

        comptime if has_accelerator():
            ref dy = output.gradients()
            if dy.is_on_gpu():
                if (
                    not image.is_on_gpu()
                    or not kernel.is_on_gpu()
                    or not bias.is_on_gpu()
                ):
                    panic(
                        "ConvTTBackward: mixed devices — image, kernel and"
                        " bias must share one GPU"
                    )
                try:
                    var result = ConvTt[Self.dtype].backward(
                        dy.buffer().layout(),
                        dy.buffer().device_state.value(),
                        image.buffer().layout(),
                        image.buffer().device_state.value(),
                        kernel.buffer().layout(),
                        kernel.buffer().device_state.value(),
                        bwd_arg.N,
                        bwd_arg.C_in,
                        bwd_arg.H_in,
                        bwd_arg.W_in,
                        bwd_arg.C_out,
                        bwd_arg.KH,
                        bwd_arg.KW,
                        bwd_arg.H_out,
                        bwd_arg.W_out,
                        bwd_arg.stride,
                        bwd_arg.dilation,
                        bwd_arg.pad_top,
                        bwd_arg.pad_left,
                        sync=True,
                    )
                    var gi_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(result[0], result[1])
                    var gk_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(result[2], result[3])
                    var gb_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(result[4], result[5])
                    # Grads are computed only if needed, but ids are ALWAYS
                    # appended (fanin-completion signal, like the CPU core).
                    if kernel.requires_grad:
                        kernel.update_grad(
                            Gradbox[Self.dtype](gk_ndb^), AddTensor, None
                        )
                    parent_ids.append(kernel._id)
                    if image.requires_grad:
                        image.update_grad(
                            Gradbox[Self.dtype](gi_ndb^), AddTensor, None
                        )
                    parent_ids.append(image._id)
                    if bias.requires_grad:
                        bias.update_grad(
                            Gradbox[Self.dtype](gb_ndb^), AddTensor, None
                        )
                    parent_ids.append(bias._id)
                    dy.zero_grad()
                except e:
                    panic(
                        "ConvTTBackward GPU backward failed: " + String(e)
                    )
                return
        panic(
            "ConvTTBackward requires GPU grads (ConvTT is"
            " GPU-only; CPU path is Conv)"
        )


struct ConvTT[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """Batched, multi-channel 2D convolution forward on GPU."""

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        kernel: Tensor[Self.dtype],
        bias: Tensor[Self.dtype],
        stride: Int = 1,
        dilation: Int = 1,
        pad_top: Int = 0,
        pad_bottom: Int = 0,
        pad_left: Int = 0,
        pad_right: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        if not image.is_on_gpu():
            panic(
                "ConvTT requires GPU input: move the tensors with to_gpu()"
                " (CPU path is Conv)"
            )
        if not kernel.is_on_gpu() or not bias.is_on_gpu():
            panic(
                "ConvTT: mixed devices — move image, kernel and bias to the"
                " same GPU"
            )
        ref image_shape = image.shape()
        ref kernel_shape = kernel.shape()
        if image_shape.rank() != 4:
            panic("ConvTT: image must be 4D (N, C_in, H_in, W_in)")
        if kernel_shape.rank() != 4:
            panic("ConvTT: kernel must be 4D (C_out, C_in, KH, KW)")
        if kernel_shape[1] != image_shape[1]:
            panic("ConvTT: kernel input channels must match input channels")
        if bias.shape().rank() != 1 or bias.shape()[0] != kernel_shape[0]:
            panic("ConvTT: bias must have shape (C_out,)")

        try:
            var result = ConvTt[Self.dtype].forward(
                image.buffer.layout(),
                image.buffer.device_state.value(),
                kernel.buffer.layout(),
                kernel.buffer.device_state.value(),
                bias.buffer.device_state.value(),
                stride,
                dilation,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                sync=sync,
            )
            var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                result[0], result[1]
            )
            # Dims before the move below uninitializes out_ndb.
            var H_out = out_ndb.shape[2]
            var W_out = out_ndb.shape[3]
            var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

            # Setup gradient tracking: the backward gathers from the saved
            # parents (needs_parent_data, like the CPU Conv) — image and
            # kernel ride the ancestry, pad_bottom/right are output-size
            # only and need no saving.
            comptime if track_grad:
                var grad_required = requires_grad.or_else(
                    image.requires_grad
                    or kernel.requires_grad
                    or bias.requires_grad
                )
                if grad_required:
                    out.requires_grad_(True)
                    var backwardFn = BackwardFn(
                        ConvTTBwdArg(
                            image_shape[0],
                            image_shape[1],
                            image_shape[2],
                            image_shape[3],
                            kernel_shape[0],
                            kernel_shape[2],
                            kernel_shape[3],
                            H_out,
                            W_out,
                            stride,
                            dilation,
                            pad_top,
                            pad_left,
                        ),
                        ConvTTBackward[Self.dtype](),
                    )
                    backwardFn.needs_parent_data = True
                    out.add_ancestry(backwardFn^, image, kernel, bias)

            return out^
        except e:
            panic("ConvTT GPU forward failed: " + String(e))
            # Unreachable — satisfies definite assignment.
            return Tensor[Self.dtype].zeros(Shape(1))


@fieldwise_init
struct ConvTT2D[dtype: DType](LayerTrait):
    """ConvTT layer wrapper for Sequential integration.

    Mirrors `net.Conv2D` (same init vocabulary, same `parameters()` /
    `to_gpu()` / `train()` / `eval()` contract) but runs the GPU-native
    `ConvTT` op, so a `to_gpu`'d model trains end-to-end on-device with no
    per-op transfers. Appends directly to `MixedSequential` (LayerTrait);
    legacy `Sequential` membership would need `net.mojo` Variant surgery
    (deliberately not done here).

    Two deliberate narrowings vs `Conv2D`:
    - `padding` is a symmetric `Int` (same pad on all four sides). The raw
      `ConvTT.forward` takes asymmetric pads; stride-2 "same" on odd
      spatial dims needs it — use the op directly there.
    - `bias` is always allocated (`bias=False` panics): `ConvTT` has no
      bias-free path, and a per-forward zeros substitute would reintroduce
      per-batch allocation traffic.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    var weight: Tensor[Self.dtype]  # (out_channels, in_channels, K, K)
    var bias: Optional[Tensor[Self.dtype]]  # (out_channels,) — always set
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var dilation: Int
    var pad: Int
    var training: Bool

    def __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
        bias: Bool = True,
        bias_zero: Bool = True,
        init_seed: Optional[Int] = None,
        init_method: String = "he",
    ):
        if not bias:
            panic("ConvTT2D requires bias=True: ConvTT has no bias-free path")
        if padding < 0:
            panic("ConvTT2D padding must be non-negative")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = padding
        self.training = True

        # Fan counts kernel taps: each output taps in_channels*K*K inputs.
        var weight_shape = Shape(
            out_channels, in_channels, kernel_size, kernel_size
        )
        var fan_in = in_channels * kernel_size * kernel_size
        var fan_out = out_channels * kernel_size * kernel_size
        self.weight, self.bias = Weights[Self.dtype].initialize(
            weight_shape,
            fan_in,
            fan_out,
            init_seed=init_seed,
            init_method=init_method,
            requires_grad=True,
            bias_len=out_channels,
            bias_zero=bias_zero,
            device=None,
        )
        if not self.bias:
            panic("ConvTT2D bias init failed (unreachable)")
        print("ConvTT2D initialized:")
        print("  Shape:", weight_shape)
        print("  In channels:", in_channels)
        print("  Out channels:", out_channels)
        print("  Kernel size:", kernel_size, "×", kernel_size)
        print("  Parameters:", self.num_parameters())

    def __init__(out self, *, copy: Self):
        self.weight = copy.weight
        self.bias = copy.bias
        self.in_channels = copy.in_channels
        self.out_channels = copy.out_channels
        self.kernel_size = copy.kernel_size
        self.stride = copy.stride
        self.dilation = copy.dilation
        self.pad = copy.pad
        self.training = copy.training

    def __init__(out self, *, deinit move: Self):
        self.weight = move.weight^
        self.bias = move.bias^
        self.in_channels = move.in_channels
        self.out_channels = move.out_channels
        self.kernel_size = move.kernel_size
        self.stride = move.stride
        self.dilation = move.dilation
        self.pad = move.pad
        self.training = move.training

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref img_shape = x.shape()

        if img_shape.rank() != 4:
            panic(
                "ConvTT2D input must be 4D: (N, C, H, W), got shape: ",
                String(img_shape),
            )

        if img_shape[1] != self.in_channels:
            panic(
                "ConvTT2D input channels mismatch: expected ",
                String(self.in_channels),
                ", got ",
                String(img_shape[1]),
            )

        if self.training:
            return ConvTT[Self.dtype].forward[track_grad=True](
                x,
                self.weight,
                self.bias.value(),
                self.stride,
                self.dilation,
                self.pad,
                self.pad,
                self.pad,
                self.pad,
                requires_grad=True,
                sync=sync,
            )
        else:
            return ConvTT[Self.dtype].forward[track_grad=False](
                x,
                self.weight,
                self.bias.value(),
                self.stride,
                self.dilation,
                self.pad,
                self.pad,
                self.pad,
                self.pad,
                requires_grad=False,
                sync=sync,
            )

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            Pointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        if self.bias:
            params.append(
                Pointer(to=self.bias.value())
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        var w = Pointer(to=self.weight).unsafe_mut_cast[True]()
        result.append(
            NamedParameter(
                prefix + "weight",
                w.as_unsafe_any_origin().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
            )
        )
        if self.bias:
            var b = Pointer(to=self.bias.value()).unsafe_mut_cast[True]()
            result.append(
                NamedParameter(
                    prefix + "bias",
                    b.as_unsafe_any_origin().unsafe_origin_cast[
                        MutUntrackedOrigin
                    ](),
                )
            )
        return result^

    def num_parameters(self) -> Int:
        var count = self.weight.numels()
        if self.bias:
            count += self.bias.value().numels()
        return count

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> ConvTT2D[Self.dtype]:
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var out = self
        out.weight = weight_gpu^
        if out.bias:
            var bias_gpu = out.bias.value().to_gpu(gpu=gpu, stop_grad=True)
            out.bias = bias_gpu^
        return out^

    def to_cpu(self) raises -> ConvTT2D[Self.dtype]:
        var weight_cpu = self.weight.to_cpu(stop_grad=True)
        var out = self
        out.weight = weight_cpu^
        if out.bias:
            var bias_cpu = out.bias.value().to_cpu(stop_grad=True)
            out.bias = bias_cpu^
        return out^
