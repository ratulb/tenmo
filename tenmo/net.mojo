from std.memory.alloc import unsafe_alloc
from .tensor import Tensor
from .shared.shapes import Shape
from .gradbox import Gradbox
from .shared.panic import panic
from std.utils import Variant
from .matmul import Matmul
from .addition import Adder
from .pad import Padding
from .cnn import Conv2dFused
from .pooling import MaxPool2d
from .dropout import Dropout
from .layernorm import LayerNorm
from .embedding import Embedding
from .positional import PositionalEmbedding
from .shared import Reduction
from .shared.mnemonics import (
    mm,
    mv,
    vm,
    dot,
)
from .blashandle import BLASHandleLite
from .blas_ndbuffer import BLASCache
from .gpu.device import Device, GPU
from .named_parameter import NamedParameter
from .layer_trait import LayerTrait
from .backpropagation import CopyFn, DestroyerFn
from .weight_init import Weights


struct Linear[dtype: DType, mode: Int = mm](LayerTrait & Writable):
    """Fully connected layer: y = xW + b.

    Weights, input and output are all `dtype`; this is a homogeneous
    layer. OutputDType inherits InputDType from the LayerTrait default.

    Crossing a dtype boundary is the CONTAINER's job, not the layer's:
    `MixedSequential`/`Seq` insert a grad-tracked `to_dtype` wherever the
    running tail's dtype differs from the next layer's InputDType
    (net.mojo `needs_cast`). To use a different dtype for a segment,
    append `to_dtype` or a layer that bridges — not a phantom input dtype.

    HISTORY: this was `Linear[InT, OutT = InT, mode]`, and
    `InT` was never read — `InputDType` was set to `OutT` because the
    checker cannot prove `InT == OutT` inside branch bodies. So
    `Linear[f32, f64]` and `Linear[f64, f64]` were the same layer, and
    all 31 two-arg call sites were misleading. The unused param is gone;
    see scripts/check_layer_dtypes.py CHECK 2, which is what found it.
    """

    comptime InputDType = Self.dtype

    var weight: Tensor[Self.dtype]
    var bias: Optional[Tensor[Self.dtype]]
    var in_features: Int
    var out_features: Int
    var training: Bool

    def __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "uniform",  # normal, uniform, xavier/glorot, kaiming/he, zero
        bias: Bool = True,
        bias_zero: Bool = True,
        device: Optional[Device] = None,
    ):
        """
        Initialize Linear layer with configurable weight initialization.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            init_seed: Random seed for reproducibility.
            init_method: Weight initialization method (see Weights.initialize):
                - "normal": N(0, 1).
                - "uniform": U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
                - "xavier": Xavier/Glorot uniform (good for tanh/sigmoid).
                - "he"/"kaiming": He/Kaiming normal (good for ReLU with standardized inputs).
                - "zero": Weights initialized to zero.
            bias: If True, create bias parameter. If False, no bias is allocated.
            bias_zero: If True and bias=True, initialize bias to zeros.
            device: If given, place the parameters on this device.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.training = True
        self.weight, self.bias = Weights[Self.dtype].initialize(
            in_features,
            out_features,
            init_seed=init_seed,
            init_method=init_method,
            requires_grad=True,
            bias=bias,
            bias_zero=bias_zero,
            device=device,
        )

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref xs_shape = x.shape()
        ref weight_shape = self.weight.shape()

        if xs_shape[-1] != weight_shape[0]:
            panic(
                "Linear forward: input dim mismatch: input shape → ",
                String(xs_shape),
                "and  weights shape → ",
                String(weight_shape),
            )
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=True, mode=Self.mode
            ](x, self.weight, sync=sync)
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=True](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        else:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=Self.mode
            ](x, self.weight, sync=sync)
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=False](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        return result^

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
        """Set to training mode - enables gradient tracking."""
        self.training = True

    def eval(mut self):
        """Set to evaluation mode - disables gradient tracking."""
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))

    def to_gpu(
        self, gpu: Optional[GPU] = None
    ) raises -> Linear[Self.dtype, Self.mode]:
        """Move this Linear layer to GPU.

        The returned layer owns GPU-resident parameter leaves; the source
        layer is left untouched (caller discards it).
        """
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var out = self
        out.weight = weight_gpu^
        if out.bias:
            var bias_gpu = out.bias.value().to_gpu(gpu=gpu, stop_grad=True)
            out.bias = bias_gpu^
        return out^

    def to_cpu(self) raises -> Linear[Self.dtype, Self.mode]:
        """Move this Linear layer back to CPU.

        The returned layer owns CPU-resident parameter leaves; the source
        layer is left untouched (caller discards it).
        """
        var weight_cpu = self.weight.to_cpu(stop_grad=True)
        var out = self
        out.weight = weight_cpu^
        if out.bias:
            var bias_cpu = out.bias.value().to_cpu(stop_grad=True)
            out.bias = bias_cpu^
        return out^

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "[input="
            + String(self.in_features)
            + " → "
            + "output="
            + String(self.out_features)
            + "]"
        )

    @no_inline
    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Linear")


struct LinearBLAS[dtype: DType, mode: Int = mm](ImplicitlyCopyable):
    """Fully connected layer: y = xW + b."""

    # NOTE: deliberately NOT a LayerTrait — rides the homogeneous
    # Layer[dtype] Variant only (Module/SequentialBLAS dispatch by isa[]).
    # Without InputDType/OutputDType it cannot enter MixedSequential/Seq;
    # routing is direct (lite attached + contiguous -> BLAS, else native).


    var weight: Tensor[Self.dtype]
    var bias: Optional[Tensor[Self.dtype]]
    var in_features: Int
    var out_features: Int
    var training: Bool
    var blas_lite: Optional[BLASHandleLite[Self.dtype]]

    def __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "uniform",  # normal, uniform, xavier/glorot, kaiming/he, zero
        bias: Bool = True,
        bias_zero: Bool = True,
    ):
        """
                Initialize LinearBLAS layer with configurable weight initialization.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            init_seed: Random seed for reproducibility.
            init_method: Weight initialization method (see Weights.initialize):
                - "normal": N(0, 1).
                - "uniform": U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
                - "xavier": Xavier/Glorot uniform (good for tanh/sigmoid).
                - "he"/"kaiming": He/Kaiming normal (good for ReLU with standardized inputs).
                - "zero": Weights initialized to zero.
            bias: If True, create bias parameter. If False, no bias is allocated.
            bias_zero: If True and bias=True, initialize bias to zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.training = True
        self.blas_lite = None

        self.weight, self.bias = Weights[Self.dtype].initialize(
            in_features,
            out_features,
            init_seed=init_seed,
            init_method=init_method,
            requires_grad=True,
            bias=bias,
            bias_zero=bias_zero,
            device=None,
        )

    def __call__(
        mut self, mut xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref xs_shape = xs.shape()
        ref weight_shape = self.weight.shape()

        if xs_shape[-1] != weight_shape[0]:
            panic(
                "LinearBLAS forward: input dim mismatch: input shape → ",
                String(xs_shape),
                "and  weights shape → ",
                String(weight_shape),
            )

        # Direct routing: BLAS when a lite is attached and both operands
        # are contiguous, native otherwise. No runtime profiling.
        if (
            self.blas_lite
            and self.weight.is_contiguous()
            and xs.is_contiguous()
        ):
            return self.matmul_blas(xs, sync=sync)
        return self.matmul(xs, sync=sync)

    def to_gpu(
        self, gpu: Optional[GPU] = None
    ) raises -> Linear[Self.dtype, Self.mode]:
        panic(
            "LinearBLAS does not support GPU — use Linear[dtype] for GPU models"
        )
        return Linear[Self.dtype, Self.mode](
            self.in_features, self.out_features
        )

    def to_cpu(self) raises -> Self:
        panic("LinearBLAS does not support GPU — nothing to transfer back")
        return self

    @always_inline
    def matmul(
        mut self, mut xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=True, mode=Self.mode
            ](xs, self.weight, sync=sync)
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=True](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        else:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=Self.mode
            ](xs, self.weight, sync=sync)
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=False](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        return result^

    @always_inline
    def matmul_blas(
        mut self, mut xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = self.blas_lite.value().matmul[track_grad=True](
                xs,
                self.weight,
                transpose_A=False,
                transpose_B=False,
                sync=sync,
            )
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=True](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        else:
            var matmul_out = self.blas_lite.value().matmul[track_grad=False](
                xs,
                self.weight,
                transpose_A=False,
                transpose_B=False,
                sync=sync,
            )
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=False](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        return result^

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
        """Set to training mode - enables gradient tracking."""
        self.training = True

    def eval(mut self):
        """Set to evaluation mode - disables gradient tracking."""
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


struct ReLU[dtype: DType](LayerTrait & RegisterPassable & Writable):
    var training: Bool
    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("ReLU")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("ReLU")

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return x.relu[track_grad=True](sync=sync)
        else:
            return x.relu[track_grad=False](sync=sync)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


struct FakeQuant[dtype: DType](LayerTrait & Writable):
    """Quantization-aware fake-quantization layer.

        out = scale * clamp(round(x / scale), qmin, qmax)

    Simulates int8 inference during training while storage stays float, so
    the network learns weights and activations that survive the grid. The
    backward pass is a straight-through estimator: `d/dx = 1` (the true
    value is 0 almost everywhere, since `round` is a staircase) while
    `d/dscale` keeps its true value. See `tenmo/fakequant.mojo`.

    Drop it in a `Sequential` after an activation to quantize activations,
    or before an activation to quantize pre-activations. `training` only
    matters for `eval()`: QAT normally keeps the quantizer active at eval
    time, because the deployed model is quantized too. Toggling it off gives
    you the float model's accuracy for comparison.
    """

    var scale: Tensor[Self.dtype]
    var qmin: Scalar[Self.dtype]
    var qmax: Scalar[Self.dtype]
    var training: Bool

    # Homogeneous layer: consumes and produces Tensor[Self.dtype].
    comptime InputDType = Self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("FakeQuant")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("FakeQuant")

    def __init__(
        out self,
        scale: Scalar[Self.dtype],
        qmin: Scalar[Self.dtype],
        qmax: Scalar[Self.dtype],
        requires_grad: Bool = False,
    ):
        """A per-tensor quantizer.

        Args:
            scale: Quantization step. Must be > 0. The grid spacing, so
                smaller means finer (and more error when simulated as int8).
            qmin: Lowest representable level (e.g. -128.0 for int8).
            qmax: Highest representable level (e.g. 127.0 for int8).
            requires_grad: Give `scale` a gradient so it can be learned
                (LSQ-style). Off by default; on it, the scale becomes a
                leaf parameter.
        """
        self.scale = Tensor[Self.dtype].scalar(
            scale, requires_grad=requires_grad
        )
        self.qmin = qmin
        self.qmax = qmax
        self.training = True

    def __init__(out self, *, copy: Self):
        self.scale = copy.scale
        self.qmin = copy.qmin
        self.qmax = copy.qmax
        self.training = copy.training

    # No `where` clause here on purpose: `LayerTrait.__call__` is unconstrained,
    # so an impl that adds one stops conforming (and putting the constraint on
    # the trait would wrongly restrict every layer to float). Dispatch at
    # comptime instead and reject non-float dtypes loudly.
    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            if self.training:
                return x.fake_quant[track_grad=True](
                    self.scale, self.qmin, self.qmax, sync=sync
                )
            else:
                return x.fake_quant[track_grad=False](
                    self.scale, self.qmin, self.qmax, sync=sync
                )
        else:
            panic(
                "FakeQuant layer requires a floating-point dtype; got ",
                String(Self.dtype),
                ". An integer grid is not simulated: there is no float-valued",
                " output to snap back onto.",
            )
            return x

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        # Only a LEARNABLE scale is a parameter. With the default
        # `requires_grad=False` the scale is a constant, and handing the
        # optimizer a constant would just add noise.
        var result = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        if self.scale.requires_grad:
            result.append(
                Pointer(to=self.scale)
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        return result^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        if self.scale.requires_grad:
            var sp = Pointer(to=self.scale).unsafe_mut_cast[True]()
            result.append(
                NamedParameter(
                    prefix + "scale",
                    sp.as_unsafe_any_origin().unsafe_origin_cast[
                        MutUntrackedOrigin
                    ](),
                )
            )
        return result^

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


struct GeLU[dtype: DType](LayerTrait & RegisterPassable & Writable):
    var training: Bool
    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("GeLU")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("GeLU")

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return x.gelu[track_grad=True](sync=sync)
        else:
            return x.gelu[track_grad=False](sync=sync)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


struct Sigmoid[dtype: DType](LayerTrait & RegisterPassable & Writable):
    var training: Bool
    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sigmoid")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Sigmoid")

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        comptime assert Self.dtype.is_floating_point()
        if self.training:
            return x.sigmoid[track_grad=True](sync=sync)
        else:
            return x.sigmoid[track_grad=False](sync=sync)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


struct Tanh[dtype: DType](LayerTrait & RegisterPassable & Writable):
    var training: Bool
    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Tanh")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Tanh")

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        comptime assert Self.dtype.is_floating_point()
        if self.training:
            return x.tanh[track_grad=True](sync=sync)
        else:
            return x.tanh[track_grad=False](sync=sync)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


# Refer to operators & matmul
# Defined in operators

# comptime dot = ?  # dot product
# comptime vm = ?  # vector & tensor matmul
# comptime mv = ?  # tensor & vector matmul
# comptime mm = ?  # tensor & tensor matmul

# Homogeneous single-dtype dispatch Variant for Module/Sequential(_BLAS).
# Mixed-dtype chains do NOT go through here — they hold LayerTrait objects
# directly (MixedSequential heap-erased, Seq variadic). Membership does not
# imply LayerTrait conformance (LinearBLAS has no InputDType/OutputDType).
comptime Layer[dtype: DType] = Variant[
    Linear[dtype],
    LinearBLAS[dtype, mm],
    ReLU[dtype],
    GeLU[dtype],
    Sigmoid[dtype],
    Tanh[dtype],
    Dropout[dtype],
    Conv2D[dtype],
    Flatten[dtype],
    MaxPool2d[dtype],
    LayerNorm[dtype],
    Embedding[dtype],
    PositionalEmbedding[dtype],
    FakeQuant[dtype],
]


@fieldwise_init
struct Module[dtype: DType](ImplicitlyCopyable & Writable):
    var layer: Layer[Self.dtype]

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Module")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Module")

    def __call__(
        mut self, mut xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.layer.isa[Linear[Self.dtype]]():
            return self.layer[Linear[Self.dtype]](xs, sync=sync)
        if self.layer.isa[LinearBLAS[Self.dtype]]():
            return self.layer[LinearBLAS[Self.dtype, mm]](xs, sync=sync)
        elif self.layer.isa[ReLU[Self.dtype]]():
            return self.layer[ReLU[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[GeLU[Self.dtype]]():
            return self.layer[GeLU[Self.dtype]](xs, sync=sync)

        elif self.layer.isa[Sigmoid[Self.dtype]]():
            return self.layer[Sigmoid[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[Tanh[Self.dtype]]():
            return self.layer[Tanh[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[Dropout[Self.dtype]]():
            return self.layer[Dropout[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[Conv2D[Self.dtype]]():
            return self.layer[Conv2D[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[Flatten[Self.dtype]]():
            return self.layer[Flatten[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[Embedding[Self.dtype]]():
            return self.layer[Embedding[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[MaxPool2d[Self.dtype]]():
            return self.layer[MaxPool2d[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            return self.layer[LayerNorm[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            return self.layer[PositionalEmbedding[Self.dtype]](xs, sync=sync)
        elif self.layer.isa[FakeQuant[Self.dtype]]():
            return self.layer[FakeQuant[Self.dtype]](xs, sync=sync)

        else:
            panic("Unknown module type")
            return Tensor[Self.dtype].scalar(0)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        if self.layer.isa[Linear[Self.dtype]]():
            return self.layer[Linear[Self.dtype]].parameters()
        elif self.layer.isa[LinearBLAS[Self.dtype]]():
            return self.layer[LinearBLAS[Self.dtype]].parameters()
        elif self.layer.isa[Conv2D[Self.dtype]]():
            return self.layer[Conv2D[Self.dtype]].parameters()
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            return self.layer[LayerNorm[Self.dtype]].parameters()
        elif self.layer.isa[Embedding[Self.dtype]]():
            return self.layer[Embedding[Self.dtype]].parameters()
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            return self.layer[PositionalEmbedding[Self.dtype]].parameters()
        elif self.layer.isa[FakeQuant[Self.dtype]]():
            return self.layer[FakeQuant[Self.dtype]].parameters()

        else:
            return List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        if self.layer.isa[Linear[Self.dtype]]():
            return self.layer[Linear[Self.dtype]].named_parameters(prefix)
        elif self.layer.isa[LinearBLAS[Self.dtype]]():
            return self.layer[LinearBLAS[Self.dtype]].named_parameters(prefix)
        elif self.layer.isa[Conv2D[Self.dtype]]():
            return self.layer[Conv2D[Self.dtype]].named_parameters(prefix)
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            return self.layer[LayerNorm[Self.dtype]].named_parameters(prefix)
        elif self.layer.isa[Embedding[Self.dtype]]():
            return self.layer[Embedding[Self.dtype]].named_parameters(prefix)
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            return self.layer[PositionalEmbedding[Self.dtype]].named_parameters(
                prefix
            )
        elif self.layer.isa[FakeQuant[Self.dtype]]():
            return self.layer[FakeQuant[Self.dtype]].named_parameters(prefix)
        else:
            return List[NamedParameter[Self.dtype]]()

    def num_parameters(self) -> Int:
        if self.layer.isa[Linear[Self.dtype]]():
            return self.layer[Linear[Self.dtype]].num_parameters()
        if self.layer.isa[LinearBLAS[Self.dtype]]():
            return self.layer[LinearBLAS[Self.dtype, mm]].num_parameters()
        elif self.layer.isa[ReLU[Self.dtype]]():
            return self.layer[ReLU[Self.dtype]].num_parameters()
        elif self.layer.isa[GeLU[Self.dtype]]():
            return self.layer[GeLU[Self.dtype]].num_parameters()

        elif self.layer.isa[Sigmoid[Self.dtype]]():
            return self.layer[Sigmoid[Self.dtype]].num_parameters()
        elif self.layer.isa[Tanh[Self.dtype]]():
            return self.layer[Tanh[Self.dtype]].num_parameters()
        elif self.layer.isa[Dropout[Self.dtype]]():
            return self.layer[Dropout[Self.dtype]].num_parameters()
        elif self.layer.isa[Conv2D[Self.dtype]]():
            return self.layer[Conv2D[Self.dtype]].num_parameters()
        elif self.layer.isa[Flatten[Self.dtype]]():
            return self.layer[Flatten[Self.dtype]].num_parameters()
        elif self.layer.isa[Embedding[Self.dtype]]():
            return self.layer[Embedding[Self.dtype]].num_parameters()
        elif self.layer.isa[MaxPool2d[Self.dtype]]():
            return self.layer[MaxPool2d[Self.dtype]].num_parameters()
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            return self.layer[LayerNorm[Self.dtype]].num_parameters()
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            return self.layer[PositionalEmbedding[Self.dtype]].num_parameters()

        else:
            return 0

    def zero_grad(self):
        """Zero all parameter gradients."""
        for parameter in self.parameters():
            parameter[].zero_grad()

    def train(mut self):
        """Set module to training mode."""
        if self.layer.isa[Linear[Self.dtype]]():
            self.layer[Linear[Self.dtype]].train()
        if self.layer.isa[LinearBLAS[Self.dtype]]():
            self.layer[LinearBLAS[Self.dtype, mm]].train()
        elif self.layer.isa[ReLU[Self.dtype]]():
            self.layer[ReLU[Self.dtype]].train()
        elif self.layer.isa[GeLU[Self.dtype]]():
            self.layer[GeLU[Self.dtype]].train()

        elif self.layer.isa[Sigmoid[Self.dtype]]():
            self.layer[Sigmoid[Self.dtype]].train()
        elif self.layer.isa[Tanh[Self.dtype]]():
            self.layer[Tanh[Self.dtype]].train()
        elif self.layer.isa[Dropout[Self.dtype]]():
            self.layer[Dropout[Self.dtype]].train()
        elif self.layer.isa[Conv2D[Self.dtype]]():
            self.layer[Conv2D[Self.dtype]].train()
        elif self.layer.isa[Flatten[Self.dtype]]():
            self.layer[Flatten[Self.dtype]].train()
        elif self.layer.isa[Embedding[Self.dtype]]():
            self.layer[Embedding[Self.dtype]].train()
        elif self.layer.isa[MaxPool2d[Self.dtype]]():
            self.layer[MaxPool2d[Self.dtype]].train()
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            self.layer[LayerNorm[Self.dtype]].train()
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            self.layer[PositionalEmbedding[Self.dtype]].train()

    def eval(mut self):
        """Set module to evaluation mode."""
        if self.layer.isa[Linear[Self.dtype]]():
            self.layer[Linear[Self.dtype]].eval()
        elif self.layer.isa[LinearBLAS[Self.dtype]]():
            self.layer[LinearBLAS[Self.dtype]].eval()
        elif self.layer.isa[ReLU[Self.dtype]]():
            self.layer[ReLU[Self.dtype]].eval()
        elif self.layer.isa[GeLU[Self.dtype]]():
            self.layer[GeLU[Self.dtype]].eval()

        elif self.layer.isa[Sigmoid[Self.dtype]]():
            self.layer[Sigmoid[Self.dtype]].eval()
        elif self.layer.isa[Tanh[Self.dtype]]():
            self.layer[Tanh[Self.dtype]].eval()
        elif self.layer.isa[Dropout[Self.dtype]]():
            self.layer[Dropout[Self.dtype]].eval()
        elif self.layer.isa[Conv2D[Self.dtype]]():
            self.layer[Conv2D[Self.dtype]].eval()
        elif self.layer.isa[Flatten[Self.dtype]]():
            self.layer[Flatten[Self.dtype]].eval()
        elif self.layer.isa[Embedding[Self.dtype]]():
            self.layer[Embedding[Self.dtype]].eval()
        elif self.layer.isa[MaxPool2d[Self.dtype]]():
            self.layer[MaxPool2d[Self.dtype]].eval()
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            self.layer[LayerNorm[Self.dtype]].eval()
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            self.layer[PositionalEmbedding[Self.dtype]].eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Module[Self.dtype]:
        """Move this module to GPU.

        Dispatches to the appropriate layer's to_gpu().
        Activation layers (ReLU, GeLU Sigmoid, Tanh, Dropout, Flatten, MaxPool2d)
        are returned unchanged — they have no parameters.
        LinearBLAS panics — use Linear for GPU models.

        Args:
            gpu: Target GPU. Uses default GPU if None.

        Returns:
            New Module with GPU parameters.
        """
        if self.layer.isa[Linear[Self.dtype]]():
            var l = self.layer[Linear[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu))
            )
        elif self.layer.isa[LinearBLAS[Self.dtype]]():
            var l = self.layer[LinearBLAS[Self.dtype, mm]]
            _ = l.to_gpu(gpu)  # panics here
            return self  # unreachable
        elif self.layer.isa[Conv2D[Self.dtype]]():
            var l = self.layer[Conv2D[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu))
            )
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            var l = self.layer[LayerNorm[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu))
            )
        elif self.layer.isa[Embedding[Self.dtype]]():
            var l = self.layer[Embedding[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu))
            )
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            var l = self.layer[PositionalEmbedding[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu))
            )

        else:
            # RELU, GELU, SIGMOID, TANH, DROPOUT, FLATTEN, MAXPOOL2D
            # No parameters — return unchanged
            return self

    def to_cpu(self) raises -> Module[Self.dtype]:
        if self.layer.isa[Linear[Self.dtype]]():
            var l = self.layer[Linear[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()))
        elif self.layer.isa[LinearBLAS[Self.dtype]]():
            var l = self.layer[LinearBLAS[Self.dtype, mm]]
            _ = l.to_cpu()  # panics
            return self  # unreachable
        elif self.layer.isa[Conv2D[Self.dtype]]():
            var l = self.layer[Conv2D[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()))
        elif self.layer.isa[LayerNorm[Self.dtype]]():
            var l = self.layer[LayerNorm[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()))
        elif self.layer.isa[Embedding[Self.dtype]]():
            var l = self.layer[Embedding[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()))
        elif self.layer.isa[PositionalEmbedding[Self.dtype]]():
            var l = self.layer[PositionalEmbedding[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()))

        else:
            # RELU, GELU, SIGMOID, TANH, DROPOUT, FLATTEN, MAXPOOL2D — no-op
            return self


@fieldwise_init
struct Sequential[dtype: DType](Copyable & Writable):
    var modules: List[Module[Self.dtype]]

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sequential")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Sequential")

    def __init__(out self):
        self.modules = List[Module[Self.dtype]]()

    def append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            if m.layer.isa[LinearBLAS[Self.dtype]]():
                panic(
                    "LinearBLAS layer can not be added to Sequential. Use"
                    " SequentialBLAS"
                )
            self.modules.append(m)

    def __call__(
        mut self, xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var out = xs
        for i in range(len(self.modules)):
            var m = self.modules[i]
            out = m(out, sync=sync)
        return out

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        for module in self.modules:
            params.extend(module.parameters())
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        for i in range(len(self.modules)):
            var module_prefix = prefix + String(i) + "."
            result.extend(self.modules[i].named_parameters(module_prefix))
        return result^

    def num_parameters(self) -> Int:
        var total: Int = 0
        for parameter in self.parameters():
            total += parameter[].numels()
        return total

    def train(mut self):
        """Set all modules to training mode."""
        for i in range(len(self.modules)):
            self.modules[i].train()

    def eval(mut self):
        """Set all modules to evaluation mode."""
        for i in range(len(self.modules)):
            self.modules[i].eval()

    def to_gpu(
        self, gpu: Optional[GPU] = None, stop_grad: Bool = True
    ) raises -> Sequential[Self.dtype]:
        """Move all layers in this Sequential model to GPU.

        Each layer's to_gpu() is called. Layers with no parameters
        (ReLU, Sigmoid etc.) are returned unchanged.
        LinearBLAS layers will panic — use Linear for GPU models.

        Args:
            gpu: Target GPU. Uses default GPU if None.
            stop_grad: Whether to stop gradients at the transfer boundary.

        Returns:
            New Sequential with all parameterised layers on GPU.

        Example:
            ```mojo
            var model = Sequential[DType.float32]()
            model.append(
                Linear[DType.float32](784, 128, init_method="he").into(),
                ReLU[DType.float32]().into(),
                Linear[DType.float32](128, 10).into(),
            )
            var model_gpu = model.to_gpu()
            var optimizer = SGD[DType.float32](model_gpu.parameters(), lr=0.01, momentum=0.9)

            for epoch in range(epochs):
                for batch in train_loader:
                    var x_gpu = batch.features.to_gpu()
                    var loss = criterion(model_gpu(x_gpu), batch.labels.to_gpu())
                    optimizer.zero_grad()
                    var l = loss.sum()
                    l.backward()
                    optimizer.step()
                    # Read GPU grads if needed:
                    # model_gpu.parameters()[0][].grad().to_cpu().print()
            ```
        """
        var out = Sequential[Self.dtype]()
        for i in range(len(self.modules)):
            out.modules.append(self.modules[i].to_gpu(gpu))
        return out^

    def to_cpu(self) raises -> Sequential[Self.dtype]:
        """Move all layers back to CPU after training.

        Example:
            var model = model.to_gpu(stop_grad=True)
            # ... training loop ...
            model = model.to_cpu(stop_grad=True)  # persist weights
        """
        var out = Sequential[Self.dtype]()
        for i in range(len(self.modules)):
            out.modules.append(self.modules[i].to_cpu())
        return out^


@fieldwise_init
struct SequentialBLAS[dtype: DType](Copyable):
    var modules: List[Module[Self.dtype]]

    def __init__(out self):
        self.modules = List[Module[Self.dtype]]()

        # BLAS status
        if BLASCache.is_available():
            print("SequentialBLAS: BLAS acceleration enabled")
        else:
            print("SequentialBLAS: BLAS not available")

    def append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            if BLASCache.is_available() and m.layer.isa[LinearBLAS[Self.dtype]]():
                var linear = m.layer[LinearBLAS[Self.dtype, mm]]
                linear.blas_lite = BLASHandleLite[Self.dtype].from_cache()
                self.modules.append(linear^.into())
                continue

            self.modules.append(m)

    def __call__(
        mut self, xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var out = xs
        for i in range(len(self.modules)):
            var m = self.modules[i]
            out = m(out, sync=sync)
        return out

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        for module in self.modules:
            params.extend(module.parameters())
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        for i in range(len(self.modules)):
            var module_prefix = prefix + String(i) + "."
            result.extend(self.modules[i].named_parameters(module_prefix))
        return result^

    def num_parameters(self) -> Int:
        var total: Int = 0
        for parameter in self.parameters():
            total += parameter[].numels()
        return total

    def train(mut self):
        """Set all modules to training mode."""
        for i in range(len(self.modules)):
            self.modules[i].train()

    def eval(mut self):
        """Set all modules to evaluation mode."""
        for i in range(len(self.modules)):
            self.modules[i].eval()


# ModuleList — ordered container of modules
@fieldwise_init
struct ModuleListIterator[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    dtype: DType,
    forward: Bool = True,
](ImplicitlyCopyable & Sized & Iterable & Iterator):
    var index: Int
    var src: Pointer[ModuleList[Self.dtype], Self.origin]

    comptime Element = Module[Self.dtype]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    @always_inline
    def __iter__(ref self) -> Self:
        return self

    def __next__(mut self) -> Self.Element:
        comptime if Self.forward:
            var idx = self.index
            self.index += 1
            return self.src[].modules[idx]
        else:
            self.index -= 1
            return self.src[].modules[self.index]

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        comptime if Self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int
        comptime if Self.forward:
            iter_len = len(self.src[]) - self.index
        else:
            iter_len = self.index
        return (iter_len, {iter_len})


@fieldwise_init
struct ModuleList[dtype: DType](Copyable & Sized & Iterable):
    """Ordered container for modules.

    Like PyTorch's ModuleList — stores a list of modules and delegates
    parameters(), named_parameters(), num_parameters(), train(), eval(),
    to_gpu(), to_cpu(), and zero_grad() through to contained modules.

    Does NOT have __call__ — it's a container, not a forward chain.
    Does NOT guard against LinearBLAS (unlike Sequential).
    """

    var modules: List[Module[Self.dtype]]

    def __init__(out self):
        self.modules = List[Module[Self.dtype]]()

    def __init__(out self, *ms: Module[Self.dtype]):
        self.modules = List[Module[Self.dtype]]()
        for m in ms:
            self.modules.append(m)

    def append(mut self, m: Module[Self.dtype]):
        self.modules.append(m)

    def extend(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            self.modules.append(m)

    def insert(mut self, idx: Int, m: Module[Self.dtype]):
        self.modules.insert(idx, m)

    def __len__(self) -> Int:
        return len(self.modules)

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return ModuleListIterator[origin_of(self), Self.dtype](
            0, Pointer(to=self)
        )

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = ModuleListIterator[iterable_origin, Self.dtype]

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        for module in self.modules:
            params.extend(module.parameters())
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        for i in range(len(self.modules)):
            var module_prefix = prefix + String(i) + "."
            result.extend(self.modules[i].named_parameters(module_prefix))
        return result^

    def num_parameters(self) -> Int:
        var total: Int = 0
        for parameter in self.parameters():
            total += parameter[].numels()
        return total

    def train(mut self):
        """Set all modules to training mode."""
        for i in range(len(self.modules)):
            self.modules[i].train()

    def eval(mut self):
        """Set all modules to evaluation mode."""
        for i in range(len(self.modules)):
            self.modules[i].eval()

    def zero_grad(mut self):
        """Zero gradients for all modules."""
        for i in range(len(self.modules)):
            self.modules[i].zero_grad()

    def to_gpu(
        self, gpu: Optional[GPU] = None, stop_grad: Bool = True
    ) raises -> ModuleList[Self.dtype]:
        var out = ModuleList[Self.dtype]()
        for i in range(len(self.modules)):
            out.modules.append(self.modules[i].to_gpu(gpu))
        return out^

    def to_cpu(self) raises -> ModuleList[Self.dtype]:
        var out = ModuleList[Self.dtype]()
        for i in range(len(self.modules)):
            out.modules.append(self.modules[i].to_cpu())
        return out^


@fieldwise_init
struct Conv2D[dtype: DType](LayerTrait):
    """
        Conv2D layer wrapper for Sequential integration.

    Stores weights and bias as trainable parameters.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    var weight: Tensor[Self.dtype]  # (out_channels, in_channels, KH, KW)
    var bias: Optional[Tensor[Self.dtype]]  # (out_channels,) or None
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var dilation: Int
    var padding: Padding
    var training: Bool
    var delegate: Conv2dFused[Self.dtype]

    def __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        bias: Bool = True,
        bias_zero: Bool = True,
        init_seed: Optional[Int] = None,
        init_method: String = "he",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding.copy()
        self.training = True

        # Fan counts kernel taps: each output taps in_channels*kH*kW inputs.
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
            bias_len=out_channels if bias else 0,
            bias_zero=bias_zero,
            device=None,
        )

        self.delegate = Conv2dFused[Self.dtype]()
        print("Conv2D initialized:")
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
        self.padding = copy.padding.copy()
        self.training = copy.training
        self.delegate = copy.delegate

    def __init__(out self, *, deinit move: Self):
        self.weight = move.weight^
        self.bias = move.bias^
        self.in_channels = move.in_channels
        self.out_channels = move.out_channels
        self.kernel_size = move.kernel_size
        self.stride = move.stride
        self.dilation = move.dilation
        self.padding = move.padding^
        self.training = move.training
        self.delegate = move.delegate^

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref img_shape = x.shape()

        if img_shape.rank() != 4:
            panic(
                "Conv2D input must be 4D: (N, C, H, W), got shape: ",
                String(img_shape),
            )

        if img_shape[1] != self.in_channels:
            panic(
                "Conv2D input channels mismatch: expected ",
                String(self.in_channels),
                ", got ",
                String(img_shape[1]),
            )

        if self.training:
            return Conv2dFused[Self.dtype].forward[track_grad=True](
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
                requires_grad=True,
                sync=sync,
            )
        else:
            return Conv2dFused[Self.dtype].forward[track_grad=False](
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
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

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Conv2D[Self.dtype]:
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var out = self
        out.weight = weight_gpu^
        if out.bias:
            var bias_gpu = out.bias.value().to_gpu(gpu=gpu, stop_grad=True)
            out.bias = bias_gpu^
        return out^

    def to_cpu(self) raises -> Conv2D[Self.dtype]:
        var weight_cpu = self.weight.to_cpu(stop_grad=True)
        var out = self
        out.weight = weight_cpu^
        if out.bias:
            var bias_cpu = out.bias.value().to_cpu(stop_grad=True)
            out.bias = bias_cpu^
        return out^


@fieldwise_init
struct Flatten[dtype: DType](LayerTrait & RegisterPassable & Writable):
    """
        Flatten spatial dimensions: (N, C, H, W) → (N, C*H*W).
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype
    var training: Bool

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Flatten")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("Flatten")

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var xr = x
        ref shape = xr.shape()

        _ = """if shape.rank() != 4:
            panic("Flatten expects 4D input: (N, C, H, W)")

        var batch_size = shape[0]
        var flattened_size = shape[1] * shape[2] * shape[3]"""
        if shape.rank() < 2:
            panic("Flatten expects at least 2D input")

        var batch_size = shape[0]

        # Calculate flattened size (all dimensions except batch)
        var flattened_size = 1
        for i in range(1, shape.rank()):
            flattened_size *= shape[i]

        if self.training:
            return xr.reshape[track_grad=True](batch_size, flattened_size)
        else:
            return xr.reshape[track_grad=False](batch_size, flattened_size)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self))


# MixedSequential (merged from mixed_net.mojo)
#
# Runtime-erased mixed-dtype chain: one container holding layers of
# DIFFERENT comptime dtypes,
# connected by real grad-tracked boundary casts (Tensor.to_dtype ->
# ToDtypeBackward). The autograd graph is untouched - a boundary cast is an
# ordinary node.
#
# Mechanics are the third application of the blob + copy/destroy/fn-pointer
# triple proven by BackwardFn and ParentNode: each appended layer is
# move-initialized into its own heap slot; per-capability closures (forward,
# param/mode/transfer collection) are instantiated at append time against
# the concrete layer type and stored as erased fn pointers. Stored state
# references no graph/container types (GPU recursion invariant I1); chain
# dtype consistency is enforced at append time (I2), head/tail annotations
# at forward time unconditionally (I4).
#
# LayerTrait itself stays in tenmo/layer_trait.mojo - it breaks the
# net <-> {dropout, layernorm, embedding, pooling} import cycle, so those
# layer modules must not import net.mojo.

# Erased vtable vocabulary — raw bytes in, raw bytes out; no graph types.

comptime ForwardFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
    Bool,
) thin -> Pointer[UInt8, MutAnyOrigin]


comptime BoundaryForwardFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
    DType,
    Bool,
) thin -> Pointer[UInt8, MutAnyOrigin]


comptime CollectFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
) thin -> None


comptime NamedCollectFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
) thin -> None


comptime IntFn = def(Pointer[UInt8, MutAnyOrigin]) thin -> Int


comptime ModeFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Bool,
) thin -> None


comptime ZeroGradFn = def(Pointer[UInt8, MutAnyOrigin]) thin -> None


comptime TransferFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Bool,
    Pointer[UInt8, MutAnyOrigin],
) thin -> None


def make_module_destroyer[M: LayerTrait]() -> DestroyerFn:
    def destroy(p: Pointer[UInt8, MutAnyOrigin]) -> None:
        var mp = p.unsafe_bitcast[M]()
        mp.unsafe_deinit_pointee()
        mp.unsafe_free()

    return destroy


def make_module_copier[M: LayerTrait]() -> CopyFn:
    def copy_it(
        src: Pointer[UInt8, MutAnyOrigin]
    ) -> Pointer[UInt8, MutAnyOrigin]:
        var dst = unsafe_alloc[M](1)
        var sp = src.unsafe_bitcast[M]()
        dst.unsafe_write(sp[].copy())
        return dst.unsafe_bitcast[UInt8]().as_unsafe_any_origin()

    return copy_it


def make_forward[M: LayerTrait]() -> ForwardFn:
    """Plain hop: consume Tensor[M.InputDType] blob, produce output blob."""

    def fwd(
        m_ptr: Pointer[UInt8, MutAnyOrigin],
        in_ptr: Pointer[UInt8, MutAnyOrigin],
        sync: Bool,
    ) -> Pointer[UInt8, MutAnyOrigin]:
        comptime TI = Tensor[M.InputDType]
        comptime TO = Tensor[M.OutputDType]
        var mp = m_ptr.unsafe_bitcast[M]()
        var ip = in_ptr.unsafe_bitcast[TI]()
        var y = mp[].__call__(ip[], sync=sync)
        ip.unsafe_deinit_pointee()
        ip.unsafe_free()
        var op = unsafe_alloc[TO](1)
        op.unsafe_write(y^)
        return op.unsafe_bitcast[UInt8]().as_unsafe_any_origin()

    return fwd


def _boundary_hop[
    SrcDT: DType, M: LayerTrait
](
    m_ptr: Pointer[UInt8, MutAnyOrigin],
    in_ptr: Pointer[UInt8, MutAnyOrigin],
    sync: Bool,
) -> Pointer[UInt8, MutAnyOrigin]:
    comptime TI = Tensor[SrcDT]
    comptime TO = Tensor[M.OutputDType]
    var ip = in_ptr.unsafe_bitcast[TI]()
    var xc = ip[].to_dtype[M.InputDType]()
    var mp = m_ptr.unsafe_bitcast[M]()
    var y = mp[].__call__(xc^, sync=sync)
    ip.unsafe_deinit_pointee()
    ip.unsafe_free()
    var op = unsafe_alloc[TO](1)
    op.unsafe_write(y^)
    return op.unsafe_bitcast[UInt8]().as_unsafe_any_origin()


def make_boundary_forward[M: LayerTrait]() -> BoundaryForwardFn:
    """Boundary hop: grad-tracked cast src→M.InputDType, then the layer.

    The incoming dtype is only known at runtime here, so the cast leg
    dispatches over f32/f16/f64 (bfloat16 is excluded — Epsilon.value
    panics at compile time for it on this build).
    """

    def fwd(
        m_ptr: Pointer[UInt8, MutAnyOrigin],
        in_ptr: Pointer[UInt8, MutAnyOrigin],
        src_dtype: DType,
        sync: Bool,
    ) -> Pointer[UInt8, MutAnyOrigin]:
        if src_dtype == DType.float32:
            return _boundary_hop[DType.float32, M](m_ptr, in_ptr, sync)
        elif src_dtype == DType.float16:
            return _boundary_hop[DType.float16, M](m_ptr, in_ptr, sync)
        elif src_dtype == DType.float64:
            return _boundary_hop[DType.float64, M](m_ptr, in_ptr, sync)
        else:
            var msg = String(
                "MixedSequential: unsupported boundary source dtype "
            )
            msg += String(src_dtype)
            panic(msg)
        # unreachable — satisfies definite-return after panic (NoneType);
        # Pointer is non-nullable on this build, so hand back a dummy byte.
        var dummy = unsafe_alloc[UInt8](1)
        return dummy.as_unsafe_any_origin()

    return fwd


def make_param_collector[M: LayerTrait]() -> CollectFn:
    def collect(
        m_ptr: Pointer[UInt8, MutAnyOrigin],
        lst_ptr: Pointer[UInt8, MutAnyOrigin],
    ) -> None:
        comptime LT = List[Pointer[Tensor[M.OutputDType], MutAnyOrigin]]
        var mp = m_ptr.unsafe_bitcast[M]()
        var params = mp[].parameters()
        var lp = lst_ptr.unsafe_bitcast[LT]()
        lp[].extend(params^)

    return collect


def make_named_collector[M: LayerTrait]() -> NamedCollectFn:
    def collect(
        m_ptr: Pointer[UInt8, MutAnyOrigin],
        prefix_ptr: Pointer[UInt8, MutAnyOrigin],
        lst_ptr: Pointer[UInt8, MutAnyOrigin],
    ) -> None:
        comptime LT = List[NamedParameter[M.OutputDType]]
        var pp = prefix_ptr.unsafe_bitcast[String]()
        var pre = pp[]
        var mp = m_ptr.unsafe_bitcast[M]()
        var named = mp[].named_parameters(pre)
        var lp = lst_ptr.unsafe_bitcast[LT]()
        lp[].extend(named^)

    return collect


def make_num_params[M: LayerTrait]() -> IntFn:
    def count(m_ptr: Pointer[UInt8, MutAnyOrigin]) -> Int:
        var mp = m_ptr.unsafe_bitcast[M]()
        return mp[].num_parameters()

    return count


def make_mode_setter[M: LayerTrait]() -> ModeFn:
    def set_mode(m_ptr: Pointer[UInt8, MutAnyOrigin], training: Bool) -> None:
        var mp = m_ptr.unsafe_bitcast[M]()
        if training:
            mp[].train()
        else:
            mp[].eval()

    return set_mode


def make_zero_grader[M: LayerTrait]() -> ZeroGradFn:
    def zero(m_ptr: Pointer[UInt8, MutAnyOrigin]) -> None:
        var mp = m_ptr.unsafe_bitcast[M]()
        mp[].zero_grad()

    return zero


def make_transfer[M: LayerTrait]() -> TransferFn:
    """Replace the stored layer in place with its transferred copy."""

    def transfer(
        m_ptr: Pointer[UInt8, MutAnyOrigin],
        to_gpu_flag: Bool,
        gpu_blob: Pointer[UInt8, MutAnyOrigin],
    ) -> None:
        try:
            var mp = m_ptr.unsafe_bitcast[M]()
            var m = mp[].copy()
            var gp = gpu_blob.unsafe_bitcast[Optional[GPU]]()
            var gpu = gp[]
            var new_m = m.to_gpu(gpu) if to_gpu_flag else m.to_cpu()
            mp.unsafe_deinit_pointee()
            mp.unsafe_write(new_m^)
        except e:
            print(e)
            panic("MixedSequential device transfer failed")

    return transfer


@fieldwise_init
struct AnyModule(ImplicitlyCopyable):
    """AnyModule — one erased layer record (blob + triple + capability fns).
    Stored fields reference no graph/container types (invariant I1).
    """
    var ptr: Pointer[UInt8, MutUntrackedOrigin]
    var input_dtype: DType
    var io_dtype: DType
    var src_dtype: DType
    var destroy_fn: DestroyerFn
    var copy_fn: CopyFn
    var forward_fn: ForwardFn
    var cast_forward_fn: Optional[BoundaryForwardFn]
    var collect_params_fn: Optional[CollectFn]
    var named_params_fn: Optional[NamedCollectFn]
    var num_params_fn: Optional[IntFn]
    var mode_fn: Optional[ModeFn]
    var zero_grad_fn: Optional[ZeroGradFn]
    var transfer_fn: Optional[TransferFn]

    def __deinit__(deinit self):
        self.destroy_fn(self.ptr.as_unsafe_any_origin())

    def __init__(out self, *, deinit move: Self):
        self.ptr = move.ptr
        self.input_dtype = move.input_dtype
        self.io_dtype = move.io_dtype
        self.src_dtype = move.src_dtype
        self.destroy_fn = move.destroy_fn
        self.copy_fn = move.copy_fn
        self.forward_fn = move.forward_fn
        self.cast_forward_fn = move.cast_forward_fn
        self.collect_params_fn = move.collect_params_fn
        self.named_params_fn = move.named_params_fn
        self.num_params_fn = move.num_params_fn
        self.mode_fn = move.mode_fn
        self.zero_grad_fn = move.zero_grad_fn
        self.transfer_fn = move.transfer_fn

    def __init__(out self, *, copy: Self):
        self.ptr = copy.copy_fn(
            copy.ptr.as_unsafe_any_origin()
        ).unsafe_origin_cast[MutUntrackedOrigin]()
        self.input_dtype = copy.input_dtype
        self.io_dtype = copy.io_dtype
        self.src_dtype = copy.src_dtype
        self.destroy_fn = copy.destroy_fn
        self.copy_fn = copy.copy_fn
        self.forward_fn = copy.forward_fn
        self.cast_forward_fn = copy.cast_forward_fn
        self.collect_params_fn = copy.collect_params_fn
        self.named_params_fn = copy.named_params_fn
        self.num_params_fn = copy.num_params_fn
        self.mode_fn = copy.mode_fn
        self.zero_grad_fn = copy.zero_grad_fn
        self.transfer_fn = copy.transfer_fn


# MixedSequential — the container.


struct MixedSequential(Copyable):
    """Runtime-erased layer chain accepting heterogeneous comptime dtypes.

    Appending a layer whose InputDType differs from the running tail's
    OutputDType inserts a real grad-tracked boundary cast; backward needs
    zero changes. Homogeneous models are the trivial case (no casts).

    Chain dtypes come solely from each layer's InputDType/OutputDType (see
    LayerTrait): a misdeclared dtype inserts wrong casts or reinterprets
    blobs at the seam.
    """

    var records: List[AnyModule]
    var head_dtype: Optional[DType]
    var tail_dtype: Optional[DType]

    def __init__(out self):
        self.records = List[AnyModule]()
        self.head_dtype = None
        self.tail_dtype = None

    def __init__(out self, *, deinit move: Self):
        self.records = move.records^
        self.head_dtype = move.head_dtype
        self.tail_dtype = move.tail_dtype

    def __init__(out self, *, copy: Self):
        self.records = copy.records.copy()
        self.head_dtype = copy.head_dtype
        self.tail_dtype = copy.tail_dtype

    def __len__(self) -> Int:
        return len(self.records)

    def append[M: LayerTrait](mut self, var m: M):
        """Append a layer, inserting a grad-tracked cast when dtypes change."""
        var p = unsafe_alloc[M](1)
        p.unsafe_write(m^)

        var needs_cast = False
        if self.tail_dtype:
            needs_cast = self.tail_dtype.value() != M.InputDType

        var src_dtype = (
            self.tail_dtype.value() if self.tail_dtype else M.InputDType
        )

        var cast_fn: Optional[BoundaryForwardFn] = None
        if needs_cast:
            cast_fn = make_boundary_forward[M]()
        var collect_fn: Optional[CollectFn] = make_param_collector[M]()
        var named_fn: Optional[NamedCollectFn] = make_named_collector[M]()
        var num_fn: Optional[IntFn] = make_num_params[M]()
        var mode_f: Optional[ModeFn] = make_mode_setter[M]()
        var zero_fn: Optional[ZeroGradFn] = make_zero_grader[M]()
        var xfer_fn: Optional[TransferFn] = make_transfer[M]()

        self.records.append(
            AnyModule(
                p.unsafe_bitcast[UInt8]().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                M.InputDType,
                M.OutputDType,
                src_dtype,
                make_module_destroyer[M](),
                make_module_copier[M](),
                make_forward[M](),
                cast_fn,
                collect_fn,
                named_fn,
                num_fn,
                mode_f,
                zero_fn,
                xfer_fn,
            )
        )
        if not self.head_dtype:
            self.head_dtype = Optional[DType](M.InputDType)
        self.tail_dtype = Optional[DType](M.OutputDType)

    def forward[
        In: DType, Out: DType
    ](mut self, x: Tensor[In], sync: Bool = True) -> Tensor[Out]:
        """Run the chain; In/Out annotations are checked unconditionally."""
        if not self.head_dtype or not self.tail_dtype:
            panic("MixedSequential.forward: empty model")
        if In != self.head_dtype.value():
            panic(
                "MixedSequential.forward: input dtype annotation ",
                String(In),
                " does not match built chain head ",
                String(self.head_dtype.value()),
            )
        if Out != self.tail_dtype.value():
            panic(
                "MixedSequential.forward: output dtype annotation ",
                String(Out),
                " does not match built chain tail ",
                String(self.tail_dtype.value()),
            )

        comptime TI = Tensor[In]
        comptime TO = Tensor[Out]
        var slot = unsafe_alloc[TI](1)
        slot.unsafe_write(x.copy())

        var cur = slot.unsafe_bitcast[UInt8]().as_unsafe_any_origin()

        for i in range(len(self.records)):
            if self.records[i].cast_forward_fn:
                cur = self.records[i].cast_forward_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(),
                    cur,
                    self.records[i].src_dtype,
                    sync,
                )
            else:
                cur = self.records[i].forward_fn(
                    self.records[i].ptr.as_unsafe_any_origin(), cur, sync
                )

        var tail_ptr = cur.unsafe_bitcast[TO]()
        var out = tail_ptr[]
        tail_ptr.unsafe_deinit_pointee()
        tail_ptr.unsafe_free()
        return out

    def parameters_of[
        D: DType
    ](ref self,) -> List[Pointer[Tensor[D], MutAnyOrigin]]:
        """Collect parameter pointers of one dtype segment (multi-SGD story)."""
        var result = List[Pointer[Tensor[D], MutAnyOrigin]]()
        var lst_ptr = (
            Pointer(to=result)
            .unsafe_origin_cast[MutAnyOrigin]()
            .unsafe_bitcast[UInt8]()
        )
        for i in range(len(self.records)):
            if (
                self.records[i].io_dtype == D
                and self.records[i].collect_params_fn
            ):
                self.records[i].collect_params_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(), lst_ptr
                )
        return result^

    def named_parameters_of[
        D: DType
    ](ref self, prefix: String) -> List[NamedParameter[D]]:
        """Dtype-filtered named parameters (records are heterogeneous)."""
        var result = List[NamedParameter[D]]()
        var lst_ptr = (
            Pointer(to=result)
            .unsafe_origin_cast[MutAnyOrigin]()
            .unsafe_bitcast[UInt8]()
        )
        for i in range(len(self.records)):
            if (
                self.records[i].io_dtype == D
                and self.records[i].named_params_fn
            ):
                var pfx = prefix + String(i) + "."
                var pfx_ptr = (
                    Pointer(to=pfx)
                    .unsafe_origin_cast[MutAnyOrigin]()
                    .unsafe_bitcast[UInt8]()
                )
                self.records[i].named_params_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(),
                    pfx_ptr,
                    lst_ptr,
                )
        return result^

    def num_parameters(self) -> Int:
        var total = 0
        for i in range(len(self.records)):
            if self.records[i].num_params_fn:
                total += self.records[i].num_params_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin()
                )
        return total

    def train(mut self):
        for i in range(len(self.records)):
            if self.records[i].mode_fn:
                self.records[i].mode_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(), True
                )

    def eval(mut self):
        for i in range(len(self.records)):
            if self.records[i].mode_fn:
                self.records[i].mode_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(), False
                )

    def zero_grad(mut self):
        for i in range(len(self.records)):
            if self.records[i].zero_grad_fn:
                self.records[i].zero_grad_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin()
                )

    def to_gpu(self, gpu: Optional[GPU] = None) raises:
        """Transfer every stored layer to GPU, replacing records in place."""
        var gpu_opt = gpu
        var gpu_blob = (
            Pointer(to=gpu_opt)
            .unsafe_origin_cast[MutAnyOrigin]()
            .unsafe_bitcast[UInt8]()
        )
        for i in range(len(self.records)):
            if self.records[i].transfer_fn:
                self.records[i].transfer_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(), True, gpu_blob
                )

    def to_cpu(self) raises:
        var dummy_gpu = Optional[GPU](None)
        var gpu_blob = (
            Pointer(to=dummy_gpu)
            .unsafe_origin_cast[MutAnyOrigin]()
            .unsafe_bitcast[UInt8]()
        )
        for i in range(len(self.records)):
            if self.records[i].transfer_fn:
                self.records[i].transfer_fn.value()(
                    self.records[i].ptr.as_unsafe_any_origin(), False, gpu_blob
                )
