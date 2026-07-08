from .tensor import Tensor
from .shapes import Shape
from .gradbox import Gradbox
from std.math import sqrt
from .common_utils import panic, now, Epsilon
from std.utils import Variant
from .forwards import (
    Matmul,
    Adder,
    Multiplicator,
    Subtractor,
    Clip,
    Padding,
    Conv2dFused,
    MaxPool2d,
    Dropout,
    LayerNorm,
    Embedding,
)
from tenmo.shared import Reduction
from .bceloss import BCEWithLogitsLoss, BCELoss
from .mnemonics import (
    mm,
    mv,
    vm,
    dot,
    LINEAR,
    LINEAR_BLAS,
    RELU,
    SIGMOID,
    TANH,
    DROPOUT,
    CONV2D,
    FLATTEN,
    MAXPOOL2D,
    LAYER_NORM,
    EMBEDDING,
)
from .blashandle import BLASHandle, BLASHandleLite
from std.utils.numerics import neg_inf
from std.algorithm import parallelize
from .ndbuffer import NDBuffer
from std.random import seed, random_float64
from std.sys import simd_width_of
from .device import GPU
from .named_parameter import NamedParameter


struct Linear[dtype: DType, mode: Int = mm](
    ImplicitlyCopyable & Movable & Writable
):
    """Fully connected layer: y = xW + b."""

    comptime TAG = LINEAR

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
        init_method: String = "standard",  # "standard", "xavier", "he"
        bias: Bool = True,
        bias_zero: Bool = True,
    ):
        """
        Initialize Linear layer with configurable weight initialization.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            init_seed: Random seed for reproducibility.
            init_method: Weight initialization method:
                - "standard": Uniform[-0.1, 0.1] (good for [0,1] normalized inputs).
                - "xavier": Xavier/Glorot uniform (good for tanh/sigmoid).
                - "he": He/Kaiming normal (good for ReLU with standardized inputs).
            bias: If True, create bias parameter. If False, no bias is allocated.
            bias_zero: If True and bias=True, initialize bias to zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.training = True

        if init_method == "xavier":
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(in_features + out_features))
            )
            self.weight = Tensor[Self.dtype].rand(
                shape=Shape(in_features, out_features),
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                if not bias_zero:
                    self.bias = Optional(
                        Tensor[Self.dtype].rand(
                            Shape(out_features),
                            min=-limit,
                            max=limit,
                            init_seed=init_seed,
                            requires_grad=True,
                        )
                    )
                else:
                    self.bias = Optional(
                        Tensor[Self.dtype].zeros(
                            Shape(out_features), requires_grad=True
                        )
                    )
            else:
                self.bias = None

        elif init_method == "he":
            var std = sqrt(2.0 / Float64(in_features))
            self.weight = Tensor[Self.dtype].randn(
                shape=Shape(in_features, out_features),
                mean=0.0,
                std=std,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                if not bias_zero:
                    self.bias = Optional(
                        Tensor[Self.dtype].randn(
                            Shape(out_features),
                            mean=0.0,
                            std=std * 0.01,
                            init_seed=init_seed,
                            requires_grad=True,
                        )
                    )
                else:
                    self.bias = Optional(
                        Tensor[Self.dtype].zeros(
                            Shape(out_features), requires_grad=True
                        )
                    )
            else:
                self.bias = None

        else:  # "standard" or default
            var limit = Scalar[Self.dtype](0.1)
            self.weight = Tensor[Self.dtype].rand(
                shape=Shape(in_features, out_features),
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                if not bias_zero:
                    self.bias = Optional(
                        Tensor[Self.dtype].rand(
                            Shape(out_features),
                            min=-limit,
                            max=limit,
                            init_seed=init_seed,
                            requires_grad=True,
                        )
                    )
                else:
                    self.bias = Optional(
                        Tensor[Self.dtype].zeros(
                            Shape(out_features), requires_grad=True
                        )
                    )
            else:
                self.bias = None

        if self.weight.requires_grad:
            self.weight.buffer.buffer.shared()

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

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        if self.bias:
            params.append(
                UnsafePointer(to=self.bias.value())
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        var w = UnsafePointer(to=self.weight).unsafe_mut_cast[True]()
        result.append(
            NamedParameter(prefix + "weight", w.as_unsafe_any_origin())
        )
        if self.bias:
            var b = UnsafePointer(to=self.bias.value()).unsafe_mut_cast[True]()
            result.append(
                NamedParameter(prefix + "bias", b.as_unsafe_any_origin())
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
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    def to_gpu(
        deinit self, gpu: Optional[GPU] = None
    ) raises -> Linear[Self.dtype, Self.mode]:
        """Move this Linear layer to GPU.
        Consumes self — original CPU Linear is destroyed.
        """
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var out = self^
        out.weight = weight_gpu^
        if out.bias:
            var bias_gpu = out.bias.value().to_gpu(gpu=gpu, stop_grad=True)
            out.bias = bias_gpu^
        return out^

    def to_cpu(deinit self) raises -> Linear[Self.dtype, Self.mode]:
        """Move this Linear layer back to CPU.
        Consumes self — original GPU Linear is destroyed.
        """
        var weight_cpu = self.weight.to_cpu(stop_grad=True)
        var out = self^
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


@fieldwise_init
struct Profile(RegisterPassable & ImplicitlyCopyable):
    """Profile for a specific batch size."""

    var use_blas: Bool
    var profiled: Bool
    var call_count: Int
    var time_native: Float64
    var time_blas: Float64
    var profile_samples: Int  # Samples per method

    def __init__(out self, profile_samples: Int = 10):
        self.use_blas = False
        self.profiled = False
        self.call_count = 0
        self.time_native = 0.0
        self.time_blas = 0.0
        self.profile_samples = profile_samples


struct LinearBLAS[dtype: DType, mode: Int = mm](ImplicitlyCopyable & Movable):
    """Fully connected layer: y = xW + b."""

    comptime TAG = LINEAR_BLAS

    var weight: Tensor[Self.dtype]
    var bias: Optional[Tensor[Self.dtype]]
    var in_features: Int
    var out_features: Int
    var training: Bool
    var blas_lite: Optional[BLASHandleLite[Self.dtype]]
    var train_profile: Profile
    var validation_profile: Profile

    def __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "standard",  # "standard", "xavier", "he"
        bias: Bool = True,
        bias_zero: Bool = True,
        profile_samples: Int = 10,
    ):
        """
        Initialize LinearBLAS layer with configurable weight initialization.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            init_seed: Random seed for reproducibility.
            init_method: Weight initialization method:
                - "standard": Uniform[-0.1, 0.1] (good for [0,1] normalized inputs).
                - "xavier": Xavier/Glorot uniform (good for tanh/sigmoid).
                - "he": He/Kaiming normal (good for ReLU with standardized inputs).
            bias: If True, create bias parameter. If False, no bias is allocated.
            bias_zero: If True and bias=True, initialize bias to zeros.
            profile_samples: Samples per method for profiling:
                - 0: Skip profiling, always use native
                - 1-3: Fast profiling (may be noisy)
                - 5-10: Recommended (default: 10)
                - >10: Extra stable, slower startup.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.training = True
        self.blas_lite = None
        self.train_profile = Profile(profile_samples)
        self.validation_profile = Profile(profile_samples)

        if profile_samples == 0:
            self.train_profile.profiled = True
            self.train_profile.use_blas = False
            self.validation_profile.profiled = True
            self.validation_profile.use_blas = False

        if init_method == "xavier":
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(in_features + out_features))
            )
            self.weight = Tensor[Self.dtype].rand(
                shape=Shape(in_features, out_features),
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                if not bias_zero:
                    self.bias = Optional(
                        Tensor[Self.dtype].rand(
                            Shape(out_features),
                            min=-limit,
                            max=limit,
                            init_seed=init_seed,
                            requires_grad=True,
                        )
                    )
                else:
                    self.bias = Optional(
                        Tensor[Self.dtype].zeros(
                            Shape(out_features), requires_grad=True
                        )
                    )
            else:
                self.bias = None

        elif init_method == "he":
            var std = sqrt(2.0 / Float64(in_features))
            self.weight = Tensor[Self.dtype].randn(
                shape=Shape(in_features, out_features),
                mean=0.0,
                std=std,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                if not bias_zero:
                    self.bias = Optional(
                        Tensor[Self.dtype].randn(
                            Shape(out_features),
                            mean=0.0,
                            std=std * 0.01,
                            init_seed=init_seed,
                            requires_grad=True,
                        )
                    )
                else:
                    self.bias = Optional(
                        Tensor[Self.dtype].zeros(
                            Shape(out_features), requires_grad=True
                        )
                    )
            else:
                self.bias = None

        else:  # "standard" or default
            var limit = Scalar[Self.dtype](0.1)
            self.weight = Tensor[Self.dtype].rand(
                shape=Shape(in_features, out_features),
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                if not bias_zero:
                    self.bias = Optional(
                        Tensor[Self.dtype].rand(
                            Shape(out_features),
                            min=-limit,
                            max=limit,
                            init_seed=init_seed,
                            requires_grad=True,
                        )
                    )
                else:
                    self.bias = Optional(
                        Tensor[Self.dtype].zeros(
                            Shape(out_features), requires_grad=True
                        )
                    )
            else:
                self.bias = None

        self.weight.buffer.buffer.shared()
        if self.bias:
            self.bias.value().buffer.buffer.shared()

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

        ref profile = (
            self.train_profile if self.training else self.validation_profile
        )
        if profile.profiled:
            if profile.use_blas:
                return self.matmul_blas(xs, sync=sync)
            else:
                return self.matmul(xs, sync=sync)

        else:
            var curr_profile = profile.copy()
            var can_profile = (
                self.blas_lite
                and self.weight.is_contiguous()
                and xs.is_contiguous()
            )

            if not can_profile:
                curr_profile.profiled = True
                curr_profile.use_blas = False

                if self.training:
                    self.train_profile = curr_profile^
                else:
                    self.validation_profile = curr_profile^

                return self.matmul(xs, sync=sync)

            elif curr_profile.call_count < curr_profile.profile_samples:
                var start = now()
                var result = self.matmul(xs, sync=sync)
                curr_profile.time_native += now() - start
                curr_profile.call_count += 1

                if self.training:
                    self.train_profile = curr_profile^
                else:
                    self.validation_profile = curr_profile^

                return result^

            else:
                var start = now()
                var result = self.matmul_blas(xs, sync=sync)
                curr_profile.time_blas += now() - start
                curr_profile.call_count += 1

                if curr_profile.call_count == curr_profile.profile_samples * 2:
                    curr_profile.use_blas = (
                        curr_profile.time_blas < curr_profile.time_native
                    )
                    curr_profile.profiled = True

                    print(
                        "LinearBLAS layer profiling complete.",
                        "Profiled samples: ",
                        curr_profile.profile_samples,
                        "Training: ",
                        self.training,
                    )
                    print(
                        "  in_features: ",
                        self.in_features,
                        "out_features: ",
                        self.out_features,
                        "batch_size: ",
                        xs_shape[0],
                    )
                    print(
                        "  Native matmul calls):",
                        curr_profile.time_native,
                        "sec",
                    )
                    print(
                        "  BLAS matmul calls:",
                        curr_profile.time_blas,
                        "sec",
                    )
                    print(
                        "  Selected:",
                        "BLAS" if curr_profile.use_blas else "Native",
                    )
                if self.training:
                    self.train_profile = curr_profile^
                else:
                    self.validation_profile = curr_profile^

                return result^

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
                xs, self.weight, transpose_A=False, transpose_B=False
            )
            if self.bias:
                result = Adder[Self.dtype].forward[track_grad=True](
                    matmul_out^, self.bias.value(), sync=sync
                )
            else:
                result = matmul_out^

        else:
            var matmul_out = self.blas_lite.value().matmul[track_grad=False](
                xs, self.weight, transpose_A=False, transpose_B=False
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
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        if self.bias:
            params.append(
                UnsafePointer(to=self.bias.value())
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        var w = UnsafePointer(to=self.weight).unsafe_mut_cast[True]()
        result.append(
            NamedParameter(prefix + "weight", w.as_unsafe_any_origin())
        )
        if self.bias:
            var b = UnsafePointer(to=self.bias.value()).unsafe_mut_cast[True]()
            result.append(
                NamedParameter(prefix + "bias", b.as_unsafe_any_origin())
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
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)


struct ReLU[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var training: Bool
    comptime TAG = RELU

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return x.relu[track_grad=True](sync=sync)
        else:
            return x.relu[track_grad=False](sync=sync)

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        return List[NamedParameter[Self.dtype]]()

    def num_parameters(self) -> Int:
        return 0

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self

    def to_cpu(self) raises -> Self:
        """No-op — no parameters to move."""
        return self


struct Sigmoid[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var training: Bool
    comptime TAG = SIGMOID

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return x.sigmoid[track_grad=True](sync=sync)
        else:
            return x.sigmoid[track_grad=False](sync=sync)

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        return List[NamedParameter[Self.dtype]]()

    def num_parameters(self) -> Int:
        return 0

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self

    def to_cpu(self) raises -> Self:
        """No-op — no parameters to move."""
        return self


struct Tanh[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var training: Bool
    comptime TAG = TANH

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return x.tanh[track_grad=True](sync=sync)
        else:
            return x.tanh[track_grad=False](sync=sync)

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        return List[NamedParameter[Self.dtype]]()

    def num_parameters(self) -> Int:
        return 0

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self

    def to_cpu(self) raises -> Self:
        """No-op — no parameters to move."""
        return self


# Refer to operators & matmul
# Defined in operators

# comptime dot = ?  # dot product
# comptime vm = ?  # vector & tensor matmul
# comptime mv = ?  # tensor & vector matmul
# comptime mm = ?  # tensor & tensor matmul

comptime Layer[dtype: DType] = Variant[
    Linear[dtype, mm],
    LinearBLAS[dtype, mm],
    ReLU[dtype],
    Sigmoid[dtype],
    Tanh[dtype],
    Dropout[dtype],
    Conv2D[dtype],
    Flatten[dtype],
    MaxPool2d[dtype],
    LayerNorm[dtype],
    Embedding[dtype],
]


@fieldwise_init
struct Module[dtype: DType](ImplicitlyCopyable & Movable):
    var layer: Layer[Self.dtype]
    var tag: Int

    def __call__(
        mut self, mut xs: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.tag == LINEAR:
            return self.layer[Linear[Self.dtype, mm]](xs, sync=sync)
        if self.tag == LINEAR_BLAS:
            return self.layer[LinearBLAS[Self.dtype, mm]](xs, sync=sync)
        elif self.tag == RELU:
            return self.layer[ReLU[Self.dtype]](xs, sync=sync)
        elif self.tag == SIGMOID:
            return self.layer[Sigmoid[Self.dtype]](xs, sync=sync)
        elif self.tag == TANH:
            return self.layer[Tanh[Self.dtype]](xs, sync=sync)
        elif self.tag == DROPOUT:
            return self.layer[Dropout[Self.dtype]](xs, sync=sync)
        elif self.tag == CONV2D:
            return self.layer[Conv2D[Self.dtype]](xs, sync=sync)
        elif self.tag == FLATTEN:
            return self.layer[Flatten[Self.dtype]](xs, sync=sync)
        elif self.tag == EMBEDDING:
            return self.layer[Embedding[Self.dtype]](xs, sync=sync)
        elif self.tag == MAXPOOL2D:
            return self.layer[MaxPool2d[Self.dtype]](xs, sync=sync)
        elif self.tag == LAYER_NORM:
            return self.layer[LayerNorm[Self.dtype]](xs, sync=sync)

        else:
            panic("Unknown module type")
            return Tensor[Self.dtype].scalar(0)

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        if self.tag == LINEAR:
            return self.layer[Linear[Self.dtype]].parameters()
        elif self.tag == LINEAR_BLAS:
            return self.layer[LinearBLAS[Self.dtype]].parameters()
        elif self.tag == CONV2D:
            return self.layer[Conv2D[Self.dtype]].parameters()
        elif self.tag == LAYER_NORM:
            return self.layer[LayerNorm[Self.dtype]].parameters()
        elif self.tag == EMBEDDING:
            return self.layer[Embedding[Self.dtype]].parameters()

        else:
            return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        if self.tag == LINEAR:
            return self.layer[Linear[Self.dtype]].named_parameters(prefix)
        elif self.tag == LINEAR_BLAS:
            return self.layer[LinearBLAS[Self.dtype]].named_parameters(prefix)
        elif self.tag == CONV2D:
            return self.layer[Conv2D[Self.dtype]].named_parameters(prefix)
        elif self.tag == LAYER_NORM:
            return self.layer[LayerNorm[Self.dtype]].named_parameters(prefix)
        else:
            return List[NamedParameter[Self.dtype]]()

    def num_parameters(self) -> Int:
        if self.tag == LINEAR:
            return self.layer[Linear[Self.dtype, mm]].num_parameters()
        if self.tag == LINEAR_BLAS:
            return self.layer[LinearBLAS[Self.dtype, mm]].num_parameters()
        elif self.tag == RELU:
            return self.layer[ReLU[Self.dtype]].num_parameters()
        elif self.tag == SIGMOID:
            return self.layer[Sigmoid[Self.dtype]].num_parameters()
        elif self.tag == TANH:
            return self.layer[Tanh[Self.dtype]].num_parameters()
        elif self.tag == DROPOUT:
            return self.layer[Dropout[Self.dtype]].num_parameters()
        elif self.tag == CONV2D:
            return self.layer[Conv2D[Self.dtype]].num_parameters()
        elif self.tag == FLATTEN:
            return self.layer[Flatten[Self.dtype]].num_parameters()
        elif self.tag == EMBEDDING:
            return self.layer[Embedding[Self.dtype]].num_parameters()
        elif self.tag == MAXPOOL2D:
            return self.layer[MaxPool2d[Self.dtype]].num_parameters()
        elif self.tag == LAYER_NORM:
            return self.layer[LayerNorm[Self.dtype]].num_parameters()

        else:
            return 0

    def zero_grad(self):
        """Zero all parameter gradients."""
        for parameter in self.parameters():
            parameter[].zero_grad()

    def train(mut self):
        """Set module to training mode."""
        if self.tag == LINEAR:
            self.layer[Linear[Self.dtype, mm]].train()
        if self.tag == LINEAR_BLAS:
            self.layer[LinearBLAS[Self.dtype, mm]].train()
        elif self.tag == RELU:
            self.layer[ReLU[Self.dtype]].train()
        elif self.tag == SIGMOID:
            self.layer[Sigmoid[Self.dtype]].train()
        elif self.tag == TANH:
            self.layer[Tanh[Self.dtype]].train()
        elif self.tag == DROPOUT:
            self.layer[Dropout[Self.dtype]].train()
        elif self.tag == CONV2D:
            self.layer[Conv2D[Self.dtype]].train()
        elif self.tag == FLATTEN:
            self.layer[Flatten[Self.dtype]].train()
        elif self.tag == EMBEDDING:
            self.layer[Embedding[Self.dtype]].train()
        elif self.tag == MAXPOOL2D:
            self.layer[MaxPool2d[Self.dtype]].train()
        elif self.tag == LAYER_NORM:
            self.layer[LayerNorm[Self.dtype]].train()

    def eval(mut self):
        """Set module to evaluation mode."""
        if self.tag == LINEAR:
            self.layer[Linear[Self.dtype]].eval()
        elif self.tag == LINEAR_BLAS:
            self.layer[LinearBLAS[Self.dtype]].eval()
        elif self.tag == RELU:
            self.layer[ReLU[Self.dtype]].eval()
        elif self.tag == SIGMOID:
            self.layer[Sigmoid[Self.dtype]].eval()
        elif self.tag == TANH:
            self.layer[Tanh[Self.dtype]].eval()
        elif self.tag == DROPOUT:
            self.layer[Dropout[Self.dtype]].eval()
        elif self.tag == CONV2D:
            self.layer[Conv2D[Self.dtype]].eval()
        elif self.tag == FLATTEN:
            self.layer[Flatten[Self.dtype]].eval()
        elif self.tag == EMBEDDING:
            self.layer[Embedding[Self.dtype]].eval()
        elif self.tag == MAXPOOL2D:
            self.layer[MaxPool2d[Self.dtype]].eval()
        elif self.tag == LAYER_NORM:
            self.layer[LayerNorm[Self.dtype]].eval()

    def to_gpu(
        mut self, gpu: Optional[GPU] = None
    ) raises -> Module[Self.dtype]:
        """Move this module to GPU.

        Dispatches to the appropriate layer's to_gpu().
        Activation layers (ReLU, Sigmoid, Tanh, Dropout, Flatten, MaxPool2d)
        are returned unchanged — they have no parameters.
        LinearBLAS panics — use Linear for GPU models.

        Args:
            gpu: Target GPU. Uses default GPU if None.

        Returns:
            New Module with GPU parameters.
        """
        if self.tag == LINEAR:
            var l = self.layer[Linear[Self.dtype, mm]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu)), self.tag
            )
        elif self.tag == LINEAR_BLAS:
            var l = self.layer[LinearBLAS[Self.dtype, mm]]
            _ = l.to_gpu(gpu)  # panics here
            return self  # unreachable
        elif self.tag == CONV2D:
            var l = self.layer[Conv2D[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu)), self.tag
            )
        elif self.tag == LAYER_NORM:
            var l = self.layer[LayerNorm[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu)), self.tag
            )
        elif self.tag == EMBEDDING:
            var l = self.layer[Embedding[Self.dtype]]
            return Module[Self.dtype](
                Layer[Self.dtype](l.to_gpu(gpu)), self.tag
            )

        else:
            # RELU, SIGMOID, TANH, DROPOUT, FLATTEN, MAXPOOL2D
            # No parameters — return unchanged
            return self

    def to_cpu(mut self) raises -> Module[Self.dtype]:
        if self.tag == LINEAR:
            var l = self.layer[Linear[Self.dtype, mm]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()), self.tag)
        elif self.tag == LINEAR_BLAS:
            var l = self.layer[LinearBLAS[Self.dtype, mm]]
            _ = l.to_cpu()  # panics
            return self  # unreachable
        elif self.tag == CONV2D:
            var l = self.layer[Conv2D[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()), self.tag)
        elif self.tag == LAYER_NORM:
            var l = self.layer[LayerNorm[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()), self.tag)
        elif self.tag == EMBEDDING:
            var l = self.layer[Embedding[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_cpu()), self.tag)

        else:
            # RELU, SIGMOID, TANH, DROPOUT, FLATTEN, MAXPOOL2D — no-op
            return self


@fieldwise_init
struct Sequential[dtype: DType](Copyable & Movable):
    var modules: List[Module[Self.dtype]]

    def __init__(out self):
        self.modules = List[Module[Self.dtype]]()

    def append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            if m.tag == LINEAR_BLAS:
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
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
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
        mut self, gpu: Optional[GPU] = None, stop_grad: Bool = True
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

    def to_cpu(mut self) raises -> Sequential[Self.dtype]:
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
struct SequentialBLAS[dtype: DType](Copyable & Movable):
    var modules: List[Module[Self.dtype]]
    var blas_handle: BLASHandle[Self.dtype]

    def __init__(out self):
        self.modules = List[Module[Self.dtype]]()
        self.blas_handle = BLASHandle[Self.dtype]()

        # BLAS status
        if self.blas_handle.is_initialized():
            print("SequeentialBLAS: BLAS acceleration enabled")
        else:
            print(
                "SequeentialBLAS: BLAS not available -",
                self.blas_handle.get_error(),
            )

    def append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            if self.blas_handle.is_initialized() and m.tag == LINEAR_BLAS:
                var linear = m.layer[LinearBLAS[Self.dtype, mm]]
                linear.blas_lite = self.blas_handle.lite_handle()
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
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
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


# -----------------------------------------
# ModuleList — ordered container of modules
# -----------------------------------------
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
struct ModuleList[dtype: DType](Copyable & Movable & Sized & Iterable):
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
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
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
        mut self, gpu: Optional[GPU] = None, stop_grad: Bool = True
    ) raises -> ModuleList[Self.dtype]:
        var out = ModuleList[Self.dtype]()
        for i in range(len(self.modules)):
            out.modules.append(self.modules[i].to_gpu(gpu))
        return out^

    def to_cpu(mut self) raises -> ModuleList[Self.dtype]:
        var out = ModuleList[Self.dtype]()
        for i in range(len(self.modules)):
            out.modules.append(self.modules[i].to_cpu())
        return out^


# -----------------------------------------
# Mean Squared Error Loss
# -----------------------------------------
@fieldwise_init
struct MSELoss[dtype: DType = DType.float32](RegisterPassable):
    var training: Bool

    def __init__(out self):
        self.training = True

    def __call__(
        self,
        preds: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        if self.training:
            return preds.mse[track_grad=True](target, sync=sync)
        else:
            return preds.mse[track_grad=False](target, sync=sync)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False


@fieldwise_init
struct Conv2D[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Conv2D layer wrapper for Sequential integration.

    Stores weights and bias as trainable parameters.
    """

    comptime TAG = CONV2D

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

        var weight_shape = Shape(
            out_channels, in_channels, kernel_size, kernel_size
        )
        var fan_in = in_channels * kernel_size * kernel_size
        if init_method == "xavier":
            var fan_out = out_channels * kernel_size * kernel_size
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(fan_in + fan_out))
            )
            self.weight = Tensor[Self.dtype].rand(
                shape=weight_shape,
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                self.bias = Optional(
                    Tensor[Self.dtype].rand(
                        Shape(out_channels),
                        min=-limit,
                        max=limit,
                        init_seed=init_seed,
                        requires_grad=True,
                    )
                )
            else:
                self.bias = None

        elif init_method == "he":
            var std = sqrt(2.0 / Float64(fan_in))
            self.weight = Tensor[Self.dtype].randn(
                shape=weight_shape,
                mean=0.0,
                std=std,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                self.bias = Optional(
                    Tensor[Self.dtype].randn(
                        Shape(out_channels),
                        mean=0.0,
                        std=std * 0.01,
                        init_seed=init_seed,
                        requires_grad=True,
                    )
                )
            else:
                self.bias = None

        else:  # "standard"
            var limit = Scalar[Self.dtype](0.1)
            self.weight = Tensor[Self.dtype].rand(
                shape=weight_shape,
                min=-limit,
                max=limit,
                init_seed=init_seed,
                requires_grad=True,
            )
            if bias:
                self.bias = Optional(
                    Tensor[Self.dtype].rand(
                        Shape(out_channels),
                        min=-limit,
                        max=limit,
                        init_seed=init_seed,
                        requires_grad=True,
                    )
                )
            else:
                self.bias = None

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
        mut self, image: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref img_shape = image.shape()

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
            return self.delegate[track_grad=True](
                image,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
                requires_grad=True,
                sync=sync,
            )
        else:
            return self.delegate[track_grad=False](
                image,
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
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        if self.bias:
            params.append(
                UnsafePointer(to=self.bias.value())
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        var w = UnsafePointer(to=self.weight).unsafe_mut_cast[True]()
        result.append(
            NamedParameter(prefix + "weight", w.as_unsafe_any_origin())
        )
        if self.bias:
            var b = UnsafePointer(to=self.bias.value()).unsafe_mut_cast[True]()
            result.append(
                NamedParameter(prefix + "bias", b.as_unsafe_any_origin())
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
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    def to_gpu(
        deinit self, gpu: Optional[GPU] = None
    ) raises -> Conv2D[Self.dtype]:
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var out = self^
        out.weight = weight_gpu^
        if out.bias:
            var bias_gpu = out.bias.value().to_gpu(gpu=gpu, stop_grad=True)
            out.bias = bias_gpu^
        return out^

    def to_cpu(deinit self) raises -> Conv2D[Self.dtype]:
        var weight_cpu = self.weight.to_cpu(stop_grad=True)
        var out = self^
        out.weight = weight_cpu^
        if out.bias:
            var bias_cpu = out.bias.value().to_cpu(stop_grad=True)
            out.bias = bias_cpu^
        return out^


@fieldwise_init
struct Flatten[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """
    Flatten spatial dimensions: (N, C, H, W) → (N, C*H*W).
    """

    comptime TAG = FLATTEN
    var training: Bool

    def __init__(out self):
        self.training = True

    def __init__(out self, *, copy: Self):
        self.training = copy.training

    def __call__(
        self, mut x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref shape = x.shape()

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
            return x.reshape[track_grad=True](batch_size, flattened_size)
        else:
            return x.reshape[track_grad=False](batch_size, flattened_size)

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        return List[NamedParameter[Self.dtype]]()

    def num_parameters(self) -> Int:
        return 0

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self

    def to_cpu(self) raises -> Self:
        """No-op — no parameters to move."""
        return self
