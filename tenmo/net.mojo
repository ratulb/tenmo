from .tensor import Tensor
from .shapes import Shape
from .gradbox import Gradbox
from std.math import sqrt
from .common_utils import panic, now
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
)
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
)
from .blashandle import BLASHandle, BLASHandleLite
from std.utils.numerics import neg_inf
from std.algorithm import parallelize
from .ndbuffer import NDBuffer
from std.random import seed, random_float64
from std.sys import simd_width_of
from .device import GPU

@fieldwise_init
struct Linear[dtype: DType, mode: Int = mm](ImplicitlyCopyable & Movable):
    """Fully connected layer: y = xW + b."""

    comptime TAG = LINEAR

    var weight: Tensor[Self.dtype]
    var bias: Tensor[Self.dtype]
    var in_features: Int
    var out_features: Int
    var training: Bool

    fn __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "standard",  # "standard", "xavier", "he"
        bias_zero: Bool = True,
        weight_factor: Scalar[Self.dtype] = Scalar[Self.dtype](1),
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
            bias_zero: If True, initialize bias to zeros.
            weight_factor: Scaling factor for weight initialization.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.training = True

        if init_method == "xavier":
            # Xavier/Glorot uniform initialization
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(in_features + out_features))
            )
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=Shape(in_features, out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if not bias_zero:
                self.bias = Tensor[Self.dtype].rand(
                    Shape(out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].zeros(
                    Shape(out_features), requires_grad=True
                )

        elif init_method == "he":
            # He/Kaiming normal initialization (for ReLU)
            var std = sqrt(2.0 / Float64(in_features))
            self.weight = (
                Tensor[Self.dtype].randn(
                    shape=Shape(in_features, out_features),
                    mean=0.0,
                    std=std,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if not bias_zero:
                self.bias = Tensor[Self.dtype].randn(
                    Shape(out_features),
                    mean=0.0,
                    std=std * 0.01,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].zeros(
                    Shape(out_features), requires_grad=True
                )

        else:  # "standard" or default
            # Simple uniform initialization (good for [0,1] normalized inputs)
            var limit = Scalar[Self.dtype](0.1)
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=Shape(in_features, out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if not bias_zero:
                self.bias = Tensor[Self.dtype].rand(
                    Shape(out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].zeros(
                    Shape(out_features), requires_grad=True
                )

    fn __call__(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
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
            ](xs, self.weight)
            result = Adder[Self.dtype].forward[track_grad=True](
                matmul_out^, self.bias
            )

        else:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=Self.mode
            ](xs, self.weight)
            result = Adder[Self.dtype].forward[track_grad=False](
                matmul_out^, self.bias
            )

        return result^

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_any_origin()
        )
        params.append(
            UnsafePointer(to=self.bias).unsafe_mut_cast[True]().as_any_origin()
        )

        return params^

    fn num_parameters(self) -> Int:
        return self.weight.numels() + self.bias.numels()

    fn train(mut self):
        """Set to training mode - enables gradient tracking."""
        self.training = True

    fn eval(mut self):
        """Set to evaluation mode - disables gradient tracking."""
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    fn to_gpu(
        deinit self,
        gpu: Optional[GPU] = None
    ) raises -> Linear[Self.dtype, Self.mode]:
        """Move this Linear layer to GPU.
        Consumes self — original CPU Linear is destroyed.
        """
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var bias_gpu   = self.bias.to_gpu(gpu=gpu, stop_grad=True)
        var out = self^
        out.weight = weight_gpu^
        out.bias   = bias_gpu^
        return out^

@fieldwise_init
struct Profile(RegisterPassable & ImplicitlyCopyable):
    """Profile for a specific batch size."""

    var use_blas: Bool
    var profiled: Bool
    var call_count: Int
    var time_native: Float64
    var time_blas: Float64
    var profile_samples: Int  # Samples per method

    fn __init__(out self, profile_samples: Int = 10):
        self.use_blas = False
        self.profiled = False
        self.call_count = 0
        self.time_native = 0.0
        self.time_blas = 0.0
        self.profile_samples = profile_samples


@fieldwise_init
struct LinearBLAS[dtype: DType, mode: Int = mm](ImplicitlyCopyable & Movable):
    """Fully connected layer: y = xW + b."""

    comptime TAG = LINEAR_BLAS

    var weight: Tensor[Self.dtype]
    var bias: Tensor[Self.dtype]
    var in_features: Int
    var out_features: Int
    var training: Bool
    var blas_lite: Optional[BLASHandleLite[Self.dtype]]
    var train_profile: Profile
    var validation_profile: Profile

    fn __init__(
        out self,
        in_features: Int,
        out_features: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "standard",  # "standard", "xavier", "he"
        bias_zero: Bool = True,
        weight_factor: Scalar[Self.dtype] = Scalar[Self.dtype](1),
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
            bias_zero: If True, initialize bias to zeros.
            weight_factor: Scaling factor for weight initialization.
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

        # If profiling disabled, skip directly to native
        if profile_samples == 0:
            self.train_profile.profiled = True
            self.train_profile.use_blas = False
            self.validation_profile.profiled = True
            self.validation_profile.use_blas = False

        if init_method == "xavier":
            # Xavier/Glorot uniform initialization
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(in_features + out_features))
            )
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=Shape(in_features, out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if not bias_zero:
                self.bias = Tensor[Self.dtype].rand(
                    Shape(out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].zeros(
                    Shape(out_features), requires_grad=True
                )

        elif init_method == "he":
            # He/Kaiming normal initialization (for ReLU)
            var std = sqrt(2.0 / Float64(in_features))
            self.weight = (
                Tensor[Self.dtype].randn(
                    shape=Shape(in_features, out_features),
                    mean=0.0,
                    std=std,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if not bias_zero:
                self.bias = Tensor[Self.dtype].randn(
                    Shape(out_features),
                    mean=0.0,
                    std=std * 0.01,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].zeros(
                    Shape(out_features), requires_grad=True
                )

        else:  # "standard" or default
            # Simple uniform initialization (good for [0,1] normalized inputs)
            var limit = Scalar[Self.dtype](0.1)
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=Shape(in_features, out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if not bias_zero:
                self.bias = Tensor[Self.dtype].rand(
                    Shape(out_features),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].zeros(
                    Shape(out_features), requires_grad=True
                )

    fn __call__(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
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
                return self.matmul_blas(xs)
            else:
                return self.matmul(xs)

        else:  # We are still in profiling phase
            var curr_profile = profile.copy()
            # Check if we can/should profile
            var can_profile = (
                self.blas_lite  # BLAS is available
                and self.weight.is_contiguous()
                and xs.is_contiguous()
            )

            if not can_profile:
                # Skip profiling - just use native and mark as profiled
                curr_profile.profiled = True
                curr_profile.use_blas = False

                if self.training:
                    self.train_profile = curr_profile^

                else:
                    self.validation_profile = curr_profile^

                return self.matmul(xs)

            # Perform profiling
            elif curr_profile.call_count < curr_profile.profile_samples:
                # First Profile -> profile_samples: measure native matmul
                var start = now()
                var result = self.matmul(xs)
                curr_profile.time_native += now() - start
                curr_profile.call_count += 1

                if self.training:
                    self.train_profile = curr_profile^
                else:
                    self.validation_profile = curr_profile^

                return result^

            else:  # curr_profile.call_count < curr_profile.profile_samples * 2:
                # Next 'profile_samples' calls: measure BLAS matmul
                var start = now()
                var result = self.matmul_blas(xs)
                curr_profile.time_blas += now() - start
                curr_profile.call_count += 1

                # After profile_samples * 2 call, finalize decision
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

    fn to_gpu(self, gpu: Optional[GPU] = None) raises -> Linear[Self.dtype, Self.mode]:
        """LinearBLAS does not support GPU.

        BLAS operates on CPU memory. Use Linear for GPU models.
        """
        panic(
         "LinearBLAS does not support GPU — use Linear[dtype] for GPU models"
        )
        # Unreachable — satisfies compiler
        return Linear[Self.dtype, Self.mode](self.in_features, self.out_features)

    @always_inline
    fn matmul(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=True, mode=Self.mode
            ](xs, self.weight)
            result = Adder[Self.dtype].forward[track_grad=True](
                matmul_out^, self.bias
            )

        else:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=Self.mode
            ](xs, self.weight)
            result = Adder[Self.dtype].forward[track_grad=False](
                matmul_out^, self.bias
            )

        return result^

    @always_inline
    fn matmul_blas(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = self.blas_lite.value().matmul[track_grad=True](
                xs, self.weight, transpose_A=False, transpose_B=False
            )
            result = Adder[Self.dtype].forward[track_grad=True](
                matmul_out^, self.bias
            )

        else:
            var matmul_out = self.blas_lite.value().matmul[track_grad=False](
                xs, self.weight, transpose_A=False, transpose_B=False
            )
            result = Adder[Self.dtype].forward[track_grad=False](
                matmul_out^, self.bias
            )

        return result^

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_any_origin()
        )
        params.append(
            UnsafePointer(to=self.bias).unsafe_mut_cast[True]().as_any_origin()
        )

        return params^

    fn num_parameters(self) -> Int:
        return self.weight.numels() + self.bias.numels()

    fn train(mut self):
        """Set to training mode - enables gradient tracking."""
        self.training = True

    fn eval(mut self):
        """Set to evaluation mode - disables gradient tracking."""
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)


struct ReLU[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var training: Bool
    comptime TAG = RELU

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.training:
            return x.relu[track_grad=True]()
        else:
            return x.relu[track_grad=False]()

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    fn to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self

struct Sigmoid[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var training: Bool
    comptime TAG = SIGMOID

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training

    fn __call__(
        self, x: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return x.sigmoid[track_grad=True]()
        else:
            return x.sigmoid[track_grad=False]()

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    fn to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self


struct Tanh[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var training: Bool
    comptime TAG = TANH

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training

    fn __call__(
        self, x: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return x.tanh[track_grad=True]()
        else:
            return x.tanh[track_grad=False]()

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    fn to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
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
]


@fieldwise_init
struct Module[dtype: DType](ImplicitlyCopyable & Movable):
    var layer: Layer[Self.dtype]
    var tag: Int

    fn __call__(
        mut self, mut xs: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.tag == LINEAR:
            return self.layer[Linear[Self.dtype, mm]](xs)
        if self.tag == LINEAR_BLAS:
            return self.layer[LinearBLAS[Self.dtype, mm]](xs)
        elif self.tag == RELU:
            return self.layer[ReLU[Self.dtype]](xs)
        elif self.tag == SIGMOID:
            return self.layer[Sigmoid[Self.dtype]](xs)
        elif self.tag == TANH:
            return self.layer[Tanh[Self.dtype]](xs)
        elif self.tag == DROPOUT:
            return self.layer[Dropout[Self.dtype]](xs)
        elif self.tag == CONV2D:
            return self.layer[Conv2D[Self.dtype]](xs)
        elif self.tag == FLATTEN:
            return self.layer[Flatten[Self.dtype]](xs)
        elif self.tag == MAXPOOL2D:
            return self.layer[MaxPool2d[Self.dtype]](xs)
        else:
            panic("Unknown module type")
            return Tensor[Self.dtype].scalar(0)

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        if self.tag == LINEAR:
            return self.layer[Linear[Self.dtype]].parameters()
        elif self.tag == LINEAR_BLAS:
            return self.layer[LinearBLAS[Self.dtype]].parameters()
        elif self.tag == CONV2D:
            return self.layer[Conv2D[Self.dtype]].parameters()

        else:
            return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
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
        elif self.tag == MAXPOOL2D:
            return self.layer[MaxPool2d[Self.dtype]].num_parameters()
        else:
            return 0

    fn zero_grad(self):
        """Zero all parameter gradients."""
        for parameter in self.parameters():
            parameter[].zero_grad()

    fn train(mut self):
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
        elif self.tag == MAXPOOL2D:
            self.layer[MaxPool2d[Self.dtype]].train()

    fn eval(mut self):
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
        elif self.tag == MAXPOOL2D:
            self.layer[MaxPool2d[Self.dtype]].eval()

    fn to_gpu(mut self, gpu: Optional[GPU] = None) raises -> Module[Self.dtype]:
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
            return Module[Self.dtype](Layer[Self.dtype](l.to_gpu(gpu)), self.tag)
        elif self.tag == LINEAR_BLAS:
            var l = self.layer[LinearBLAS[Self.dtype, mm]]
            _ = l.to_gpu(gpu)  # panics here
            return self        # unreachable
        elif self.tag == CONV2D:
            var l = self.layer[Conv2D[Self.dtype]]
            return Module[Self.dtype](Layer[Self.dtype](l.to_gpu(gpu)), self.tag)
        else:
            # RELU, SIGMOID, TANH, DROPOUT, FLATTEN, MAXPOOL2D
            # No parameters — return unchanged
            return self

@fieldwise_init
struct Sequential[dtype: DType](Copyable & Movable):
    var modules: List[Module[Self.dtype]]

    fn __init__(out self):
        self.modules = List[Module[Self.dtype]]()

    fn append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            if m.tag == LINEAR_BLAS:
                panic(
                    "LinearBLAS layer can not be added to Sequential. Use"
                    " SequentialBLAS"
                )
            self.modules.append(m)

    fn __call__(
        mut self, xs: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var out = xs
        for i in range(len(self.modules)):
            ref m = self.modules[i]
            out = m(out)
        return out

    fn to_gpu(mut self, gpu: Optional[GPU] = None, stop_grad: Bool=True) raises -> Sequential[Self.dtype]:
        """Move all layers in this Sequential model to GPU.

        Each layer's to_gpu() is called. Layers with no parameters
        (ReLU, Sigmoid etc.) are returned unchanged.
        LinearBLAS layers will panic — use Linear for GPU models.

        Args:
            gpu: Target GPU. Uses default GPU if None.

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
            var optimizer = SGD(model_gpu.parameters(), lr=0.01, momentum=0.9)

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

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        for module in self.modules:
            params.extend(module.parameters())
        return params^

    fn num_parameters(self) -> Int:
        var total: Int = 0
        for parameter in self.parameters():
            total += parameter[].numels()
        return total

    fn train(mut self):
        """Set all modules to training mode."""
        for i in range(len(self.modules)):
            self.modules[i].train()

    fn eval(mut self):
        """Set all modules to evaluation mode."""
        for i in range(len(self.modules)):
            self.modules[i].eval()


@fieldwise_init
struct SequentialBLAS[dtype: DType](Copyable & Movable):
    var modules: List[Module[Self.dtype]]
    var blas_handle: BLASHandle[Self.dtype]

    fn __init__(out self):
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

    fn append(mut self, *ms: Module[Self.dtype]):
        for m in ms:
            if self.blas_handle.is_initialized() and m.tag == LINEAR_BLAS:
                var linear = m.layer[LinearBLAS[Self.dtype, mm]]
                linear.blas_lite = self.blas_handle.lite_handle()
                self.modules.append(linear^.into())
                continue

            self.modules.append(m)

    fn __call__(
        mut self, xs: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var out = xs
        for i in range(len(self.modules)):
            ref m = self.modules[i]
            out = m(out)
        return out

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        for module in self.modules:
            params.extend(module.parameters())
        return params^

    fn num_parameters(self) -> Int:
        var total: Int = 0
        for parameter in self.parameters():
            total += parameter[].numels()
        return total

    fn train(mut self):
        """Set all modules to training mode."""
        for i in range(len(self.modules)):
            self.modules[i].train()

    fn eval(mut self):
        """Set all modules to evaluation mode."""
        for i in range(len(self.modules)):
            self.modules[i].eval()


# -----------------------------------------
# Mean Squared Error Loss
# -----------------------------------------
@fieldwise_init
struct MSELoss[dtype: DType = DType.float32](RegisterPassable):
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __call__(
        self, preds: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype]:
        if self.training:
            return preds.mse[track_grad=True](target)
        else:
            return preds.mse[track_grad=False](target)

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False


@fieldwise_init
struct BCELoss[dtype: DType = DType.float32](RegisterPassable):
    var training: Bool
    var epsilon: Scalar[Self.dtype]

    fn __init__(
        out self, epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9)
    ):
        self.training = True
        self.epsilon = epsilon

    # Instance method - respects training mode
    fn __call__(
        self, pred: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return Self.forward[track_grad=True](pred, target, self.epsilon)
        else:
            return Self.forward[track_grad=False](pred, target, self.epsilon)

    # Static method - can be called directly or via instance
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        pred: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        # Clip for numerical stability
        var pred_safe = Clip[Self.dtype].forward[track_grad](
            pred, epsilon, 1 - epsilon
        )

        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        var log_pred = pred_safe.log[track_grad]()
        var term1 = Multiplicator[Self.dtype].forward[track_grad](
            target, log_pred
        )

        var one = Tensor[Self.dtype].scalar(1)
        var one_minus_target = Subtractor[Self.dtype].forward[track_grad](
            one, target
        )
        var one_minus_pred = Subtractor[Self.dtype].forward[track_grad](
            one, pred_safe
        )
        var log_one_minus_pred = one_minus_pred.log[track_grad]()
        var term2 = Multiplicator[Self.dtype].forward[track_grad](
            one_minus_target, log_one_minus_pred
        )

        var sum_terms = Adder[Self.dtype].forward[track_grad](term1, term2)
        var neg_one = Tensor[Self.dtype].scalar(-1)
        var loss = Multiplicator[Self.dtype].forward[track_grad](
            sum_terms, neg_one
        )

        return loss.mean[track_grad]()

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False


@fieldwise_init
struct BCEWithLogitsLoss[dtype: DType = DType.float32](RegisterPassable):
    var training: Bool
    var epsilon: Scalar[Self.dtype]

    fn __init__(
        out self, epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9)
    ):
        self.training = True
        self.epsilon = epsilon

    # Instance method - respects training mode
    fn __call__(
        self, logits: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return Self.forward[track_grad=True](logits, target, self.epsilon)
        else:
            return Self.forward[track_grad=False](logits, target, self.epsilon)

    # Static method - can be called directly or via instance
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        # Apply sigmoid to convert logits to probabilities
        var pred_probs = logits.sigmoid[track_grad]()

        # Clip for numerical stability
        var probs_safe = Clip[Self.dtype].forward[track_grad](
            pred_probs, epsilon, 1 - epsilon
        )

        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        var log_probs = probs_safe.log[track_grad]()
        var term1 = Multiplicator[Self.dtype].forward[track_grad](
            target, log_probs
        )

        var one = Tensor[Self.dtype].scalar(1)
        var one_minus_target = Subtractor[Self.dtype].forward[track_grad](
            one, target
        )
        var one_minus_probs = Subtractor[Self.dtype].forward[track_grad](
            one, probs_safe
        )
        var log_one_minus = one_minus_probs.log[track_grad]()
        var term2 = Multiplicator[Self.dtype].forward[track_grad](
            one_minus_target, log_one_minus
        )

        var sum_terms = Adder[Self.dtype].forward[track_grad](term1, term2)
        var neg_one = Tensor[Self.dtype].scalar(-1)
        var loss = Multiplicator[Self.dtype].forward[track_grad](
            sum_terms, neg_one
        )

        return loss.mean[track_grad]()

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False


@fieldwise_init
struct Conv2D[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Conv2D layer wrapper for Sequential integration.

    Stores weights and bias as trainable parameters.
    """

    comptime TAG = CONV2D

    var weight: Tensor[Self.dtype]  # (out_channels, in_channels, KH, KW)
    var bias: Tensor[Self.dtype]  # (out_channels,)
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var dilation: Int
    var padding: Padding
    var training: Bool
    var delegate: Conv2dFused[Self.dtype]

    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        bias: Bool = True,
        init_seed: Optional[Int] = None,
        init_method: String = "he",  # "he" for ReLU, "xavier" for tanh/sigmoid
        weight_factor: Scalar[Self.dtype] = Scalar[Self.dtype](1),
    ):
        """
        Initialize Conv2D layer.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB).
            out_channels: Number of output feature maps/filters.
            kernel_size: Size of square kernel (kernel_h = kernel_w).
            stride: Stride for convolution.
            dilation: Dilation factor.
            padding: "valid", "same", int, or custom.
            bias: Whether to include bias term.
            init_seed: Optional seed.
            init_method: Weight initialization ("xavier", "he", "standard").
            weight_factor: Scaling factor for weight initialization.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.training = True

        # Initialize weights: (out_channels, in_channels, kernel_size, kernel_size)
        var weight_shape = Shape(
            out_channels, in_channels, kernel_size, kernel_size
        )
        var fan_in = in_channels * kernel_size * kernel_size
        if init_method == "xavier":
            # Xavier/Glorot uniform initialization
            var fan_out = out_channels * kernel_size * kernel_size
            var limit = Scalar[Self.dtype](
                sqrt(6.0 / Float64(fan_in + fan_out))
            )
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=weight_shape,
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if bias:
                self.bias = Tensor[Self.dtype].rand(
                    Shape(out_channels),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                # Create a dummy bias - would not be used
                self.bias = Tensor[Self.dtype].scalar(0)

        elif init_method == "he":
            # He/Kaiming normal initialization (for ReLU)
            var std = sqrt(2.0 / Float64(fan_in))
            self.weight = (
                Tensor[Self.dtype].randn(
                    shape=weight_shape,
                    mean=0.0,
                    std=std,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if bias:
                self.bias = Tensor[Self.dtype].randn(
                    Shape(out_channels),
                    mean=0.0,
                    std=std * 0.01,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                # Create a dummy bias - would not be used
                self.bias = Tensor[Self.dtype].scalar(0)

        else:  # "standard" or default
            # Simple uniform initialization (good for [0,1] normalized inputs)
            var limit = Scalar[Self.dtype](0.1)
            self.weight = (
                Tensor[Self.dtype].rand(
                    shape=weight_shape,
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
                * weight_factor
            )
            if bias:
                self.bias = Tensor[Self.dtype].rand(
                    Shape(out_channels),
                    min=-limit,
                    max=limit,
                    init_seed=init_seed,
                    requires_grad=True,
                )
            else:
                self.bias = Tensor[Self.dtype].scalar(0)

        self.delegate = Conv2dFused[Self.dtype]()
        print("Conv2D initialized:")
        print("  Shape:", weight_shape)
        print("  In channels:", in_channels)
        print("  Out channels:", out_channels)
        print("  Kernel size:", kernel_size, "×", kernel_size)
        print("  Parameters:", self.num_parameters())

    fn __call__(mut self, image: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        """
        Forward pass.

        Args:
            image: (batch_size, in_channels, height, width).

        Returns:
            Output: (batch_size, out_channels, out_height, out_width).
        """
        ref img_shape = image.shape()

        # Validate input
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

        # Forward pass
        if self.training:
            return self.delegate[track_grad=True](
                image,
                self.weight,
                bias=Optional(self.bias) if self.bias.requires_grad else None,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
                requires_grad=True,
            )
        else:
            return self.delegate[track_grad=False](
                image,
                self.weight,
                bias=Optional(self.bias) if self.bias.requires_grad else None,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
                requires_grad=False,
            )

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        """Return trainable parameters for optimizer."""
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

        params.append(
            UnsafePointer(to=self.weight)
            .unsafe_mut_cast[True]()
            .as_any_origin()
        )

        if self.bias.shape().rank() > 0:  # Has actual bias
            params.append(
                UnsafePointer(to=self.bias)
                .unsafe_mut_cast[True]()
                .as_any_origin()
            )

        return params^

    fn num_parameters(self) -> Int:
        """Count total parameters."""
        var count = self.weight.numels()
        if self.bias.shape().rank() > 0:
            count += self.bias.numels()
        return count

    fn train(mut self):
        """Set to training mode."""
        self.training = True

    fn eval(mut self):
        """Set to evaluation mode."""
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        """Convert to Module for Sequential."""
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    fn to_gpu(deinit self, gpu: Optional[GPU] = None) raises -> Conv2D[Self.dtype]:
        """Move this Conv2D layer to GPU.

        Consumes self.

        Weights and bias become permanent GPU residents.
        Gradients accumulate on GPU.

        Args:
            gpu: Target GPU. Uses default GPU if None.

        Returns:
            New Conv2D layer with GPU weights and bias.
        """
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var bias_gpu   = self.bias.to_gpu(gpu=gpu, stop_grad=True)

        var out = self^
        out.weight = weight_gpu^
        out.bias = bias_gpu^

        return out^

struct Flatten[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """
    Flatten spatial dimensions: (N, C, H, W) → (N, C*H*W).
    """

    comptime TAG = FLATTEN
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training

    fn __call__(self, mut x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
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

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

    fn to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layer have no parameters to move."""
        return self

