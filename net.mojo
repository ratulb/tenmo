from tenmo import Tensor
from shapes import Shape
from gradbox import Gradbox
from math import sqrt
from common_utils import panic, now
from utils import Variant
from forwards import (
    Matmul,
    Adder,
    Multiplicator,
    Subtractor,
    Clip,
    Padding,
    Conv2dFused,
    MaxPool2d,
)
from operators import mm, mv, vm, dot
from blashandle import BLASHandle, BLASHandleLite
from utils.numerics import neg_inf
from algorithm import parallelize
from ndbuffer import NDBuffer
from random import seed, random_float64
from sys import simd_width_of

alias LINEAR = 0
alias LINEAR_BLAS = 1
alias RELU = 2
alias SIGMOID = 3
alias TANH = 4
alias DROPOUT = 5
alias CONV2D = 6
alias FLATTEN = 7
alias MAXPOOL2D = 8


@fieldwise_init
struct Linear[dtype: DType, mode: Int = mm](ImplicitlyCopyable & Movable):
    """Fully connected layer: y = xW + b."""

    alias TAG = LINEAR

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
                sqrt(6.0 / (in_features + out_features))
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
                xs_shape.__str__(),
                "and  weights shape → ",
                weight_shape.__str__(),
            )
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=True, mode=mode
            ](xs, self.weight)
            result = Adder[Self.dtype].forward[track_grad=True](
                matmul_out^, self.bias
            )

        else:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=mode
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


@fieldwise_init
@register_passable
struct Profile(ImplicitlyCopyable):
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

    alias TAG = LINEAR_BLAS

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
                sqrt(6.0 / (in_features + out_features))
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
                xs_shape.__str__(),
                "and  weights shape → ",
                weight_shape.__str__(),
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

    @always_inline
    fn matmul(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var result: Tensor[Self.dtype]

        if self.training:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=True, mode=mode
            ](xs, self.weight)
            result = Adder[Self.dtype].forward[track_grad=True](
                matmul_out^, self.bias
            )

        else:
            var matmul_out = Matmul[Self.dtype].forward[
                track_grad=False, mode=mode
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


@register_passable
struct ReLU[dtype: DType](ImplicitlyCopyable):
    var training: Bool
    alias TAG = RELU

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

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

@fieldwise_init
@register_passable
struct Dropout[dtype: DType](ImplicitlyCopyable):
    """
    Optimized Dropout layer.

    1. Direct buffer manipulation (no intermediate tensors)
    2. SIMD vectorization
    3. Fused mask generation and scaling
    4. Fast random number generation
    5. Zero overhead in eval mode
    """

    var training: Bool
    var p: Scalar[Self.dtype]
    var scale: Scalar[Self.dtype]
    var seed: Int  # For reproducible randomness

    alias TAG = DROPOUT

    fn __init__(out self, p: Scalar[Self.dtype] = Scalar[Self.dtype](0.5)):
        """Initialize Dropout layer."""
        if p < 0.0 or p >= 1.0:
            panic("Dropout probability must be in [0, 1)")

        self.training = True
        self.p = p
        self.scale = Scalar[Self.dtype](1.0) / (Scalar[Self.dtype](1.0) - p)
        self.seed = 42  # Default seed

    fn __copyinit__(out self, other: Self):
        self.training = other.training
        self.p = other.p
        self.scale = other.scale
        self.seed = other.seed

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        """Forward pass with optimized dropout."""

        if not self.training or self.p == 0.0:
            # Eval mode or no dropout
            return x

        if self.p == 1.0:
            # Drop everything: return zeros
            return Tensor[Self.dtype].zeros(x.shape())

        # Training mode: Apply dropout
        # 1. Generate random values
        # 2. Compare with threshold
        # 3. Scale survivors
        # 4. Multiply with input
        var output = Tensor[dtype].zeros(x.shape())

        var x_ptr = x.buffer.data_buffer().data
        var out_ptr = output.buffer.data_buffer().data

        var total_elements = x.numels()

        alias simd_w = simd_width_of[dtype]()

        # Vectorized constants
        var threshold_vec = SIMD[dtype, simd_w](self.p)
        var scale_vec = SIMD[Self.dtype, simd_w](self.scale)
        var zero_vec = SIMD[Self.dtype, simd_w](0)

        # SIMD vectorized dropout
        var i = 0
        var vec_end = (total_elements // simd_w) * simd_w

        for _ in range(vec_end // simd_w):
            # Load input values
            var x_vec = x_ptr.load[width=simd_w](i)

            # Generate random values [0, 1)
            var rand_vec = SIMD[dtype, simd_w](0)

            @parameter
            for v in range(simd_w):
                rand_vec[v] = random_float64(0.0, 1.0).cast[dtype]()

            # Create mask: 1 if rand > p, else 0
            # Using select: selects scale if condition true, else 0
            var mask_vec = (rand_vec.gt(threshold_vec)).select(
                scale_vec, zero_vec
            )

            # Apply mask and scale in one operation
            var result_vec = x_vec * mask_vec

            # Store result
            out_ptr.store[width=simd_w](i, result_vec)
            i += simd_w

        # Scalar tail
        for j in range(vec_end, total_elements):
            var x_val = x_ptr[j]
            var rand_val = random_float64(0.0, 1.0).cast[dtype]()

            if rand_val > self.p:
                out_ptr[j] = x_val * self.scale
            else:
                out_ptr[j] = 0.0

        # Setup gradient tracking
        if x.requires_grad:
            output.requires_grad_(True)
            # Dropout backward is handled automatically through multiplication

        return output^

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

    fn set_seed(mut self, seed_val: Int):
        """Set random seed for reproducibility."""
        self.seed = seed_val
        seed(seed_val)

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)

@register_passable
struct Sigmoid[dtype: DType](ImplicitlyCopyable):
    var training: Bool
    alias TAG = SIGMOID

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
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


@register_passable
struct Tanh[dtype: DType](ImplicitlyCopyable):
    var training: Bool
    alias TAG = TANH

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
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


# Refer to operators & matmul
# Defined in operators

# alias dot = 28  # dot product
# alias vm = 29  # vector & tensor matmul
# alias mv = 30  # tensor & vector matmul
# alias mm = 31  # tensor & tensor matmul

alias Layer[dtype: DType] = Variant[
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

    fn __call__(mut self, mut xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
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

    fn __call__(mut self, xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var out = xs
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
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

    fn __call__(mut self, xs: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var out = xs
        for i in range(len(self.modules)):
            var ref m = self.modules[i]
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
@register_passable
struct MSELoss[dtype: DType = DType.float32]:
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
@register_passable
struct BCELoss[dtype: DType = DType.float32]:
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
    ) -> Tensor[Self.dtype]:
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
    ) -> Tensor[Self.dtype]:
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
@register_passable
struct BCEWithLogitsLoss[dtype: DType = DType.float32]:
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
    ) -> Tensor[Self.dtype]:
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
    ) -> Tensor[Self.dtype]:
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
struct SGD[dtype: DType, //](ImplicitlyCopyable & Movable):
    """Stochastic Gradient Descent."""

    var parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]
    var lr: Scalar[Self.dtype]
    var momentum: Scalar[Self.dtype]
    var velocities: List[Gradbox[Self.dtype]]  # For momentum

    fn __init__(
        out self,
        parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]],
        lr: Scalar[Self.dtype] = 0.01,
        momentum: Scalar[Self.dtype] = 0.0,
    ):
        self.parameters = parameters.copy()
        self.lr = lr
        self.momentum = momentum
        self.velocities = List[Gradbox[Self.dtype]]()

        # Initialize velocities
        if momentum > 0:
            for parameter in self.parameters:
                self.velocities.append(
                    Gradbox[Self.dtype].zeros(parameter[].shape(), share=False)
                )

    fn __copyinit__(out self, existing: Self):
        self.parameters = existing.parameters.copy()
        self.lr = existing.lr
        self.momentum = existing.momentum
        self.velocities = existing.velocities.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.parameters = existing.parameters^
        self.lr = existing.lr
        self.momentum = existing.momentum
        self.velocities = existing.velocities^

    fn step(mut self):
        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if parameter.requires_grad and parameter.has_grad():
                ref grad = parameter.gradients()[]
                if self.momentum > 0:
                    # v = momentum * v + grad
                    self.velocities[i] = (
                        self.velocities[i] * self.momentum + grad
                    )
                    # param -= lr * v
                    parameter -= self.velocities[i] * self.lr
                else:
                    # param -= lr * grad
                    parameter -= grad * self.lr

    fn zero_grad(self):
        for parameter in self.parameters:
            parameter[].zero_grad()

    fn set_lr(mut self, lr: Scalar[Self.dtype]):
        self.lr = lr


@fieldwise_init
struct Conv2D[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Conv2D layer wrapper for Sequential integration.

    Stores weights and bias as trainable parameters.
    """

    alias TAG = CONV2D

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
            var limit = Scalar[Self.dtype](sqrt(6.0 / (fan_in + fan_out)))
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
                img_shape.__str__(),
            )

        if img_shape[1] != self.in_channels:
            panic(
                "Conv2D input channels mismatch: expected ",
                self.in_channels.__str__(),
                ", got ",
                img_shape[1].__str__(),
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


@register_passable
struct Flatten[dtype: DType](ImplicitlyCopyable):
    """
    Flatten spatial dimensions: (N, C, H, W) → (N, C*H*W).
    """

    alias TAG = FLATTEN
    var training: Bool

    fn __init__(out self):
        self.training = True

    fn __copyinit__(out self, other: Self):
        self.training = other.training

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


@fieldwise_init
@register_passable
struct MaxPool2d_old[dtype: DType](ImplicitlyCopyable):
    """
    Batched, multi-channel 2D Max Pooling.

    Supports:
    - Arbitrary kernel size and stride
    - Optional padding
    - Overlapping and non-overlapping pooling
    - Full gradient tracking

    Args:
        input: (N, C, H_in, W_in)
        kernel_size: Size of the pooling window
        stride: Stride for pooling (defaults to kernel_size for non-overlapping)
        padding: Zero-padding added to input

    Returns:
        output: (N, C, H_out, W_out)

    Note:
        - H_out = (H_in + 2*padding - kernel_size) // stride + 1
        - W_out = (W_in + 2*padding - kernel_size) // stride + 1
    """

    alias TAG = MAXPOOL2D
    var training: Bool
    var kernel_size: Int
    var stride: Int
    var padding: Int

    fn __init__(
        out self,
        kernel_size: Int = 2,
        stride: Optional[Int] = None,
        padding: Int = 0,
    ):
        self.training = True
        self.kernel_size = kernel_size
        self.stride = stride.or_else(kernel_size)
        self.padding = padding

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.training:
            return Self.forward[track_grad=True](
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                requires_grad=True,
            )
        else:
            return Self.forward[track_grad=False](
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                requires_grad=False,
            )

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        input_tensor: Tensor[Self.dtype],
        kernel_size: Int = 2,
        stride: Optional[Int] = None,
        padding: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        ref input_shape = input_tensor.shape()
        if input_shape.rank() != 4:
            panic("MaxPool2d expects 4D input: (N, C, H_in, W_in)")

        var N = input_shape[0]
        var C = input_shape[1]
        var H_in = input_shape[2]
        var W_in = input_shape[3]

        var KH = kernel_size
        var KW = kernel_size
        var s = stride.or_else(
            kernel_size
        )  # Default stride = kernel_size (non-overlapping)
        var pad = padding

        # Calculate output dimensions
        var H_out = (H_in + 2 * pad - KH) // s + 1
        var W_out = (W_in + 2 * pad - KW) // s + 1

        if H_out <= 0 or W_out <= 0:
            panic(
                "Invalid MaxPool2d parameters lead to non-positive output size."
                " H_out="
                + String(H_out)
                + ", W_out="
                + String(W_out)
            )

        # Output tensor and argmax mask for gradient routing
        var output = Tensor[Self.dtype].zeros(N, C, H_out, W_out)
        var argmax_mask = NDBuffer[DType.int64].zeros(Shape(N, C, H_out, W_out))

        # Parallelize over (N * C) for better load balancing
        @parameter
        fn pool_for_batch_channel(idx: Int):
            var n = idx // C
            var c = idx % C

            # Process all output spatial positions for this (n, c)
            for out_y in range(H_out):
                for out_x in range(W_out):
                    # Calculate input window start (accounting for padding)
                    var in_y_start = out_y * s - pad
                    var in_x_start = out_x * s - pad

                    # Find max in the pooling window
                    var max_val = neg_inf[Self.dtype]()  # Large negative value
                    var max_idx = -1  # -1 indicates invalid (all padding)

                    # Scan the pooling window
                    for ky in range(KH):
                        for kx in range(KW):
                            var in_y = in_y_start + ky
                            var in_x = in_x_start + kx

                            # Check if within valid input bounds (not in padding region)
                            if (
                                in_y >= 0
                                and in_y < H_in
                                and in_x >= 0
                                and in_x < W_in
                            ):
                                var val = input_tensor[n, c, in_y, in_x]
                                if val > max_val:
                                    max_val = val
                                    max_idx = (
                                        in_y * W_in + in_x
                                    )  # Flatten to single index

                    # Store results
                    output[n, c, out_y, out_x] = max_val
                    argmax_mask[[n, c, out_y, out_x]] = max_idx

        # Parallelize over all (batch, channel) combinations
        parallelize[pool_for_batch_channel](N * C)

        # Setup gradient tracking
        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(
                input_tensor.requires_grad
            )
            if grad_required:
                output.requires_grad_(True)
                _ = """var backward_fn = MaxPool2dBackward[Self.dtype](
                    kernel_size=kernel_size,
                    stride=s,
                    padding=pad,
                    input_shape=input_shape,
                    argmax_mask=argmax_mask,
                ).into_backward_fn()
                output.backwardFn = Optional(backward_fn^)"""
                output.add_ancestry(input_tensor)

        return output^

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


fn main():
    pass
