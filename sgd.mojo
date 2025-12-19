from tenmo import Tensor
from gradbox import Gradbox
from math import sqrt


@fieldwise_init
struct SGD[dtype: DType, //](ImplicitlyCopyable & Movable):
    """Robust SGD with momentum, weight decay, and gradient clipping."""

    var parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]
    var lr: Scalar[Self.dtype]
    var momentum: Scalar[Self.dtype]
    var weight_decay: Scalar[Self.dtype]
    var clip_norm: Scalar[Self.dtype]  # Max gradient norm (0 = disabled)
    var clip_value: Scalar[Self.dtype]  # Max gradient value (0 = disabled)

    var velocities: List[Gradbox[Self.dtype]]
    var use_momentum: Bool

    fn __init__(
        out self,
        parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]],
        lr: Scalar[Self.dtype] = 0.01,
        momentum: Scalar[Self.dtype] = 0.0,
        weight_decay: Scalar[Self.dtype] = 0.0,
        clip_norm: Scalar[Self.dtype] = 0.0,
        clip_value: Scalar[Self.dtype] = 0.0,
    ):
        """
        Initialize SGD optimizer.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor (0 = vanilla SGD).
            weight_decay: L2 regularization strength.
            clip_norm: Maximum gradient norm (0 = no norm clipping).
            clip_value: Maximum gradient value (0 = no value clipping).

        Note: If both clip_norm and clip_value are set, norm clipping is applied first.
        """
        self.parameters = parameters.copy()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.use_momentum = momentum > 0
        self.velocities = List[Gradbox[Self.dtype]]()
        if self.use_momentum:
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i]
                self.velocities.append(
                    Gradbox[Self.dtype].zeros(parameter[].shape(), share=False)
                )

    fn __copyinit__(out self, existing: Self):
        self.parameters = existing.parameters.copy()
        self.lr = existing.lr
        self.momentum = existing.momentum
        self.weight_decay = existing.weight_decay
        self.clip_norm = existing.clip_norm
        self.clip_value = existing.clip_value
        self.use_momentum = existing.use_momentum
        self.velocities = existing.velocities.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.parameters = existing.parameters^
        self.lr = existing.lr
        self.momentum = existing.momentum
        self.weight_decay = existing.weight_decay
        self.clip_norm = existing.clip_norm
        self.clip_value = existing.clip_value
        self.use_momentum = existing.use_momentum
        self.velocities = existing.velocities^

    fn _clip_gradients(mut self):
        """
        Apply gradient clipping.
        """
        var pre_clip_norm: Scalar[Self.dtype] = 0.0

        # === 1. NORM CLIPPING (if enabled) ===
        if self.clip_norm > 0:
            var total_norm_sq: Scalar[Self.dtype] = 0.0

            # Calculate total gradient norm across all parameters
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if parameter.requires_grad and parameter.has_grad():
                    ref grad = parameter.gradients()[]
                    total_norm_sq += (grad * grad).sum().item()

            var total_norm = sqrt(total_norm_sq)
            pre_clip_norm = total_norm

            # Apply clipping if norm exceeds threshold
            if total_norm > self.clip_norm:
                var clip_coef = self.clip_norm / (
                    total_norm + Scalar[Self.dtype](1e-8)
                )

                for i in range(len(self.parameters)):
                    ref parameter = self.parameters[i][]
                    if parameter.requires_grad and parameter.has_grad():
                        ref grad = (
                            parameter.gradients()[]
                        )  # UnsafePointer Reference to gradbox
                        grad *= clip_coef  # Inplace update

        # === 2. VALUE CLIPPING (if enabled) ===
        if self.clip_value > 0:
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if parameter.requires_grad and parameter.has_grad():
                    ref grad = parameter.gradients()[]
                    grad.clamp_in_place(-self.clip_value, self.clip_value)

    @always_inline
    fn step(mut self):
        """
        Perform one optimization step.

        """
        # 1. Apply gradient clipping first (prevents explosions)
        self._clip_gradients()

        # 2. Update parameters
        if self.use_momentum:
            # Momentum SGD: v = β*v + g, θ = θ - η*v
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if parameter.requires_grad and parameter.has_grad():
                    ref grad = parameter.gradients()[]

                    # Apply L2 weight decay to gradient.
                    if self.weight_decay > 0:
                        # Update velocity: v = momentum * v + grad
                        self.velocities[i] = (
                            self.velocities[i] * self.momentum
                            + grad
                            + parameter.__mul__[track_grad=False](
                                self.weight_decay
                            )
                        )
                    else:
                        self.velocities[i] = (
                            self.velocities[i] * self.momentum + grad
                        )
                    # Update parameter: param -= lr * velocity
                    parameter -= self.velocities[i] * self.lr
        else:
            # Vanilla SGD: θ = θ - η*g
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if parameter.requires_grad and parameter.has_grad():
                    ref grad = parameter.gradients()[]

                    # Apply L2 weight decay to gradient.
                    # Update parameter: param -= lr * grad
                    if self.weight_decay > 0:
                        parameter -= (
                            grad
                            + parameter.__mul__[track_grad=False](
                                self.weight_decay
                            )
                        ) * self.lr
                    else:
                        parameter -= grad * self.lr

    @always_inline
    fn zero_grad(self):
        """Zero out all gradients."""
        for i in range(len(self.parameters)):
            self.parameters[i][].zero_grad()

    fn set_lr(mut self, lr: Scalar[Self.dtype]):
        """Update learning rate."""
        self.lr = lr

    fn get_lr(self) -> Scalar[Self.dtype]:
        """Get current learning rate."""
        return self.lr

    fn set_clip_norm(mut self, clip_norm: Scalar[Self.dtype]):
        """Update gradient norm clipping threshold."""
        self.clip_norm = clip_norm

    fn set_clip_value(mut self, clip_value: Scalar[Self.dtype]):
        """Update gradient value clipping threshold."""
        self.clip_value = clip_value

    fn set_weight_decay(mut self, weight_decay: Scalar[Self.dtype]):
        """Update weight decay strength."""
        self.weight_decay = weight_decay


fn main() raises:
    test_sgd_clipping()


from shapes import Shape


fn test_sgd_clipping():
    """Test that gradient clipping works correctly."""
    alias dtype = DType.float32

    # Create a parameter with exploding gradient
    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params_list = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_list.append(UnsafePointer(to=param))

    # Create SGD with tight clipping
    var optimizer = SGD[dtype](
        parameters=params_list^,
        lr=0.01,
        clip_norm=0.1,  # Very tight clipping
        clip_value=0.05,
    )

    # Set an artificially large gradient
    param.seed_grad(Scalar[dtype](10.0))

    # Before step: gradient should be huge
    print("Before clipping:")
    print("  Gradient norm:", param.grad().norm())

    # Apply step (includes clipping)
    optimizer.step()

    # Check parameter was updated (not NaN)
    print("After clipping + step:")
    print("  Parameter updated successfully")
    # print("  Max value:", param.max().item())
    param.print()

    # Verify no explosion
    # assert_true(not param.has_nan(), "Parameter should not have NaN")
