from tenmo import Tensor
from gradbox import Gradbox
from math import sqrt
from sys import simd_width_of
from common_utils import panic


@fieldwise_init
struct SGD[dtype: DType, //](ImplicitlyCopyable & Movable):
    """
    SGD with momentum, weight decay, and gradient clipping.
    """

    var parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]
    var lr: Scalar[Self.dtype]
    var momentum: Scalar[Self.dtype]
    var weight_decay: Scalar[Self.dtype]
    var clip_norm: Scalar[Self.dtype]
    var clip_value: Scalar[Self.dtype]
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
        if clip_norm < 0 or clip_value < 0:
            panic("Clip_norm and clip_value must be >= 0")
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

    fn compute_grad_norm(self) -> Scalar[Self.dtype]:
        """Compute total gradient norm with SIMD optimization."""
        var total_norm_sq: Scalar[Self.dtype] = 0.0
        alias simd_w = simd_width_of[Self.dtype]()

        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if parameter.requires_grad and parameter.has_grad():
                ref grad = parameter.gradients()[]
                var grad_ptr = grad.buffer.data_buffer().data
                var num_elements = grad.num_elements()

                # Norm computation
                var norm_vec = SIMD[Self.dtype, simd_w](0)
                var j = 0
                var vec_end = (num_elements // simd_w) * simd_w

                for _ in range(vec_end // simd_w):
                    var g_vec = grad_ptr.load[width=simd_w](j)
                    norm_vec += g_vec * g_vec
                    j += simd_w

                total_norm_sq += norm_vec.reduce_add()

                # Scalar tail
                for k in range(vec_end, num_elements):
                    var g = grad_ptr[k]
                    total_norm_sq += g * g

        return sqrt(total_norm_sq)

    fn clip_gradients(mut self):
        """Optimized gradient clipping with SIMD."""
        alias simd_w = simd_width_of[Self.dtype]()

        # 1. NORM CLIPPING (if enabled)
        if self.clip_norm > 0:
            var total_norm = self.compute_grad_norm()

            if total_norm > self.clip_norm:
                var clip_coef = self.clip_norm / total_norm

                # Apply clipping with SIMD
                for i in range(len(self.parameters)):
                    ref parameter = self.parameters[i][]
                    if parameter.requires_grad and parameter.has_grad():
                        ref grad = parameter.gradients()[]
                        var grad_ptr = grad.buffer.data_buffer().data
                        var num_elements = grad.num_elements()

                        var clip_vec = SIMD[Self.dtype, simd_w](clip_coef)
                        var j = 0
                        var vec_end = (num_elements // simd_w) * simd_w

                        for _ in range(vec_end // simd_w):
                            var g_vec = grad_ptr.load[width=simd_w](j)
                            grad_ptr.store[width=simd_w](j, g_vec * clip_vec)
                            j += simd_w

                        # Scalar tail
                        for k in range(vec_end, num_elements):
                            grad_ptr[k] *= clip_coef

        # === 2. VALUE CLIPPING (if enabled) ===
        if self.clip_value > 0:
            var min_val = -self.clip_value
            var max_val = self.clip_value
            var min_vec = SIMD[Self.dtype, simd_w](min_val)
            var max_vec = SIMD[Self.dtype, simd_w](max_val)

            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if parameter.requires_grad and parameter.has_grad():
                    ref grad = parameter.gradients()[]
                    var grad_ptr = grad.buffer.data_buffer().data
                    var num_elements = grad.num_elements()

                    var j = 0
                    var vec_end = (num_elements // simd_w) * simd_w

                    for _ in range(vec_end // simd_w):
                        var g_vec = grad_ptr.load[width=simd_w](j)
                        var clamped = g_vec.clamp(min_vec, max_vec)
                        grad_ptr.store[width=simd_w](j, clamped)
                        j += simd_w

                    # Scalar tail
                    for k in range(vec_end, num_elements):
                        var g = grad_ptr[k]
                        grad_ptr[k] = max(min_val, min(max_val, g))

    @always_inline
    fn step(mut self):
        """
        Optimized parameter update with SIMD and fused operations.

        Single pass per parameter:
        1. Clip gradient (if needed)
        2. Update velocity (if momentum)
        3. Apply weight decay
        4. Update parameter

        All fused into one loop.
        """
        # 1. Apply gradient clipping
        self.clip_gradients()

        # 2. Update parameters with SIMD
        alias simd_w = simd_width_of[Self.dtype]()

        var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
        var momentum_vec = SIMD[Self.dtype, simd_w](self.momentum)
        var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)

        # Could parallelize over parameters - but with less cores overhead of parallel setup may dominate
        # @parameter
        # fn update_param(i: Int):
        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not (parameter.requires_grad and parameter.has_grad()):
                continue

            ref grad = parameter.gradients()[]
            var param_ptr = parameter.buffer.data_buffer().data
            var grad_ptr = grad.buffer.data_buffer().data
            var num_elements = parameter.num_elements()

            # Momentum SGD: v = β*v + g, θ = θ - η*v
            if self.use_momentum:
                ref velocity = self.velocities[i]
                var vel_ptr = velocity.buffer.data_buffer().data

                # Fused momentum update
                var j = 0
                var vec_end = (num_elements // simd_w) * simd_w

                for _ in range(vec_end // simd_w):
                    var p_vec = param_ptr.load[width=simd_w](j)
                    var g_vec = grad_ptr.load[width=simd_w](j)
                    var v_vec = vel_ptr.load[width=simd_w](j)

                    # Apply weight decay to gradient
                    if self.weight_decay > 0:
                        g_vec += p_vec * wd_vec

                    # Update velocity: v = momentum * v + g
                    v_vec = momentum_vec * v_vec + g_vec
                    vel_ptr.store[width=simd_w](j, v_vec)

                    # Update parameter: p -= lr * v
                    p_vec -= lr_vec * v_vec
                    param_ptr.store[width=simd_w](j, p_vec)

                    j += simd_w

                # Scalar tail
                for k in range(vec_end, num_elements):
                    var p = param_ptr[k]
                    var g = grad_ptr[k]
                    var v = vel_ptr[k]

                    if self.weight_decay > 0:
                        g += p * self.weight_decay

                    v = self.momentum * v + g
                    vel_ptr[k] = v
                    param_ptr[k] = p - self.lr * v
            else:
                # Vanilla SGD (fused)
                var j = 0
                var vec_end = (num_elements // simd_w) * simd_w

                for _ in range(vec_end // simd_w):
                    var p_vec = param_ptr.load[width=simd_w](j)
                    var g_vec = grad_ptr.load[width=simd_w](j)

                    # Apply weight decay
                    if self.weight_decay > 0:
                        g_vec += p_vec * wd_vec

                    # Update: p -= lr * g
                    p_vec -= lr_vec * g_vec
                    param_ptr.store[width=simd_w](j, p_vec)

                    j += simd_w

                # Scalar tail
                for k in range(vec_end, num_elements):
                    var p = param_ptr[k]
                    var g = grad_ptr[k]

                    if self.weight_decay > 0:
                        g += p * self.weight_decay

                    param_ptr[k] = p - self.lr * g

        # parallelize[update_param](len(self.parameters))

    @always_inline
    fn zero_grad(self):
        """Zero out all gradients."""
        for i in range(len(self.parameters)):
            self.parameters[i][].zero_grad()

    fn set_lr(mut self, lr: Scalar[Self.dtype]):
        self.lr = lr

    fn get_lr(self) -> Scalar[Self.dtype]:
        return self.lr

    fn set_clip_norm(mut self, clip_norm: Scalar[Self.dtype]):
        self.clip_norm = clip_norm

    fn set_clip_value(mut self, clip_value: Scalar[Self.dtype]):
        self.clip_value = clip_value

    fn set_weight_decay(mut self, weight_decay: Scalar[Self.dtype]):
        self.weight_decay = weight_decay
