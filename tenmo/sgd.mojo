from .tensor import Tensor
from .gradbox import Gradbox
from std.math import sqrt
from std.sys import simd_width_of, has_accelerator
from .common_utils import panic


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

    def __init__(
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
                ref parameter = self.parameters[i][]

                comptime if has_accelerator():
                    if parameter.is_on_gpu():
                        self.velocities.append(
                            Gradbox[Self.dtype].full(
                                parameter.shape(),
                                Scalar[Self.dtype](0),
                                share=False,
                                device=parameter.device(),
                            )
                        )
                    else:
                        self.velocities.append(
                            Gradbox[Self.dtype].zeros(
                                parameter.shape(), share=False
                            )
                        )
                else:
                    self.velocities.append(
                        Gradbox[Self.dtype].zeros(
                            parameter.shape(), share=False
                        )
                    )

    def __copyinit__(out self, copy: Self):
        self.parameters = copy.parameters.copy()
        self.lr = copy.lr
        self.momentum = copy.momentum
        self.weight_decay = copy.weight_decay
        self.clip_norm = copy.clip_norm
        self.clip_value = copy.clip_value
        self.use_momentum = copy.use_momentum
        self.velocities = copy.velocities.copy()

    def __moveinit__(out self, deinit take: Self):
        self.parameters = take.parameters^
        self.lr = take.lr
        self.momentum = take.momentum
        self.weight_decay = take.weight_decay
        self.clip_norm = take.clip_norm
        self.clip_value = take.clip_value
        self.use_momentum = take.use_momentum
        self.velocities = take.velocities^


    @always_inline
    def _step_no_momentum[
        simd_w: Int
    ](
        self,
        param_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
    ):
        var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
        var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w

        for _ in range(vec_end // simd_w):
            var p_vec = param_ptr.load[width=simd_w](j)
            var g_vec = grad_ptr.load[width=simd_w](j)
            if self.weight_decay > 0:
                g_vec += p_vec * wd_vec
            p_vec -= lr_vec * g_vec
            param_ptr.store[width=simd_w](j, p_vec)
            j += simd_w

        for k in range(vec_end, num_elements):
            var p = param_ptr[k]
            var g = grad_ptr[k]
            if self.weight_decay > 0:
                g += p * self.weight_decay
            param_ptr[k] = p - self.lr * g

    @always_inline
    def _apply_momentum[
        simd_w: Int
    ](
        self,
        param_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        vel_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
    ):
        var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
        var momentum_vec = SIMD[Self.dtype, simd_w](self.momentum)
        var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w

        for _ in range(vec_end // simd_w):
            var p_vec = param_ptr.load[width=simd_w](j)
            var g_vec = grad_ptr.load[width=simd_w](j)
            var v_vec = vel_ptr.load[width=simd_w](j)
            if self.weight_decay > 0:
                g_vec += p_vec * wd_vec
            v_vec = momentum_vec * v_vec + g_vec
            vel_ptr.store[width=simd_w](j, v_vec)
            p_vec -= lr_vec * v_vec
            param_ptr.store[width=simd_w](j, p_vec)
            j += simd_w

        for k in range(vec_end, num_elements):
            var p = param_ptr[k]
            var g = grad_ptr[k]
            var v = vel_ptr[k]
            if self.weight_decay > 0:
                g += p * self.weight_decay
            v = self.momentum * v + g
            vel_ptr[k] = v
            param_ptr[k] = p - self.lr * v

    @always_inline
    def _zero_rows_ptr[
        simd_w: Int
    ](
        self,
        ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        row_width: Int,
        indices: IntArray,
    ):
        var zero_vec = SIMD[Self.dtype, simd_w](0)
        for k in range(len(indices)):
            var row = indices[k]
            var base = row * row_width
            var j = 0
            var vec_end = (row_width // simd_w) * simd_w
            for _ in range(vec_end // simd_w):
                ptr.store[width=simd_w](base + j, zero_vec)
                j += simd_w
            for col in range(vec_end, row_width):
                ptr[base + col] = 0

    @always_inline
    def _step_no_momentum_sparse[
        simd_w: Int
    ](
        self,
        param_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        row_width: Int,
        indices: IntArray,
    ):
        var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
        var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)
        for k in range(len(indices)):
            var row = indices[k]
            var base = row * row_width
            var j = 0
            var vec_end = (row_width // simd_w) * simd_w
            for _ in range(vec_end // simd_w):
                var p_vec = param_ptr.load[width=simd_w](base + j)
                var g_vec = grad_ptr.load[width=simd_w](base + j)
                if self.weight_decay > 0:
                    g_vec += p_vec * wd_vec
                p_vec -= lr_vec * g_vec
                param_ptr.store[width=simd_w](base + j, p_vec)
                j += simd_w
            for col in range(vec_end, row_width):
                var p = param_ptr[base + col]
                var g = grad_ptr[base + col]
                if self.weight_decay > 0:
                    g += p * self.weight_decay
                param_ptr[base + col] = p - self.lr * g

    @always_inline
    def _apply_momentum_sparse[
        simd_w: Int
    ](
        self,
        param_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        vel_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        row_width: Int,
        indices: IntArray,
    ):
        var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
        var momentum_vec = SIMD[Self.dtype, simd_w](self.momentum)
        var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)
        for k in range(len(indices)):
            var row = indices[k]
            var base = row * row_width
            var j = 0
            var vec_end = (row_width // simd_w) * simd_w
            for _ in range(vec_end // simd_w):
                var p_vec = param_ptr.load[width=simd_w](base + j)
                var g_vec = grad_ptr.load[width=simd_w](base + j)
                var v_vec = vel_ptr.load[width=simd_w](base + j)
                if self.weight_decay > 0:
                    g_vec += p_vec * wd_vec
                v_vec = momentum_vec * v_vec + g_vec
                vel_ptr.store[width=simd_w](base + j, v_vec)
                p_vec -= lr_vec * v_vec
                param_ptr.store[width=simd_w](base + j, p_vec)
                j += simd_w
            for col in range(vec_end, row_width):
                var p = param_ptr[base + col]
                var g = grad_ptr[base + col]
                var v = vel_ptr[base + col]
                if self.weight_decay > 0:
                    g += p * self.weight_decay
                v = self.momentum * v + g
                vel_ptr[base + col] = v
                param_ptr[base + col] = p - self.lr * v

    @always_inline
    def _apply_clip_norm_to_ptr[
        simd_w: Int
    ](
        self,
        grad_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
        clip_coef: Scalar[Self.dtype],
    ):
        var clip_vec = SIMD[Self.dtype, simd_w](clip_coef)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w
        for _ in range(vec_end // simd_w):
            var g_vec = grad_ptr.load[width=simd_w](j)
            grad_ptr.store[width=simd_w](j, g_vec * clip_vec)
            j += simd_w
        for k in range(vec_end, num_elements):
            grad_ptr[k] *= clip_coef

    @always_inline
    def _apply_clip_value_to_ptr[
        simd_w: Int
    ](
        self,
        grad_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
    ):
        var min_val = -self.clip_value
        var max_val = self.clip_value
        var min_vec = SIMD[Self.dtype, simd_w](min_val)
        var max_vec = SIMD[Self.dtype, simd_w](max_val)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w
        for _ in range(vec_end // simd_w):
            var g_vec = grad_ptr.load[width=simd_w](j)
            grad_ptr.store[width=simd_w](j, g_vec.clamp(min_vec, max_vec))
            j += simd_w
        for k in range(vec_end, num_elements):
            grad_ptr[k] = max(min_val, min(max_val, grad_ptr[k]))


    def compute_grad_norm(self) -> Scalar[Self.dtype]:
        var total_norm_sq: Scalar[Self.dtype] = 0.0
        comptime simd_w = simd_width_of[Self.dtype]()

        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not (parameter.requires_grad and parameter.has_grad()):
                continue
            ref grad = parameter.gradients()[]
            var num_elements = grad.num_elements()

            comptime if has_accelerator():
                if parameter.is_on_gpu():
                    try:
                        var ds = grad.buffer.device_state.value()
                        with ds.buffer.map_to_host() as host:
                            var grad_ptr = host.unsafe_ptr().bitcast[
                                Scalar[Self.dtype]
                            ]()
                            var norm_vec = SIMD[Self.dtype, simd_w](0)
                            var j = 0
                            var vec_end = (num_elements // simd_w) * simd_w
                            for _ in range(vec_end // simd_w):
                                var g_vec = grad_ptr.load[width=simd_w](j)
                                norm_vec += g_vec * g_vec
                                j += simd_w
                            total_norm_sq += norm_vec.reduce_add()
                            for k in range(vec_end, num_elements):
                                var g = grad_ptr[k]
                                total_norm_sq += g * g
                    except e:
                        panic("SGD.compute_grad_norm GPU failed: " + String(e))
                    continue

            # CPU path
            var grad_ptr = grad.data_ptr()
            var norm_vec = SIMD[Self.dtype, simd_w](0)
            var j = 0
            var vec_end = (num_elements // simd_w) * simd_w
            for _ in range(vec_end // simd_w):
                var g_vec = grad_ptr.load[width=simd_w](j)
                norm_vec += g_vec * g_vec
                j += simd_w
            total_norm_sq += norm_vec.reduce_add()
            for k in range(vec_end, num_elements):
                var g = grad_ptr[k]
                total_norm_sq += g * g

        return sqrt(total_norm_sq)


    def clip_gradients(mut self):
        comptime simd_w = simd_width_of[Self.dtype]()

        if self.clip_norm > 0:
            var total_norm = self.compute_grad_norm()
            if total_norm > self.clip_norm:
                var clip_coef = self.clip_norm / total_norm
                for i in range(len(self.parameters)):
                    ref parameter = self.parameters[i][]
                    if not (parameter.requires_grad and parameter.has_grad()):
                        continue
                    ref grad = parameter.gradients()[]
                    var num_elements = grad.num_elements()

                    comptime if has_accelerator():
                        if parameter.is_on_gpu():
                            try:
                                var ds = grad.buffer.device_state.value()
                                with ds.buffer.map_to_host() as host:
                                    self._apply_clip_norm_to_ptr[simd_w](
                                        host.unsafe_ptr().bitcast[
                                            Scalar[Self.dtype]
                                        ](),
                                        num_elements,
                                        clip_coef,
                                    )
                            except e:
                                panic("SGD.clip_norm GPU failed: " + String(e))
                            continue

                    self._apply_clip_norm_to_ptr[simd_w](
                        grad.data_ptr(), num_elements, clip_coef
                    )

        if self.clip_value > 0:
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if not (parameter.requires_grad and parameter.has_grad()):
                    continue
                ref grad = parameter.gradients()[]
                var num_elements = grad.num_elements()

                comptime if has_accelerator():
                    if parameter.is_on_gpu():
                        try:
                            var ds = grad.buffer.device_state.value()
                            with ds.buffer.map_to_host() as host:
                                self._apply_clip_value_to_ptr[simd_w](
                                    host.unsafe_ptr().bitcast[
                                        Scalar[Self.dtype]
                                    ](),
                                    num_elements,
                                )
                        except e:
                            panic("SGD.clip_value GPU failed: " + String(e))
                        continue

                self._apply_clip_value_to_ptr[simd_w](
                    grad.data_ptr(), num_elements
                )


    @always_inline
    def step(mut self, indices: IntArray = IntArray()):
        """
        Optimized parameter update.

        When `indices` is provided and non-empty, performs a sparse row-wise
        update: only the rows specified in `indices` are updated (2D parameters
        only). Non-2D parameters are skipped in sparse mode.

        Single pass per parameter:
        1. Clip gradient (if needed)
        2. Update velocity (if momentum)
        3. Apply weight decay
        4. Update parameter

        Args:
            indices: Row indices for sparse update (default: empty = dense).
        """

        self.clip_gradients()
        comptime simd_w = simd_width_of[Self.dtype]()
        var is_sparse = len(indices) > 0

        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not (parameter.requires_grad and parameter.has_grad()):
                continue

            ref grad = parameter.gradients()[]
            var num_elements = parameter.num_elements()

            if is_sparse:
                var shape = parameter.shape()
                if shape.rank() != 2:
                    continue
                var row_width = shape[1]

                comptime if has_accelerator():
                    if parameter.is_on_gpu():
                        try:
                            var param_ds = parameter.buffer.device_state.value()
                            var grad_ds = grad.buffer.device_state.value()
                            if self.use_momentum:
                                ref velocity = self.velocities[i]
                                var vel_ds = velocity.buffer.device_state.value()
                                with param_ds.buffer.map_to_host() as param_host, grad_ds.buffer.map_to_host() as grad_host, vel_ds.buffer.map_to_host() as vel_host:
                                    self._apply_momentum_sparse[simd_w](
                                        param_host.unsafe_ptr().bitcast[Scalar[Self.dtype]](),
                                        grad_host.unsafe_ptr().bitcast[Scalar[Self.dtype]](),
                                        vel_host.unsafe_ptr().bitcast[Scalar[Self.dtype]](),
                                        row_width, indices,
                                    )
                            else:
                                with param_ds.buffer.map_to_host() as param_host, grad_ds.buffer.map_to_host() as grad_host:
                                    self._step_no_momentum_sparse[simd_w](
                                        param_host.unsafe_ptr().bitcast[Scalar[Self.dtype]](),
                                        grad_host.unsafe_ptr().bitcast[Scalar[Self.dtype]](),
                                        row_width, indices,
                                    )
                            param_ds.sync()
                        except e:
                            panic("SGD.step GPU sparse failed: " + String(e))
                        continue

                # CPU sparse path
                if self.use_momentum:
                    ref velocity = self.velocities[i]
                    self._apply_momentum_sparse[simd_w](
                        parameter.data_ptr(), grad.data_ptr(),
                        velocity.data_ptr(), row_width, indices,
                    )
                else:
                    self._step_no_momentum_sparse[simd_w](
                        parameter.data_ptr(), grad.data_ptr(),
                        row_width, indices,
                    )
                continue

            # Dense path (existing behavior)
            comptime if has_accelerator():
                if parameter.is_on_gpu():
                    try:
                        var param_ds = parameter.buffer.device_state.value()
                        var grad_ds = grad.buffer.device_state.value()

                        if self.use_momentum:
                            ref velocity = self.velocities[i]
                            var vel_ds = velocity.buffer.device_state.value()
                            with param_ds.buffer.map_to_host() as param_host, grad_ds.buffer.map_to_host() as grad_host, vel_ds.buffer.map_to_host() as vel_host:
                                self._apply_momentum[simd_w](
                                    param_host.unsafe_ptr().bitcast[
                                        Scalar[Self.dtype]
                                    ](),
                                    grad_host.unsafe_ptr().bitcast[
                                        Scalar[Self.dtype]
                                    ](),
                                    vel_host.unsafe_ptr().bitcast[
                                        Scalar[Self.dtype]
                                    ](),
                                    num_elements,
                                )
                        else:
                            with param_ds.buffer.map_to_host() as param_host, grad_ds.buffer.map_to_host() as grad_host:
                                self._step_no_momentum[simd_w](
                                    param_host.unsafe_ptr().bitcast[
                                        Scalar[Self.dtype]
                                    ](),
                                    grad_host.unsafe_ptr().bitcast[
                                        Scalar[Self.dtype]
                                    ](),
                                    num_elements,
                                )

                        param_ds.sync()
                    except e:
                        panic("SGD.step GPU failed: " + String(e))
                    continue

            # CPU dense path
            if self.use_momentum:
                ref velocity = self.velocities[i]
                self._apply_momentum[simd_w](
                    parameter.data_ptr(),
                    grad.data_ptr(),
                    velocity.data_ptr(),
                    num_elements,
                )
            else:
                self._step_no_momentum[simd_w](
                    parameter.data_ptr(),
                    grad.data_ptr(),
                    num_elements,
                )

    @always_inline
    def zero_grad(mut self, indices: IntArray = IntArray()):
        """
        Zero gradients.

        When `indices` is provided and non-empty, only zero the specified
        rows of 2D gradient buffers (sparse mode). Falls back to dense
        zeroing for non-2D parameters.

        Args:
            indices: Row indices for sparse zeroing (default: empty = dense).
        """
        var is_sparse = len(indices) > 0
        comptime simd_w = simd_width_of[Self.dtype]()
        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not parameter.has_grad():
                continue
            ref grad = parameter.gradients()[]
            if is_sparse:
                var shape = parameter.shape()
                if shape.rank() != 2:
                    continue
                self._zero_rows_ptr[simd_w](
                    grad.data_ptr(), shape[1], indices,
                )
                if self.use_momentum:
                    ref velocity = self.velocities[i]
                    self._zero_rows_ptr[simd_w](
                        velocity.data_ptr(), shape[1], indices,
                    )
            else:
                parameter.zero_grad()

    def set_lr(mut self, lr: Scalar[Self.dtype]):
        self.lr = lr

    def get_lr(self) -> Scalar[Self.dtype]:
        return self.lr

    def set_clip_norm(mut self, clip_norm: Scalar[Self.dtype]):
        self.clip_norm = clip_norm

    def set_clip_value(mut self, clip_value: Scalar[Self.dtype]):
        self.clip_value = clip_value

    def set_weight_decay(mut self, weight_decay: Scalar[Self.dtype]):
        self.weight_decay = weight_decay
