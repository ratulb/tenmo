from .tensor import Tensor
from .gradbox import Gradbox
from .shared.intarray import IntArray
from std.math import sqrt, pow
from std.sys import simd_width_of, has_accelerator
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from .shared.panic import panic
from .kernels import AdamWKernel
from .numpy_interop import to_ndarray, ndarray_ptr, checked_ndarray_ptr


@fieldwise_init
struct AdamW[dtype: DType](ImplicitlyCopyable & Writable):
    """
        AdamW — Adam with decoupled weight decay.

    Per-parameter adaptive step sizes from the first/second gradient moments
    (m/v states, one Gradbox per parameter each), bias-corrected, with the
    weight-decay term applied OUTSIDE the adaptive denominator:

        m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g*g
        p -= lr * ( (m/c1) / (sqrt(v/c2) + eps) + wd*p )

    where c1 = 1-b1^t, c2 = 1-b2^t.
    SGD stays as the reference optimizer; AdamW is the transformer default.
    """

    var parameters: List[Pointer[Tensor[Self.dtype], MutUntrackedOrigin]]
    var lr: Scalar[Self.dtype]
    var beta1: Scalar[Self.dtype]
    var beta2: Scalar[Self.dtype]
    var eps: Scalar[Self.dtype]
    var weight_decay: Scalar[Self.dtype]
    var clip_norm: Scalar[Self.dtype]
    var clip_value: Scalar[Self.dtype]
    var m_states: List[Gradbox[Self.dtype]]
    var v_states: List[Gradbox[Self.dtype]]
    var step_count: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write("AdamW")

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("AdamW")

    def __init__(
        out self,
        parameters: List[Pointer[Tensor[Self.dtype], MutAnyOrigin]],
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.95,
        eps: Scalar[Self.dtype] = 1e-8,
        weight_decay: Scalar[Self.dtype] = 0.1,
        clip_norm: Scalar[Self.dtype] = 0.0,
        clip_value: Scalar[Self.dtype] = 0.0,
    ):
        if clip_norm < 0 or clip_value < 0:
            panic("Clip_norm and clip_value must be >= 0")
        self.parameters = List[
            Pointer[Tensor[Self.dtype], MutUntrackedOrigin]
        ](capacity=len(parameters))
        for p in parameters:
            self.parameters.append(p.unsafe_origin_cast[MutUntrackedOrigin]())
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.m_states = List[Gradbox[Self.dtype]]()
        self.v_states = List[Gradbox[Self.dtype]]()
        self.step_count = 0

        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]

            comptime if has_accelerator():
                if parameter.is_on_gpu():
                    self.m_states.append(
                        Gradbox[Self.dtype].full(
                            parameter.shape(),
                            Scalar[Self.dtype](0),
                            device=parameter.device(),
                        )
                    )
                    self.v_states.append(
                        Gradbox[Self.dtype].full(
                            parameter.shape(),
                            Scalar[Self.dtype](0),
                            device=parameter.device(),
                        )
                    )
                else:
                    self.m_states.append(
                        Gradbox[Self.dtype].zeros(parameter.shape())
                    )
                    self.v_states.append(
                        Gradbox[Self.dtype].zeros(parameter.shape())
                    )
            else:
                self.m_states.append(
                    Gradbox[Self.dtype].zeros(parameter.shape())
                )
                self.v_states.append(
                    Gradbox[Self.dtype].zeros(parameter.shape())
                )

    def __init__(out self, *, copy: Self):
        self.parameters = copy.parameters.copy()
        self.lr = copy.lr
        self.beta1 = copy.beta1
        self.beta2 = copy.beta2
        self.eps = copy.eps
        self.weight_decay = copy.weight_decay
        self.clip_norm = copy.clip_norm
        self.clip_value = copy.clip_value
        self.m_states = copy.m_states.copy()
        self.v_states = copy.v_states.copy()
        self.step_count = copy.step_count

    def __init__(out self, *, deinit move: Self):
        self.parameters = move.parameters^
        self.lr = move.lr
        self.beta1 = move.beta1
        self.beta2 = move.beta2
        self.eps = move.eps
        self.weight_decay = move.weight_decay
        self.clip_norm = move.clip_norm
        self.clip_value = move.clip_value
        self.m_states = move.m_states^
        self.v_states = move.v_states^
        self.step_count = move.step_count

    @always_inline
    def _adamw_step[
        simd_w: Int
    ](
        self,
        param_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        grad_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        m_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        v_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
        bias_correction1: Scalar[Self.dtype],
        bias_correction2: Scalar[Self.dtype],
    ):
        # One fused pass over (p, g, m, v): update moments, correct the
        # zero-init bias, apply the adaptive step + decoupled decay.
        # Correction scalars are hoisted — computed once per step(), not here.
        var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
        var b1_vec = SIMD[Self.dtype, simd_w](self.beta1)
        var b2_vec = SIMD[Self.dtype, simd_w](self.beta2)
        var omb1_vec = SIMD[Self.dtype, simd_w](
            Scalar[Self.dtype](1) - self.beta1
        )
        var omb2_vec = SIMD[Self.dtype, simd_w](
            Scalar[Self.dtype](1) - self.beta2
        )
        var eps_vec = SIMD[Self.dtype, simd_w](self.eps)
        var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)
        var c1_vec = SIMD[Self.dtype, simd_w](bias_correction1)
        var c2_vec = SIMD[Self.dtype, simd_w](bias_correction2)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w

        for _ in range(vec_end // simd_w):
            var p_vec = param_ptr.unsafe_load[width=simd_w](j)
            var g_vec = grad_ptr.unsafe_load[width=simd_w](j)
            var m_vec = m_ptr.unsafe_load[width=simd_w](j)
            var v_vec = v_ptr.unsafe_load[width=simd_w](j)
            m_vec = b1_vec * m_vec + omb1_vec * g_vec
            v_vec = b2_vec * v_vec + omb2_vec * g_vec * g_vec
            m_ptr.unsafe_store[width=simd_w](j, m_vec)
            v_ptr.unsafe_store[width=simd_w](j, v_vec)
            var m_hat = m_vec / c1_vec
            var v_hat = v_vec / c2_vec
            # eps sits INSIDE the denominator: m_hat / (sqrt(v_hat) + eps).
            p_vec -= lr_vec * (
                m_hat / (sqrt(v_hat) + eps_vec) + wd_vec * p_vec
            )
            param_ptr.unsafe_store[width=simd_w](j, p_vec)
            j += simd_w

        for k in range(vec_end, num_elements):
            var p = param_ptr[unsafe_offset=k]
            var g = grad_ptr[unsafe_offset=k]
            var m = m_ptr[unsafe_offset=k]
            var v = v_ptr[unsafe_offset=k]
            m = self.beta1 * m + (Scalar[Self.dtype](1) - self.beta1) * g
            v = self.beta2 * v + (Scalar[Self.dtype](1) - self.beta2) * g * g
            m_ptr[unsafe_offset=k] = m
            v_ptr[unsafe_offset=k] = v
            var m_hat = m / bias_correction1
            var v_hat = v / bias_correction2
            param_ptr[unsafe_offset=k] = p - self.lr * (
                m_hat / (sqrt(v_hat) + self.eps) + self.weight_decay * p
            )

    @always_inline
    def _apply_clip_norm_to_ptr[
        simd_w: Int
    ](
        self,
        grad_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
        clip_coef: Scalar[Self.dtype],
    ):
        var clip_vec = SIMD[Self.dtype, simd_w](clip_coef)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w
        for _ in range(vec_end // simd_w):
            var g_vec = grad_ptr.unsafe_load[width=simd_w](j)
            grad_ptr.unsafe_store[width=simd_w](j, g_vec * clip_vec)
            j += simd_w
        for k in range(vec_end, num_elements):
            grad_ptr[unsafe_offset=k] *= clip_coef

    @always_inline
    def _apply_clip_value_to_ptr[
        simd_w: Int
    ](
        self,
        grad_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        num_elements: Int,
    ):
        var min_val = -self.clip_value
        var max_val = self.clip_value
        var min_vec = SIMD[Self.dtype, simd_w](min_val)
        var max_vec = SIMD[Self.dtype, simd_w](max_val)
        var j = 0
        var vec_end = (num_elements // simd_w) * simd_w
        for _ in range(vec_end // simd_w):
            var g_vec = grad_ptr.unsafe_load[width=simd_w](j)
            grad_ptr.unsafe_store[width=simd_w](
                j, g_vec.clamp(min_vec, max_vec)
            )
            j += simd_w
        for k in range(vec_end, num_elements):
            grad_ptr[unsafe_offset=k] = max(
                min_val, min(max_val, grad_ptr[unsafe_offset=k])
            )

    def compute_grad_norm(self) -> Scalar[Self.dtype]:
        var total_norm_sq: Scalar[Self.dtype] = 0.0
        comptime simd_w = simd_width_of[Self.dtype]()

        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not (parameter.requires_grad and parameter.has_grad()):
                continue
            ref grad = parameter.gradients()
            var num_elements = grad.num_elements()

            comptime if has_accelerator():
                if parameter.is_on_gpu():
                    try:
                        var ds = grad.buffer().device_state.value()
                        with ds.buffer.map_to_host() as host:
                            var grad_ptr = (
                                host.unsafe_ptr()
                                .unsafe_bitcast[Scalar[Self.dtype]]()
                                .unsafe_origin_cast[MutAnyOrigin]()
                            )
                            var norm_vec = SIMD[Self.dtype, simd_w](0)
                            var j = 0
                            var vec_end = (num_elements // simd_w) * simd_w
                            for _ in range(vec_end // simd_w):
                                var g_vec = grad_ptr.unsafe_load[width=simd_w](
                                    j
                                )
                                norm_vec += g_vec * g_vec
                                j += simd_w
                            total_norm_sq += norm_vec.reduce_add()
                            for k in range(vec_end, num_elements):
                                var g = grad_ptr[unsafe_offset=k]
                                total_norm_sq += g * g
                    except e:
                        panic(
                            "AdamW.compute_grad_norm GPU failed: " + String(e)
                        )
                    continue

            # CPU path
            var grad_ptr = grad.data_ptr()
            var norm_vec = SIMD[Self.dtype, simd_w](0)
            var j = 0
            var vec_end = (num_elements // simd_w) * simd_w
            for _ in range(vec_end // simd_w):
                var g_vec = grad_ptr.unsafe_load[width=simd_w](j)
                norm_vec += g_vec * g_vec
                j += simd_w
            total_norm_sq += norm_vec.reduce_add()
            for k in range(vec_end, num_elements):
                var g = grad_ptr[unsafe_offset=k]
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
                    ref grad = parameter.gradients()
                    var num_elements = grad.num_elements()

                    comptime if has_accelerator():
                        if parameter.is_on_gpu():
                            try:
                                var ds = grad.buffer().device_state.value()
                                with ds.buffer.map_to_host() as host:
                                    self._apply_clip_norm_to_ptr[simd_w](
                                        host.unsafe_ptr()
                                        .unsafe_bitcast[Scalar[Self.dtype]]()
                                        .unsafe_origin_cast[MutAnyOrigin](),
                                        num_elements,
                                        clip_coef,
                                    )
                            except e:
                                panic(
                                    "AdamW.clip_norm GPU failed: " + String(e)
                                )
                            continue

                    self._apply_clip_norm_to_ptr[simd_w](
                        grad.data_ptr(), num_elements, clip_coef
                    )

        if self.clip_value > 0:
            for i in range(len(self.parameters)):
                ref parameter = self.parameters[i][]
                if not (parameter.requires_grad and parameter.has_grad()):
                    continue
                ref grad = parameter.gradients()
                var num_elements = grad.num_elements()

                comptime if has_accelerator():
                    if parameter.is_on_gpu():
                        try:
                            var ds = grad.buffer().device_state.value()
                            with ds.buffer.map_to_host() as host:
                                self._apply_clip_value_to_ptr[simd_w](
                                    host.unsafe_ptr()
                                    .unsafe_bitcast[Scalar[Self.dtype]]()
                                    .unsafe_origin_cast[MutAnyOrigin](),
                                    num_elements,
                                )
                        except e:
                            panic(
                                "AdamW.clip_value GPU failed: " + String(e)
                            )
                        continue

                self._apply_clip_value_to_ptr[simd_w](
                    grad.data_ptr(), num_elements
                )

    @always_inline
    def step(mut self):
        """
                Single AdamW update over all parameters with gradients.

        Order: clip raw grads (before they enter the moments) → bump the
        step counter → hoist bias-correction scalars → one fused SIMD pass
        per parameter (CPU) or one fused kernel launch (GPU).

        v1 is dense-only: no sparse row-wise path (the
        bandwidth arithmetic makes dense passes over wte negligible).
        """
        self.clip_gradients()
        self.step_count += 1
        var t = Scalar[Self.dtype](self.step_count)
        var bias_correction1 = Scalar[Self.dtype](1) - pow(self.beta1, t)
        var bias_correction2 = Scalar[Self.dtype](1) - pow(self.beta2, t)
        comptime simd_w = simd_width_of[Self.dtype]()

        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not (parameter.requires_grad and parameter.has_grad()):
                continue

            ref grad = parameter.gradients()
            var num_elements = parameter.num_elements()

            # Dense GPU path — fused kernel, no CPU round-trip
            comptime if has_accelerator():
                if parameter.is_on_gpu():
                    try:
                        ref m_state = self.m_states[i]
                        ref v_state = self.v_states[i]
                        var param_ndb = parameter.buffer.copy()
                        var grad_ndb = grad.buffer().copy()
                        var m_ndb = m_state.buffer().copy()
                        var v_ndb = v_state.buffer().copy()
                        AdamWKernel[Self.dtype].launch(
                            param_ndb.layout(),
                            param_ndb.device_state.value(),
                            grad_ndb.layout(),
                            grad_ndb.device_state.value(),
                            m_ndb.layout(),
                            m_ndb.device_state.value(),
                            v_ndb.layout(),
                            v_ndb.device_state.value(),
                            num_elements,
                            self.lr,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            self.weight_decay,
                            bias_correction1,
                            bias_correction2,
                            sync=False,
                        )
                    except e:
                        panic("AdamW.step GPU failed: " + String(e))
                    continue

            # CPU dense path — one fused SIMD pass
            ref m_state = self.m_states[i]
            ref v_state = self.v_states[i]
            self._adamw_step[simd_w](
                parameter.data_ptr(),
                grad.data_ptr(),
                m_state.data_ptr(),
                v_state.data_ptr(),
                num_elements,
                bias_correction1,
                bias_correction2,
            )

    @always_inline
    def zero_grad(mut self):
        for i in range(len(self.parameters)):
            ref parameter = self.parameters[i][]
            if not parameter.has_grad():
                continue
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

    def state_dict(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "AdamW"
        state["lr"] = np.array([Float64(self.lr)])
        state["beta1"] = np.array([Float64(self.beta1)])
        state["beta2"] = np.array([Float64(self.beta2)])
        state["eps"] = np.array([Float64(self.eps)])
        state["weight_decay"] = np.array([Float64(self.weight_decay)])
        state["clip_norm"] = np.array([Float64(self.clip_norm)])
        state["clip_value"] = np.array([Float64(self.clip_value)])
        # step_count as float64: exact below 2**53, same idiom as the scalars.
        state["step_count"] = np.array([Float64(self.step_count)])
        var m_list = Python.list()
        for i in range(len(self.m_states)):
            m_list.append(to_ndarray(self.m_states[i]))
        state["m_states"] = m_list
        var v_list = Python.list()
        for i in range(len(self.v_states)):
            v_list.append(to_ndarray(self.v_states[i]))
        state["v_states"] = v_list
        return state

    @staticmethod
    def load_state_dict(
        state: PythonObject,
        parameters: List[Pointer[Tensor[Self.dtype], MutAnyOrigin]],
    ) raises -> Self:
        var np = Python.import_module("numpy")

        var lr = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](state["lr"], 1, "AdamW:lr").unsafe_load()
        )
        var beta1 = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](
                state["beta1"], 1, "AdamW:beta1"
            ).unsafe_load()
        )
        var beta2 = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](
                state["beta2"], 1, "AdamW:beta2"
            ).unsafe_load()
        )
        var eps = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](
                state["eps"], 1, "AdamW:eps"
            ).unsafe_load()
        )
        var weight_decay = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](
                state["weight_decay"], 1, "AdamW:weight_decay"
            ).unsafe_load()
        )
        var clip_norm = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](
                state["clip_norm"], 1, "AdamW:clip_norm"
            ).unsafe_load()
        )
        var clip_value = Scalar[Self.dtype](
            checked_ndarray_ptr[DType.float64](
                state["clip_value"], 1, "AdamW:clip_value"
            ).unsafe_load()
        )
        var step_count = Int(
            checked_ndarray_ptr[DType.float64](
                state["step_count"], 1, "AdamW:step_count"
            ).unsafe_load()
        )

        var opt = Self(
            parameters,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            clip_value=clip_value,
        )
        opt.step_count = step_count

        if state.__contains__("m_states"):
            var saved_m = state["m_states"]
            for i in range(len(opt.m_states)):
                var m_nd = saved_m[i]
                ref dst_m = opt.m_states[i]
                var dst_m_ndb = dst_m.buffer()
                var src_ptr = checked_ndarray_ptr[Self.dtype](
                    m_nd, dst_m_ndb.numels(), "AdamW:m_states"
                )
                unsafe_memcpy(
                    dest=dst_m_ndb.data_ptr().unsafe_mut_cast[True](),
                    src=src_ptr,
                    count=dst_m_ndb.numels(),
                )

        if state.__contains__("v_states"):
            var saved_v = state["v_states"]
            for i in range(len(opt.v_states)):
                var v_nd = saved_v[i]
                ref dst_v = opt.v_states[i]
                var dst_v_ndb = dst_v.buffer()
                var src_ptr = checked_ndarray_ptr[Self.dtype](
                    v_nd, dst_v_ndb.numels(), "AdamW:v_states"
                )
                unsafe_memcpy(
                    dest=dst_v_ndb.data_ptr().unsafe_mut_cast[True](),
                    src=src_ptr,
                    count=dst_v_ndb.numels(),
                )

        return opt
