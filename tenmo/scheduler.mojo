from std.math import cos, pi
from std.python import Python, PythonObject
from .numpy_interop import ndarray_ptr


# StepLR — decay LR by gamma every step_size epochs


@fieldwise_init
struct StepLR[dtype: DType](ImplicitlyCopyable):
    var base_lr: Scalar[Self.dtype]
    var step_size: Int
    var gamma: Scalar[Self.dtype]
    var last_epoch: Int

    def __init__(
        out self,
        base_lr: Scalar[Self.dtype],
        step_size: Int,
        gamma: Scalar[Self.dtype],
    ):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = -1

    def __init__(out self, *, copy: Self):
        self.base_lr = copy.base_lr
        self.step_size = copy.step_size
        self.gamma = copy.gamma
        self.last_epoch = copy.last_epoch

    def __init__(out self, *, deinit move: Self):
        self.base_lr = move.base_lr
        self.step_size = move.step_size
        self.gamma = move.gamma
        self.last_epoch = move.last_epoch

    def step(mut self) -> Scalar[Self.dtype]:
        self.last_epoch += 1
        if self.step_size <= 0:
            return self.base_lr
        var k = self.last_epoch // self.step_size
        var factor = Scalar[Self.dtype](1)
        for _ in range(k):
            factor *= self.gamma
        return self.base_lr * factor

    def get_last_lr(self) -> Scalar[Self.dtype]:
        if self.last_epoch < 0:
            return self.base_lr
        var k = self.last_epoch // self.step_size
        var factor = Scalar[Self.dtype](1)
        for _ in range(k):
            factor *= self.gamma
        return self.base_lr * factor

    def state_dict(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "StepLR"
        state["lr"] = np.array([Float64(self.base_lr)])
        state["step_size"] = np.array([Int64(self.step_size)], dtype=np.int64)
        state["gamma"] = np.array([Float64(self.gamma)])
        state["last_epoch"] = np.array([Int64(self.last_epoch)], dtype=np.int64)
        return state

    def load_state_dict(mut self, state: PythonObject) raises:
        self.base_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["lr"]).unsafe_load()
        )
        self.step_size = Int(
            ndarray_ptr[DType.int64](state["step_size"]).unsafe_load()
        )
        self.gamma = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["gamma"]).unsafe_load()
        )
        self.last_epoch = Int(
            ndarray_ptr[DType.int64](state["last_epoch"]).unsafe_load()
        )


# MultiStepLR — decay LR by gamma at each milestone epoch


@fieldwise_init
struct MultiStepLR[dtype: DType](ImplicitlyCopyable):
    var base_lr: Scalar[Self.dtype]
    var milestones: List[Int]
    var gamma: Scalar[Self.dtype]
    var last_epoch: Int

    def __init__(
        out self,
        base_lr: Scalar[Self.dtype],
        milestones: List[Int],
        gamma: Scalar[Self.dtype],
    ):
        self.base_lr = base_lr
        self.milestones = milestones.copy()
        self.gamma = gamma
        self.last_epoch = -1

    def __init__(out self, *, copy: Self):
        self.base_lr = copy.base_lr
        self.milestones = copy.milestones.copy()
        self.gamma = copy.gamma
        self.last_epoch = copy.last_epoch

    def __init__(out self, *, deinit move: Self):
        self.base_lr = move.base_lr
        self.milestones = move.milestones^
        self.gamma = move.gamma
        self.last_epoch = move.last_epoch

    def step(mut self) -> Scalar[Self.dtype]:
        self.last_epoch += 1
        var factor = Scalar[Self.dtype](1)
        for i in range(len(self.milestones)):
            if self.last_epoch >= self.milestones[i]:
                factor *= self.gamma
        return self.base_lr * factor

    def get_last_lr(self) -> Scalar[Self.dtype]:
        if self.last_epoch < 0:
            return self.base_lr
        var factor = Scalar[Self.dtype](1)
        for i in range(len(self.milestones)):
            if self.last_epoch >= self.milestones[i]:
                factor *= self.gamma
        return self.base_lr * factor

    def state_dict(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "MultiStepLR"
        state["lr"] = np.array([Float64(self.base_lr)])
        var num_ms = len(self.milestones)
        var ms_arr = np.zeros(num_ms, dtype=np.int64)
        for i in range(num_ms):
            ms_arr[i] = self.milestones[i]
        state["num_milestones"] = np.array([Int64(num_ms)], dtype=np.int64)
        state["milestones"] = ms_arr
        state["gamma"] = np.array([Float64(self.gamma)])
        state["last_epoch"] = np.array([Int64(self.last_epoch)], dtype=np.int64)
        return state

    def load_state_dict(mut self, state: PythonObject) raises:
        self.base_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["lr"]).unsafe_load()
        )
        var num_ms = Int(
            ndarray_ptr[DType.int64](state["num_milestones"]).unsafe_load()
        )
        var ms_arr = state["milestones"]
        self.milestones = List[Int]()
        for i in range(num_ms):
            self.milestones.append(
                Int(ndarray_ptr[DType.int64](ms_arr)[unsafe_offset=i])
            )
        self.gamma = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["gamma"]).unsafe_load()
        )
        self.last_epoch = Int(
            ndarray_ptr[DType.int64](state["last_epoch"]).unsafe_load()
        )


# CosineAnnealingLR — cosine decay from base_lr to eta_min in T_max epochs


@fieldwise_init
struct CosineAnnealingLR[dtype: DType](ImplicitlyCopyable):
    var base_lr: Scalar[Self.dtype]
    var T_max: Int
    var eta_min: Scalar[Self.dtype]
    var last_epoch: Int

    def __init__(
        out self,
        base_lr: Scalar[Self.dtype],
        T_max: Int,
        eta_min: Scalar[Self.dtype],
    ):
        self.base_lr = base_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = -1

    def __init__(out self, *, copy: Self):
        self.base_lr = copy.base_lr
        self.T_max = copy.T_max
        self.eta_min = copy.eta_min
        self.last_epoch = copy.last_epoch

    def __init__(out self, *, deinit move: Self):
        self.base_lr = move.base_lr
        self.T_max = move.T_max
        self.eta_min = move.eta_min
        self.last_epoch = move.last_epoch

    def step(mut self) -> Scalar[Self.dtype]:
        self.last_epoch += 1
        if self.T_max <= 0:
            return self.eta_min
        var cos_val = cos(pi * Float64(self.last_epoch) / Float64(self.T_max))
        var ratio = (1 + cos_val) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * Scalar[
            Self.dtype
        ](ratio)

    def get_last_lr(self) -> Scalar[Self.dtype]:
        if self.last_epoch < 0:
            return self.base_lr
        if self.T_max <= 0:
            return self.eta_min
        var cos_val = cos(pi * Float64(self.last_epoch) / Float64(self.T_max))
        var ratio = (1 + cos_val) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * Scalar[
            Self.dtype
        ](ratio)

    def state_dict(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "CosineAnnealingLR"
        state["lr"] = np.array([Float64(self.base_lr)])
        state["T_max"] = np.array([Int64(self.T_max)], dtype=np.int64)
        state["eta_min"] = np.array([Float64(self.eta_min)])
        state["last_epoch"] = np.array([Int64(self.last_epoch)], dtype=np.int64)
        return state

    def load_state_dict(mut self, state: PythonObject) raises:
        self.base_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["lr"]).unsafe_load()
        )
        self.T_max = Int(ndarray_ptr[DType.int64](state["T_max"]).unsafe_load())
        self.eta_min = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["eta_min"]).unsafe_load()
        )
        self.last_epoch = Int(
            ndarray_ptr[DType.int64](state["last_epoch"]).unsafe_load()
        )


@fieldwise_init
struct WarmupCosineLR[dtype: DType](ImplicitlyCopyable):
    """WarmupCosineLR.
    Linear warmup 0 → max_lr, then cosine decay to min_lr
    LLM training driver. This is the learn-rate plan
    as a stateful struct:
    the scheduler owns the schedule, the optimizer only reads `lr` each
    step via `optim.set_lr(sched.step())`.
    Phase shape over optimizer step t (0-based):
      t < warmup_steps : max_lr * t / warmup_steps   (linear ramp from 0)
      t >= max_steps   : min_lr                       (floor, never below)
      else             : min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))
                         with progress = (t-warmup)/(max-warmup)
    Design notes (all load-bearing, do not "simplify"):
    - `last_step` counts OPTIMIZER steps, not epochs (-1 before the first
      `step()`), and persists via state_dict — the same resume-correctness
      rule as AdamW's step counter: a resume that resets the
      counter would replay warmup mid-training and silently spike the LR.
    - `lr_at(step)` is a pure query (no mutation) so the driver can probe
      the schedule (logging, pilot re-tuning) without advancing it.
    - Warmup from EXACTLY 0 (`lr_at(0) == 0`): the first update is a
      no-op step, which is the standard nanoGPT/GPT warmup convention.
    - Degenerate configs degrade gracefully, never divide by zero:
      warmup_steps <= 0 skips warmup (pure cosine from max_lr);
      max_steps <= warmup_steps collapses to warmup-then-floor.
    - Arithmetic runs in Float64 and casts once, mirroring
      CosineAnnealingLR above — the schedule must be bit-stable across
      resume (state holds Ints + float64 round-trip, not dtype scalars).
    """
    var max_lr: Scalar[Self.dtype]
    var min_lr: Scalar[Self.dtype]
    var warmup_steps: Int
    var max_steps: Int
    var last_step: Int

    def __init__(
        out self,
        max_lr: Scalar[Self.dtype],
        min_lr: Scalar[Self.dtype],
        warmup_steps: Int,
        max_steps: Int,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.last_step = -1

    def __init__(out self, *, copy: Self):
        self.max_lr = copy.max_lr
        self.min_lr = copy.min_lr
        self.warmup_steps = copy.warmup_steps
        self.max_steps = copy.max_steps
        self.last_step = copy.last_step

    def __init__(out self, *, deinit move: Self):
        self.max_lr = move.max_lr
        self.min_lr = move.min_lr
        self.warmup_steps = move.warmup_steps
        self.max_steps = move.max_steps
        self.last_step = move.last_step

    def lr_at(self, step: Int) -> Scalar[Self.dtype]:
        """Learn rate at optimizer step `step` (pure query, see phases)."""
        if step <= 0:
            return Scalar[Self.dtype](0)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * Scalar[Self.dtype](
                Float64(step) / Float64(self.warmup_steps)
            )
        if step >= self.max_steps:
            return self.min_lr
        var span = self.max_steps - self.warmup_steps
        if span <= 0:
            return self.min_lr
        var progress = Float64(step - self.warmup_steps) / Float64(span)
        var coeff = 0.5 * (1 + cos(pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * Scalar[
            Self.dtype
        ](coeff)

    def step(mut self) -> Scalar[Self.dtype]:
        self.last_step += 1
        return self.lr_at(self.last_step)

    def get_last_lr(self) -> Scalar[Self.dtype]:
        if self.last_step < 0:
            return Scalar[Self.dtype](0)
        return self.lr_at(self.last_step)

    def state_dict(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "WarmupCosineLR"
        state["max_lr"] = np.array([Float64(self.max_lr)])
        state["min_lr"] = np.array([Float64(self.min_lr)])
        state["warmup_steps"] = np.array(
            [Int64(self.warmup_steps)], dtype=np.int64
        )
        state["max_steps"] = np.array(
            [Int64(self.max_steps)], dtype=np.int64
        )
        state["last_step"] = np.array(
            [Int64(self.last_step)], dtype=np.int64
        )
        return state

    def load_state_dict(mut self, state: PythonObject) raises:
        self.max_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["max_lr"]).unsafe_load()
        )
        self.min_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["min_lr"]).unsafe_load()
        )
        self.warmup_steps = Int(
            ndarray_ptr[DType.int64](state["warmup_steps"]).unsafe_load()
        )
        self.max_steps = Int(
            ndarray_ptr[DType.int64](state["max_steps"]).unsafe_load()
        )
        self.last_step = Int(
            ndarray_ptr[DType.int64](state["last_step"]).unsafe_load()
        )
