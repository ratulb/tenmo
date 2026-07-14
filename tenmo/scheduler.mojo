from std.math import cos, pi
from std.python import Python, PythonObject
from .numpy_interop import ndarray_ptr


# ═══════════════════════════════════════════════════════════════════════
# StepLR — decay LR by gamma every step_size epochs
# ═══════════════════════════════════════════════════════════════════════

@fieldwise_init
struct StepLR[dtype: DType](ImplicitlyCopyable & Movable):
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
        np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "StepLR"
        state["lr"] = np.array([Float64(self.base_lr)])
        state["step_size"] = np.array([Int64(self.step_size)], dtype=np.int64)
        state["gamma"] = np.array([Float64(self.gamma)])
        state["last_epoch"] = np.array([Int64(self.last_epoch)], dtype=np.int64)
        return state

    def load_state_dict(mut self, state: PythonObject) raises:
        self.base_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["lr"]).load()
        )
        self.step_size = Int(
            ndarray_ptr[DType.int64](state["step_size"]).load()
        )
        self.gamma = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["gamma"]).load()
        )
        self.last_epoch = Int(
            ndarray_ptr[DType.int64](state["last_epoch"]).load()
        )


# ═══════════════════════════════════════════════════════════════════════
# MultiStepLR — decay LR by gamma at each milestone epoch
# ═══════════════════════════════════════════════════════════════════════

@fieldwise_init
struct MultiStepLR[dtype: DType](ImplicitlyCopyable & Movable):
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
        np = Python.import_module("numpy")
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
            ndarray_ptr[DType.float64](state["lr"]).load()
        )
        var num_ms = Int(
            ndarray_ptr[DType.int64](state["num_milestones"]).load()
        )
        var ms_arr = state["milestones"]
        self.milestones = List[Int]()
        for i in range(num_ms):
            self.milestones.append(
                Int(ndarray_ptr[DType.int64](ms_arr)[i])
            )
        self.gamma = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["gamma"]).load()
        )
        self.last_epoch = Int(
            ndarray_ptr[DType.int64](state["last_epoch"]).load()
        )


# ═══════════════════════════════════════════════════════════════════════
# CosineAnnealingLR — cosine decay from base_lr to eta_min in T_max epochs
# ═══════════════════════════════════════════════════════════════════════

@fieldwise_init
struct CosineAnnealingLR[dtype: DType](ImplicitlyCopyable & Movable):
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
        return self.eta_min + (self.base_lr - self.eta_min) * Scalar[Self.dtype](ratio)

    def get_last_lr(self) -> Scalar[Self.dtype]:
        if self.last_epoch < 0:
            return self.base_lr
        if self.T_max <= 0:
            return self.eta_min
        var cos_val = cos(pi * Float64(self.last_epoch) / Float64(self.T_max))
        var ratio = (1 + cos_val) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * Scalar[Self.dtype](ratio)

    def state_dict(self) raises -> PythonObject:
        np = Python.import_module("numpy")
        var state: PythonObject = {}
        state["type"] = "CosineAnnealingLR"
        state["lr"] = np.array([Float64(self.base_lr)])
        state["T_max"] = np.array([Int64(self.T_max)], dtype=np.int64)
        state["eta_min"] = np.array([Float64(self.eta_min)])
        state["last_epoch"] = np.array([Int64(self.last_epoch)], dtype=np.int64)
        return state

    def load_state_dict(mut self, state: PythonObject) raises:
        self.base_lr = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["lr"]).load()
        )
        self.T_max = Int(
            ndarray_ptr[DType.int64](state["T_max"]).load()
        )
        self.eta_min = Scalar[Self.dtype](
            ndarray_ptr[DType.float64](state["eta_min"]).load()
        )
        self.last_epoch = Int(
            ndarray_ptr[DType.int64](state["last_epoch"]).load()
        )
