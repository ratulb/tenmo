# tenmo/checkpoint.mojo — Phase 4a: CPU-only checkpoint module
#
# Provides:
#   Checkpoint       — data container (model + optimizer + scheduler + metadata)
#   save_state       — serialize full training state to disk
#   load_state       — deserialize Checkpoint from disk (auto-detects old/new format)
#   apply_to_model   — copy model_state ndarrays into a CPU model
#   save_weights     — model-only persist for inference
#   load_weights     — model-only restore for inference

from std.python import Python, PythonObject
from std.memory import memcpy
from .net import Sequential
from .optim import SGD
from .numpy_interop import to_ndarray, ndarray_ptr


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint — opaque data container for serialized training state
#
# NOT dtype-parameterized — dtype is only needed at save/load time
# when converting between ndarrays and Tensors.
#
# Fields:
#   model_state:     {name: ndarray} — CPU ndarrays of model weights
#   optimizer_state: {type, lr, ..., velocities} — hyperparams + velocity buffers
#   scheduler_state: {type, param1, ...} — always empty in Phase 4a (slot for 4b)
#   metadata:        {epoch, step, loss, config, ...} — arbitrary user dict
# ═══════════════════════════════════════════════════════════════════════

@fieldwise_init
struct Checkpoint(ImplicitlyCopyable & Movable):
    var model_state: PythonObject
    var optimizer_state: PythonObject
    var scheduler_state: PythonObject
    var metadata: PythonObject

    def __init__(out self) raises:
        self.model_state = {}
        self.optimizer_state = {}
        self.scheduler_state = {}
        self.metadata = {}

    def __init__(out self, *, copy: Self):
        self.model_state = copy.model_state
        self.optimizer_state = copy.optimizer_state
        self.scheduler_state = copy.scheduler_state
        self.metadata = copy.metadata

    def __init__(out self, *, deinit move: Self):
        self.model_state = move.model_state
        self.optimizer_state = move.optimizer_state
        self.scheduler_state = move.scheduler_state
        self.metadata = move.metadata


# ═══════════════════════════════════════════════════════════════════════
# save_state — serialize full training state to disk
#
# Two overloads:
#   1. (path, model, metadata)     — model weights + metadata
#   2. (path, model, optimizer, metadata) — model + optimizer + metadata
#
# Scheduler state slot always empty in Phase 4a.
# Returns Checkpoint for optional in-memory inspection.
# ═══════════════════════════════════════════════════════════════════════

def save_state[
    dtype: DType, //
](
    path: String,
    model: Sequential[dtype],
    metadata: PythonObject,
) raises -> Checkpoint:
    np = Python.import_module("numpy")

    var model_state: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        model_state[p.name] = to_ndarray(tensor_ptr[])

    var top: PythonObject = {}
    top["model"] = model_state
    top["optimizer"] = {}
    top["scheduler"] = {}
    top["metadata"] = metadata

    np.save(path, top)

    var ckpt = Checkpoint()
    ckpt.model_state = model_state^
    ckpt.optimizer_state = {}
    ckpt.scheduler_state = {}
    ckpt.metadata = metadata
    return ckpt^


def save_state[
    dtype: DType, //
](
    path: String,
    model: Sequential[dtype],
    optimizer: SGD[dtype],
    metadata: PythonObject,
) raises -> Checkpoint:
    np = Python.import_module("numpy")

    var model_state: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        model_state[p.name] = to_ndarray(tensor_ptr[])

    var optimizer_state = optimizer.state_dict()

    var top: PythonObject = {}
    top["model"] = model_state
    top["optimizer"] = optimizer_state
    top["scheduler"] = {}
    top["metadata"] = metadata

    np.save(path, top)

    var ckpt = Checkpoint()
    ckpt.model_state = model_state^
    ckpt.optimizer_state = optimizer_state^
    ckpt.scheduler_state = {}
    ckpt.metadata = metadata
    return ckpt^


# ═══════════════════════════════════════════════════════════════════════
# load_state — deserialize Checkpoint from disk
#
# Auto-detects format:
#   New: data has "model" key → structured dict
#   Old: no "model" key → flat dict {name: ndarray} treated as model_state
# ═══════════════════════════════════════════════════════════════════════

def load_state(path: String) raises -> Checkpoint:
    np = Python.import_module("numpy")
    var data = np.load(path, allow_pickle=True).item()

    var ckpt = Checkpoint()

    if data.__contains__("model"):
        ckpt.model_state = data["model"]
        ckpt.optimizer_state = data["optimizer"] if data.__contains__("optimizer") else {}
        ckpt.scheduler_state = data["scheduler"] if data.__contains__("scheduler") else {}
        ckpt.metadata = data["metadata"] if data.__contains__("metadata") else {}
    else:
        ckpt.model_state = data
        ckpt.optimizer_state = {}
        ckpt.scheduler_state = {}
        ckpt.metadata = {}

    return ckpt^


# ═══════════════════════════════════════════════════════════════════════
# apply_to_model — copy model_state ndarrays into a CPU model's tensors
#
# CPU only in Phase 4a — direct memcpy to tensor data_ptr.
# GPU-aware version deferred to Phase 4b.
#
# Silently skips keys in checkpoint that don't exist in the model
# (safe partial loading, same as old load_checkpoint).
# ═══════════════════════════════════════════════════════════════════════

def apply_to_model[
    dtype: DType, //
](
    mut model: Sequential[dtype], checkpoint: Checkpoint
) raises:
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if checkpoint.model_state.__contains__(key):
            var nd = checkpoint.model_state[key]
            var src_ptr = ndarray_ptr[dtype](nd)
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


# ═══════════════════════════════════════════════════════════════════════
# save_weights — model-only persist for inference / deployment
# ═══════════════════════════════════════════════════════════════════════

def save_weights[
    dtype: DType, //
](path: String, model: Sequential[dtype]) raises:
    var _ = save_state(path, model, {})


# ═══════════════════════════════════════════════════════════════════════
# load_weights — model-only restore for inference / deployment
#
# Works with both old-format (flat dict) and new-format (structured) files.
# ═══════════════════════════════════════════════════════════════════════

def load_weights[
    dtype: DType, //
](mut model: Sequential[dtype], path: String) raises:
    var ckpt = load_state(path)
    apply_to_model(model, ckpt)
