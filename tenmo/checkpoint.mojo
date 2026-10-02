"""Checkpoint — serialize and restore training state.

`Checkpoint` (model + optimizer + scheduler + metadata), `save_state`,
`load_state`, `apply_to_model`, `save_weights`, `load_weights`,
`save_best_if_improved`, `save_step_checkpoint`. CPU only: weights move
through ndarrays and `unsafe_memcpy`.
"""

from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from .net import Sequential
from .optim import SGD
from .gpt import GPTModel
from .encoder import BertForMLM, BertForSequenceClassification
from .adamw import AdamW
from .numpy_interop import to_ndarray, ndarray_ptr, checked_ndarray_ptr


@fieldwise_init
struct Checkpoint(ImplicitlyCopyable):
    """Serialized training state. NOT dtype-parameterized — dtype is only
    needed at save/load time when converting between ndarrays and Tensors.

    Fields:
      model_state:     {name: ndarray} — CPU ndarrays of model weights
      optimizer_state: {type, lr, ..., velocities} — hyperparams + buffers
      scheduler_state: {type, param1, ...} — empty unless a scheduler was passed
      metadata:        {epoch, step, loss, config, ...} — arbitrary user dict
    """
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


def save_state[
    dtype: DType, //
](
    path: String,
    model: Sequential[dtype],
    metadata: PythonObject,
) raises -> Checkpoint:
    """Serialize model weights + metadata. Optimizer/scheduler stay empty.
    Returns the Checkpoint for optional in-memory inspection.
    """
    var np = Python.import_module("numpy")

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
    var np = Python.import_module("numpy")

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


def load_state(path: String) raises -> Checkpoint:
    """Deserialize a Checkpoint, auto-detecting format: a "model" key means
    the structured layout, otherwise the whole dict is taken as `model_state`
    (legacy flat files).
    """
    var np = Python.import_module("numpy")
    var data = np.load(path, allow_pickle=True).item()

    var ckpt = Checkpoint()

    if data.__contains__("model"):
        ckpt.model_state = data["model"]
        ckpt.optimizer_state = data["optimizer"] if data.__contains__(
            "optimizer"
        ) else {}
        ckpt.scheduler_state = data["scheduler"] if data.__contains__(
            "scheduler"
        ) else {}
        ckpt.metadata = data["metadata"] if data.__contains__(
            "metadata"
        ) else {}
    else:
        ckpt.model_state = data
        ckpt.optimizer_state = {}
        ckpt.scheduler_state = {}
        ckpt.metadata = {}

    return ckpt^


def apply_to_model[
    dtype: DType, //
](mut model: Sequential[dtype], checkpoint: Checkpoint) raises:
    """Copy `model_state` ndarrays into a CPU model's tensors via
    `unsafe_memcpy`. Silently skips checkpoint keys the model lacks, so
    partial loads (e.g. MLM weights into a classifier) work unchanged.
    """
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if checkpoint.model_state.__contains__(key):
            var nd = checkpoint.model_state[key]
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            var src_ptr = checked_ndarray_ptr[dtype](
                nd, t.numels(), "apply_to_model:" + key
            )
            unsafe_memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


def save_weights[
    dtype: DType, //
](path: String, model: Sequential[dtype]) raises:
    """Model-only persist for inference / deployment."""
    var _ = save_state(path, model, {})


def load_weights[
    dtype: DType, //
](mut model: Sequential[dtype], path: String) raises:
    """Model-only restore; reads either checkpoint format."""
    var ckpt = load_state(path)
    apply_to_model(model, ckpt)


def save_state[
    dtype: DType, //
](
    path: String,
    model: GPTModel[dtype],
    optimizer: AdamW[dtype],
    sched_state: PythonObject,
    metadata: PythonObject,
) raises -> Checkpoint:
    """GPT training state: same on-disk format as the `Sequential`
    overloads, so `load_state` reads either without branching.

    `sched_state` is the opaque `WarmupCosineLR.state_dict()` (or `{}`).
    Tied weights stay symmetric: `named_parameters` lists `wte` once, so
    save/load needs no re-tying step.
    """
    var np = Python.import_module("numpy")

    var model_state: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        model_state[p.name] = to_ndarray(tensor_ptr[])

    var optimizer_state = optimizer.state_dict()

    var top: PythonObject = {}
    top["model"] = model_state
    top["optimizer"] = optimizer_state
    top["scheduler"] = sched_state
    top["metadata"] = metadata

    np.save(path, top)

    var ckpt = Checkpoint()
    ckpt.model_state = model_state^
    ckpt.optimizer_state = optimizer_state^
    ckpt.scheduler_state = sched_state
    ckpt.metadata = metadata
    return ckpt^


def apply_to_model[
    dtype: DType, //
](mut model: GPTModel[dtype], checkpoint: Checkpoint) raises:
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if checkpoint.model_state.__contains__(key):
            var nd = checkpoint.model_state[key]
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            var src_ptr = checked_ndarray_ptr[dtype](
                nd, t.numels(), "apply_to_model:" + key
            )
            unsafe_memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


def load_weights[
    dtype: DType, //
](mut model: GPTModel[dtype], path: String) raises:
    var ckpt = load_state(path)
    apply_to_model(model, ckpt)


def save_state[
    dtype: DType, //
](
    path: String,
    model: BertForMLM[dtype],
    optimizer: AdamW[dtype],
    sched_state: PythonObject,
    metadata: PythonObject,
) raises -> Checkpoint:
    """`BertForMLM` training state, same format as the GPT overloads.
    `named_parameters` prefixes names `emb.` / `blk{i}.` / `head.`.
    """
    var np = Python.import_module("numpy")

    var model_state: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        model_state[p.name] = to_ndarray(tensor_ptr[])

    var optimizer_state = optimizer.state_dict()

    var top: PythonObject = {}
    top["model"] = model_state
    top["optimizer"] = optimizer_state
    top["scheduler"] = sched_state
    top["metadata"] = metadata

    np.save(path, top)

    var ckpt = Checkpoint()
    ckpt.model_state = model_state^
    ckpt.optimizer_state = optimizer_state^
    ckpt.scheduler_state = sched_state
    ckpt.metadata = metadata
    return ckpt^


def apply_to_model[
    dtype: DType, //
](mut model: BertForMLM[dtype], checkpoint: Checkpoint) raises:
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if checkpoint.model_state.__contains__(key):
            var nd = checkpoint.model_state[key]
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            var src_ptr = checked_ndarray_ptr[dtype](
                nd, t.numels(), "apply_to_model:" + key
            )
            unsafe_memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


def load_weights[
    dtype: DType, //
](mut model: BertForMLM[dtype], path: String) raises:
    var ckpt = load_state(path)
    apply_to_model(model, ckpt)


def save_state[
    dtype: DType, //
](
    path: String,
    model: BertForSequenceClassification[dtype],
    optimizer: AdamW[dtype],
    sched_state: PythonObject,
    metadata: PythonObject,
) raises -> Checkpoint:
    """`BertForSequenceClassification` training state, same format again.
    Applying an MLM checkpoint fills `emb.`/`blk{i}.` and skips the fresh
    `head.clf.` keys, so pretraining transfers with no surgery.
    """
    var np = Python.import_module("numpy")

    var model_state: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        model_state[p.name] = to_ndarray(tensor_ptr[])

    var optimizer_state = optimizer.state_dict()

    var top: PythonObject = {}
    top["model"] = model_state
    top["optimizer"] = optimizer_state
    top["scheduler"] = sched_state
    top["metadata"] = metadata

    np.save(path, top)

    var ckpt = Checkpoint()
    ckpt.model_state = model_state^
    ckpt.optimizer_state = optimizer_state^
    ckpt.scheduler_state = sched_state
    ckpt.metadata = metadata
    return ckpt^


def apply_to_model[
    dtype: DType, //
](mut model: BertForSequenceClassification[dtype], checkpoint: Checkpoint) raises:
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if checkpoint.model_state.__contains__(key):
            var nd = checkpoint.model_state[key]
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            var src_ptr = checked_ndarray_ptr[dtype](
                nd, t.numels(), "apply_to_model:" + key
            )
            unsafe_memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


def load_weights[
    dtype: DType, //
](mut model: BertForSequenceClassification[dtype], path: String) raises:
    var ckpt = load_state(path)
    apply_to_model(model, ckpt)


def save_best_if_improved[
    dtype: DType, //
](
    path_prefix: String,
    model: Sequential[dtype],
    current_loss: Float64,
    best_loss: Float64,
    metadata: PythonObject,
) raises -> Float64:
    """Always saves `{path_prefix}_latest.npy`; also saves
    `{path_prefix}_best.npy` when `current_loss < best_loss`. Returns the
    new best for the caller's tracker.
    """
    var _ = save_state(path_prefix + "_latest.npy", model, metadata)
    var new_best = best_loss
    if current_loss < best_loss:
        var _ = save_state(path_prefix + "_best.npy", model, metadata)
        new_best = current_loss
    return new_best


def save_best_if_improved[
    dtype: DType, //
](
    path_prefix: String,
    model: Sequential[dtype],
    optimizer: SGD[dtype],
    current_loss: Float64,
    best_loss: Float64,
    metadata: PythonObject,
) raises -> Float64:
    var _ = save_state(path_prefix + "_latest.npy", model, optimizer, metadata)
    var new_best = best_loss
    if current_loss < best_loss:
        var _ = save_state(
            path_prefix + "_best.npy", model, optimizer, metadata
        )
        new_best = current_loss
    return new_best


def save_step_checkpoint[
    dtype: DType, //
](
    path_prefix: String,
    step: Int,
    model: Sequential[dtype],
    metadata: PythonObject,
) raises:
    """Snapshot to `{path_prefix}_step_{step}.npy`."""
    var path = path_prefix + "_step_" + String(step) + ".npy"
    var _ = save_state(path, model, metadata)


def save_step_checkpoint[
    dtype: DType, //
](
    path_prefix: String,
    step: Int,
    model: Sequential[dtype],
    optimizer: SGD[dtype],
    metadata: PythonObject,
) raises:
    var path = path_prefix + "_step_" + String(step) + ".npy"
    var _ = save_state(path, model, optimizer, metadata)
