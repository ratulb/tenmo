from tenmo.tensor import Tensor
from tenmo.net import Sequential, Linear, ReLU, Conv2D, Flatten
from tenmo.numpy_interop import save_checkpoint, load_checkpoint, to_ndarray, ndarray_ptr
from tenmo.named_parameter import NamedParameter
from tenmo.shapes import Shape
from tenmo.forwards import LayerNorm
from tenmo.embedding import Embedding
from tenmo.optim import SGD
from tenmo.checkpoint import Checkpoint, save_state, load_state, apply_to_model, save_weights, load_weights, save_best_if_improved, save_step_checkpoint
from std.python import Python, PythonObject
from std.testing import assert_true, assert_equal, TestSuite


def test_named_params_linear() raises:
    var lin = Linear[DType.float32](4, 3)
    var params = lin.named_parameters("")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "weight")
    assert_equal(params[1].name, "bias")


def test_named_params_linear_with_prefix() raises:
    var lin = Linear[DType.float32](4, 3)
    var params = lin.named_parameters("0.")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "0.weight")
    assert_equal(params[1].name, "0.bias")


def test_named_params_relu_empty() raises:
    var relu = ReLU[DType.float32]()
    var params = relu.named_parameters("")
    assert_equal(len(params), 0)


def test_named_params_embedding() raises:
    """Embedding named_parameters returns one entry: 'weight'."""
    var emb = Embedding[DType.float32](10, 64)
    var params = emb.named_parameters("")
    assert_equal(len(params), 1)
    assert_equal(params[0].name, "weight")


def test_named_params_embedding_frozen() raises:
    """Embedding with freeze=True has no trainable parameters."""
    var emb = Embedding[DType.float32](10, 64, freeze=True)
    var params = emb.named_parameters("")
    assert_equal(len(params), 0)


def test_named_params_sequential_with_embedding() raises:
    """Sequential containing Linear + Embedding enumerates both."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](64, 32).into())
    model.append(Embedding[DType.float32](10, 64).into())
    var params = model.named_parameters("")
    # Linear: weight + bias = 2; Embedding: weight = 1
    assert_equal(len(params), 3)
    assert_equal(params[0].name, "0.weight")
    assert_equal(params[1].name, "0.bias")
    assert_equal(params[2].name, "1.weight")


def test_named_params_sequential_simple() raises:
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into())
    var params = model.named_parameters("")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "0.weight")
    assert_equal(params[1].name, "0.bias")


def test_named_params_sequential_conv() raises:
    var model = Sequential[DType.float32]()
    model.append(
        Conv2D[DType.float32](1, 2, 3, bias=True).into(),
        Flatten[DType.float32]().into(),
        Linear[DType.float32](8, 4).into(),
    )
    var params = model.named_parameters("")
    # Conv2D: weight + bias = 2; Linear: weight + bias = 2
    assert_equal(len(params), 4)
    assert_equal(params[0].name, "0.weight")
    assert_equal(params[1].name, "0.bias")
    assert_equal(params[2].name, "2.weight")
    assert_equal(params[3].name, "2.bias")


def test_checkpoint_roundtrip_linear() raises:
    var lin = Linear[DType.float32](4, 3)
    var model = Sequential[DType.float32]()
    model.append(lin.into())
    var original_w = Tensor[DType.float32].rand(shape=Shape(4, 3))
    var original_b = Tensor[DType.float32].rand(Shape(3))
    model.modules[0].layer[Linear[DType.float32]].weight = original_w
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(original_b)
    var tmp_path = "/tmp/test_ckpt_linear.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3).into())
    load_checkpoint(model2, tmp_path)
    var w_loaded = model2.modules[0].layer[Linear[DType.float32]].weight
    var b_loaded = model2.modules[0].layer[Linear[DType.float32]].bias.value()
    assert_true(w_loaded.all_close(original_w))
    assert_true(b_loaded.all_close(original_b))


def test_checkpoint_roundtrip_multi_layer() raises:
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into(), Linear[DType.float32](3, 2).into())
    var orig_w0 = Tensor[DType.float32].rand(Shape(4, 3))
    var orig_b0 = Tensor[DType.float32].rand(Shape(3))
    var orig_w1 = Tensor[DType.float32].rand(Shape(3, 2))
    var orig_b1 = Tensor[DType.float32].rand(Shape(2))
    model.modules[0].layer[Linear[DType.float32]].weight = orig_w0
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(orig_b0)
    model.modules[2].layer[Linear[DType.float32]].weight = orig_w1
    model.modules[2].layer[Linear[DType.float32]].bias = Optional(orig_b1)
    var tmp_path = "/tmp/test_ckpt_multi.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into(), Linear[DType.float32](3, 2).into())
    load_checkpoint(model2, tmp_path)
    var w0_loaded = model2.modules[0].layer[Linear[DType.float32]].weight
    var b0_loaded = model2.modules[0].layer[Linear[DType.float32]].bias.value()
    var w1_loaded = model2.modules[2].layer[Linear[DType.float32]].weight
    var b1_loaded = model2.modules[2].layer[Linear[DType.float32]].bias.value()
    assert_true(w0_loaded.all_close(orig_w0))
    assert_true(b0_loaded.all_close(orig_b0))
    assert_true(w1_loaded.all_close(orig_w1))
    assert_true(b1_loaded.all_close(orig_b1))


def test_checkpoint_roundtrip_conv2d() raises:
    var model = Sequential[DType.float32]()
    model.append(Conv2D[DType.float32](1, 2, 3, bias=True).into(), Flatten[DType.float32]().into())
    var orig_w = Tensor[DType.float32].rand(shape=Shape(2, 1, 3, 3))
    var orig_b = Tensor[DType.float32].rand(Shape(2))
    model.modules[0].layer[Conv2D[DType.float32]].weight = orig_w
    model.modules[0].layer[Conv2D[DType.float32]].bias = Optional(orig_b)
    var tmp_path = "/tmp/test_ckpt_conv.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Conv2D[DType.float32](1, 2, 3, bias=True).into(), Flatten[DType.float32]().into())
    load_checkpoint(model2, tmp_path)
    var w_loaded = model2.modules[0].layer[Conv2D[DType.float32]].weight
    var b_loaded = model2.modules[0].layer[Conv2D[DType.float32]].bias.value()
    assert_true(w_loaded.all_close(orig_w))
    assert_true(b_loaded.all_close(orig_b))


def test_checkpoint_roundtrip_embedding() raises:
    """Save and load a Sequential with a single Embedding layer."""
    var model = Sequential[DType.float32]()
    model.append(Embedding[DType.float32](10, 64).into())
    var orig_w = Tensor[DType.float32].rand(Shape(10, 64), requires_grad=True)
    model.modules[0].layer[Embedding[DType.float32]].weight = orig_w
    var tmp_path = "/tmp/test_ckpt_emb.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Embedding[DType.float32](10, 64).into())
    load_checkpoint(model2, tmp_path)
    var w_loaded = model2.modules[0].layer[Embedding[DType.float32]].weight
    assert_true(w_loaded.all_close(orig_w))


def test_checkpoint_roundtrip_full_mix() raises:
    """Save and load a Sequential with Linear + Embedding + LayerNorm."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](64, 32).into())
    model.append(Embedding[DType.float32](10, 64).into())
    model.append(LayerNorm[DType.float32](32).into())
    var orig_w0 = Tensor[DType.float32].rand(Shape(64, 32))
    var orig_b0 = Tensor[DType.float32].rand(Shape(32))
    var orig_w1 = Tensor[DType.float32].rand(Shape(10, 64), requires_grad=True)
    var orig_gamma = Tensor[DType.float32].rand(Shape(32))
    var orig_beta = Tensor[DType.float32].rand(Shape(32))
    model.modules[0].layer[Linear[DType.float32]].weight = orig_w0
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(orig_b0)
    model.modules[1].layer[Embedding[DType.float32]].weight = orig_w1
    model.modules[2].layer[LayerNorm[DType.float32]].gamma = orig_gamma
    model.modules[2].layer[LayerNorm[DType.float32]].beta = orig_beta
    var tmp_path = "/tmp/test_ckpt_full_mix.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](64, 32).into())
    model2.append(Embedding[DType.float32](10, 64).into())
    model2.append(LayerNorm[DType.float32](32).into())
    load_checkpoint(model2, tmp_path)
    assert_true(model2.modules[0].layer[Linear[DType.float32]].weight.all_close(orig_w0))
    assert_true(model2.modules[0].layer[Linear[DType.float32]].bias.value().all_close(orig_b0))
    assert_true(model2.modules[1].layer[Embedding[DType.float32]].weight.all_close(orig_w1))
    assert_true(model2.modules[2].layer[LayerNorm[DType.float32]].gamma.all_close(orig_gamma))
    assert_true(model2.modules[2].layer[LayerNorm[DType.float32]].beta.all_close(orig_beta))


def test_checkpoint_pytorch_compatible_format() raises:
    np = Python.import_module("numpy")
    var state_dict: PythonObject = {}
    state_dict["0.weight"] = to_ndarray(Tensor[DType.float32].ones(Shape(4, 3)))
    state_dict["0.bias"] = to_ndarray(Tensor[DType.float32].zeros(Shape(3)))
    var tmp_path = "/tmp/test_ckpt_pytorch.npy"
    np.save(tmp_path, state_dict)
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    load_checkpoint(model, tmp_path)
    var w = model.modules[0].layer[Linear[DType.float32]].weight
    var b = model.modules[0].layer[Linear[DType.float32]].bias.value()
    # Note: saved as float64, loaded into float32 — values should match
    assert_true(w.all_close(Tensor[DType.float32].ones(Shape(4, 3))))
    assert_true(b.all_close(Tensor[DType.float32].zeros(Shape(3))))


def test_load_numpy_dict() raises:
    np = Python.import_module("numpy")
    var manual_dict: PythonObject = {}
    var t = Tensor[DType.float64].arange(6)
    t = t.reshape(2, 3)
    manual_dict["my_weight"] = to_ndarray(t)
    var tmp_path = "/tmp/test_manual_dict.npy"
    np.save(tmp_path, manual_dict)
    var loaded = np.load(tmp_path, allow_pickle=True).item()
    assert_true(loaded.__contains__("my_weight"))


def test_sgd_state_dict_no_momentum() raises:
    """SGD without momentum produces state_dict with no velocities key."""
    var param = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    var params = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))
    var opt = SGD[DType.float32](params^, lr=0.1)
    var state = opt.state_dict()
    assert_true(state.__contains__("type"))
    assert_true(state["type"] == "SGD")
    assert_true(state.__contains__("lr"))
    assert_true(state.__contains__("momentum"))
    assert_true(state.__contains__("weight_decay"))
    assert_true(state.__contains__("velocities") == False)


def test_sgd_state_dict_with_momentum() raises:
    """SGD with momentum produces state_dict with velocities list."""
    var param = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    var params = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))
    var opt = SGD[DType.float32](params^, lr=0.1, momentum=0.9)
    var state = opt.state_dict()
    assert_true(state.__contains__("velocities"))
    assert_equal(len(state["velocities"]), 1)


def test_sgd_load_state_dict_full() raises:
    """SGD.load_state_dict restores all hyperparameters exactly."""
    var param = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    var params = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))
    var opt = SGD[DType.float32](
        params^, lr=0.01, momentum=0.9, weight_decay=1e-4,
        clip_norm=1.0, clip_value=0.5,
    )
    var state = opt.state_dict()
    var params2 = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    var param_copy = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    params2.append(UnsafePointer(to=param_copy))
    var restored = SGD[DType.float32].load_state_dict(state, params2^)
    var restored_state = restored.state_dict()
    assert_true(
        ndarray_ptr[DType.float64](state["lr"]).load() ==
        ndarray_ptr[DType.float64](restored_state["lr"]).load()
    )
    assert_true(
        ndarray_ptr[DType.float64](state["momentum"]).load() ==
        ndarray_ptr[DType.float64](restored_state["momentum"]).load()
    )
    assert_true(
        ndarray_ptr[DType.float64](state["weight_decay"]).load() ==
        ndarray_ptr[DType.float64](restored_state["weight_decay"]).load()
    )
    assert_true(
        ndarray_ptr[DType.float64](state["clip_norm"]).load() ==
        ndarray_ptr[DType.float64](restored_state["clip_norm"]).load()
    )
    assert_true(
        ndarray_ptr[DType.float64](state["clip_value"]).load() ==
        ndarray_ptr[DType.float64](restored_state["clip_value"]).load()
    )


def test_sgd_velocity_values_preserved() raises:
    """Velocity buffer values survive state_dict roundtrip."""
    var param = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    var params = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))
    var opt = SGD[DType.float32](params^, lr=0.1, momentum=0.9)
    param.seed_grad(Scalar[DType.float32](1.0))
    opt.step()
    var state = opt.state_dict()
    var params2 = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    var param_copy = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    params2.append(UnsafePointer(to=param_copy))
    var restored = SGD[DType.float32].load_state_dict(state, params2^)
    var orig_vel = to_ndarray(opt.velocities[0])
    var loaded_vel = to_ndarray(restored.velocities[0])
    np = Python.import_module("numpy")
    assert_true(np.array_equal(orig_vel, loaded_vel))


def test_sgd_state_dict_defaults() raises:
    """SGD state_dict roundtrips with zero-valued optional hyperparams."""
    var param = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    var params = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))
    var opt = SGD[DType.float32](params^, lr=0.01, weight_decay=0.0, clip_norm=0.0, clip_value=0.0)
    var state = opt.state_dict()
    var params2 = List[UnsafePointer[Tensor[DType.float32], MutAnyOrigin]]()
    var param_copy = Tensor[DType.float32].ones(Shape(2, 2), requires_grad=True)
    params2.append(UnsafePointer(to=param_copy))
    var restored = SGD[DType.float32].load_state_dict(state, params2^)
    var restored_state = restored.state_dict()
    assert_true(
        ndarray_ptr[DType.float64](state["lr"]).load() ==
        ndarray_ptr[DType.float64](restored_state["lr"]).load()
    )
    assert_true(
        ndarray_ptr[DType.float64](state["weight_decay"]).load() ==
        ndarray_ptr[DType.float64](restored_state["weight_decay"]).load()
    )
    assert_true(
        ndarray_ptr[DType.float64](state["clip_norm"]).load() ==
        ndarray_ptr[DType.float64](restored_state["clip_norm"]).load()
    )


def test_named_params_layernorm() raises:
    var ln = LayerNorm[DType.float32](8)
    var params = ln.named_parameters("")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "gamma")
    assert_equal(params[1].name, "beta")


# ═══════════════════════════════════════════════════════════════════════
# Phase 4a — Checkpoint module tests
# ═══════════════════════════════════════════════════════════════════════


def test_save_state_model_only() raises:
    """Save model weights with save_state and verify load_state restores them."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into())
    var orig_w = Tensor[DType.float32].rand(Shape(4, 3))
    var orig_b = Tensor[DType.float32].rand(Shape(3))
    model.modules[0].layer[Linear[DType.float32]].weight = orig_w
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(orig_b)
    var tmp_path = "/tmp/test_save_state_model_only.npy"
    var ckpt = save_state(tmp_path, model, {})
    assert_true(ckpt.model_state.__contains__("0.weight"))
    assert_true(ckpt.model_state.__contains__("0.bias"))
    var loaded = load_state(tmp_path)
    assert_true(loaded.model_state.__contains__("0.weight"))
    assert_true(loaded.model_state.__contains__("0.bias"))
    assert_equal(len(loaded.model_state), 2)


def test_save_state_with_optimizer() raises:
    """Save optimizer state with save_state and verify via state_dict roundtrip."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var params = model.parameters()
    var opt = SGD[DType.float32](params^, lr=0.01, momentum=0.9)
    var tmp_path = "/tmp/test_save_state_with_optimizer.npy"
    var ckpt = save_state(tmp_path, model, opt, {})
    assert_true(ckpt.optimizer_state.__contains__("type"))
    assert_true(ckpt.optimizer_state["type"] == "SGD")
    assert_true(ckpt.optimizer_state.__contains__("lr"))
    assert_true(ckpt.optimizer_state.__contains__("momentum"))
    assert_true(ckpt.optimizer_state.__contains__("velocities"))
    var loaded = load_state(tmp_path)
    assert_true(loaded.optimizer_state.__contains__("type"))
    assert_true(
        ndarray_ptr[DType.float64](ckpt.optimizer_state["lr"]).load()
        == ndarray_ptr[DType.float64](loaded.optimizer_state["lr"]).load()
    )


def test_save_state_with_metadata() raises:
    """Store user-provided metadata with save_state and verify load_state recovers it."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var meta: PythonObject = {}
    meta["epoch"] = 42
    meta["step"] = 1000
    meta["best_loss"] = 0.123
    meta["config"] = "test-config"
    var tmp_path = "/tmp/test_save_state_metadata.npy"
    var ckpt = save_state(tmp_path, model, meta)
    assert_true(ckpt.metadata.__contains__("epoch"))
    assert_true(ckpt.metadata["epoch"] == 42)
    assert_true(ckpt.metadata["step"] == 1000)
    assert_true(ckpt.metadata["best_loss"] == 0.123)
    assert_true(ckpt.metadata["config"] == "test-config")
    var loaded = load_state(tmp_path)
    assert_true(loaded.metadata["epoch"] == 42)
    assert_true(loaded.metadata["step"] == 1000)


def test_apply_to_model_cpu() raises:
    """Copy weights from checkpoint into a CPU model via apply_to_model."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var orig_w = Tensor[DType.float32].rand(Shape(4, 3))
    var orig_b = Tensor[DType.float32].rand(Shape(3))
    model.modules[0].layer[Linear[DType.float32]].weight = orig_w
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(orig_b)
    var tmp_path = "/tmp/test_apply_to_model_cpu.npy"
    var ckpt = save_state(tmp_path, model, {})
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3).into())
    apply_to_model(model2, ckpt)
    assert_true(model2.modules[0].layer[Linear[DType.float32]].weight.all_close(orig_w))
    assert_true(model2.modules[0].layer[Linear[DType.float32]].bias.value().all_close(orig_b))


def test_apply_to_model_skips_missing_keys() raises:
    """Skip checkpoint keys not in the model via apply_to_model (safe partial loading)."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var orig_w = Tensor[DType.float32].rand(Shape(4, 3))
    var orig_b = Tensor[DType.float32].rand(Shape(3))
    model.modules[0].layer[Linear[DType.float32]].weight = orig_w
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(orig_b)
    var tmp_path = "/tmp/test_apply_to_model_skip.npy"
    var ckpt = save_state(tmp_path, model, {})
    # model2 has different architecture (no bias)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3, bias=False).into())
    apply_to_model(model2, ckpt)
    assert_true(model2.modules[0].layer[Linear[DType.float32]].weight.all_close(orig_w))


def test_backward_compat_old_format() raises:
    """Detect old flat-format dict (no 'model' key) and verify load_state wraps it."""
    np = Python.import_module("numpy")
    var state_dict: PythonObject = {}
    state_dict["0.weight"] = to_ndarray(Tensor[DType.float32].ones(Shape(4, 3)))
    state_dict["0.bias"] = to_ndarray(Tensor[DType.float32].zeros(Shape(3)))
    var tmp_path = "/tmp/test_backward_compat_old.npy"
    np.save(tmp_path, state_dict)
    var ckpt = load_state(tmp_path)
    assert_true(ckpt.model_state.__contains__("0.weight"))
    assert_true(ckpt.model_state.__contains__("0.bias"))
    # old-format: optimizer/scheduler/metadata should be empty
    assert_true(ckpt.optimizer_state == PythonObject({}))
    assert_true(ckpt.scheduler_state == PythonObject({}))
    assert_true(ckpt.metadata == PythonObject({}))


def test_backward_compat_old_load_weights() raises:
    """Verify load_weights works with old flat-format files."""
    np = Python.import_module("numpy")
    var state_dict: PythonObject = {}
    state_dict["0.weight"] = to_ndarray(Tensor[DType.float32].full(Shape(4, 3), 2.0))
    state_dict["0.bias"] = to_ndarray(Tensor[DType.float32].ones(Shape(3)))
    var tmp_path = "/tmp/test_backward_compat_old_lw.npy"
    np.save(tmp_path, state_dict)
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    load_weights(model, tmp_path)
    assert_true(model.modules[0].layer[Linear[DType.float32]].weight.all_close(
        Tensor[DType.float32].full(Shape(4, 3), 2.0)
    ))
    assert_true(model.modules[0].layer[Linear[DType.float32]].bias.value().all_close(
        Tensor[DType.float32].ones(Shape(3))
    ))


def test_save_weights_roundtrip() raises:
    """Verify save_weights + load_weights roundtrip preserves model weights."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into())
    var orig_w = Tensor[DType.float32].rand(Shape(4, 3))
    var orig_b = Tensor[DType.float32].rand(Shape(3))
    model.modules[0].layer[Linear[DType.float32]].weight = orig_w
    model.modules[0].layer[Linear[DType.float32]].bias = Optional(orig_b)
    var tmp_path = "/tmp/test_save_weights_roundtrip.npy"
    save_weights(tmp_path, model)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into())
    load_weights(model2, tmp_path)
    assert_true(model2.modules[0].layer[Linear[DType.float32]].weight.all_close(orig_w))
    assert_true(model2.modules[0].layer[Linear[DType.float32]].bias.value().all_close(orig_b))


def test_load_state_new_format() raises:
    """Detect new format via 'model' key in load_state."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var tmp_path = "/tmp/test_load_state_new_format.npy"
    var saved = save_state(tmp_path, model, {})
    var loaded = load_state(tmp_path)
    assert_true(loaded.model_state.__contains__("0.weight"))
    # verify it's NOT the old-format fallback — optimizer should be empty dict, not absent
    assert_true(loaded.optimizer_state.__contains__("type") == False)


def test_checkpoint_struct_default_construction() raises:
    """Default Checkpoint has all fields as empty dicts."""
    var ckpt = Checkpoint()
    assert_equal(len(ckpt.model_state), 0)
    assert_equal(len(ckpt.optimizer_state), 0)
    assert_equal(len(ckpt.scheduler_state), 0)
    assert_equal(len(ckpt.metadata), 0)


def test_save_state_with_optimizer_no_momentum() raises:
    """Save state with optimizer (no momentum) — velocities key absent."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var params = model.parameters()
    var opt = SGD[DType.float32](params^, lr=0.01)
    var tmp_path = "/tmp/test_save_state_opt_nomom.npy"
    var ckpt = save_state(tmp_path, model, opt, {})
    assert_true(ckpt.optimizer_state.__contains__("type"))
    assert_true(ckpt.optimizer_state.__contains__("lr"))
    assert_true(ckpt.optimizer_state.__contains__("velocities") == False)


# ═══════════════════════════════════════════════════════════════════════
# Phase 5 — Best-model & step-based tracking tests
# ═══════════════════════════════════════════════════════════════════════


def test_save_best_if_improved_saves_latest() raises:
    """Always saves _latest.npy regardless of loss."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var prefix = "/tmp/test_best_latest"
    var result = save_best_if_improved(prefix, model, Float64(0.5), Float64(1.0), {})
    np = Python.import_module("numpy")
    var latest = np.load(prefix + "_latest.npy", allow_pickle=True).item()
    assert_true(latest.__contains__("model"))
    assert_true(result == Float64(0.5))


def test_save_best_if_improved_saves_best() raises:
    """Saves _best.npy when current_loss < best_loss."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var prefix = "/tmp/test_best_improved"
    var result = save_best_if_improved(prefix, model, Float64(0.3), Float64(1.0), {})
    np = Python.import_module("numpy")
    var best = np.load(prefix + "_best.npy", allow_pickle=True).item()
    assert_true(best.__contains__("model"))
    assert_true(result == Float64(0.3))


def test_save_best_if_improved_no_improvement() raises:
    """Does NOT save _best.npy when current_loss >= best_loss."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var prefix = "/tmp/test_best_no_impr"
    # First save creates a best
    var _ = save_best_if_improved(prefix, model, Float64(0.3), Float64(1.0), {})
    # Second save with worse loss should NOT overwrite best
    var result = save_best_if_improved(prefix, model, Float64(0.5), Float64(0.3), {})
    np = Python.import_module("numpy")
    var best = np.load(prefix + "_best.npy", allow_pickle=True).item()
    # Best should still have the 0.3 model data (we verify by checking it exists)
    assert_true(best.__contains__("model"))
    assert_true(result == Float64(0.3))


def test_save_best_if_improved_returns_min() raises:
    """Returns min(current_loss, best_loss)."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    # current_loss > best_loss → return best_loss
    var r1 = save_best_if_improved("/tmp/test_best_return1", model, Float64(0.5), Float64(0.3), {})
    assert_true(r1 == Float64(0.3))
    # current_loss < best_loss → return current_loss
    var r2 = save_best_if_improved("/tmp/test_best_return2", model, Float64(0.1), Float64(0.3), {})
    assert_true(r2 == Float64(0.1))
    # equal → return current (also equals best)
    var r3 = save_best_if_improved("/tmp/test_best_return3", model, Float64(0.3), Float64(0.3), {})
    assert_true(r3 == Float64(0.3))


def test_save_best_if_improved_with_optimizer() raises:
    """Full state (model + optimizer) saved in both _latest and _best."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var params = model.parameters()
    var opt = SGD[DType.float32](params^, lr=0.01, momentum=0.9)
    var prefix = "/tmp/test_best_with_opt"
    var _ = save_best_if_improved(prefix, model, opt, Float64(0.3), Float64(1.0), {})
    np = Python.import_module("numpy")
    var latest = np.load(prefix + "_latest.npy", allow_pickle=True).item()
    var best = np.load(prefix + "_best.npy", allow_pickle=True).item()
    assert_true(latest["optimizer"].__contains__("type"))
    assert_true(latest["optimizer"]["type"] == "SGD")
    assert_true(best["optimizer"].__contains__("type"))
    assert_true(best["optimizer"]["type"] == "SGD")


def test_save_step_checkpoint_basic() raises:
    """Save to {prefix}_step_{N}.npy via save_step_checkpoint."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    save_step_checkpoint("/tmp/test_step_basic", 42, model, {})
    np = Python.import_module("numpy")
    var data = np.load("/tmp/test_step_basic_step_42.npy", allow_pickle=True).item()
    assert_true(data.__contains__("model"))
    assert_true(data.__contains__("optimizer"))


def test_save_step_checkpoint_with_optimizer() raises:
    """Include optimizer state in save_step_checkpoint."""
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](4, 3).into())
    var params = model.parameters()
    var opt = SGD[DType.float32](params^, lr=0.01, momentum=0.9)
    save_step_checkpoint("/tmp/test_step_with_opt", 99, model, opt, {})
    np = Python.import_module("numpy")
    var data = np.load("/tmp/test_step_with_opt_step_99.npy", allow_pickle=True).item()
    assert_true(data["optimizer"].__contains__("type"))
    assert_true(data["optimizer"]["type"] == "SGD")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
