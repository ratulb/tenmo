from tenmo.tensor import Tensor
from tenmo.net import Sequential, Linear, ReLU, Conv2D, Flatten
from tenmo.numpy_interop import save_checkpoint, load_checkpoint
from tenmo.named_parameter import NamedParameter
from tenmo.shapes import Shape
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
    model.modules[0].layer[Linear[DType.float32]].bias = original_b
    var tmp_path = "/tmp/test_ckpt_linear.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3).into())
    load_checkpoint(model2, tmp_path)
    var w_loaded = model2.modules[0].layer[Linear[DType.float32]].weight
    var b_loaded = model2.modules[0].layer[Linear[DType.float32]].bias
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
    model.modules[0].layer[Linear[DType.float32]].bias = orig_b0
    model.modules[2].layer[Linear[DType.float32]].weight = orig_w1
    model.modules[2].layer[Linear[DType.float32]].bias = orig_b1
    var tmp_path = "/tmp/test_ckpt_multi.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Linear[DType.float32](4, 3).into(), ReLU[DType.float32]().into(), Linear[DType.float32](3, 2).into())
    load_checkpoint(model2, tmp_path)
    var w0_loaded = model2.modules[0].layer[Linear[DType.float32]].weight
    var b0_loaded = model2.modules[0].layer[Linear[DType.float32]].bias
    var w1_loaded = model2.modules[2].layer[Linear[DType.float32]].weight
    var b1_loaded = model2.modules[2].layer[Linear[DType.float32]].bias
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
    model.modules[0].layer[Conv2D[DType.float32]].bias = orig_b
    var tmp_path = "/tmp/test_ckpt_conv.npy"
    save_checkpoint(model, tmp_path)
    var model2 = Sequential[DType.float32]()
    model2.append(Conv2D[DType.float32](1, 2, 3, bias=True).into(), Flatten[DType.float32]().into())
    load_checkpoint(model2, tmp_path)
    var w_loaded = model2.modules[0].layer[Conv2D[DType.float32]].weight
    var b_loaded = model2.modules[0].layer[Conv2D[DType.float32]].bias
    assert_true(w_loaded.all_close(orig_w))
    assert_true(b_loaded.all_close(orig_b))


def test_checkpoint_pytorch_compatible_format() raises:
    from tenmo.numpy_interop import to_ndarray
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
    var b = model.modules[0].layer[Linear[DType.float32]].bias
    # Note: saved as float64, loaded into float32 — values should match
    assert_true(w.all_close(Tensor[DType.float32].ones(Shape(4, 3))))
    assert_true(b.all_close(Tensor[DType.float32].zeros(Shape(3))))


def test_load_numpy_dict() raises:
    from tenmo.numpy_interop import to_ndarray
    np = Python.import_module("numpy")
    var manual_dict: PythonObject = {}
    var t = Tensor[DType.float64].arange(6)
    t = t.reshape(2, 3)
    manual_dict["my_weight"] = to_ndarray(t)
    var tmp_path = "/tmp/test_manual_dict.npy"
    np.save(tmp_path, manual_dict)
    var loaded = np.load(tmp_path, allow_pickle=True).item()
    assert_true(loaded.__contains__("my_weight"))


def test_named_params_layernorm() raises:
    from tenmo.forwards import LayerNorm
    var ln = LayerNorm[DType.float32](8)
    var params = ln.named_parameters("")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "gamma")
    assert_equal(params[1].name, "beta")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
