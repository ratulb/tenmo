from std.testing import assert_true, TestSuite
from std.utils.numerics import isinf, isnan

from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.blas_ndbuffer import BLASCache
from tenmo.blashandle import BLASHandleLite
from tenmo.shared.mnemonics import mm
from tenmo.net import Linear, LinearBLAS, Sequential, SequentialBLAS


def _blas_available() -> Bool:
    """True iff the OpenBLAS library loaded (flag-independent)."""
    return BLASCache.is_available()


def _blas_enabled() -> Bool:
    """True iff both the -D BLAS opt-in AND the library are active.

    Gate for the Tensor.matmul-vs-BLASHandleLite consistency check: without
    the opt-in, Tensor.matmul is native and both sides would just be two
    different native computes disagreeing on numerics.
    """
    return BLASCache.is_enabled() and BLASCache.is_available()


def test_sequentialblas_appends_lite_for_linearblas() raises:
    """SequentialBLAS wires a BLASHandleLite into every LinearBLAS it appends.

    When the library is absent the module is appended unchanged (native
    LinearBLAS), so the layer never holds an unusable lite.
    """
    comptime dtype = DType.float32
    var model = SequentialBLAS[dtype]()
    model.append(LinearBLAS[dtype](6, 4, init_seed=7).into())

    var m = model.modules[0]
    ref linear = m.layer[LinearBLAS[dtype, mm]]
    if _blas_available():
        assert_true(
            linear.blas_lite != None,
            "SequentialBLAS must attach a lite when BLAS is available",
        )
    else:
        assert_true(
            linear.blas_lite == None,
            "SequentialBLAS must NOT attach a lite when BLAS is missing",
        )


def test_linearblas_matmul_blas_matches_native_f32() raises:
    """BLASHandleLite-backed matmul_blas == native matmul (fwd + grads, f32)."""
    if not _blas_available():
        return
    comptime dtype = DType.float32
    var layer = LinearBLAS[dtype](6, 4, init_seed=7)
    layer.blas_lite = BLASHandleLite[dtype].from_cache()

    var x = Tensor[dtype].randn(Shape(3, 6))

    # Forward equivalence.
    var y_native = layer.matmul(x)
    var y_blas = layer.matmul_blas(x)
    assert_true(
        y_native.all_close[atol=1e-4](y_blas),
        "LinearBLAS: BLAS forward != native forward (f32)",
    )

    # Gradient equivalence for weight and bias.
    var loss_native = layer.matmul(x).sum()
    loss_native.backward()
    var grad_w_native = layer.weight.grad().clone()
    var grad_b_native = layer.bias.value().grad().clone()

    layer.weight.zero_grad()
    layer.bias.value().zero_grad()

    var loss_blas = layer.matmul_blas(x).sum()
    loss_blas.backward()
    assert_true(
        grad_w_native.all_close[atol=1e-4](layer.weight.grad()),
        "LinearBLAS: BLAS dW != native dW (f32)",
    )
    assert_true(
        grad_b_native.all_close[atol=1e-4](layer.bias.value().grad()),
        "LinearBLAS: BLAS db != native db (f32)",
    )


def test_linearblas_matmul_blas_matches_native_f64() raises:
    """BLASHandleLite-backed matmul_blas == native matmul (fwd + grads, f64)."""
    if not _blas_available():
        return
    comptime dtype = DType.float64
    var layer = LinearBLAS[dtype](6, 4, init_seed=7)
    layer.blas_lite = BLASHandleLite[dtype].from_cache()

    var x = Tensor[dtype].randn(Shape(3, 6))

    var y_native = layer.matmul(x)
    var y_blas = layer.matmul_blas(x)
    assert_true(
        y_native.all_close[atol=1e-10](y_blas),
        "LinearBLAS: BLAS forward != native forward (f64)",
    )

    var loss_native = layer.matmul(x).sum()
    loss_native.backward()
    var grad_w_native = layer.weight.grad().clone()
    var grad_b_native = layer.bias.value().grad().clone()

    layer.weight.zero_grad()
    layer.bias.value().zero_grad()

    var loss_blas = layer.matmul_blas(x).sum()
    loss_blas.backward()
    assert_true(
        grad_w_native.all_close[atol=1e-10](layer.weight.grad()),
        "LinearBLAS: BLAS dW != native dW (f64)",
    )
    assert_true(
        grad_b_native.all_close[atol=1e-10](layer.bias.value().grad()),
        "LinearBLAS: BLAS db != native db (f64)",
    )


def test_sequentialblas_forward_matches_sequential() raises:
    """SequentialBLAS == Sequential forward + param plumbing.

    Routing is direct (lite attached: BLAS; otherwise native); whichever
    path is taken must agree with Sequential on identical seeds.
    """
    comptime dtype = DType.float32
    var model = SequentialBLAS[dtype]()
    model.append(LinearBLAS[dtype](6, 4, init_seed=7).into())
    model.append(LinearBLAS[dtype](4, 2, init_seed=11).into())

    var ref_model = Sequential[dtype]()
    ref_model.append(Linear[dtype](6, 4, init_seed=7).into())
    ref_model.append(Linear[dtype](4, 2, init_seed=11).into())

    var x = Tensor[dtype].randn(Shape(3, 6))
    var y_blas = model(x)
    var y_ref = ref_model(x)
    assert_true(
        y_blas.all_close[atol=1e-4](y_ref),
        "SequentialBLAS forward != Sequential forward",
    )

    # Parameter plumbing must expose the same number of params.
    assert_true(
        len(model.parameters()) == len(ref_model.parameters()),
        "SequentialBLAS/Sequential parameter count mismatch",
    )

    # Backward must run cleanly and produce finite grads.
    var loss = model(x).sum()
    loss.backward()
    for param in model.parameters():
        var grad = param[].grad()
        assert_true(
            not isnan(grad.sum().item()) and not isinf(grad.sum().item()),
            "grad not finite",
        )


def test_sequentialblas_matmul_blas_routing() raises:
    """The lite attached by SequentialBLAS actually routes via BLAS at the
    layer level, and agrees with the native path on the same layer."""
    if not _blas_available():
        return
    comptime dtype = DType.float64
    var model = SequentialBLAS[dtype]()
    model.append(LinearBLAS[dtype](6, 4, init_seed=7).into())

    var x = Tensor[dtype].randn(Shape(3, 6))
    var m = model.modules[0]
    ref linear = m.layer[LinearBLAS[dtype, mm]]

    var y_blas = linear.matmul_blas(x)
    var y_native = linear.matmul(x)
    assert_true(
        y_blas.all_close[atol=1e-10](y_native),
        "SequentialBLAS-attached lite: BLAS forward != native forward",
    )


def test_tensor_matmul_vs_blashandle_consistency() raises:
    """Tensor.matmul (BLAS-routed) == BLASHandleLite.matmul on matching inputs.

    Only meaningful when the -D BLAS opt-in routes Tensor.matmul; otherwise
    Tensor.matmul is a different native kernel and the two are not guaranteed
    to agree bit-for-bit.
    """
    if not _blas_enabled():
        return
    comptime dtype = DType.float32
    var A = Tensor[dtype].randn(Shape(5, 7))
    var B = Tensor[dtype].randn(Shape(7, 4))

    var C_tensor = A.matmul(B)
    var lite = BLASHandleLite[dtype].from_cache()
    var C_blas = lite.matmul(A, B)
    assert_true(
        C_tensor.all_close[atol=1e-4](C_blas),
        "routed Tensor.matmul != BLASHandleLite.matmul",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()