from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape


def test_cast_forward_values() raises:
    comptime f32 = DType.float32

    var x = Tensor[f32].full(Shape(4), 1.5)
    var y = x.to_dtype[DType.float64]()
    var expected = Tensor[DType.float64].full(Shape(4), 1.5)
    assert_true(
        y.all_close(expected),
        "Cast: f32→f64 forward values preserved",
    )


def test_cast_grad_flow_simple() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var x = Tensor[f32].full(Shape(4), 2.0, requires_grad=True)
    var y = x.to_dtype[f64]()
    var out = y.sum()
    out.backward()
    assert_true(
        x.grad().all_close(Tensor[f32].full(Shape(4), 1.0)),
        "Cast: gradient flows through single cast back to leaf",
    )


def test_cast_nested_f32_f64_f32() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    # Two chained casts in opposite directions — both cross-dtype edge
    # orientations (f32 parent under f64 output, f64 parent under f32 output).
    var a = Tensor[f32].full(Shape(3), 2.0, requires_grad=True)
    var b = a.to_dtype[f64]()
    var c = b.to_dtype[f32]()
    var d = c * 3.0
    var out = d.sum()
    out.backward()
    assert_true(
        a.grad().all_close(Tensor[f32].full(Shape(3), 3.0)),
        "Cast: nested f32→f64→f32 preserves gradient end-to-end",
    )


def test_cast_multi_consumer() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    # One cast node feeding two downstream ops — accumulation must cross the
    # boundary and sum in the leaf dtype.
    var x = Tensor[f32].full(Shape(4), 1.0, requires_grad=True)
    var y = x.to_dtype[f64]()
    var l1 = (y * 2.0).sum()
    var l2 = (y * 5.0).sum()
    var total = l1 + l2
    total.backward()
    assert_true(
        x.grad().all_close(Tensor[f32].full(Shape(4), 7.0)),
        "Cast: multi-consumer gradients accumulate across the boundary",
    )


def test_cast_clears_intermediate_across_boundary() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var x = Tensor[f32].full(Shape(4), 2.0, requires_grad=True)
    var y = x.to_dtype[f64]()
    var out = y.sum()

    out.backward()
    assert_true(
        y.grad().all_close(Tensor[f64].zeros(Shape(4))),
        "Cast: intermediate grad on cast node clears once consumed",
    )
    assert_true(
        x.grad().all_close(Tensor[f32].full(Shape(4), 1.0)),
        "Cast: first backward reaches leaf across the boundary",
    )

    # Second pass over the graph must still run (backward never frees
    # ancestry). Gradients ACCUMULATE at the leaf; the cleared intermediate
    # recomputes fresh each pass.
    out.backward()
    assert_true(
        x.grad().all_close(Tensor[f32].full(Shape(4), 2.0)),
        "Cast: graph re-runs; accumulation crosses boundary",
    )
    assert_true(
        y.grad().all_close(Tensor[f64].zeros(Shape(4))),
        "Cast: intermediate clears on every pass",
    )


def test_cast_no_grad_is_leaf() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var x = Tensor[f32].full(Shape(2), 1.0, requires_grad=True)
    var y = x.to_dtype[f64](requires_grad=False)
    assert_true(not y.requires_grad, "Cast: requires_grad=False override honored")
    assert_true(not y.has_ancestry(), "Cast: no-grad cast produces a leaf")

    var z = Tensor[f32].full(Shape(2), 1.0)
    var w = z.to_dtype[f64]()
    assert_true(not w.has_ancestry(), "Cast: untracked source stays a leaf")


def test_cast_int_target_stays_leaf() raises:
    comptime f32 = DType.float32

    # Float-only backward guard: casting to an int dtype never registers.
    var x = Tensor[f32].full(Shape(2), 3.0, requires_grad=True)
    var yi = x.to_dtype[DType.int32]()
    assert_true(
        not yi.requires_grad,
        "Cast: int target does not register backward",
    )
    assert_true(not yi.has_ancestry(), "Cast: int target is a leaf")


def test_cast_same_dtype_is_leaf() raises:
    comptime f32 = DType.float32

    var x = Tensor[f32].full(Shape(2), 1.0, requires_grad=True)
    var y = x.to_dtype[f32]()
    assert_true(
        not y.has_ancestry(),
        "Cast: same-dtype to_dtype remains a plain copy leaf",
    )


def test_cast_graph_gpu() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var x = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var xg = x.to_gpu()
        var y = xg.to_dtype[DType.float64]()
        var out = y.sum()
        out.backward()
        assert_true(
            x.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Cast: gradient flows GPU cast → transfer → CPU leaf",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
