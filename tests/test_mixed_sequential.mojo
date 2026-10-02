"""Tests for the runtime-erased mixed-dtype Sequential (net.mojo).

Covers forward/gradient equivalence
against manual segmentation (the correctness gate), retained double
backward accumulation across a boundary, multi-consumer fan-out,
per-dtype parameter collection, mode switching, record deep-copy
independence, GPU transfer (guarded), an f16 smoke hop, a tiny
convergence run with an f16 head segment, and end-to-end probes that
run a minimal child process to prove the runtime annotation guards
abort with precise diagnostics.

Primary equivalence chains use f32→f64 boundaries (proven cast
territory); f16 appears only in dedicated smoke/convergence tests
because f16 CPU kernels are barely exercised elsewhere in the suite.
"""

from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from std.python import Python, PythonObject
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.net import Linear, ReLU, Dropout
from tenmo.optim import SGD
from tenmo.net import MixedSequential
from std.sys.defines import get_defined_string

def make_mixed_model[
    seed: Int
]() -> MixedSequential:
    """F32 → [cast] → f64 reference chain: Linear-ReLU | Linear-ReLU."""
    comptime f32 = DType.float32
    comptime f64 = DType.float64
    var model = MixedSequential()
    model.append(Linear[f32](4, 6, init_seed=seed))
    model.append(ReLU[f32]())
    model.append(Linear[f64](6, 3, init_seed=seed))
    model.append(ReLU[f64]())
    return model^


def manual_reference_chain[
    seed: Int
](x: Tensor[DType.float32]) -> Tensor[DType.float64]:
    """Option-C oracle: explicit cast between identically seeded layers."""
    comptime f64 = DType.float64
    var l1 = Linear[DType.float32](4, 6, init_seed=seed)
    var l2 = Linear[f64](6, 3, init_seed=seed)
    var h = l1(x).relu[track_grad=True]()
    return l2(h.to_dtype[f64]()).relu[track_grad=True]()


def test_mixed_forward_values_match_manual() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var model = make_mixed_model[42]()
    var x = Tensor[f32].full(Shape(2, 4), 0.5)

    var y = model.forward[f32, f64](x)
    var expected = manual_reference_chain[42](x)

    assert_true(
        y.all_close(expected),
        "MixedSeq: forward values equal manual segmentation",
    )


def test_mixed_gradients_match_manual() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var model = make_mixed_model[43]()
    var x = Tensor[f32].full(Shape(2, 4), 1.25, requires_grad=True)
    var y = model.forward[f32, f64](x)
    var loss = y.sum()
    loss.backward()

    var xm = Tensor[f32].full(Shape(2, 4), 1.25, requires_grad=True)
    var l1 = Linear[f32](4, 6, init_seed=43)
    var l2 = Linear[f64](6, 3, init_seed=43)
    var h = l1(xm).relu[track_grad=True]()
    var ym = l2(h.to_dtype[f64]()).relu[track_grad=True]()
    var lossm = ym.sum()
    lossm.backward()

    assert_true(
        x.grad().all_close(xm.grad()),
        "MixedSeq: leaf gradient equals manual segmentation",
    )

    var model_params = model.parameters_of[f32]()
    var manual_params = l1.parameters()
    assert_true(
        len(model_params) == len(manual_params),
        "MixedSeq: f32 segment parameter count matches",
    )
    for i in range(len(manual_params)):
        assert_true(
            model_params[i][].grad().all_close(manual_params[i][].grad()),
            "MixedSeq: f32 weight gradient equals manual segmentation",
        )

    var model_params64 = model.parameters_of[f64]()
    var manual_params64 = l2.parameters()
    assert_true(
        len(model_params64) == len(manual_params64),
        "MixedSeq: f64 segment parameter count matches",
    )
    for i in range(len(manual_params64)):
        assert_true(
            model_params64[i][].grad().all_close(
                manual_params64[i][].grad()
            ),
            "MixedSeq: f64 weight gradient equals manual segmentation",
        )


def test_boundary_double_backward_accumulates() raises:
    comptime f32 = DType.float32

    var model = make_mixed_model[44]()
    var x = Tensor[f32].full(Shape(2, 4), 1.0, requires_grad=True)

    var y = model.forward[f32, DType.float64](x)
    var loss = y.sum()
    loss.backward()
    var first = x.grad()

    loss.backward()
    assert_true(
        x.grad().all_close(first + first),
        "MixedSeq: double backward across boundary accumulates at the leaf",
    )


def test_multi_consumer_fanout_across_boundary() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var model = make_mixed_model[45]()
    var x = Tensor[f32].full(Shape(2, 4), 0.75, requires_grad=True)
    var y = model.forward[f32, f64](x)

    var w1 = Tensor[f64].full(Shape(2, 3), 2.0)
    var w2 = Tensor[f64].full(Shape(2, 3), 5.0)
    var total = (y * w1).sum() + (y * w2).sum()
    total.backward()

    # Same fan-out on the manual chain.
    var xm = Tensor[f32].full(Shape(2, 4), 0.75, requires_grad=True)
    var l1 = Linear[f32](4, 6, init_seed=45)
    var l2 = Linear[f64](6, 3, init_seed=45)
    var h = l1(xm).relu[track_grad=True]()
    var ym = l2(h.to_dtype[f64]()).relu[track_grad=True]()
    var manual_total = ((ym * w1).sum() + (ym * w2).sum())
    manual_total.backward()

    assert_true(
        x.grad().all_close(xm.grad()),
        "MixedSeq: fan-out across a boundary accumulates into the leaf",
    )


def test_parameters_of_segments_and_identity() raises:
    comptime f32 = DType.float32

    var model = make_mixed_model[46]()

    var p32 = model.parameters_of[f32]()
    var p64 = model.parameters_of[DType.float64]()
    # Each Linear contributes weight+bias pointers (2); activations none.
    assert_true(len(p32) == 2, "MixedSeq: two f32 parameter tensors")
    assert_true(len(p64) == 2, "MixedSeq: two f64 parameter tensors")
    # Linear(4,6)+bias=30 plus Linear(6,3)+bias=21.
    assert_true(
        model.num_parameters() == 51,
        "MixedSeq: total parameter count over both segments",
    )

    # Pointer identity: grads collected through parameters_of live on the
    # real weights — container zero_grad must clear them.
    var x = Tensor[f32].full(Shape(2, 4), 1.0, requires_grad=True)
    var loss = model.forward[f32, DType.float64](x).sum()
    loss.backward()
    var zeros = Tensor[f32].zeros(Shape(4, 6))
    var pre = p32[0][].grad()
    assert_true(
        not pre.all_close(zeros), "MixedSeq: grad populated"
    )
    model.zero_grad()
    assert_true(
        p32[0][].grad().all_close(zeros),
        "MixedSeq: zero_grad reaches live weights through erased records",
    )


def test_train_eval_flips_dropout_through_container() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var model = MixedSequential()
    model.append(Dropout[f32](Scalar[f32](0.5)))
    model.append(Linear[f64](64, 4, init_seed=47))

    var x = Tensor[f32].full(Shape(8, 64), 1.0)

    model.eval()
    var y_eval = model.forward[f32, f64](x)

    # Eval dropout is identity → deterministic reference.
    var l2 = Linear[f64](64, 4, init_seed=47)
    var expected = l2(x.to_dtype[f64]())
    assert_true(
        y_eval.all_close(expected),
        "MixedSeq: eval-mode dropout is identity through container",
    )

    model.train()
    var y_train = model.forward[f32, f64](x)
    var delta = (y_train - y_eval).abs().sum().item()
    assert_true(
        delta > 0.0,
        "MixedSeq: train-mode dropout randomizes forward through container",
    )


def test_midchain_float_boundary_is_installed() raises:
    """The f32->f64 seam must be a real grad-tracked cast, not a reinterp.

    Added with the Linear[InT,OutT] -> Linear[dtype,mode]
    collapse. That rewrite changed every `Linear[f32, f64]` in this file
    to `Linear[f64]`, which READS like a reordering of the chain — so pin
    the mechanism directly: record 2 arrives with src=f32, demands in=f64,
    and therefore carries a boundary forward fn. If the collapse ever
    collapsed the seam away, values/gradients would still be checked by
    the tests above but this would say why they were right.
    """
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var model = make_mixed_model[53]()

    assert_true(
        model.records[0].src_dtype == f32
        and model.records[1].src_dtype == f32,
        "MixedSeq: f32 prefix needs no cast",
    )
    assert_true(
        model.records[0].cast_forward_fn == None
        and model.records[1].cast_forward_fn == None,
        "MixedSeq: f32 prefix carries no boundary cast",
    )

    # The seam: tail is f32, this layer is f64.
    assert_true(
        model.records[2].src_dtype == f32,
        "MixedSeq: seam record receives an f32 tensor",
    )
    assert_true(
        model.records[2].input_dtype == f64
        and model.records[2].io_dtype == f64,
        "MixedSeq: seam layer is purely f64 in and out",
    )
    assert_true(
        model.records[2].cast_forward_fn != None,
        "MixedSeq: a grad-tracked cast IS installed at the f32->f64 seam",
    )

    # After the seam there is nothing left to bridge.
    assert_true(
        model.records[3].cast_forward_fn == None
        and model.records[3].src_dtype == f64,
        "MixedSeq: f64 suffix needs no cast",
    )


def test_chain_dtype_bookkeeping_and_deep_copy() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var model = make_mixed_model[48]()
    assert_true(model.__len__() == 4, "MixedSeq: four module records")
    assert_true(
        model.head_dtype.value() == f32, "MixedSeq: head dtype recorded"
    )
    assert_true(
        model.tail_dtype.value() == f64, "MixedSeq: tail dtype recorded"
    )

    # Deep-copy independence: the returned copy survives the original's
    # destruction and still forwards correctly (shallow records would
    # double-free / read freed layer blobs here).
    var survivor = make_then_drop_original[49]()
    var x = Tensor[f32].full(Shape(2, 4), 0.5)
    var expected = manual_reference_chain[49](x)
    assert_true(
        survivor.forward[f32, f64](x).all_close(expected),
        "MixedSeq: copied container owns independent layer blobs",
    )


def make_then_drop_original[
    seed: Int
]() -> MixedSequential:
    var m = make_mixed_model[seed]()
    var c = m.copy()
    _ = m
    return c^


def test_f16_boundary_smoke() raises:
    comptime f32 = DType.float32
    comptime f16 = DType.float16

    var model = MixedSequential()
    model.append(Linear[f32](4, 6, init_seed=49))
    model.append(ReLU[f32]())
    model.append(Linear[f16](6, 3, init_seed=49))

    var x = Tensor[f32].full(Shape(2, 4), 0.5)
    var y = model.forward[f32, f16](x)

    var l1 = Linear[f32](4, 6, init_seed=49)
    var l2 = Linear[f16](6, 3, init_seed=49)
    var h = l1(x).relu[track_grad=True]()
    var expected = l2(h.to_dtype[f16]())

    assert_true(
        y.all_close(expected),
        "MixedSeq: f16 boundary hop matches manual segmentation",
    )


def test_convergence_xor_f16_head() raises:
    comptime f32 = DType.float32
    comptime f16 = DType.float16

    var xs = Tensor[f32].d2([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    var ys = Tensor[f16].d2([[0.0], [1.0], [1.0], [0.0]])

    var model = MixedSequential()
    model.append(Linear[f32](2, 8, init_method="he", init_seed=7))
    model.append(ReLU[f32]())
    model.append(Linear[f32](8, 8, init_method="he", init_seed=8))
    model.append(ReLU[f32]())
    model.append(Linear[f16](8, 1, init_method="uniform", init_seed=9))

    var sgd32 = SGD[f32](
        model.parameters_of[f32](),
        lr=Scalar[f32](0.1),
        momentum=Scalar[f32](0.9),
    )
    var sgd16 = SGD[f16](
        model.parameters_of[f16](),
        lr=Scalar[f16](0.1),
        momentum=Scalar[f16](0.9),
    )

    var out0 = model.forward[f32, f16](xs)
    var diff0 = out0 - ys
    var first_loss = (diff0 * diff0).sum()

    var last_loss = first_loss
    for _ in range(200):
        var out = model.forward[f32, f16](xs)
        var diff = out - ys
        last_loss = (diff * diff).sum()
        model.zero_grad()
        last_loss.backward()
        sgd32.step()
        sgd16.step()

    print("XOR f16-head initial loss:", first_loss.item())
    print("XOR f16-head final loss:", last_loss.item())
    assert_true(
        last_loss.item() < first_loss.item() * 0.5,
        "MixedSeq: f16-head XOR training reduces loss (smoke)",
    )


def test_gpu_round_trip() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64
    comptime if has_accelerator():
        var model = make_mixed_model[50]()
        var x = Tensor[f32].full(Shape(2, 4), 0.5)
        var cpu_out = model.forward[f32, f64](x)

        model.to_gpu()
        var xg = x.to_gpu()
        var gpu_out = model.forward[f32, f64](xg)

        assert_true(
            gpu_out.to_cpu().all_close(cpu_out),
            "MixedSeq: GPU round trip preserves mixed-chain forward",
        )


# ============================================================================
# Runtime annotation-guard probes.
#
# forward[In, Out] checks its head/tail annotations at RUNTIME and reports
# mismatches via panic → abort() — fatal, and uncatchable in-process
# (assert_raises sees raised Errors, not aborts). To prove the guards fire
# with clear messages, the test below spawns the minimal probe harness
# tests/test_mixed_seq_annotation_probes.mojo: each child invocation
# performs exactly one mismatched call and dies by the guard under test;
# we assert non-zero exit plus the exact diagnostic text. If a guard ever
# stops firing, the child reaches its own trailing panic instead and the
# message assertion fails. The harness is a separate MINIMAL file because
# the child JIT runs alongside this resident process — re-executing THIS
# file OOM-killed a 15 GB box (~10 GB RSS peak). Children are warm-cache
# recompiles (seconds); the mojo cache this process just built is shared.
# ============================================================================


def _spawn_annotation_probe(name: String) raises -> PythonObject:
    """Run guard probe `name` from the minimal probe harness in a child."""
    var script = (
        "__import__('subprocess').run("
        + "['pixi', 'run', 'mojo', '-I', '.', "
        + "'tests/test_mixed_seq_annotation_probes.mojo', "
        + "'--probe-" + name + "'], "
        + "capture_output=True, text=True, timeout=1200)"
    )
    return Python.evaluate(script)


def test_annotation_guards_abort_with_clear_messages() raises:
    # NOTE: children execute only under -D subprocess=1 (else vacuous
    # pass) — e.g. `pixi run mojo -I . -D subprocess=1 tests/test_mixed_sequential.mojo`.
    comptime subprocess = get_defined_string["subprocess", ""]()
    comptime if not subprocess == "":
        var head = _spawn_annotation_probe("head")
        var head_out = String(head.stdout) + String(head.stderr)
        assert_true(
            String(head.returncode) != "0",
            "MixedSeq: head-mismatch probe exits non-zero",
        )
        assert_true(
            head_out.find(
                "input dtype annotation float64 does not match built chain head float32"
            ) >= 0,
            "MixedSeq: head mismatch reports a precise diagnostic",
        )

        var tail = _spawn_annotation_probe("tail")
        var tail_out = String(tail.stdout) + String(tail.stderr)
        assert_true(
            String(tail.returncode) != "0",
            "MixedSeq: tail-mismatch probe exits non-zero",
        )
        assert_true(
            tail_out.find(
                "output dtype annotation float32 does not match built chain tail float64"
            ) >= 0,
            "MixedSeq: tail mismatch reports a precise diagnostic",
        )

        var empty = _spawn_annotation_probe("empty")
        var empty_out = String(empty.stdout) + String(empty.stderr)
        assert_true(
            String(empty.returncode) != "0",
            "MixedSeq: empty-model probe exits non-zero",
        )
        assert_true(
            empty_out.find("MixedSequential.forward: empty model") >= 0,
            "MixedSeq: empty-model forward reports a precise diagnostic",
        )
    else:
        pass


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
