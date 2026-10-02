"""Tests for the static variadic mixed-dtype chain (static_seq.mojo).

The gates mirror the
MixedSequential suite but everything here is compile-time resolved:
seam casts instantiate statically via the `_run_from` recursion, the
head dtype annotation is a comptime assert, and the output dtype is the
computed `Seq.OutDT` member.
"""

from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.net import Linear, ReLU, Dropout
from tenmo.optim import SGD
from tenmo.static_seq import Seq


def make_static_pair[
    seed: Int
]() -> Tuple[
    Seq[
        Linear[DType.float32],
        ReLU[DType.float32],
        Linear[DType.float64],
        ReLU[DType.float64],
    ],
    Linear[DType.float32],
    Linear[DType.float64],
]:
    """Model + identically seeded manual layers for oracle comparison."""
    comptime Model = Seq[
        Linear[DType.float32],
        ReLU[DType.float32],
        Linear[DType.float64],
        ReLU[DType.float64],
    ]
    var model = Model(
        Tuple(
            Linear[DType.float32](4, 6, init_seed=seed),
            ReLU[DType.float32](),
            Linear[DType.float64](6, 3, init_seed=seed),
            ReLU[DType.float64](),
        )
    )
    return (
        model^,
        Linear[DType.float32](4, 6, init_seed=seed),
        Linear[DType.float64](6, 3, init_seed=seed),
    )


def test_static_forward_values_match_manual() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var paired = make_static_pair[42]()
    var model = paired[0].copy()
    var l1 = paired[1]
    var l2 = paired[2]

    var x = Tensor[f32].full(Shape(2, 4), 0.5)
    var y = model[DType.float32, DType.float64](x)

    var h = l1(x).relu[track_grad=True]()
    var expected = l2(h.to_dtype[f64]()).relu[track_grad=True]()

    assert_true(
        y.all_close(expected),
        "StaticSeq: forward equals manual segmentation across seam cast",
    )
    assert_true(y.dtype == f64, "StaticSeq: output dtype is chain tail")


def test_static_gradients_match_manual() raises:
    comptime f32 = DType.float32

    var paired = make_static_pair[43]()
    var model = paired[0].copy()
    var l1 = paired[1]
    var l2 = paired[2]

    var x = Tensor[f32].full(Shape(2, 4), 1.25, requires_grad=True)
    var loss = model[DType.float32, DType.float64](x).sum()
    loss.backward()

    var xm = Tensor[f32].full(Shape(2, 4), 1.25, requires_grad=True)
    var h = l1(xm).relu[track_grad=True]()
    var ym = l2(h.to_dtype[DType.float64]()).relu[track_grad=True]()
    var lossm = ym.sum()
    lossm.backward()

    assert_true(
        x.grad().all_close(xm.grad()),
        "StaticSeq: leaf gradient equals manual segmentation",
    )

    var p32 = model.parameters_of[f32]()
    var p64 = model.parameters_of[DType.float64]()
    assert_true(len(p32) == 2, "StaticSeq: two f32 parameter tensors")
    assert_true(len(p64) == 2, "StaticSeq: two f64 parameter tensors")

    var m32 = l1.parameters()
    for i in range(len(m32)):
        assert_true(
            p32[i][].grad().all_close(m32[i][].grad()),
            "StaticSeq: f32 weight gradient equals manual",
        )
    var m64 = l2.parameters()
    for i in range(len(m64)):
        assert_true(
            p64[i][].grad().all_close(m64[i][].grad()),
            "StaticSeq: f64 weight gradient equals manual",
        )


def test_static_double_backward_accumulates() raises:
    comptime f32 = DType.float32

    var paired = make_static_pair[44]()
    var model = paired[0].copy()
    var x = Tensor[f32].full(Shape(2, 4), 1.0, requires_grad=True)

    var loss = model[DType.float32, DType.float64](x).sum()
    loss.backward()
    var first = x.grad()

    loss.backward()
    assert_true(
        x.grad().all_close(first + first),
        "StaticSeq: double backward across static seam accumulates at the leaf",
    )


def test_static_fanout_across_seam() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    var paired = make_static_pair[45]()
    var model = paired[0].copy()
    var l1 = paired[1]
    var l2 = paired[2]

    var x = Tensor[f32].full(Shape(2, 4), 0.75, requires_grad=True)
    var y = model[DType.float32, DType.float64](x)

    var w1 = Tensor[f64].full(Shape(2, 3), 2.0)
    var w2 = Tensor[f64].full(Shape(2, 3), 5.0)
    var total = (y * w1).sum() + (y * w2).sum()
    total.backward()

    var xm = Tensor[f32].full(Shape(2, 4), 0.75, requires_grad=True)
    var h = l1(xm).relu[track_grad=True]()
    var ym = l2(h.to_dtype[f64]()).relu[track_grad=True]()
    var manual_total = ((ym * w1).sum() + (ym * w2).sum())
    manual_total.backward()

    assert_true(
        x.grad().all_close(xm.grad()),
        "StaticSeq: fan-out across a static seam accumulates into the leaf",
    )


def test_static_param_counts_and_zero_grad() raises:
    comptime f32 = DType.float32

    var paired = make_static_pair[46]()
    var model = paired[0].copy()

    assert_true(model.__len__() == 4, "StaticSeq: four layers")
    # Linear(4,6)+bias=30 plus Linear(6,3)+bias=21.
    assert_true(model.num_parameters() == 51, "StaticSeq: total parameters")

    var x = Tensor[f32].full(Shape(2, 4), 1.0, requires_grad=True)
    var loss = model[DType.float32, DType.float64](x).sum()
    loss.backward()

    var p32 = model.parameters_of[f32]()
    var zeros = Tensor[f32].zeros(Shape(4, 6))
    assert_true(
        not p32[0][].grad().all_close(zeros), "StaticSeq: grad populated"
    )
    model.zero_grad()
    assert_true(
        p32[0][].grad().all_close(zeros),
        "StaticSeq: zero_grad reaches live weights",
    )


def test_static_train_eval_dropout() raises:
    comptime f32 = DType.float32
    comptime f64 = DType.float64

    comptime Model = Seq[
        Dropout[DType.float32],
        Linear[DType.float64],
    ]
    var model = Model(
        Tuple(
            Dropout[f32](Scalar[f32](0.5)),
            Linear[f64](64, 4, init_seed=47),
        )
    )
    var x = Tensor[f32].full(Shape(8, 64), 1.0)

    model.eval()
    var y_eval = model[DType.float32, DType.float64](x)

    var l2 = Linear[f64](64, 4, init_seed=47)
    var expected = l2(x.to_dtype[f64]())
    assert_true(
        y_eval.all_close(expected),
        "StaticSeq: eval-mode dropout is identity through static chain",
    )

    model.train()
    var y_train = model[DType.float32, DType.float64](x)
    var delta = (y_train - y_eval).abs().sum().item()
    assert_true(
        delta > 0.0,
        "StaticSeq: train-mode dropout randomizes forward",
    )


def test_static_convergence_xor_f16_head() raises:
    comptime f32 = DType.float32
    comptime f16 = DType.float16

    comptime Model = Seq[
        Linear[f32],
        ReLU[f32],
        Linear[f32],
        ReLU[f32],
        Linear[f16],
    ]
    var model = Model(
        Tuple(
            Linear[f32](2, 8, init_method="he", init_seed=7),
            ReLU[f32](),
            Linear[f32](8, 8, init_method="he", init_seed=8),
            ReLU[f32](),
            Linear[f16](8, 1, init_method="uniform", init_seed=9),
        )
    )

    var xs = Tensor[f32].d2([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    var ys = Tensor[f16].d2([[0.0], [1.0], [1.0], [0.0]])

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

    var out0 = model[DType.float32, DType.float16](xs)
    var diff0 = out0 - ys
    var first_loss = (diff0 * diff0).sum()

    var last_loss = first_loss
    for _ in range(200):
        var out = model[DType.float32, DType.float16](xs)
        var diff = out - ys
        last_loss = (diff * diff).sum()
        model.zero_grad()
        last_loss.backward()
        sgd32.step()
        sgd16.step()

    print("XOR static f16-head initial loss:", first_loss.item())
    print("XOR static f16-head final loss:", last_loss.item())
    assert_true(
        last_loss.item() < first_loss.item() * 0.5,
        "StaticSeq: f16-head XOR training reduces loss (smoke)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
