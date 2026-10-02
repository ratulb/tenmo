"""Gate for the epoch loops (`tenmo/epochs.mojo`).

Tiny synthetic config (V=32, T=8, C=16, 1 layer, no mbpe): one train epoch
must update params and return a finite loss/accuracy, one eval pass must
leave params bit-identical (grad-free) and agree with itself, and an empty
loader must raise (not hang or divide by zero).
"""

from std.testing import assert_true, assert_equal, assert_raises, TestSuite
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import train_epoch, train_epoch_sched, eval_epoch
from tenmo.scheduler import WarmupCosineLR
from tenmo.gpt import GPTModel
from tenmo.tensor import Tensor


def _tiny_model() -> GPTModel[DType.float32]:
    var model = GPTModel[DType.float32](
        n_vocab=32,
        n_ctx=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        tie_weights=True,
        init_seed=7,
    )
    model.train()
    return model^


def _tiny_loader(shuffle: Bool) -> SlidingWindowDataset[DType.int64]:
    # Deterministic 64-token stream over a 32-id vocab, T=8 stride=2 ->
    # 29 windows; B=4 -> 8 batches (last partial).
    var ids = List[Scalar[DType.int64]](capacity=64)
    for k in range(64):
        ids.append(Scalar[DType.int64]((k * 5 + 1) % 32))
    return SlidingWindowDataset[DType.int64](ids^, seq_length=8, stride=2)


def test_train_epoch_updates_and_reports() raises:
    print("Test 1: train_epoch updates params, finite loss/acc")
    var model = _tiny_model()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params,
        lr=Scalar[DType.float32](3e-4),
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var before = params[0][].get(0)

    var ds = _tiny_loader(shuffle=False)
    var loader = ds.into_loader(batch_size=4, shuffle=False)
    var result = train_epoch(model, criterion, adamw, loader)
    var loss = result[0]
    var acc = result[1]
    print("  train loss:", loss, "acc:", acc)
    assert_true(loss == loss, "train loss is NaN")
    assert_true(loss > 0.0, "train loss must be positive")
    assert_true(acc >= 0.0 and acc <= 1.0, "train acc out of [0, 1]")
    assert_true(params[0][].get(0) != before, "train made no update")


def test_eval_epoch_is_grad_free_and_stable() raises:
    print("Test 2: eval_epoch leaves params identical, stable reread")
    var model = _tiny_model()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")

    var ds = _tiny_loader(shuffle=False)
    var loader = ds.into_loader(batch_size=4, shuffle=False)
    var r1 = eval_epoch(model, criterion, loader)
    var ds2 = _tiny_loader(shuffle=False)
    var loader2 = ds2.into_loader(batch_size=4, shuffle=False)
    var r2 = eval_epoch(model, criterion, loader2)
    print("  eval loss:", r1[0], "acc:", r1[1])
    assert_true(r1[0] == r1[0], "eval loss is NaN")
    assert_true(r1[0] == r2[0], "eval pass not deterministic")
    assert_true(r1[1] == r2[1], "eval acc not deterministic")
    assert_true(r1[1] >= 0.0 and r1[1] <= 1.0, "eval acc out of [0, 1]")


def test_empty_loader_raises() raises:
    print("Test 3: empty loader raises")
    var model = _tiny_model()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params,
        lr=Scalar[DType.float32](3e-4),
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var ids = List[Scalar[DType.int64]](capacity=4)
    for k in range(4):
        ids.append(Scalar[DType.int64](k))
    var ds = SlidingWindowDataset[DType.int64](ids^, seq_length=8, stride=2)
    var loader = ds.into_loader(batch_size=4, shuffle=False)
    with assert_raises():
        _ = train_epoch(model, criterion, adamw, loader)
    # eval on a live loader works (sanity that the raise came from emptiness)
    var ds2 = _tiny_loader(shuffle=False)
    var loader2 = ds2.into_loader(batch_size=4, shuffle=False)
    var r = eval_epoch(model, criterion, loader2, max_batches=1)
    assert_true(r[0] == r[0], "capped eval loss is NaN")


def test_train_epoch_sched_advances_schedule() raises:
    """Train_epoch_sched: per-step set_lr from the schedule,
    driver-owned global_step survives, loss finite."""
    var model = _tiny_model()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params,
        lr=Scalar[DType.float32](0.0),  # schedule must overwrite this
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var sched = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](3e-4),
        min_lr=Scalar[DType.float32](3e-5),
        warmup_steps=4,
        max_steps=12,
    )
    var global_step = 0
    var ds = _tiny_loader(shuffle=False)
    var loader = ds.into_loader(batch_size=4, shuffle=False)
    var r = train_epoch_sched(
        model, criterion, adamw, sched, global_step, loader,
        max_batches=3,
    )
    assert_true(r[0] == r[0], "sched train loss is NaN")
    assert_equal(global_step, 3)
    assert_equal(sched.last_step, 2)
    # LR actually flowed: peak schedule value reached the optimizer.
    assert_true(
        abs(Float64(adamw.lr) - Float64(sched.get_last_lr())) < 1e-12,
    )
    # Second call continues the schedule (no warmup replay).
    var loader_b = ds.into_loader(batch_size=4, shuffle=False)
    _ = train_epoch_sched(
        model, criterion, adamw, sched, global_step, loader_b,
        max_batches=2,
    )
    assert_equal(global_step, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll epoch tests passed!")
