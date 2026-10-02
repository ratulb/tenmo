"""Epoch loops for the Mojo LLM path.

`train_epoch` / `eval_epoch` run one full pass of a token-stream
`WindowLoader` through a `GPTModel`: forward, shifted cross-entropy,
(train only) backward + AdamW step, with running mean loss and token
accuracy. This is the mechanical piece filling the Phase-3
gap — the Python side (`python-binding/tenmo.py`) only covers
`Sequential` classifiers, so the LLM path gets its own loops here,
concrete over `GPTModel` + `AdamW` + `CrossEntropyLoss` (the GPT
stack). Generalizing over `LayerTrait` models is blocked on the
trait system: `M.InputDType` never unifies with the loader's concrete
index dtype at the `model(xb)` call site.

Mode contract: `train_epoch` forces `model.train()` / `criterion.train()`
on entry; `eval_epoch` forces `eval()` on entry and restores `train()` on
exit, so an eval pass between training epochs can never silently leave
the model graph-free (eval builds no autograd graph by design).

Index discipline: class targets are `int64` (`CrossEntropyLoss`'s default
`target_dtype`), so the loader must be `WindowLoader[int64]` — exactly
what `SlidingWindowDataset[int64]` builds — and the model must use the
default `index_dtype`.
"""

from .accuracy import Accuracy
from .adamw import AdamW
from .crossentropy import CrossEntropyLoss
from .dataloader import WindowLoader
from .gpt import GPTModel
from .scheduler import WarmupCosineLR
from std.time import perf_counter_ns


def train_epoch[OutT: DType](
    mut model: GPTModel[OutT],
    mut criterion: CrossEntropyLoss[OutT],
    mut optimizer: AdamW[OutT],
    mut loader: WindowLoader[DType.int64],
    max_batches: Int = -1,
    log_every: Int = 0,
) raises -> Tuple[Float64, Float64] where OutT.is_floating_point():
    """One training epoch: (mean batch loss, token accuracy).

    Constant-LR form (optimizer's `lr` untouched). For the
    warmup→cosine plan see `train_epoch_sched`. Consumes `loader` (a
    fresh `into_loader` per epoch). Batches are cloned off the loader's
    persistent gather buffers before use. Raises on NaN loss or an empty
    loader. `max_batches >= 0` caps the pass (smoke tests); `log_every >
    0` prints every N batches.
    """
    # Dummy schedule, never consulted (`use_sched=False`): keeps one loop
    # body for both entry points instead of two drifting copies.
    var dummy = WarmupCosineLR[OutT](
        max_lr=Scalar[OutT](0),
        min_lr=Scalar[OutT](0),
        warmup_steps=0,
        max_steps=0,
    )
    var step = 0
    return _train_epoch_impl(
        model, criterion, optimizer, False, dummy, step, loader,
        max_batches, log_every,
    )


def train_epoch_sched[OutT: DType](
    mut model: GPTModel[OutT],
    mut criterion: CrossEntropyLoss[OutT],
    mut optimizer: AdamW[OutT],
    mut sched: WarmupCosineLR[OutT],
    mut global_step: Int,
    mut loader: WindowLoader[DType.int64],
    max_batches: Int = -1,
    log_every: Int = 0,
) raises -> Tuple[Float64, Float64] where OutT.is_floating_point():
    """One training epoch under a warmup→cosine schedule.

    Identical to `train_epoch`, except each optimizer step is preceded
    by `optimizer.set_lr(sched.step())` and `global_step` advances, so
    the schedule survives across epochs (the driver owns the counter —
    resetting it per epoch would replay warmup). Returns (mean batch
    loss, token accuracy) with the same NaN/empty guards.
    """
    return _train_epoch_impl(
        model, criterion, optimizer, True, sched, global_step, loader,
        max_batches, log_every,
    )


def _train_epoch_impl[OutT: DType](
    mut model: GPTModel[OutT],
    mut criterion: CrossEntropyLoss[OutT],
    mut optimizer: AdamW[OutT],
    use_sched: Bool,
    mut sched: WarmupCosineLR[OutT],
    mut global_step: Int,
    mut loader: WindowLoader[DType.int64],
    max_batches: Int,
    log_every: Int,
) raises -> Tuple[Float64, Float64] where OutT.is_floating_point():
    """Shared epoch body (see the two public entry points)."""
    model.train()
    criterion.train()
    var t_entry = perf_counter_ns()
    var total_loss = 0.0
    var total_correct = 0.0
    var total_tokens = 0
    var n_batches = 0
    while loader.__has_next__():
        if max_batches >= 0 and n_batches >= max_batches:
            break
        ref b = loader.__next__()
        var xb = b.features.clone()
        var yb = b.labels.clone()

        var logits = model(xb)  # (B, T, V)
        var loss = criterion(logits.permute([0, 2, 1]), yb)  # (B, V, T)

        var v = Float64(loss.item())
        if v != v:
            raise Error(
                "train_epoch: loss NaN at batch " + String(n_batches + 1)
            )
        var acc = Accuracy[OutT, DType.int64].token_accuracy(logits, yb)
        total_loss += v
        total_correct += acc * Float64(yb.numels())
        total_tokens += yb.numels()
        n_batches += 1

        optimizer.zero_grad()
        loss.backward()
        if use_sched:
            optimizer.set_lr(sched.step())
            global_step += 1
        optimizer.step()

        if log_every > 0 and n_batches % log_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_entry) / 1e9
            print(
                "  batch", n_batches, "loss:", v, "acc:", acc,
                "elapsed_s:", elapsed,
            )

    if n_batches == 0:
        raise Error("train_epoch: loader yielded no batches")
    return (
        total_loss / Float64(n_batches),
        total_correct / Float64(total_tokens),
    )


def eval_epoch[OutT: DType](
    mut model: GPTModel[OutT],
    mut criterion: CrossEntropyLoss[OutT],
    mut loader: WindowLoader[DType.int64],
    max_batches: Int = -1,
) raises -> Tuple[Float64, Float64] where OutT.is_floating_point():
    """One eval pass (no graph, no optimizer): (mean batch loss, accuracy).

    Forces `eval()` on entry, restores `train()` on exit (see module doc).
    Same batch cloning and empty-loader guard as `train_epoch`.
    """
    model.eval()
    criterion.eval()
    var total_loss = 0.0
    var total_correct = 0.0
    var total_tokens = 0
    var n_batches = 0
    while loader.__has_next__():
        if max_batches >= 0 and n_batches >= max_batches:
            break
        ref b = loader.__next__()
        var xb = b.features.clone()
        var yb = b.labels.clone()

        var logits = model(xb)  # (B, T, V), grad-free under eval()
        var loss = criterion(logits.permute([0, 2, 1]), yb)

        var v = Float64(loss.item())
        if v != v:
            raise Error(
                "eval_epoch: loss NaN at batch " + String(n_batches + 1)
            )
        var acc = Accuracy[OutT, DType.int64].token_accuracy(logits, yb)
        total_loss += v
        total_correct += acc * Float64(yb.numels())
        total_tokens += yb.numels()
        n_batches += 1

    model.train()
    criterion.train()
    if n_batches == 0:
        raise Error("eval_epoch: loader yielded no batches")
    return (
        total_loss / Float64(n_batches),
        total_correct / Float64(total_tokens),
    )
