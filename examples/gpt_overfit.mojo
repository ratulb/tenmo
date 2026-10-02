"""
GPT overfit smoke.
======================================
Trains a tiny GPT on ONE fixed batch until it memorizes it (loss -> ~0).

This is the LLM path's first end-to-end exercise: vendored tiny-shakespeare
excerpt -> mbpe BPETokenizer -> SlidingWindowDataset -> WindowLoader ->
GPTModel -> CrossEntropy -> AdamW. Memorizing one batch proves capacity and
plumbing (every parameter receives grads, the shift-by-one targets line up);
a stuck loss here can only blame the wiring, never data scale or schedules.

Run from the repo root: `./example.sh gpt_overfit`.
"""

from bpe.tokenizer import BPETokenizer
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.gpt import GPTModel
from tenmo.tensor import Tensor
from std.pathlib import Path


def main() raises:
    # ---- Corpus: vendored tiny-shakespeare excerpt (~12 KB) ----
    var text = Path("examples/data/tinyshakespeare_excerpt.txt").read_text()
    print("corpus bytes:", text.byte_length())

    # ---- Tokenize: mbpe trained on the excerpt itself ----
    var tokenizer = BPETokenizer()
    var corpus = List[String]()
    corpus.append(text)
    tokenizer.train(corpus, 512)
    var ids = tokenizer.encode(text)
    print("tokens:", len(ids))
    var id_min = ids[0]
    var id_max = ids[0]
    for i in range(len(ids)):
        if ids[i] < id_min:
            id_min = ids[i]
        if ids[i] > id_max:
            id_max = ids[i]
    print("id range:", id_min, "..", id_max)
    if id_min < 0 or id_max >= 512:
        raise Error("gpt_overfit: token ID out of vocab range")

    # ---- Window: flat ID stream -> shift-by-one windows ----
    var scalars = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        scalars.append(Scalar[DType.int64](ids[i]))
    var ds = SlidingWindowDataset[DType.int64](scalars^, seq_length=32, stride=8)
    print("windows:", len(ds))

    # ---- The one batch: first loader batch, cloned off the loader buffers --
    # (clones isolate the loop from the loader's persistent gather buffers).
    var loader = ds.into_loader(batch_size=4, shuffle=False)
    if not loader.__has_next__():
        raise Error("gpt_overfit: no windows (excerpt too short?)")
    ref b0 = loader.__next__()
    var xb = b0.features.clone()
    var yb = b0.labels.clone()
    print("batch:", xb.shape(), yb.shape())

    # ---- Model: 2 layers, C=64, tied head (~150K params) ----
    var model = GPTModel[DType.float32](
        n_vocab=512,
        n_ctx=32,
        n_embd=64,
        n_head=2,
        n_layer=2,
        tie_weights=True,
        init_seed=42,
    )
    model.train()
    var params = model.parameters()
    var n_params = 0
    for i in range(len(params)):
        n_params += params[i][].numels()
    print("parameters:", n_params)

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    criterion.train()
    var learning_rate = Scalar[DType.float32](3e-4)

    # ---- Memorize with AdamW (the LLM path's optimizer) ----
    # Calibrated: lr=3e-4 constant, no warmup. (Bigger steps
    # diverge at this width — lr=1e-3 climbs after ~50 steps, SGD lr=0.5
    # NaNs by step 25 — while lr=1e-4 descends monotonically; no graph or
    # data bug, just transformer lr sensitivity.)
    var adamw = AdamW[DType.float32](
        params,
        lr=learning_rate,
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var loss0 = 0.0
    var lossN = 0.0
    for step in range(800):
        var logits = model(xb)  # (B, T, V)
        var loss = criterion(logits.permute([0, 2, 1]), yb)  # (B, V, T)

        var v = Float64(loss.item())
        if v != v:
            raise Error(
                "gpt_overfit: loss NaN at step " + String(step + 1)
            )
        if step == 0:
            loss0 = v
        lossN = v

        adamw.zero_grad()
        loss.backward()
        adamw.step()

        if (step + 1) % 100 == 0:
            print("step", step + 1, "loss:", v)

    print("loss0=", loss0, "lossN=", lossN)

    # ---- Post-turn gradient check: analytic vs finite differences ----
    # A single-step FD check at init cannot catch cross-step accumulation
    # bugs, so this runs HERE (post-trajectory): matching grads while the
    # loss climbs means overshoot dynamics (schedule around it); a mismatch
    # means the graph itself is wrong (hunt the op, not the lr).
    var fd_logits = model(xb)
    var fd_perm = fd_logits.permute([0, 2, 1])
    var fd_loss = criterion(fd_perm, yb)
    adamw.zero_grad()
    fd_loss.backward()

    # wte is the unique (512, 64) table; ln_f gamma is params[-2]
    # (LayerNorm.parameters appends [gamma, beta], ln_f goes last when tied).
    var wte_idx = -1
    for i in range(len(params)):
        var sh = params[i][].shape()
        if sh.rank() == 2 and sh.dims[0] == 512 and sh.dims[1] == 64:
            wte_idx = i
    if wte_idx < 0:
        raise Error("gradcheck: wte table not found in parameters()")
    var ln_idx = len(params) - 2

    var fd_eps = Float32(1e-2)
    var grad_ok = True

    var probe_row = Int(xb[0, 0])
    var analytic_wte = Float64(params[wte_idx][].gradients().get(probe_row * 64))
    var wte_orig = params[wte_idx][][probe_row, 0]
    params[wte_idx][][probe_row, 0] = wte_orig + fd_eps
    var fwd_p = model(xb)
    var perm_p = fwd_p.permute([0, 2, 1])
    var lp_wte = Float64(criterion(perm_p, yb).item())
    params[wte_idx][][probe_row, 0] = wte_orig - fd_eps
    var fwd_m = model(xb)
    var perm_m = fwd_m.permute([0, 2, 1])
    var lm_wte = Float64(criterion(perm_m, yb).item())
    params[wte_idx][][probe_row, 0] = wte_orig
    var fd_wte = (lp_wte - lm_wte) / (2.0 * Float64(fd_eps))
    var dw = analytic_wte - fd_wte
    if dw < 0:
        dw = -dw
    var fw = fd_wte
    if fw < 0:
        fw = -fw
    print(
        "gradcheck wte[row=",
        probe_row,
        "]: analytic=",
        analytic_wte,
        " fd=",
        fd_wte,
    )
    if dw > 0.05 * (fw + 1e-8) + 1e-5:
        print("gradcheck: wte MISMATCH")
        grad_ok = False

    var analytic_ln = Float64(params[ln_idx][].gradients().get(0))
    var ln_orig = params[ln_idx][][0]
    params[ln_idx][][0] = ln_orig + fd_eps
    var fl_p = model(xb)
    var pl_p = fl_p.permute([0, 2, 1])
    var lp_ln = Float64(criterion(pl_p, yb).item())
    params[ln_idx][][0] = ln_orig - fd_eps
    var fl_m = model(xb)
    var pl_m = fl_m.permute([0, 2, 1])
    var lm_ln = Float64(criterion(pl_m, yb).item())
    params[ln_idx][][0] = ln_orig
    var fd_ln = (lp_ln - lm_ln) / (2.0 * Float64(fd_eps))
    var dl = analytic_ln - fd_ln
    if dl < 0:
        dl = -dl
    var fl = fd_ln
    if fl < 0:
        fl = -fl
    print("gradcheck ln_f.gamma[0]: analytic=", analytic_ln, " fd=", fd_ln)
    if dl > 0.05 * (fl + 1e-8) + 1e-5:
        print("gradcheck: ln_f.gamma MISMATCH")
        grad_ok = False

    if not grad_ok:
        raise Error(
            "gpt_overfit: analytic grads do NOT match finite differences —"
            " graph bug, hunt the op (not the lr)"
        )
    print("gradcheck passed: backward is exact; the turn is dynamics, not wiring.")
    if lossN > 0.5:
        raise Error(
            "gpt_overfit: grads exact but loss did not collapse (loss0="
            + String(loss0)
            + " lossN="
            + String(lossN)
            + ") — needs an lr schedule, not a bugfix"
        )
    print("overfit smoke passed: one batch memorized.")
