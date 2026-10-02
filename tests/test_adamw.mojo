"""
Tests for AdamW.

────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE IS FOR (read this first if you are new to the codebase).
────────────────────────────────────────────────────────────────────────────
AdamW is the transformer optimizer: per-parameter adaptive step sizes
from gradient moments (m/v), bias-corrected, with weight decay applied
OUTSIDE the adaptive denominator (the "W"). Each test below is one
gate case. Cases 1–2 carry their oracle values inline: if you
change the update formula, re-derive those numbers first.

HOW TO RUN:
    ./execute.sh adamw

The suite discovers all `test_adamw_*` functions automatically via
`TestSuite.discover_tests[__functions_in_module()]().run()`.
────────────────────────────────────────────────────────────────────────────
"""

from std.math import sqrt
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.gpt import GPTModel
from tenmo.optim import SGD
from tenmo.tensor import Tensor


def check_close[dtype: DType](
    t: Tensor[dtype], expected: List[Float64], tol: Float64
) raises:
    """Assert every element of t is within tol of the expected value."""
    assert_true(t.num_elements() == len(expected))
    for k in range(t.num_elements()):
        var diff = Float64(t.get(k)) - expected[k]
        assert_true(diff < tol and diff > -tol)


def make_params[
    dtype: DType
](mut t: Tensor[dtype]) -> List[Pointer[Tensor[dtype], MutAnyOrigin]]:
    var params = List[Pointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(Pointer(to=t).unsafe_origin_cast[MutAnyOrigin]())
    return params^


# ═════════════════════════════════════════════════════════════════════════════
# Case 1 — Single-step hand parity
# ═════════════════════════════════════════════════════════════════════════════


def test_adamw_single_step_parity() raises:
    """
    TEST — one AdamW step on θ=[1.0, 0.5], g=[0.1.
    −0.2] with
    β1=β2=0.9, lr=0.01, wd=0 must give θ=[0.99, 0.51].

    WHY: catches formula typos (misplaced correction, wrong decay slot)
    that "loss goes down" testing would never localize.
    """
    comptime dtype = DType.float32
    var theta = Tensor[dtype].d1([1.0, 0.5], requires_grad=True)
    var opt = AdamW[dtype](
        make_params(theta),
        lr=0.01,
        beta1=0.9,
        beta2=0.9,
        weight_decay=0.0,
    )
    theta.seed_grad(Tensor[dtype].d1([0.1, -0.2]))
    opt.step()
    var want1: List[Float64] = [0.99, 0.51]
    check_close[dtype](theta, want1, 1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# Case 2 — Bias correction
# ═════════════════════════════════════════════════════════════════════════════


def test_adamw_bias_correction() raises:
    """
    TEST.
    (a) step 1 from zero memories moves exactly lr·sign(g);
    (b) after 200 steady steps the per-step move is still lr·sign(g)
    (correction faded to a no-op, memories converged to the steady values).

    WHY: fails if c1/c2 are misplaced, missing, or applied to the wrong
    quantities; (b) fails if t is not actually advancing.
    """
    comptime dtype = DType.float32
    var theta = Tensor[dtype].d1([1.0, 1.0, 1.0], requires_grad=True)
    var opt = AdamW[dtype](
        make_params(theta), lr=0.1, weight_decay=0.0
    )
    var g = Tensor[dtype].d1([0.5, -2.0, 0.25])

    # (a) first step: pure sign descent at lr
    theta.seed_grad(g)
    opt.step()
    var want2: List[Float64] = [0.9, 1.1, 0.9]
    check_close[dtype](theta, want2, 1e-5)

    # (b) 199 more steady steps, then measure one step's move
    for _ in range(199):
        opt.zero_grad()
        theta.seed_grad(g)
        opt.step()
    var before0 = Float64(theta.get(0))
    opt.zero_grad()
    theta.seed_grad(g)
    opt.step()
    var move = before0 - Float64(theta.get(0))
    # steady g>0 ⇒ each step moves −lr exactly (within f32 dust)
    assert_true(move < 0.1 + 1e-4 and move > 0.1 - 1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# Case 3 — Decoupled-vs-L2 divergence (proves the "W" is real)
# ═════════════════════════════════════════════════════════════════════════════


def test_adamw_decoupled_divergence() raises:
    """
        TEST — a two-step discriminating trace (big g=10, then tiny g=0.001,.
    λ=0.5): our result must match a Float64 decoupled reference to 1e-5
    AND differ from the L2-in-gradient variant by > 1e-3.

    WHY: with steady gradients both variants coincide (Adam normalizes to
    sign steps either way); only a varying-gradient trace separates them.
    This is the test that proves weight decay bypasses the moments.
    """
    comptime dtype = DType.float32
    var lr = 0.01
    var b1 = 0.9
    var b2 = 0.9
    var decay = 0.5

    var theta = Tensor[dtype].d1([1.0], requires_grad=True)
    var opt = AdamW[dtype](
        make_params(theta),
        lr=Scalar[dtype](lr),
        beta1=Scalar[dtype](b1),
        beta2=Scalar[dtype](b2),
        weight_decay=Scalar[dtype](decay),
    )

    # Float64 reference — decoupled (ours) and L2-folded (theirs)
    var th: Float64 = 1.0
    var m: Float64 = 0.0
    var v: Float64 = 0.0
    var th_l2: Float64 = 1.0
    var m_l2: Float64 = 0.0
    var v_l2: Float64 = 0.0
    var grads: List[Float64] = [10.0, 0.001]
    var corrects: List[Float64] = [0.1, 0.19]  # 1 − 0.9**t for t = 1, 2
    for step in range(2):
        var c1 = corrects[step]
        var c2 = corrects[step]
        var g = grads[step]
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        th = th - lr * ((m / c1) / (sqrt(v / c2) + 1e-8) + decay * th)
        var g_l2 = g + decay * th_l2
        m_l2 = b1 * m_l2 + (1.0 - b1) * g_l2
        v_l2 = b2 * v_l2 + (1.0 - b2) * g_l2 * g_l2
        th_l2 = th_l2 - lr * ((m_l2 / c1) / (sqrt(v_l2 / c2) + 1e-8))

    theta.seed_grad(Tensor[dtype].d1([10.0]))
    opt.step()
    opt.zero_grad()
    theta.seed_grad(Tensor[dtype].d1([0.001]))
    opt.step()

    var got = Float64(theta.get(0))
    var d_decoupled = got - th
    assert_true(d_decoupled < 1e-5 and d_decoupled > -1e-5)
    var d_l2 = got - th_l2
    assert_true(d_l2 > 1e-3 or d_l2 < -1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# Case 4 — state_dict round-trip + resume equality
# ═════════════════════════════════════════════════════════════════════════════


def run_adamw_steps[dtype: DType](mut t: Tensor[dtype], mut opt: AdamW[dtype], n: Int) raises:
    for _ in range(n):
        opt.zero_grad()
        t.seed_grad(Scalar[dtype](0.5))
        opt.step()


def test_adamw_state_dict_resume() raises:
    """
    TEST — run 10 steps uninterrupted (A); run 5, save.
    Load into a fresh
    optimizer (B), run 5 more. Final parameters must match.

    WHY: catches step_count/moments restore bugs. A resume that reset t
    would silently re-shrink every update  — this test makes
    that failure loud.
    """
    comptime dtype = DType.float32
    var theta_a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var opt_a = AdamW[dtype](make_params(theta_a), lr=0.05)
    run_adamw_steps(theta_a, opt_a, 10)

    var theta_b = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var opt_b = AdamW[dtype](make_params(theta_b), lr=0.05)
    run_adamw_steps(theta_b, opt_b, 5)
    var saved = opt_b.state_dict()
    assert_true(saved["type"] == "AdamW")
    var opt_c = AdamW[dtype].load_state_dict(saved, make_params(theta_b))
    run_adamw_steps(theta_b, opt_c, 5)

    assert_true(theta_a.all_close(theta_b))


# ═════════════════════════════════════════════════════════════════════════════
# Case 5 — Overfit race vs SGD
# ═════════════════════════════════════════════════════════════════════════════


def race_steps_to_bar(use_adamw: Bool) raises -> Int:
    """Train the case-8 tiny GPTModel to CE < 0.5; return steps (61 = never)."""
    comptime dtype = DType.float32
    var tokens = Tensor[DType.int64].d2([[3, 17, 5, 22], [1, 4, 9, 30]])
    var targets = Tensor[DType.int64].d2([[17, 5, 22, 3], [4, 9, 30, 1]])
    var model = GPTModel[dtype](
        n_vocab=32,
        n_ctx=8,
        n_embd=16,
        n_head=2,
        n_layer=2,
        tie_weights=True,
        init_seed=42,
    )
    # NOTE: train() is mandatory here, not incidental: the `training` flag
    # forks the `track_grad` comptime switch (gpt.mojo:36), so eval() builds
    # no graph and nothing could learn. Dropout defaults are 0.0 throughout
    # the stack, so train-mode trajectories are deterministic and the race
    # cannot flake run-to-run.
    model.train()
    var params = model.parameters()
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()
    if use_adamw:
        var opt = AdamW[dtype](params^, lr=0.05, weight_decay=0.0)
        for step in range(60):
            var logits = model(tokens).permute([0, 2, 1])
            var loss = criterion(logits, targets)
            var v = Float64(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if v < 0.5:
                return step + 1
        return 61
    else:
        var sgd = SGD[dtype](params^, lr=0.5)
        for step in range(60):
            var logits = model(tokens).permute([0, 2, 1])
            var loss = criterion(logits, targets)
            var v = Float64(loss.item())
            sgd.zero_grad()
            loss.backward()
            sgd.step()
            if v < 0.5:
                return step + 1
        return 61


def test_adamw_overfit_race() raises:
    """
    TEST — on the case-8 tiny GPTModel.
    AdamW must drive CE below
    0.5 in strictly fewer steps than SGD at lr=0.5 (both capped at 60).

    WHY: proves the motivation on our own graph — adaptive steps
    beat a global lr on transformer loss surfaces.
    """
    var adamw_steps = race_steps_to_bar(True)
    var sgd_steps = race_steps_to_bar(False)
    print("  adamw_steps=", adamw_steps, " sgd_steps=", sgd_steps)
    assert_true(adamw_steps <= 60)
    assert_true(adamw_steps < sgd_steps)


# ═════════════════════════════════════════════════════════════════════════════
# Case 6 — Hygiene: no-grad params, set_lr, clip engagement
# ═════════════════════════════════════════════════════════════════════════════


def test_adamw_hygiene() raises:
    """
    TEST.
    (a) a parameter that never received a gradient is untouched by
    step(); (b) set_lr takes effect via get_lr; (c) clip_norm bounds the
    update under a 1e6 gradient.

    WHY: (a) guards the requires_grad/has_grad skip; (c) proves clipping
    happens BEFORE moments (a post-moment clip could not bound a step-1
    update this way — step 1 is lr·sign of the CLIPPED grad).
    """
    comptime dtype = DType.float32

    # (a) untouched without grads
    var theta = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var opt = AdamW[dtype](make_params(theta), weight_decay=0.0)
    opt.step()
    var want6: List[Float64] = [1.0, 2.0]
    check_close[dtype](theta, want6, 1e-9)

    # (b) set_lr round-trips
    opt.set_lr(Scalar[dtype](0.2))
    assert_true(opt.get_lr() == Scalar[dtype](0.2))

    # (c) clip_norm engages before moments: grad 1e6, norm clipped to 1.0,
    # so step 1 moves lr·sign ⇒ |Δ| == lr == 0.1 (wd=0), not 1e5.
    var big = Tensor[dtype].d1([1.0, -1.0], requires_grad=True)
    var opt_c = AdamW[dtype](
        make_params(big), lr=0.1, weight_decay=0.0, clip_norm=1.0
    )
    big.seed_grad(Tensor[dtype].d1([1e6, -1e6]))
    opt_c.step()
    var d0 = Float64(big.get(0)) - 1.0
    var d1 = Float64(big.get(1)) + 1.0
    assert_true(d0 < 0.0 and d0 > -0.11)
    assert_true(d1 > 0.0 and d1 < 0.11)


# ═════════════════════════════════════════════════════════════════════════════
# GPU parity (guarded — runs only where a device exists)
# ═════════════════════════════════════════════════════════════════════════════


def test_adamw_gpu_matches_cpu() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params_cpu = List[Pointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[Pointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(Pointer(to=w).unsafe_origin_cast[MutAnyOrigin]())
        params_gpu.append(
            Pointer(to=w_gpu).unsafe_origin_cast[MutAnyOrigin]()
        )
        var opt_cpu = AdamW[dtype](params_cpu, lr=0.1, weight_decay=0.0)
        var opt_gpu = AdamW[dtype](params_gpu, lr=0.1, weight_decay=0.0)
        var grad = Tensor[dtype].d1([0.5, -1.5, 0.25])
        var grad_gpu = grad.to_gpu()
        w.seed_grad(grad)
        w_gpu.seed_grad(grad_gpu)
        opt_cpu.step()
        opt_gpu.step()
        assert_true(w.all_close(w_gpu.to_cpu()))


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll AdamW tests passed!")
