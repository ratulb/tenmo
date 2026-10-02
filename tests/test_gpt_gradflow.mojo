"""GPT gradient-flow ladder — ordered regression tests for the zero-grad hunt.

BACKGROUND (read this first — it explains why this file exists):
    During the GPT-stack build, training produced ALL-ZERO gradients:
    `loss.backward()` ran without error, but every parameter's gradbox stayed
    at zero and the loss sat stuck at ~log(V). It LOOKED like the autograd
    graph was severed somewhere inside the model — a reasonable hypothesis,
    since `GPTModel` chains the trickiest machinery in the library: nested
    field access (`self.wte_wpe.wte.weight`), a `List[TransformerBlock]` block
    loop via `ref` bindings, pre-norm residual adds, and a weight-tied output
    head (`logits = x @ wte.T`).

    So a ladder of minimal probes was written (`debug/test_*`, since removed),
    each adding exactly ONE structural element of the real model, checking
    whether gradients survived. Every rung passed. The graph was innocent.

    The real root cause was the WEIGHT INITIALIZER, not autograd: one call
    path requested `init_method="standard"` — a name from an older inline
    init scheme — which `Embedding` did not recognize. The weights came out
    all zeros, and `backward()` faithfully propagated those zeros. Backward
    was functional all along; it was differentiating a dead model.

    The fix was a single shared initializer, `Weights.initialize`
    (`tenmo/weight_init.mojo`), keyed on the `WeightStrategy` vocabulary
    (`normal / uniform / xavier(+glorot) / kaiming(+he) / zero`). A `String`
    passed as `init_method` converts `@implicit`ly to `WeightStrategy`, and
    an UNKNOWN name `panic`s at construction — loudly, instead of silently
    building a zero model. That panic-on-unknown is the lesson of this file
    made permanent.

    A NOTE ON INIT SCHEMES IN THIS FILE: standalone layers keep their own
    defaults (`Linear`/`SelfAttention` → "uniform", `Embedding` →
    "normal", `PositionalEmbedding`/`MLP` → "xavier", `Conv2D` → "he"), but
    containers FORWARD one scheme down the whole subtree — so a default
    `GPTModel` is all-xavier end to end. Rungs 3-4 use standalone defaults
    (they test loop mechanics, not GPT fidelity); rungs 5-7 pass explicit
    "xavier" everywhere to mirror a default GPTModel exactly. Rung 0 pins
    every vocabulary name so the mix can never silently rot again.

HOW TO READ THIS FILE:
    Rung 0 first: it directly encodes the lesson — every vocabulary name must
    produce live weights and live gradients. If it ever fails, the init
    contract regressed; do NOT go hunting in autograd.
    Rungs 1-7 then replay the original bisection in order, smallest structure
    first. Each rung's docstring states WHAT it isolates, HOW it checks, and
    WHAT A FAILURE MEANS (which component to suspect). If rung N fails but
    all rungs below N pass, the bug was introduced by whatever rung N adds.

RUNNING:
    ./execute.sh gptflow            # this file, top-to-bottom via main()
    ./execute.sh -d gptflow         # same, plus per-step diagnostics
                                    # (log_debug only prints at LOGGING_LEVEL=debug)
    Diagnostic chatter goes through `log_debug`, so normal runs stay quiet
    and only PASS/FAIL lines print.
"""

from std.testing import assert_true
from tenmo.tensor import Tensor
from tenmo.matmul import Matmul
from tenmo.embedding import Embedding
from tenmo.positional import PositionalEmbedding
from tenmo.attention import SelfAttention
from tenmo.layernorm import LayerNorm
from tenmo.gelu import GeLU
from tenmo.dropout import Dropout
from tenmo.net import Linear
from tenmo.gpt import MLP
from tenmo.shared.shapes import Shape
from tenmo.shared.intarray import IntArray
from tenmo.shared.panic import panic
from tenmo.shared.logging import log_debug
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.optim import SGD
from tenmo.layer_trait import LayerTrait


# ─── shared helpers ──────────────────────────────────────────────────────────
# `Tensor` without parameters means `Tensor[DType.float32]`; this whole file
# is float32 (the chunk generators assume float32-only test files).


def max_abs(t: Tensor) -> Float64:
    """Largest absolute element of a tensor's storage. Used to ask "is this
    weight / gradient alive?" — anything above 1e-9 counts as nonzero."""
    var m: Float64 = 0.0
    for k in range(t.num_elements()):
        var v = Float64(t.get(k))
        if v > m or -v > m:
            m = v if v > 0 else -v
    return m


def grad_max_abs(t: Tensor) -> Float64:
    """Largest absolute element of a parameter's gradbox. Returns 0.0.
    when the
    parameter has no gradbox at all (i.e. nothing ever accumulated into it —
    the signature of a severed graph OR a dead-zero forward)."""
    if not t.has_grad():
        return 0.0
    var gb = t.gradients()
    var m: Float64 = 0.0
    for k in range(gb.num_elements()):
        var v = Float64(gb.get(k))
        if v > m or -v > m:
            m = v if v > 0 else -v
    return m


# ─── Rung 0: the lesson itself ───────────────────────────────────────────────
# WHAT: every name in the WeightStrategy vocabulary must build a LIVE model.
# HOW: construct a Linear and an Embedding under each scheme name (plus the
#   `glorot`/`he` aliases), run one forward+backward step, and assert the
#   weights are nonzero (except "zero") and gradients flow (all schemes).
# WHY FIRST: this single test would have caught the original "standard" bug
#   in one run — an unknown name panics at construction instead of silently
#   producing a zero model, and any scheme that yields dead weights fails
#   here before anyone blames autograd.
# FAILURE MEANS: the init contract regressed — look at
#   `tenmo/weight_init.mojo` / `tenmo/shared/__init__.mojo` (WeightStrategy),
#   NOT at backward/ancestry code.


def check_scheme_live(name: String) raises:
    comptime dtype = DType.float32
    var is_zero_scheme = name == "zero"

    var lin = Linear[dtype](8, 8, init_seed=7, init_method=name)
    lin.train()
    var w_max = max_abs(lin.weight)
    log_debug("init_vocab: scheme=" + name + " weight_max=" + String(w_max))
    if is_zero_scheme:
        assert_true(w_max == 0.0, "rung 0: 'zero' scheme must build zero weights")
    else:
        assert_true(
            w_max > 1e-9, "rung 0: scheme '" + name + "' built dead-zero weights"
        )

    var x = Tensor[dtype].ones(2, 8)
    var y = lin(x)
    var loss = y.sum()
    loss.backward()
    var g_max = grad_max_abs(lin.weight)
    assert_true(
        g_max > 1e-9,
        "rung 0: scheme '" + name + "' yields zero gradients after backward",
    )

    var emb = Embedding[dtype, DType.int64](
        num_embeddings=16, embedding_dim=8, init_seed=7, init_method=name
    )
    var e_max = max_abs(emb.weight)
    if is_zero_scheme:
        assert_true(e_max == 0.0, "rung 0: 'zero' scheme must build zero embeddings")
    else:
        assert_true(
            e_max > 1e-9, "rung 0: scheme '" + name + "' built dead-zero embeddings"
        )


def test_init_vocab_all_schemes_live() raises:
    var names = List[String]()
    names.append("normal")
    names.append("uniform")
    names.append("xavier")
    names.append("glorot")  # alias of xavier — locked in so it can't drift
    names.append("kaiming")
    names.append("he")  # alias of kaiming — locked in so it can't drift
    names.append("zero")
    for i in range(len(names)):
        check_scheme_live(names[i])


# ─── Rung 1: transpose off a field keeps the graph ───────────────────────────
# WHAT: the tied output head does `weight.transpose(...)` where `weight` is
#   reached through struct fields. Checks that a transpose of a field-owned
#   tensor (not just a local variable) still records ancestry.
# HOW: pattern A transposes a direct variable copy of the embedding weight;
#   pattern B transposes `emb.weight` inline. Both feed a tracked matmul +
#   CE loss; both embedding gradboxes must be nonzero.
# FAILURE MEANS: field access (rather than something deeper) breaks the
#   origin chain backward needs — look at how transpose/matmul capture
#   parents, not at List/attention/embedding machinery.


def test_direct_and_field_transpose_keep_graph() raises:
    comptime dtype = DType.float32
    var V = 8
    var C = 4

    var tokens = Tensor[DType.int64].d2([[1, 3, 5], [2, 4, 6]])
    var targets = Tensor[DType.int64].d2([[3, 5, 1], [4, 6, 2]])

    # Pattern A: transpose of a direct variable.
    var emb_a = Embedding[dtype, DType.int64](V, C, init_seed=42)
    var x_a = emb_a(tokens)
    var weight_a = emb_a.weight
    var wt_a = weight_a.transpose(IntArray(1, 0))
    var logits_a = Matmul[dtype].forward[track_grad=True](x_a, wt_a)
    var crit_a = CrossEntropyLoss[dtype](reduction="mean")
    crit_a.train()
    var loss_a = crit_a(logits_a.permute([0, 2, 1]), targets)
    loss_a.backward()
    var gb_a = grad_max_abs(emb_a.weight)
    log_debug("rung 1: pattern A (direct var) grad_max=" + String(gb_a))

    # Pattern B: transpose of a single-level field chain.
    var emb_b = Embedding[dtype, DType.int64](V, C, init_seed=42)
    var x_b = emb_b(tokens)
    var wt_b = emb_b.weight.transpose(IntArray(1, 0))
    var logits_b = Matmul[dtype].forward[track_grad=True](x_b, wt_b)
    var crit_b = CrossEntropyLoss[dtype](reduction="mean")
    crit_b.train()
    var loss_b = crit_b(logits_b.permute([0, 2, 1]), targets)
    loss_b.backward()
    var gb_b = grad_max_abs(emb_b.weight)
    log_debug("rung 1: pattern B (field chain) grad_max=" + String(gb_b))

    assert_true(gb_a > 1e-9, "rung 1: direct-variable transpose lost gradients")
    assert_true(gb_b > 1e-9, "rung 1: field-chain transpose lost gradients")


# ─── Rung 2: nested holders, accessor methods, pointer identity ──────────────
# WHAT: `GPTModel` reaches its tied weight through TWO struct levels
#   (`self.wte_wpe.wte.weight`) and via a `wte_weight()` accessor returning a
#   `ref`. Checks all three reach-styles plus the scarier possibility that
#   `parameters()` hands the optimizer pointers to COPIES instead of the
#   real field storage (gradients would then land somewhere the optimizer
#   never reads).
# HOW: patterns 1-3 run the same embed → transpose → matmul → CE graph
#   through (1) a nested field chain, (2) the accessor method, (3) the exact
#   forward/head split `GPTModel._forward` uses. Pattern 4 compares the raw
#   data pointer behind `parameters()[0]` with the field's own pointer.
# FAILURE MEANS: nesting or the accessor severs ancestry (patterns 1-3), or
#   the optimizer is updating copies (pattern 4) — look at struct copy/ref
#   semantics and `parameters()` construction, not at attention/blocks.


@fieldwise_init
struct NestedEmbedHolder[dtype: DType, index_dtype: DType]:
    var wte: Embedding[Self.dtype, Self.index_dtype]

    def __init__(out self, V: Int, C: Int):
        self.wte = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=V, embedding_dim=C, init_seed=42
        )

    def wte_weight(ref self) -> ref[self.wte.weight] Tensor[Self.dtype]:
        return self.wte.weight

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.wte.parameters()


@fieldwise_init
struct NestedModel[dtype: DType, index_dtype: DType]:
    var emb: NestedEmbedHolder[Self.dtype, Self.index_dtype]

    def __init__(out self, V: Int, C: Int):
        self.emb = NestedEmbedHolder[Self.dtype, Self.index_dtype](V, C)

    def wte_weight(ref self) -> ref[self.emb.wte.weight] Tensor[Self.dtype]:
        return self.emb.wte.weight

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.emb.parameters()


def test_nested_holder_and_accessor_keep_graph() raises:
    comptime dtype = DType.float32
    var V = 8
    var C = 4

    var tokens = Tensor[DType.int64].d2([[1, 3, 5], [2, 4, 6]])
    var targets = Tensor[DType.int64].d2([[3, 5, 1], [4, 6, 2]])

    # Pattern 1: doubly-nested field chain, as in `self.wte_wpe.wte.weight`.
    var m1 = NestedModel[dtype, DType.int64](V, C)
    var x1 = m1.emb.wte(tokens)
    var wt1 = m1.emb.wte.weight.transpose(IntArray(1, 0))
    var crit1 = CrossEntropyLoss[dtype](reduction="mean")
    crit1.train()
    var loss1 = crit1(
        Matmul[dtype]
        .forward[track_grad=True](x1, wt1)
        .permute([0, 2, 1]),
        targets,
    )
    loss1.backward()
    var g1 = grad_max_abs(m1.emb.wte.weight)
    log_debug("rung 2: pattern 1 (nested chain) grad_max=" + String(g1))

    # Pattern 2: accessor method returning a ref.
    var m2 = NestedModel[dtype, DType.int64](V, C)
    var x2 = m2.emb.wte(tokens)
    var wt2 = m2.wte_weight().transpose(IntArray(1, 0))
    var crit2 = CrossEntropyLoss[dtype](reduction="mean")
    crit2.train()
    var loss2 = crit2(
        Matmul[dtype]
        .forward[track_grad=True](x2, wt2)
        .permute([0, 2, 1]),
        targets,
    )
    loss2.backward()
    var g2 = grad_max_abs(m2.emb.wte.weight)
    log_debug("rung 2: pattern 2 (accessor ref) grad_max=" + String(g2))

    # Pattern 3: the exact forward/head split GPTModel._forward uses.
    var m3 = NestedModel[dtype, DType.int64](V, C)
    var x3 = m3.emb.wte(tokens)
    var wt3 = m3.wte_weight().transpose(IntArray(1, 0))
    var crit3 = CrossEntropyLoss[dtype](reduction="mean")
    crit3.train()
    var loss3 = crit3(
        Matmul[dtype]
        .forward[track_grad=True](x3, wt3)
        .permute([0, 2, 1]),
        targets,
    )
    loss3.backward()
    var g3 = grad_max_abs(m3.emb.wte.weight)
    log_debug("rung 2: pattern 3 (forward/head split) grad_max=" + String(g3))

    # Pattern 4: the optimizer must point at the REAL field storage.
    var m4 = NestedModel[dtype, DType.int64](V, C)
    var params = m4.parameters()
    var field_ptr = m4.emb.wte.weight.buffer.buffer.data
    var param_ptr = params[0][].buffer.buffer.data
    if field_ptr == param_ptr:
        log_debug("rung 2: pattern 4 parameters() aliases field storage")
    else:
        log_debug("rung 2: pattern 4 MISMATCH — parameters() points at copies")

    assert_true(g1 > 1e-9, "rung 2: nested field chain lost gradients")
    assert_true(g2 > 1e-9, "rung 2: accessor-method ref lost gradients")
    assert_true(g3 > 1e-9, "rung 2: forward/head split lost gradients")
    assert_true(
        field_ptr == param_ptr,
        "rung 2: parameters() points at copies, not field storage",
    )


# ─── Rung 3: List-of-layers loop keeps upstream grads ────────────────────────
# WHAT: `GPTModel.h` is a `List[TransformerBlock]` driven via
#   `ref block = self.h[i]`. Checks the suspicion that List indexing (or the
#   `ref` reborrow) detaches the graph upstream of the loop.
# HOW: a miniature model — plain embedding holder (NOT in a List) feeding
#   two `Linear`s held in a `List`, looped exactly like `GPTModel._forward`
#   — then asserts the embedding OUTSIDE the List still gets gradients and
#   every parameter moves after one optimizer step.
# FAILURE MEANS: List indexing/`ref` severs the graph — look at
#   `List.__getitem__` aliasing and ref-reborrow semantics, not at block
#   internals (those come in rungs 4+).


@fieldwise_init
struct PlainEmbed[dtype: DType, index_dtype: DType]:
    var wte: Embedding[Self.dtype, Self.index_dtype]

    def __init__(out self, V: Int, C: Int):
        self.wte = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=V, embedding_dim=C, init_seed=42
        )

    def wte_weight(ref self) -> ref[self.wte.weight] Tensor[Self.dtype]:
        return self.wte.weight

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.wte.parameters()


@fieldwise_init
struct ChainModel[dtype: DType, index_dtype: DType]:
    var emb: PlainEmbed[Self.dtype, Self.index_dtype]
    var blocks: List[Linear[Self.dtype]]
    var n_blocks: Int

    def __init__(out self, V: Int, C: Int, n_blocks: Int):
        self.emb = PlainEmbed[Self.dtype, Self.index_dtype](V, C)
        self.blocks = List[Linear[Self.dtype]]()
        self.n_blocks = n_blocks
        for _ in range(n_blocks):
            self.blocks.append(
                Linear[Self.dtype](in_features=C, out_features=C, init_seed=42)
            )

    def __call__(mut self, tokens: Tensor[Self.index_dtype]) -> Tensor[Self.dtype]:
        var x = self.emb.wte(tokens)
        # EXACT pattern used in GPTModel._forward's block loop.
        for i in range(self.n_blocks):
            ref block = self.blocks[i]
            x = block(x)
        var wt = self.emb.wte_weight().transpose(IntArray(1, 0))
        return Matmul[Self.dtype].forward[track_grad=True](x, wt)

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.emb.parameters()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            var bp = block.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        return params^


def test_list_loop_keeps_upstream_grad() raises:
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var N_BLOCKS = 2

    var tokens = Tensor[DType.int64].d2([[1, 3, 5], [2, 4, 6]])
    var targets = Tensor[DType.int64].d2([[3, 5, 1], [4, 6, 2]])

    var model = ChainModel[dtype, DType.int64](V, C, N_BLOCKS)

    var params = model.parameters()
    var snapshots = List[Tensor[dtype]]()
    for i in range(len(params)):
        snapshots.append(params[i][].clone())

    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var logits = model(tokens)
    var loss = criterion(logits.permute([0, 2, 1]), targets)
    log_debug("rung 3: loss=" + String(Float64(loss.item())))

    optimizer.zero_grad()
    loss.backward()

    var emb_max = grad_max_abs(model.emb.wte.weight)
    log_debug("rung 3: emb grad_max=" + String(emb_max))
    for i in range(N_BLOCKS):
        ref block = model.blocks[i]
        log_debug(
            "rung 3: blocks[" + String(i) + "] grad_max="
            + String(grad_max_abs(block.weight))
        )

    optimizer.step()

    var any_unchanged = False
    for i in range(len(params)):
        var snap = snapshots[i]
        var param = params[i][]
        var changed = False
        for j in range(snap.num_elements()):
            var diff = Float64(snap.get(j)) - Float64(param.get(j))
            if diff > 1e-6 or diff < -1e-6:
                changed = True
                break
        if not changed:
            any_unchanged = True

    assert_true(
        emb_max > 1e-9,
        "rung 3: List/ref loop severed gradients upstream of the blocks",
    )
    assert_true(
        not any_unchanged,
        "rung 3: some parameter did not move after optimizer.step()",
    )


# ─── Rung 4: residual + training-dispatch block keeps grads ──────────────────
# WHAT: the previous rung used bare Linears; real blocks add the pre-norm
#   residual pattern (`x + attn(ln(x))`, `h + mlp(ln(h))`) plus the
#   `if self.training: _forward[track_grad=True] else: _forward[track_grad=False]`
#   dispatch every GPT layer uses. Attention itself is still stubbed by a
#   Linear here (it has its own test suite and is cleared separately).
# HOW: `StubBlock` mirrors `TransformerBlock` exactly except for the stub;
#   asserts upstream (embedding) and deep (mlp weights) grads plus full
#   parameter movement.
# FAILURE MEANS: the break is in the residual wiring or the
#   training-flag/`track_grad` dispatch — bisect by forcing
#   `track_grad=True` unconditionally to isolate the dispatch.


@fieldwise_init
struct DropEmbed[dtype: DType, index_dtype: DType]:
    var wte: Embedding[Self.dtype, Self.index_dtype]
    var drop: Dropout[Self.dtype]
    var training: Bool

    def __init__(out self, V: Int, C: Int):
        self.wte = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=V, embedding_dim=C, init_seed=42
        )
        self.drop = Dropout[Self.dtype](Scalar[Self.dtype](0.0))
        self.training = True

    def __call__(mut self, x: Tensor[Self.index_dtype]) -> Tensor[Self.dtype]:
        var out = self.wte(x)
        return self.drop(out)

    def wte_weight(ref self) -> ref[self.wte.weight] Tensor[Self.dtype]:
        return self.wte.weight

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.wte.parameters()

    def train(mut self):
        self.training = True
        self.drop.train()


@fieldwise_init
struct StubBlock[dtype: DType](LayerTrait):
    comptime TAG = 10101
    comptime InputDType = Self.dtype
    comptime OutputDType = Self.dtype
    var ln_1: LayerNorm[Self.dtype]
    var attn_stub: Linear[Self.dtype]
    var ln_2: LayerNorm[Self.dtype]
    var mlp: MLP[Self.dtype]
    var training: Bool

    def __init__(out self, n_embd: Int):
        self.training = True
        self.ln_1 = LayerNorm[Self.dtype](n_embd)
        self.attn_stub = Linear[Self.dtype](
            in_features=n_embd, out_features=n_embd, init_seed=42
        )
        self.ln_2 = LayerNorm[Self.dtype](n_embd)
        self.mlp = MLP[Self.dtype](n_embd, dropout_p=0.0, init_seed=42)

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def _forward[track_grad: Bool](
        mut self, x: Tensor[Self.dtype], sync: Bool
    ) -> Tensor[Self.dtype]:
        var h = self.ln_1(x, sync=sync)
        var attn_out = self.attn_stub(h, sync=sync)
        var residual_1 = x.__add__[track_grad=track_grad](attn_out)
        var h2 = self.ln_2(residual_1, sync=sync)
        var mlp_out = self.mlp(h2, sync=sync)
        return residual_1.__add__[track_grad=track_grad](mlp_out)

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.ln_1.parameters()
        var a = self.attn_stub.parameters()
        for p in range(len(a)):
            params.append(a[p])
        var ln2 = self.ln_2.parameters()
        for p in range(len(ln2)):
            params.append(ln2[p])
        var m = self.mlp.parameters()
        for p in range(len(m)):
            params.append(m[p])
        return params^

    def train(mut self):
        self.training = True
        self.ln_1.train()
        self.attn_stub.train()
        self.ln_2.train()
        self.mlp.train()

    def eval(mut self):
        self.training = False
        self.ln_1.eval()
        self.attn_stub.eval()
        self.ln_2.eval()
        self.mlp.eval()


@fieldwise_init
struct StubModel[dtype: DType, index_dtype: DType]:
    var emb: DropEmbed[Self.dtype, Self.index_dtype]
    var blocks: List[StubBlock[Self.dtype]]
    var training: Bool

    def __init__(out self, V: Int, C: Int, n_blocks: Int):
        self.emb = DropEmbed[Self.dtype, Self.index_dtype](V, C)
        self.blocks = List[StubBlock[Self.dtype]]()
        self.training = True
        for _ in range(n_blocks):
            self.blocks.append(StubBlock[Self.dtype](C))

    def __call__(mut self, tokens: Tensor[Self.index_dtype]) -> Tensor[Self.dtype]:
        var x = self.emb(tokens)
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            x = block(x)
        var wt = self.emb.wte_weight().transpose(IntArray(1, 0))
        return Matmul[Self.dtype].forward[track_grad=True](x, wt)

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.emb.parameters()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            var bp = block.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        return params^

    def train(mut self):
        self.training = True
        self.emb.train()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.train()


def test_residual_dispatch_block_keeps_grad() raises:
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var N_BLOCKS = 2

    var tokens = Tensor[DType.int64].d2([[1, 3, 5], [2, 4, 6]])
    var targets = Tensor[DType.int64].d2([[3, 5, 1], [4, 6, 2]])

    var model = StubModel[dtype, DType.int64](V, C, N_BLOCKS)
    model.train()

    var params = model.parameters()
    var snapshots = List[Tensor[dtype]]()
    for i in range(len(params)):
        snapshots.append(params[i][].clone())

    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var logits = model(tokens)
    var loss = criterion(logits.permute([0, 2, 1]), targets)
    log_debug("rung 4: loss=" + String(Float64(loss.item())))

    optimizer.zero_grad()
    loss.backward()

    var emb_max = grad_max_abs(model.emb.wte.weight)
    log_debug("rung 4: emb grad_max=" + String(emb_max))
    for i in range(N_BLOCKS):
        ref block = model.blocks[i]
        log_debug(
            "rung 4: blocks[" + String(i) + "].mlp.c_fc grad_max="
            + String(grad_max_abs(block.mlp.c_fc.weight))
        )

    optimizer.step()

    var any_unchanged = False
    for i in range(len(params)):
        var snap = snapshots[i]
        var param = params[i][]
        var changed = False
        for j in range(snap.num_elements()):
            var diff = Float64(snap.get(j)) - Float64(param.get(j))
            if diff > 1e-6 or diff < -1e-6:
                changed = True
                break
        if not changed:
            any_unchanged = True

    assert_true(
        emb_max > 1e-9,
        "rung 4: residual/training-dispatch block severed upstream gradients",
    )
    assert_true(
        not any_unchanged,
        "rung 4: some parameter did not move after optimizer.step()",
    )


# ─── Rung 5: the real SelfAttention in the loop ────────────────────────
# WHAT: rung 4 stubbed attention with a Linear. This rung swaps in the real
#   `SelfAttention` (fused QKV, head split, causal mask, output proj)
#   inside the identical block/model harness.
# HOW: same asserts as rung 4, plus per-block attention-parameter grad dumps.
# FAILURE MEANS: attention breaks the graph specifically in this composed
#   call chain (it passes its own isolated suite) — suspect state that only
#   differs here: causal-mask caching across List elements, or the
#   reshape+permute q/k/v strided-view path interacting with grad tracking.


@fieldwise_init
struct AttnBlock[dtype: DType](LayerTrait):
    comptime TAG = 10102
    comptime InputDType = Self.dtype
    comptime OutputDType = Self.dtype
    var ln_1: LayerNorm[Self.dtype]
    var attn: SelfAttention[Self.dtype]
    var ln_2: LayerNorm[Self.dtype]
    var mlp: MLP[Self.dtype]
    var training: Bool

    def __init__(out self, n_embd: Int, n_head: Int):
        self.training = True
        self.ln_1 = LayerNorm[Self.dtype](n_embd)
        # Explicit xavier (not the standalone "uniform" default): rungs 5-7
        # mirror a default-constructed GPTModel, which is all-xavier
        # end to end via init_method forwarding (see GPTModel docs).
        self.attn = SelfAttention[Self.dtype](
            n_embd, n_head, dropout_p=0.0, init_seed=42, init_method="xavier"
        )
        self.ln_2 = LayerNorm[Self.dtype](n_embd)
        self.mlp = MLP[Self.dtype](
            n_embd, dropout_p=0.0, init_seed=42, init_method="xavier"
        )

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def _forward[track_grad: Bool](
        mut self, x: Tensor[Self.dtype], sync: Bool
    ) -> Tensor[Self.dtype]:
        var h = self.ln_1(x, sync=sync)
        var attn_out = self.attn(h, sync=sync)
        var residual_1 = x.__add__[track_grad=track_grad](attn_out)
        var h2 = self.ln_2(residual_1, sync=sync)
        var mlp_out = self.mlp(h2, sync=sync)
        return residual_1.__add__[track_grad=track_grad](mlp_out)

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.ln_1.parameters()
        var a = self.attn.parameters()
        for p in range(len(a)):
            params.append(a[p])
        var ln2 = self.ln_2.parameters()
        for p in range(len(ln2)):
            params.append(ln2[p])
        var m = self.mlp.parameters()
        for p in range(len(m)):
            params.append(m[p])
        return params^

    def train(mut self):
        self.training = True
        self.ln_1.train()
        self.attn.train()
        self.ln_2.train()
        self.mlp.train()

    def eval(mut self):
        self.training = False
        self.ln_1.eval()
        self.attn.eval()
        self.ln_2.eval()
        self.mlp.eval()


@fieldwise_init
struct AttnModel[dtype: DType, index_dtype: DType]:
    var emb: DropEmbed[Self.dtype, Self.index_dtype]
    var blocks: List[AttnBlock[Self.dtype]]
    var training: Bool

    def __init__(out self, V: Int, C: Int, n_head: Int, n_blocks: Int):
        self.emb = DropEmbed[Self.dtype, Self.index_dtype](V, C)
        self.blocks = List[AttnBlock[Self.dtype]]()
        self.training = True
        for _ in range(n_blocks):
            self.blocks.append(AttnBlock[Self.dtype](C, n_head))

    def __call__(mut self, tokens: Tensor[Self.index_dtype]) -> Tensor[Self.dtype]:
        var x = self.emb(tokens)
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            x = block(x)
        var wt = self.emb.wte_weight().transpose(IntArray(1, 0))
        return Matmul[Self.dtype].forward[track_grad=True](x, wt)

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.emb.parameters()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            var bp = block.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        return params^

    def train(mut self):
        self.training = True
        self.emb.train()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.train()


def test_real_attention_block_keeps_grad() raises:
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var N_HEAD = 2
    var N_BLOCKS = 2

    var tokens = Tensor[DType.int64].d2([[1, 3, 5], [2, 4, 6]])
    var targets = Tensor[DType.int64].d2([[3, 5, 1], [4, 6, 2]])

    var model = AttnModel[dtype, DType.int64](V, C, N_HEAD, N_BLOCKS)
    model.train()

    var params = model.parameters()
    var snapshots = List[Tensor[dtype]]()
    for i in range(len(params)):
        snapshots.append(params[i][].clone())

    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var logits = model(tokens)
    var loss = criterion(logits.permute([0, 2, 1]), targets)
    log_debug("rung 5: loss=" + String(Float64(loss.item())))

    optimizer.zero_grad()
    loss.backward()

    var emb_max = grad_max_abs(model.emb.wte.weight)
    log_debug("rung 5: emb grad_max=" + String(emb_max))
    for i in range(N_BLOCKS):
        ref block = model.blocks[i]
        var ap = block.attn.parameters()
        for p in range(len(ap)):
            log_debug(
                "rung 5: blocks[" + String(i) + "].attn.param[" + String(p)
                + "] grad_max=" + String(grad_max_abs(ap[p][]))
            )

    optimizer.step()

    var any_unchanged = False
    for i in range(len(params)):
        var snap = snapshots[i]
        var param = params[i][]
        var changed = False
        for j in range(snap.num_elements()):
            var diff = Float64(snap.get(j)) - Float64(param.get(j))
            if diff > 1e-6 or diff < -1e-6:
                changed = True
                break
        if not changed:
            any_unchanged = True

    assert_true(
        emb_max > 1e-9,
        "rung 5: real SelfAttention severed upstream gradients",
    )
    assert_true(
        not any_unchanged,
        "rung 5: some parameter did not move after optimizer.step()",
    )


# ─── Rung 6: the real tok+pos embedding add ──────────────────────────────────
# WHAT: all previous rungs embedded bare token ids. The real `GPTEmbedding`
#   adds a positional table (`tok + pos`) through its OWN `track_grad`
#   dispatch branch — and `wpe` was the OTHER field that came back zero in
#   the original bug. This rung is the last structural gap to the real model.
# HOW: `MiniGPTEmb` mirrors `GPTEmbedding._forward` exactly (real wte + wpe,
#   real `position_ids`, `Tensor.add[track_grad=...]`, dropout); asserts BOTH
#   wte and wpe grads plus full parameter movement.
# FAILURE MEANS: the break is in the tok+pos add path — bisect by swapping
#   the static `Tensor.add[track_grad=...]` for the instance
#   `tok.__add__[track_grad=...]` (the residual path rung 4 already cleared).


@fieldwise_init
struct MiniGPTEmb[dtype: DType, index_dtype: DType](LayerTrait):
    comptime TAG = 10103
    comptime InputDType = Self.index_dtype
    comptime OutputDType = Self.dtype

    var wte: Embedding[Self.dtype, Self.index_dtype]
    var wpe: PositionalEmbedding[Self.dtype, Self.index_dtype]
    var drop: Dropout[Self.dtype]
    var n_ctx: Int
    var training: Bool

    def __init__(out self, n_vocab: Int, n_ctx: Int, n_embd: Int):
        self.n_ctx = n_ctx
        self.training = True
        # Explicit xavier throughout (overriding Embedding's standalone
        # "normal" default): a default GPTModel forwards one scheme to both
        # tables, so the mirror does the same.
        self.wte = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=n_vocab,
            embedding_dim=n_embd,
            init_seed=42,
            init_method="xavier",
        )
        self.wpe = PositionalEmbedding[Self.dtype, Self.index_dtype](
            n_ctx, n_embd, init_seed=42, init_method="xavier"
        )
        self.drop = Dropout[Self.dtype](Scalar[Self.dtype](0.0))

    def __call__(
        mut self, x: Tensor[Self.index_dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def _forward[track_grad: Bool](
        mut self, x: Tensor[Self.index_dtype], sync: Bool
    ) -> Tensor[Self.dtype]:
        var B = x.shape()[0]
        var T = x.shape()[1]
        var ids: Tensor[Self.index_dtype] = Tensor[Self.index_dtype].zeros(
            Shape(B, T)
        )
        try:
            ids = PositionalEmbedding[
                Self.dtype, Self.index_dtype
            ].position_ids(self.n_ctx, B, T)
        except e:
            # `position_ids` only raises on invalid geometry; our dims are
            # fixed and valid, so reaching here means the helper itself is
            # broken. LayerTrait.__call__ is non-raising, hence panic (which
            # aborts the suite) instead of a catchable test failure.
            print(e)
            panic("MiniGPTEmb.position_ids failed")
        var tok = self.wte(x, sync=sync)
        var pos = self.wpe(ids, sync=sync)
        var x_out = Tensor[Self.dtype].add[track_grad=track_grad](
            tok, pos, sync=sync
        )
        return self.drop(x_out, sync=sync)

    def wte_weight(ref self) -> ref[self.wte.weight] Tensor[Self.dtype]:
        return self.wte.weight

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.wte.parameters()
        var wpe = self.wpe.parameters()
        for p in range(len(wpe)):
            params.append(wpe[p])
        return params^

    def train(mut self):
        self.training = True
        self.wte.train()
        self.wpe.train()
        self.drop.train()

    def eval(mut self):
        self.training = False
        self.wte.eval()
        self.wpe.eval()
        self.drop.eval()


@fieldwise_init
struct MiniGPT[dtype: DType, index_dtype: DType]:
    var wte_wpe: MiniGPTEmb[Self.dtype, Self.index_dtype]
    var blocks: List[AttnBlock[Self.dtype]]
    var training: Bool

    def __init__(out self, V: Int, n_ctx: Int, C: Int, n_head: Int, n_blocks: Int):
        self.wte_wpe = MiniGPTEmb[Self.dtype, Self.index_dtype](V, n_ctx, C)
        self.blocks = List[AttnBlock[Self.dtype]]()
        self.training = True
        for _ in range(n_blocks):
            self.blocks.append(AttnBlock[Self.dtype](C, n_head))

    def __call__(mut self, tokens: Tensor[Self.index_dtype]) -> Tensor[Self.dtype]:
        var x = self.wte_wpe(tokens)
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            x = block(x)
        var wt = self.wte_wpe.wte_weight().transpose(IntArray(1, 0))
        return Matmul[Self.dtype].forward[track_grad=True](x, wt)

    def parameters(ref self) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.wte_wpe.parameters()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            var bp = block.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        return params^

    def train(mut self):
        self.training = True
        self.wte_wpe.train()
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.train()


def test_tokpos_embedding_keeps_grad() raises:
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var N_CTX = 4
    var N_HEAD = 2
    var N_BLOCKS = 2

    var tokens = Tensor[DType.int64].d2([[1, 3, 5, 2], [2, 4, 6, 1]])
    var targets = Tensor[DType.int64].d2([[3, 5, 2, 1], [4, 6, 1, 2]])

    var model = MiniGPT[dtype, DType.int64](V, N_CTX, C, N_HEAD, N_BLOCKS)
    model.train()

    var params = model.parameters()
    var snapshots = List[Tensor[dtype]]()
    for i in range(len(params)):
        snapshots.append(params[i][].clone())

    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var logits = model(tokens)
    var loss = criterion(logits.permute([0, 2, 1]), targets)
    log_debug("rung 6: loss=" + String(Float64(loss.item())))

    optimizer.zero_grad()
    loss.backward()

    var wte_max = grad_max_abs(model.wte_wpe.wte.weight)
    log_debug("rung 6: wte grad_max=" + String(wte_max))
    var wpe_params = model.wte_wpe.wpe.parameters()
    var wpe_max: Float64 = 0.0
    for p in range(len(wpe_params)):
        var m = grad_max_abs(wpe_params[p][])
        log_debug("rung 6: wpe.param[" + String(p) + "] grad_max=" + String(m))
        if m > wpe_max:
            wpe_max = m

    optimizer.step()

    var any_unchanged = False
    for i in range(len(params)):
        var snap = snapshots[i]
        var param = params[i][]
        var changed = False
        for j in range(snap.num_elements()):
            var diff = Float64(snap.get(j)) - Float64(param.get(j))
            if diff > 1e-6 or diff < -1e-6:
                changed = True
                break
        if not changed:
            any_unchanged = True

    assert_true(wte_max > 1e-9, "rung 6: tok+pos add severed wte gradients")
    assert_true(wpe_max > 1e-9, "rung 6: wpe received zero gradient")
    assert_true(
        not any_unchanged,
        "rung 6: some parameter did not move after optimizer.step()",
    )


# ─── Rung 7: the full 60-step loop stays live ────────────────────────────────
# WHAT: rungs 1-6 check a SINGLE step. Some corruptions only appear across
#   repeated forward/backward/step cycles (stale views, gradbox reuse,
#   pointer invalidation). This rung runs the real 60-step training loop on
#   the fully-mirrored stack and fails if EITHER the loss doesn't learn OR
#   the wte gradient goes stale at any step after being alive.
# HOW: tracks wte grad magnitude every step; records the first step where it
#   drops to ~0 after being nonzero (that pinpoints construction-vs-loop
#   corruption). Asserts no such step exists and loss halves.
# FAILURE AT STEP 0: the single-step graph is broken — but rung 6 covers
#   that, so look there first.
# FAILURE AT STEP k>0: the loop corrupts state — suspect List reallocations
#   invalidating the Pointers `parameters()` captured once before the loop,
#   or SGD caching stale storage. Check for anything mutating `self.blocks`
#   mid-training.
# DIVERGENCE IS NOT SEVERING: if grads first EXPLODE and the loss goes NaN,
#   the learning rate is simply too hot for this stack (a hyperparameter
#   artifact, not a graph bug). Severing reads exactly 0.0 from step 0 with
#   a finite loss. The lr below is deliberately small to keep this rung in
#   the stable regime; the per-10-step finite-loss assert names divergence
#   explicitly so the two failure modes can't be confused.


def test_sixty_step_loop_stays_live() raises:
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var N_CTX = 4
    var N_HEAD = 2
    var N_BLOCKS = 2

    var tokens = Tensor[DType.int64].d2([[1, 3, 5, 2], [2, 4, 6, 1]])
    var targets = Tensor[DType.int64].d2([[3, 5, 2, 1], [4, 6, 1, 2]])

    var model = MiniGPT[dtype, DType.int64](V, N_CTX, C, N_HEAD, N_BLOCKS)
    model.train()

    var params = model.parameters()
    var optimizer = SGD[dtype](params, lr=0.05)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var loss0: Float64 = 0.0
    var lossN: Float64 = 0.0
    var first_stale_step: Int = -1
    var prev_wte_max: Float64 = -1.0

    for step in range(60):
        var logits = model(tokens)
        var loss = criterion(logits.permute([0, 2, 1]), targets)

        var v = Float64(loss.item())
        if step == 0:
            loss0 = v
        lossN = v

        optimizer.zero_grad()
        loss.backward()

        var wte_max = grad_max_abs(model.wte_wpe.wte.weight)
        if (
            step > 0
            and prev_wte_max > 1e-9
            and wte_max <= 1e-9
            and first_stale_step == -1
        ):
            first_stale_step = step
            log_debug(
                "rung 7: wte grad went stale at step "
                + String(step)
                + " (was "
                + String(prev_wte_max)
                + ")"
            )
        prev_wte_max = wte_max

        if step % 10 == 0 or step == 59:
            assert_true(
                v == v,
                "rung 7: loss went NaN at step " + String(step)
                + " — DIVERGENCE (lr too hot), not graph severing",
            )
            log_debug(
                "rung 7: step " + String(step) + " loss=" + String(v)
                + " wte_grad_max=" + String(wte_max)
            )

        optimizer.step()

    log_debug(
        "rung 7: loss0=" + String(loss0) + " lossN=" + String(lossN)
        + " ratio=" + String(lossN / loss0)
    )
    assert_true(
        first_stale_step == -1,
        "rung 7: wte grad went stale mid-loop at step "
        + String(first_stale_step),
    )
    assert_true(
        lossN < loss0 * 0.5, "rung 7: loss did not halve over 60 steps"
    )


# ─── ordered entrypoint ──────────────────────────────────────────────────────
# Runs rungs 0-7 top-to-bottom: init vocabulary first (the actual lesson),
# then the structural bisection in the order it was originally debugged.


def main() raises:
    print("=== test_init_vocab_all_schemes_live ===")
    try:
        test_init_vocab_all_schemes_live()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_direct_and_field_transpose_keep_graph ===")
    try:
        test_direct_and_field_transpose_keep_graph()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_nested_holder_and_accessor_keep_graph ===")
    try:
        test_nested_holder_and_accessor_keep_graph()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_list_loop_keeps_upstream_grad ===")
    try:
        test_list_loop_keeps_upstream_grad()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_residual_dispatch_block_keeps_grad ===")
    try:
        test_residual_dispatch_block_keeps_grad()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_real_attention_block_keeps_grad ===")
    try:
        test_real_attention_block_keeps_grad()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_tokpos_embedding_keeps_grad ===")
    try:
        test_tokpos_embedding_keeps_grad()
        print("PASS")
    except e:
        print("FAIL:", e)

    print("=== test_sixty_step_loop_stays_live ===")
    try:
        test_sixty_step_loop_stays_live()
        print("PASS")
    except e:
        print("FAIL:", e)
