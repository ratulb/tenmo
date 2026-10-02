"""SelfAttention — multi-head self-attention, causal by default.

Forward over `x: (B, T, C)` -> `(B, T, C)`, `h = n_head`, `dh = C // h`:

    c_attn -> (B,T,3C) sliced into q/k/v (fused, GPT-2 layout)
    reshape+permute -> (B,h,T,dh); scores = q @ k^T / sqrt(dh)
    causal mask -> -inf where j > i, when `causal`
    softmax(-1), dropout, attn @ v
    permute/reshape -> (B,T,C), then c_proj

`LayerTrait` composition over existing ops — no new autograd primitive.
"""

from std.math import sqrt
from std.utils.numerics import neg_inf
from .tensor import Tensor
from .shared.shapes import Shape
from .net import Linear
from .dropout import Dropout
from .layer_trait import LayerTrait
from .gpu.device import GPU
from .named_parameter import NamedParameter
from .shared.panic import panic

@fieldwise_init
struct SelfAttention[dtype: DType](LayerTrait):
    """Multi-head causal self-attention over a sequence of `n_embd`-dim vectors.

    Two learnable sub-layers (`c_attn`, `c_proj`, both `Linear`) plus
    `dattn`. All math is in `_forward` via existing ops — no custom kernels.

    Field names match the GPT-2 / HuggingFace `transformers` convention, so
    pretrained GPT-2 weights load directly.

    Args:
        n_embd: Hidden/embedding dimension (C). Must be divisible by n_head.
        n_head: Number of attention heads (h); head_dim = n_embd // n_head.
        dropout_p: Dropout on the softmax output (training only).
        init_seed / init_method: Forwarded to both inner Linears.
        causal: True = GPT-style (query `i` sees keys `j <= i`); False =
            BERT-style bidirectional. Runtime flag, so both share one
            `_forward`.
        qkv_bias: If True, `c_attn` keeps its QKV bias (GPT-2 layout).
    """

    comptime InputDType = Self.dtype

    var c_attn: Linear[Self.dtype]
    var c_proj: Linear[Self.dtype]
    var dattn: Dropout[Self.dtype]
    var n_head: Int
    var head_dim: Int
    var n_embd: Int
    var inv_scale: Scalar[Self.dtype]
    # Lazily-built per-`T` (T,T) causal mask. Device-resident, so
    # `to_gpu()`/`to_cpu()` must clear it — see those methods.
    var causal_mask_cache: Optional[Tensor[DType.bool]]
    var training: Bool
    var causal: Bool

    def __init__(
        out self,
        n_embd: Int,
        n_head: Int,
        dropout_p: Float32 = 0.0,
        init_seed: Optional[Int] = None,
        init_method: String = "uniform",
        causal: Bool = True,
        qkv_bias: Bool = False,
    ):
        """Create a causal self-attention module.

        Panics (not `raises`) on a misconfiguration: these are programming
        errors, not expected runtime failures. `n_embd % n_head` must be 0
        because `head_dim` is an exact reshape target in `_forward`.

        Both Linears get the same `init_seed`/`init_method`, which configures
        them identically but does not give them identical weights (their
        shapes differ).

        `qkv_bias` (default False) is the library default: `c_attn` has no
        bias parameter at all and `named_parameters` omits the key. Pass
        True for the GPT-2 layout, which biases every projection.
        `c_proj` always keeps its bias.
        """
        if n_embd < 1:
            panic(
                "SelfAttention: n_embd must be >= 1, got", String(n_embd)
            )
        if n_head < 1:
            panic(
                "SelfAttention: n_head must be >= 1, got", String(n_head)
            )
        if n_embd % n_head != 0:
            panic(
                "SelfAttention: n_embd",
                String(n_embd),
                "not divisible by n_head",
                String(n_head),
            )
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.inv_scale = Scalar[Self.dtype](1.0 / sqrt(Float64(self.head_dim)))
        self.causal_mask_cache = None
        self.training = True
        self.causal = causal
        self.c_attn = Linear[Self.dtype](
            in_features=n_embd,
            out_features=3 * n_embd,
            init_seed=init_seed,
            init_method=init_method,
            bias=qkv_bias,
        )
        self.c_proj = Linear[Self.dtype](
            in_features=n_embd,
            out_features=n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.dattn = Dropout[Self.dtype](Scalar[Self.dtype](dropout_p))

    def __call__(
        mut self,
        x: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Run self-attention. `x` must be `(B, T, n_embd)`.

        Fetches the causal mask via `_base_mask` (None when not `causal`),
        then dispatches on `self.training` to `_forward[track_grad=...]`.

        For padded batches use `forward_padded` — `LayerTrait` pins this
        method's `(x, sync)` signature, so it cannot grow a mask parameter.

        Not `raises`, so every op called here must not raise (errors on these
        paths are `panic`). Mutates `causal_mask_cache`, so concurrent calls
        on one instance need external synchronization.
        """
        var T = x.shape()[1]
        var mask = self._base_mask(T, sync=sync)
        if self.training:
            return self._forward[track_grad=True](x, mask, sync=sync)
        else:
            return self._forward[track_grad=False](x, mask, sync=sync)

    def forward_padded(
        mut self,
        x: Tensor[Self.dtype],
        pad_mask: Tensor[DType.bool],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Run attention with a key-padding mask. `x` is `(B, T, n_embd)`,
        `pad_mask` is `(B, T)` bool (`True` = real token).

        Padded keys get `-inf` before softmax, so they carry exactly zero
        weight. Queries are never masked — padded query rows produce garbage
        nobody reads.

        Same as `__call__` plus the mask, which `_forward` applies as a
        second `where` after the causal mask (or on the raw scores when
        bidirectional). Separate method because `LayerTrait` pins
        `__call__`'s signature. Per-batch, so it never enters the cache.
        """
        var T = x.shape()[1]
        var mask = self._base_mask(T, sync=sync)
        if self.training:
            return self._forward[track_grad=True](
                x, mask, sync=sync, pad_mask=pad_mask
            )
        else:
            return self._forward[track_grad=False](
                x, mask, sync=sync, pad_mask=pad_mask
            )

    def _base_mask(
        mut self, T: Int, sync: Bool
    ) -> Optional[Tensor[DType.bool]]:
        """The (T,T) base mask: cached causal mask when `self.causal` is
        True, `None` otherwise (bidirectional mode — every query attends
        every key, so the mask `where` in `_forward` would be the identity
        and is skipped entirely: no all-allow tensor is even built).

        Returning `None` instead of an all-True table saves both the (T,T)
        alloc and a full O(B*h*T^2) `where` pass over the scores per call —
        pure win for BERT-side training, where every call is bidirectional.
        The shared `_forward` needs no `if bidirectional` branch for the
        math itself — one code path, two behaviors, chosen by mask presence.
        Non-`raises` like `_causal_mask` (no fallible ops on either path).
        """
        if self.causal:
            return self._causal_mask(T, sync=sync)
        return None

    def _causal_mask(mut self, T: Int, sync: Bool) -> Tensor[DType.bool]:
        """Fetch (or lazily build and cache) the (T,T) boolean causal mask.

        `mask[i, j] == True` iff `j <= i`. Built bool-native — `tril` is
        dtype-generic, so no float `ones` + `> 0.5` step.

        Safe to cache: the mask depends only on `T`, and is built with
        `track_grad=False`, so it is a leaf with no graph state to go stale.
        Keyed off the cached tensor's own shape, so there is no second
        field to drift.

        Device-resident — `to_gpu()`/`to_cpu()` must clear it. A plain
        `var out = self` copy would carry the old mask into a `where` whose
        operands are on different devices.
        """
        if self.causal_mask_cache:
            if self.causal_mask_cache.value().shape()[0] == T:
                return self.causal_mask_cache.value()
        var ones_m = Tensor[DType.bool].ones(Shape(T, T))
        var cond = ones_m.tril[track_grad=False](diagonal=0, sync=sync)
        self.causal_mask_cache = cond
        return cond

    def _forward[
        track_grad: Bool
    ](
        mut self,
        x: Tensor[Self.dtype],
        mask: Optional[Tensor[DType.bool]] = None,
        sync: Bool = True,
        pad_mask: Optional[Tensor[DType.bool]] = None,
    ) -> Tensor[Self.dtype]:
        """The attention computation (see the module docstring for the recipe).

        `track_grad` is comptime and propagated into every sub-op, so the
        whole call is uniformly recording or uniformly not.

        `mask` is the (T,T) mask prepared by `__call__`, or `None` when not
        causal. Read-only `self`: the mask cache is hoisted into
        `__call__`/`_causal_mask`, which is why only those are `mut self`.

        `pad_mask` is the optional per-batch key-padding mask.
        """
        comptime assert Self.dtype.is_floating_point()
        var B = x.shape()[0]
        var T = x.shape()[1]
        var C = x.shape()[2]
        var h = self.n_head
        var dh = self.head_dim

        # Fused QKV -> q, k, v (each (B, T, C)); columns are [q | k | v].
        var qkv = self.c_attn(x, sync=sync)  # (B, T, 3C)
        var q = qkv.slice[track_grad=track_grad](
            0, C, axis=2
        )  # (B, T, C)   -- columns [0, C)
        var k = qkv.slice[track_grad=track_grad](
            C, 2 * C, axis=2
        )  # (B, T, C)   -- columns [C, 2C)
        var v = qkv.slice[track_grad=track_grad](
            2 * C, 3 * C, axis=2
        )  # (B, T, C)   -- columns [2C, 3C)

        # Head split: (B,T,C) -> (B,h,T,dh). `k_h` uses [0,2,3,1] so it lands
        # already transposed as (B,h,dh,T), saving a separate transpose.
        # Zero-copy views, so these are strided — `matmul` handles that.
        var q_h = q.reshape[track_grad=track_grad](
            Shape(B, T, h, dh), sync=sync
        ).permute[track_grad=track_grad]([0, 2, 1, 3], sync=sync)
        var k_h = k.reshape[track_grad=track_grad](
            Shape(B, T, h, dh), sync=sync
        ).permute[track_grad=track_grad]([0, 2, 3, 1], sync=sync)
        var v_h = v.reshape[track_grad=track_grad](
            Shape(B, T, h, dh), sync=sync
        ).permute[track_grad=track_grad]([0, 2, 1, 3], sync=sync)

        # Scaled dot-product scores: (B,h,T,dh) @ (B,h,dh,T) -> (B,h,T,T).
        var scores = Tensor[Self.dtype].matmul[track_grad=track_grad](
            q_h, k_h, sync=sync
        )
        # `self.inv_scale` (== 1/sqrt(head_dim)) is cached in `__init__`.
        if sync:
            scores = scores.__mul__[track_grad=track_grad, sync=True](
                self.inv_scale
            )
        else:
            scores = scores.__mul__[track_grad=track_grad, sync=False](
                self.inv_scale
            )

        # Causal mask: keep score(i,j) iff j <= i, else -inf. `mask` is None
        # when not causal, where the `where` would be the identity.
        if mask:
            scores = Tensor[Self.dtype].where[track_grad=track_grad](
                mask.value(), scores, neg_inf[Self.dtype](), sync=sync
            )

        # Key-padding mask: (B,T) reshaped to (B,1,1,T) so `where`
        # broadcasts over heads and queries. Never enters the cache.
        if pad_mask:
            var n_batch = scores.shape()[0]
            var n_keys = scores.shape()[3]
            var pad4 = pad_mask.value().reshape[track_grad=False](
                Shape(n_batch, 1, 1, n_keys), sync=sync
            )
            scores = Tensor[Self.dtype].where[track_grad=track_grad](
                pad4, scores, neg_inf[Self.dtype](), sync=sync
            )

        # Row softmax, dropout, then attn @ v.
        var attn = scores.softmax[track_grad=track_grad](
            [-1], sync=sync
        )  # (B, h, T, T)
        attn = self.dattn(attn, sync=sync)
        var out = Tensor[Self.dtype].matmul[track_grad=track_grad](
            attn, v_h, sync=sync
        )  # (B, h, T, dh)

        # Merge heads back to (B,T,C), then c_proj.
        var merged = out.permute[track_grad=track_grad](
            [0, 2, 1, 3], sync=sync
        ).reshape[track_grad=track_grad](Shape(B, T, C), sync=sync)
        return self.c_proj(merged, sync=sync)

    def forward_step(
        mut self,
        x_1: Tensor[Self.dtype],
        k_prev: Optional[Tensor[Self.dtype]],
        v_prev: Optional[Tensor[Self.dtype]],
        sync: Bool = True,
        pad_mask: Optional[Tensor[DType.bool]] = None,
    ) -> Tuple[Tensor[Self.dtype], Tensor[Self.dtype], Tensor[Self.dtype]]:
        """Single-token inference step with an external K/V cache.

        `x_1` is `(B, 1, C)`; `k_prev`/`v_prev` are the cache so far as
        `(B, h, S, dh)`, or `None` on the first step (`Shape(0)` is illegal,
        so emptiness is `None`). Projects the new token, appends its heads to
        the cache (the step-local QKV buffer dies at return, so the cache
        must own its storage), then attends over the full key set with NO
        causal mask — every cached key is at a position `<=` the query's, so
        the mask row would be all-allow. The result is the last row
        `_forward` would compute for the same prefix.

        Inference-only: all ops are `track_grad=False` and `dattn` is the
        eval no-op (callers enforce eval mode). Returns
        `(out, k_full, v_full)`.
        """
        comptime assert Self.dtype.is_floating_point()
        var B = x_1.shape()[0]
        var C = x_1.shape()[2]
        var h = self.n_head
        var dh = self.head_dim

        var qkv = self.c_attn(x_1, sync=sync)  # (B, 1, 3C)
        var q = qkv.slice[track_grad=False](0, C, axis=2)
        var k = qkv.slice[track_grad=False](C, 2 * C, axis=2)
        var v = qkv.slice[track_grad=False](2 * C, 3 * C, axis=2)

        var q_h = q.reshape[track_grad=False](
            Shape(B, 1, h, dh), sync=sync
        ).permute[track_grad=False]([0, 2, 1, 3], sync=sync)
        var k_new = k.reshape[track_grad=False](
            Shape(B, 1, h, dh), sync=sync
        ).permute[track_grad=False]([0, 2, 1, 3], sync=sync)
        var v_new = v.reshape[track_grad=False](
            Shape(B, 1, h, dh), sync=sync
        ).permute[track_grad=False]([0, 2, 1, 3], sync=sync)

        var k_full = k_new.clone()
        var v_full = v_new.clone()
        if k_prev and v_prev:
            var k_pair = List[Tensor[Self.dtype]]()
            k_pair.append(k_prev.value())
            k_pair.append(k_new)
            k_full = Tensor[Self.dtype].concat[track_grad=False](
                k_pair^, axis=2, sync=sync
            )
            var v_pair = List[Tensor[Self.dtype]]()
            v_pair.append(v_prev.value())
            v_pair.append(v_new)
            v_full = Tensor[Self.dtype].concat[track_grad=False](
                v_pair^, axis=2, sync=sync
            )

        var kT = k_full.permute[track_grad=False]([0, 1, 3, 2], sync=sync)
        var scores = Tensor[Self.dtype].matmul[track_grad=False](
            q_h, kT, sync=sync
        )
        if sync:
            scores = scores.__mul__[track_grad=False, sync=True](self.inv_scale)
        else:
            scores = scores.__mul__[track_grad=False, sync=False](
                self.inv_scale
            )
        # Optional key-padding mask for padded-batch generation: (B,S) over
        # the cached keys, reshaped to (B,1,1,S). The causal mask stays
        # vacuous here (single query), so only padding applies.
        if pad_mask:
            var n_keys_step = scores.shape()[3]
            var pad4_step = pad_mask.value().reshape[track_grad=False](
                Shape(B, 1, 1, n_keys_step), sync=sync
            )
            scores = Tensor[Self.dtype].where[track_grad=False](
                pad4_step, scores, neg_inf[Self.dtype](), sync=sync
            )
        var attn = scores.softmax[track_grad=False]([-1], sync=sync)
        attn = self.dattn(attn, sync=sync)
        var out = Tensor[Self.dtype].matmul[track_grad=False](
            attn, v_full, sync=sync
        )
        var merged = out.permute[track_grad=False](
            [0, 2, 1, 3], sync=sync
        ).reshape[track_grad=False](Shape(B, 1, C), sync=sync)
        return (self.c_proj(merged, sync=sync), k_full^, v_full^)

    @always_inline
    def qkv_weight(
        ref self,
    ) -> ref[self.c_attn.weight] Tensor[Self.dtype]:
        """The fused `(n_embd, 3*n_embd)` QKV weight — a live alias into
        `c_attn.weight`, so writing through it mutates the module's weight.
        """
        return self.c_attn.weight

    # Delegation only, so generic code can treat `SelfAttention` like any
    # other `LayerTrait` layer.

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        """`c_attn`'s parameters followed by `c_proj`'s. Pointers, not
        copies, so an optimizer updating through this list updates the
        module's own storage. `Dropout` has no learnable parameters.
        """
        var params = self.c_attn.parameters()
        var proj = self.c_proj.parameters()
        for p in range(len(proj)):
            params.append(proj[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        """`parameters()` with dotted names — e.g. `named_parameters("attn.")`
        yields `attn.c_attn.weight`, `attn.c_attn.bias`,
        `attn.c_proj.weight`, `attn.c_proj.bias`, in that order.
        """
        var result = self.c_attn.named_parameters(prefix + "c_attn.")
        var proj = self.c_proj.named_parameters(prefix + "c_proj.")
        for p in range(len(proj)):
            result.append(proj[p])
        return result^

    def num_parameters(self) -> Int:
        """Both Linears' scalar counts, summed. Independent of `n_head` —
        head splitting reshapes the same weights rather than adding storage.
        """
        return self.c_attn.num_parameters() + self.c_proj.num_parameters()

    def train(mut self):
        """Training mode: activate dropout and propagate to both Linears.
        """
        self.training = True
        self.dattn.train()
        self.c_attn.train()
        self.c_proj.train()

    def eval(mut self):
        """Evaluation mode: disable dropout and propagate to both Linears.
        """
        self.training = False
        self.dattn.eval()
        self.c_attn.eval()
        self.c_proj.eval()

    def to_gpu(
        self, gpu: Optional[GPU] = None
    ) raises -> SelfAttention[Self.dtype]:
        """Copy with both Linears moved to GPU.

        `dattn` needs no handling: its fields are plain scalars and its
        Philox RNG derives each mask at launch time from `(seed,
        position)`, so `var out = self` already copies it correctly.

        `causal_mask_cache` MUST be reset — it is device-resident, and a
        plain copy would carry the old mask into a `where` whose operands
        are on different devices.
        """
        var out = self
        out.c_attn = self.c_attn.to_gpu(gpu=gpu)
        out.c_proj = self.c_proj.to_gpu(gpu=gpu)
        out.causal_mask_cache = None
        return out^

    def to_cpu(self) raises -> SelfAttention[Self.dtype]:
        """Copy with both Linears moved to CPU. Mirror of `to_gpu()`,
        including the required `causal_mask_cache` reset.
        """
        var out = self
        out.c_attn = self.c_attn.to_cpu()
        out.c_proj = self.c_proj.to_cpu()
        out.causal_mask_cache = None
        return out^
