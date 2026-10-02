"""
TinyStories generation demo.
==========================================================
Loads the stage-2 pilot checkpoint (`examples/data/pilot_best.npy`,
gitignored — 202 MB) into a fresh shrink-config model and
closes the prompt→text loop: mbpe-gpt2 encode -> `generate_cached`
(KV path) + `generate` (uncached) with the same seed -> id-for-id
agreement assert -> mbpe decode -> print.

Expectations: one epoch at eval loss 4.62 does NOT write coherent
stories — the sample is a plumbing proof (weights load, sampler,
cache, and decode compose on real 50k-vocab weights), not a quality
gate. Coherence is stage 4's job (CV-loss < 4.0 + grammaticality
screen). Assertions are structural: output length, prompt
prefix preserved, ids in range, cached == uncached.

Run from the repo root: `./example.sh tinystories_generate`.
"""

from bpe.tokenizer import Tokenizers
from tenmo.checkpoint import load_weights
from tenmo.generate import generate, generate_cached
from tenmo.gpt import GPTModel
from tenmo.tensor import Tensor


comptime N_VOCAB = 50257
comptime N_CTX = 256
comptime N_EMBD = 256
comptime N_HEAD = 8
comptime N_LAYER = 6
comptime N_PARAMS = 17670400
comptime CKPT_PATH = "examples/data/pilot_best.npy"
comptime PROMPT_TEXT = "Once upon a time"
comptime MAX_NEW = 60


def main() raises:
    # ---- Fresh shrink model, exact pilot pins, weights from disk ----
    var model = GPTModel[DType.float32](
        n_vocab=N_VOCAB,
        n_ctx=N_CTX,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout_p=0.1,
        tie_weights=True,
        init_method="xavier",
        init_seed=7,
        qkv_bias=True,  # pilot_best.npy was trained with biased QKV
    )
    if model.num_parameters() != N_PARAMS:
        raise Error("tinystories_generate: param count drift — pins changed?")
    load_weights(model, CKPT_PATH)
    print("loaded:", CKPT_PATH, "params:", model.num_parameters())

    # ---- Prompt -> ids (mbpe-gpt2, same tokenizer as training) ----
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var prompt_ids = gpt2.encode(PROMPT_TEXT)
    print("prompt:", PROMPT_TEXT, "->", len(prompt_ids), "ids")
    var ps = List[Scalar[DType.int64]](capacity=len(prompt_ids))
    for i in range(len(prompt_ids)):
        ps.append(Scalar[DType.int64](prompt_ids[i]))
    var prompt = Tensor[DType.int64].from_list[DType.int64](ps^)

    # ---- Cached + uncached, same seed -> must agree id-for-id ----
    var out_cached = generate_cached(
        model,
        prompt,
        max_new_tokens=MAX_NEW,
        temperature=0.8,
        top_k=50,
        init_seed=7,
    )
    var out_plain = generate(
        model,
        prompt,
        max_new_tokens=MAX_NEW,
        temperature=0.8,
        top_k=50,
        init_seed=7,
    )
    if out_cached.numels() != len(prompt_ids) + MAX_NEW:
        raise Error("tinystories_generate: cached length mismatch")
    if out_plain.numels() != out_cached.numels():
        raise Error("tinystories_generate: cached/uncached length mismatch")
    var gen = List[Int](capacity=out_cached.numels())
    for i in range(out_cached.numels()):
        var vc = Int(out_cached.get(i))
        if vc != Int(out_plain.get(i)):
            raise Error(
                "tinystories_generate: cached/uncached diverged at "
                + String(i)
            )
        if i < len(prompt_ids):
            if vc != prompt_ids[i]:
                raise Error("tinystories_generate: prompt prefix not preserved")
        if vc < 0 or vc >= N_VOCAB:
            raise Error("tinystories_generate: id out of vocab range")
        gen.append(vc)

    var text = gpt2.decode(gen^)
    print("---- continuation (1 epoch, plumbing only) ----")
    print(text)
    print("-----------------------------------------------")
    print("tinystories_generate passed: checkpoint -> text end to end.")
