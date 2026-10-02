"""Gate for `tenmo/generate.mojo`: sampler + uncached loop.

Tiny synthetic config (V=32, T=8, C=16, 1 layer): greedy/top-1 equal
argmax, top-k masks the tail across seeds, `generate` preserves the
prompt prefix, honors `max_new_tokens`/`end_id`/`n_ctx`, and repeats
bit-identically under a fixed seed: `generate_cached` reproduces
the uncached logits and id streams exactly, and refuses over-`n_ctx`
runs (no v1 cache sliding).
"""

from std.testing import assert_true, assert_equal, assert_raises, TestSuite
from tenmo.generate import sample, generate, generate_cached
from tenmo.gpt import GPTModel, KVCache
from tenmo.tensor import Tensor


def _tiny_model() -> GPTModel[DType.float32]:
    var model = GPTModel[DType.float32](
        n_vocab=32,
        n_ctx=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        tie_weights=True,
        init_seed=11,
    )
    model.train()
    return model^


def _prompt(vals: List[Int]) -> Tensor[DType.int64]:
    var s = List[Scalar[DType.int64]](capacity=len(vals))
    for i in range(len(vals)):
        s.append(Scalar[DType.int64](vals[i]))
    return Tensor[DType.int64].from_list[DType.int64](s^)


def test_sample_greedy_is_argmax() raises:
    print("Test 1: greedy and top_k=1 equal argmax")
    var logits = Tensor[DType.float32].randn(32, init_seed=3)
    var expected = Int(logits.argmax(axis=0).item())
    var g = sample(logits, temperature=0.0, top_k=50)
    var k1 = sample(logits, temperature=1.0, top_k=1, init_seed=5)
    print("  argmax:", expected, "greedy:", g, "top1:", k1)
    assert_equal(g, expected)
    assert_equal(k1, expected)


def test_sample_topk_masks_tail() raises:
    print("Test 2: top-k draws stay inside the kept set")
    var logits = Tensor[DType.float32].zeros(32) + Float32(-1000.0)
    logits[3] = 2.0
    logits[7] = 1.0
    for seed in range(20):
        var id = sample(logits, temperature=1.0, top_k=2, init_seed=seed)
        assert_true(
            id == 3 or id == 7, "top-k draw escaped the kept set"
        )
    var id0 = sample(logits, temperature=1.0, top_k=2, init_seed=4)
    var id1 = sample(logits, temperature=1.0, top_k=2, init_seed=4)
    assert_equal(id0, id1)


def test_generate_prefix_length_and_stop() raises:
    print("Test 3: generate preserves prefix, caps length, stops on end_id")
    var model = _tiny_model()
    var vals = List[Int](capacity=4)
    vals.append(1)
    vals.append(2)
    vals.append(3)
    vals.append(4)
    var prompt = _prompt(vals^)

    var out = generate(
        model, prompt, max_new_tokens=3, temperature=0.0, top_k=50
    )
    assert_equal(out.numels(), 7)
    for i in range(4):
        assert_equal(Int(out.get(i)), i + 1)

    # First greedy id becomes the stop token: capped run stops at 5.
    var probe = generate(
        model, prompt, max_new_tokens=1, temperature=0.0, top_k=50
    )
    var stop = Int(probe.get(4))
    var out2 = generate(
        model,
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        top_k=50,
        end_id=stop,
    )
    assert_equal(out2.numels(), 5)
    assert_equal(Int(out2.get(4)), stop)


def test_generate_slides_window_and_repeats() raises:
    print("Test 4: n_ctx slide works, fixed seed repeats exactly")
    var model = _tiny_model()
    var vals = List[Int](capacity=10)
    for i in range(10):
        vals.append((i * 3 + 1) % 32)
    var prompt = _prompt(vals^)

    var a = generate(
        model,
        prompt,
        max_new_tokens=3,
        temperature=1.0,
        top_k=5,
        init_seed=9,
    )
    var b = generate(
        model,
        prompt,
        max_new_tokens=3,
        temperature=1.0,
        top_k=5,
        init_seed=9,
    )
    assert_equal(a.numels(), 13)
    for i in range(13):
        assert_equal(Int(a.get(i)), Int(b.get(i)))
    for i in range(10):
        assert_equal(Int(a.get(i)), (i * 3 + 1) % 32)


def test_cached_logits_match_uncached() raises:
    print("Test 5: cached per-position logits match the batched forward")
    var model = _tiny_model()
    model.eval()

    var prow = List[Scalar[DType.int64]](capacity=6)
    for i in range(6):
        prow.append(Scalar[DType.int64]((i * 7 + 2) % 32))
    var batch = Tensor[DType.int64].from_list[DType.int64](prow^).reshape(
        1, 6
    )
    var ref_logits = model(batch)  # (1, 6, 32)

    var cache = KVCache[DType.float32]()
    var dmax: Float32 = 0.0
    for p in range(6):
        var one = List[Scalar[DType.int64]](capacity=1)
        one.append(Scalar[DType.int64]((p * 7 + 2) % 32))
        var x_1 = Tensor[DType.int64].from_list[DType.int64](one^).reshape(
            1, 1
        )
        var step = model.forward_step(x_1, p, cache)  # (1, 1, 32)
        var ref_row = ref_logits.slice(p, p + 1, 1, axis=1)
        # Contract: Tensor.get(i) is storage-level (ignores view
        # offsets), so slice views compare via strided subscripts.
        for j in range(32):
            var d = abs(step[0, 0, j] - ref_row[0, 0, j])
            if d > dmax:
                dmax = d
    print("  cached-vs-batched max abs diff:", dmax)
    assert_true(dmax < 1e-4, "cached logits diverge from batched forward")
    assert_equal(cache.seq_len(), 6)


def test_cached_stream_matches_uncached() raises:
    print("Test 6: cached and uncached id streams agree exactly")
    var model = _tiny_model()
    var vals = List[Int](capacity=4)
    vals.append(5)
    vals.append(17)
    vals.append(9)
    vals.append(23)
    var prompt = _prompt(vals^)

    var g = generate(
        model, prompt, max_new_tokens=4, temperature=0.0, top_k=50
    )
    var c = generate_cached(
        model, prompt, max_new_tokens=4, temperature=0.0, top_k=50
    )
    assert_equal(c.numels(), 8)
    for i in range(8):
        assert_equal(Int(c.get(i)), Int(g.get(i)))

    var gs = generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=5,
        init_seed=21,
    )
    var cs = generate_cached(
        model,
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=5,
        init_seed=21,
    )
    for i in range(8):
        assert_equal(Int(cs.get(i)), Int(gs.get(i)))

    # Over-n_ctx has no v1 cache sliding: must raise, not panic.
    var long_vals = List[Int](capacity=7)
    for i in range(7):
        long_vals.append(i)
    var long_prompt = _prompt(long_vals^)
    with assert_raises():
        _ = generate_cached(
            model, long_prompt, max_new_tokens=3, temperature=0.0
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll generate tests passed!")
