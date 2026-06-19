from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator


comptime F32 = DType.float32

def test_1d_single_sample() raises:
    var probs = Tensor[F32].d1([0.25, 0.25, 0.25, 0.25])
    var idx = probs.multinomial(num_samples=1)
    assert_true(idx.rank() == 1)
    assert_true(idx.shape()[0] == 1)
    var val = idx[0]
    assert_true(val >= 0 and val < 4)

def test_1d_replacement() raises:
    var probs = Tensor[F32].d1([0.1, 0.2, 0.4, 0.2, 0.1])
    var idx = probs.multinomial(num_samples=3, replacement=True)
    assert_true(idx.rank() == 1)
    assert_true(idx.shape()[0] == 3)
    for i in range(3):
        var val = idx[i]
        assert_true(val >= 0 and val < 5)

def test_2d_replacement() raises:
    var probs = Tensor[F32].d2([
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.4, 0.3],
    ])
    var idx = probs.multinomial(num_samples=2, replacement=True)
    assert_true(idx.rank() == 2)
    assert_true(idx.shape()[0] == 2)
    assert_true(idx.shape()[1] == 2)
    for b in range(2):
        for s in range(2):
            var val = idx[b, s]
            assert_true(val >= 0 and val < 4)

def test_without_replacement() raises:
    var probs = Tensor[F32].d1([0.1, 0.2, 0.3, 0.4])
    var idx = probs.multinomial(num_samples=3, replacement=False)
    assert_true(idx.rank() == 1)
    assert_true(idx.shape()[0] == 3)
    var s0 = idx[0]
    var s1 = idx[1]
    var s2 = idx[2]
    assert_true(s0 >= 0 and s0 < 4)
    assert_true(s1 >= 0 and s1 < 4)
    assert_true(s2 >= 0 and s2 < 4)
    assert_true(s0 != s1 and s0 != s2 and s1 != s2)

def test_2d_without_replacement() raises:
    var probs = Tensor[F32].d2([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ])
    var idx = probs.multinomial(num_samples=2, replacement=False)
    assert_true(idx.rank() == 2)
    assert_true(idx.shape()[0] == 2)
    assert_true(idx.shape()[1] == 2)
    for b in range(2):
        var s0 = idx[b, 0]
        var s1 = idx[b, 1]
        assert_true(s0 >= 0 and s0 < 4)
        assert_true(s1 >= 0 and s1 < 4)
        assert_true(s0 != s1)

def test_num_samples_exceeds_n() raises:
    var probs = Tensor[F32].d1([0.25, 0.25, 0.25, 0.25])
    try:
        var idx = probs.multinomial(num_samples=5, replacement=False)
        assert_true(False)
    except:
        assert_true(True)

def test_seed_reproducibility() raises:
    var probs = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0])
    var idx1 = probs.multinomial(num_samples=4, replacement=True, init_seed=42)
    var idx2 = probs.multinomial(num_samples=4, replacement=True, init_seed=42)
    for i in range(4):
        assert_true(idx1[i] == idx2[i])

def test_seed_different() raises:
    var probs = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0])
    var idx1 = probs.multinomial(num_samples=4, replacement=True, init_seed=42)
    var idx2 = probs.multinomial(num_samples=4, replacement=True, init_seed=99)
    var same = True
    for i in range(4):
        if idx1[i] != idx2[i]:
            same = False
    assert_true(not same)

def test_all_zero_probs() raises:
    var probs = Tensor[F32].d1([0.0, 0.0, 0.0])
    var idx = probs.multinomial(num_samples=1)
    assert_true(idx[0] >= 0 and idx[0] < 3)

def test_temperature() raises:
    var probs = Tensor[F32].d1([1.0, 1.0, 1.0])
    var idx = probs.multinomial(num_samples=1, temperature=0.5)
    assert_true(idx[0] >= 0 and idx[0] < 3)

def test_num_samples_zero() raises:
    var probs = Tensor[F32].d1([0.25, 0.25, 0.25, 0.25])
    try:
        var idx = probs.multinomial(num_samples=0)
        assert_true(False)
    except:
        assert_true(True)

def test_3d_input() raises:
    var probs = Tensor[F32].d3([
        [[0.25, 0.25, 0.25, 0.25]],
        [[0.25, 0.25, 0.25, 0.25]],
    ])
    try:
        var idx = probs.multinomial(num_samples=1)
        assert_true(False)
    except:
        assert_true(True)

# ── GPU tests (guarded) ────────────────────────────────────────────────────

def test_gpu_with_replacement() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d1([0.1, 0.2, 0.4, 0.2, 0.1]).to_gpu()
        var idx = probs.multinomial(num_samples=3, replacement=True, init_seed=42)
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.rank() == 1)
        assert_true(idx_cpu.shape()[0] == 3)
        for i in range(3):
            var val = idx_cpu[i]
            assert_true(val >= 0 and val < 5)

def test_gpu_without_replacement() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d1([0.1, 0.2, 0.3, 0.4]).to_gpu()
        var idx = probs.multinomial(num_samples=3, replacement=False, init_seed=42)
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.rank() == 1)
        assert_true(idx_cpu.shape()[0] == 3)
        var s0 = idx_cpu[0]
        var s1 = idx_cpu[1]
        var s2 = idx_cpu[2]
        assert_true(s0 >= 0 and s0 < 4)
        assert_true(s1 >= 0 and s1 < 4)
        assert_true(s2 >= 0 and s2 < 4)
        assert_true(s0 != s1 and s0 != s2 and s1 != s2)

def test_gpu_without_replacement_exhaust() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d1([0.1, 0.2, 0.3, 0.4]).to_gpu()
        var idx = probs.multinomial(
            num_samples=4, replacement=False, init_seed=42
        )
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.shape()[0] == 4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert_true(idx_cpu[i] != idx_cpu[j])

def test_gpu_batched() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d2([
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]).to_gpu()
        var idx = probs.multinomial(num_samples=2, replacement=True, init_seed=42)
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.rank() == 2)
        assert_true(idx_cpu.shape()[0] == 2)
        assert_true(idx_cpu.shape()[1] == 2)
        for b in range(2):
            for s in range(2):
                assert_true(idx_cpu[b, s] >= 0 and idx_cpu[b, s] < 4)

def test_gpu_batched_without_replacement() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d2([
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]).to_gpu()
        var idx = probs.multinomial(
            num_samples=2, replacement=False, init_seed=42
        )
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.rank() == 2)
        assert_true(idx_cpu.shape()[0] == 2)
        assert_true(idx_cpu.shape()[1] == 2)
        for b in range(2):
            assert_true(idx_cpu[b, 0] != idx_cpu[b, 1])
            assert_true(idx_cpu[b, 0] >= 0 and idx_cpu[b, 0] < 4)
            assert_true(idx_cpu[b, 1] >= 0 and idx_cpu[b, 1] < 4)

def test_gpu_seed_reproducibility() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d1([0.1, 0.2, 0.3, 0.4]).to_gpu()
        var idx1 = probs.multinomial(
            num_samples=5, replacement=True, init_seed=42
        )
        var idx2 = probs.multinomial(
            num_samples=5, replacement=True, init_seed=42
        )
        var idx1_cpu = idx1.to_cpu()
        var idx2_cpu = idx2.to_cpu()
        for i in range(5):
            assert_true(idx1_cpu[i] == idx2_cpu[i])

def test_gpu_num_classes_one() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d1([1.0]).to_gpu()
        var idx = probs.multinomial(num_samples=3, replacement=True, init_seed=42)
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.shape()[0] == 3)
        for i in range(3):
            assert_true(idx_cpu[i] == 0)

def test_gpu_temperature() raises:
    comptime if has_accelerator():
        var probs = Tensor[F32].d1([0.1, 0.2, 0.3, 0.4]).to_gpu()
        var idx = probs.multinomial(
            num_samples=4, replacement=True, temperature=0.5, init_seed=42
        )
        var idx_cpu = idx.to_cpu()
        assert_true(idx_cpu.shape()[0] == 4)
        for i in range(4):
            assert_true(idx_cpu[i] >= 0 and idx_cpu[i] < 4)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
