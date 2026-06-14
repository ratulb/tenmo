from tenmo.tensor import Tensor
from std.random import seed
from std.testing import assert_true, TestSuite
from tenmo.net import Dropout
from std.sys import has_accelerator
from std.random import seed
from tenmo.shapes import Shape


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

# ============================================================================
# BACKWARD PASS TESTS
# ============================================================================
# ============================================================================
# EDGE CASES AND SPECIAL TESTS
# ============================================================================

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll dropout tests passed!")


#=========



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_zeros[dtype: DType](t: Tensor[dtype]) -> Int:
    var count = 0
    for i in range(t.numels()):
        if t.get(i) == 0.0:
            count += 1
    return count

def count_nonzeros[dtype: DType](t: Tensor[dtype]) -> Int:
    return t.numels() - count_zeros(t)

def all_nonzero_close[dtype: DType](
    t: Tensor[dtype], expected_val: Scalar[dtype], atol: Scalar[dtype]
) -> Bool:
    for i in range(t.numels()):
        var v = t.get(i)
        if v != 0.0:
            if abs(v - expected_val) > atol:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORWARD — CPU
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKWARD — CPU
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAD FLOW VERIFICATION — CPU
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD — GPU
# ─────────────────────────────────────────────────────────────────────────────

def test_dropout2_fwd_gpu_eval_is_identity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.eval()
        var x_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        assert_true(out.to_cpu().all_close(x_cpu))


def test_dropout2_fwd_gpu_output_shape_preserved_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(64)).to_gpu()
        var out = dropout(x)
        assert_true(out.shape() == Shape(64))


def test_dropout2_fwd_gpu_output_shape_preserved_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(8, 8)).to_gpu()
        var out = dropout(x)
        assert_true(out.shape() == Shape(8, 8))


def test_dropout2_fwd_gpu_output_shape_preserved_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(2, 4, 8)).to_gpu()
        var out = dropout(x)
        assert_true(out.shape() == Shape(2, 4, 8))


def test_dropout2_fwd_gpu_high_p_many_zeros() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.9)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(200)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(count_zeros(out) > 150)


def test_dropout2_fwd_gpu_scale_correct_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(200)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(all_nonzero_close(out, 2.0, 1e-5))


def test_dropout2_fwd_gpu_scale_correct_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(10, 10)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(all_nonzero_close(out, 2.0, 1e-5))


def test_dropout2_fwd_gpu_scale_correct_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(2, 4, 8)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(all_nonzero_close(out, 2.0, 1e-5))


def test_dropout2_fwd_gpu_different_masks_per_call() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(100)).to_gpu()
        var out1 = dropout(x).to_cpu()
        var out2 = dropout(x).to_cpu()
        var same = True
        for i in range(100):
            if out1.get(i) != out2.get(i):
                same = False
                break
        assert_true(not same)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKWARD — GPU
# ─────────────────────────────────────────────────────────────────────────────

def test_dropout2_bwd_gpu_grad_zero_where_dropped_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(64), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(64):
            if out_cpu.get(i) == 0.0:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)
            else:
                assert_true(abs(x_cpu.grad().get(i) - 2.0) < 1e-5)


def test_dropout2_bwd_gpu_grad_zero_where_dropped_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(8, 8), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(64):
            if out_cpu.get(i) == 0.0:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)
            else:
                assert_true(abs(x_cpu.grad().get(i) - 2.0) < 1e-5)


def test_dropout2_bwd_gpu_grad_zero_where_dropped_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(2, 4, 8), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(x_cpu.numels()):
            if out_cpu.get(i) == 0.0:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)
            else:
                assert_true(abs(x_cpu.grad().get(i) - 2.0) < 1e-5)


def test_dropout2_bwd_gpu_no_grad_leaf_unaffected() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(16)).to_gpu()
        var out = dropout(x)
        assert_true(not out.requires_grad)


def test_dropout2_bwd_gpu_high_p_grad_flow() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.9)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(200), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(200):
            if out_cpu.get(i) != 0.0:
                assert_true(abs(x_cpu.grad().get(i) - 10.0) < 1e-4)
            else:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)


def test_dropout2_bwd_gpu_chained_with_linear_op() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(32), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var scaled = out * Tensor[dtype].full_like(out, 3.0)
        var loss = scaled.sum()
        loss.backward()
        for i in range(32):
            var g = x_cpu.grad().get(i)
            assert_true(abs(g) < 1e-6 or abs(g - 6.0) < 1e-5)


def test_dropout2_bwd_gpu_eval_grad_is_ones() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.eval()
        var x_cpu = Tensor[dtype].ones(Shape(16), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var loss = out.sum()
        loss.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](Tensor.ones_like(x_cpu)))


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAD FLOW VERIFICATION — GPU
# ─────────────────────────────────────────────────────────────────────────────

def test_dropout2_gradflow_gpu_mask_consistent_fwd_bwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(128), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        var consistent = True
        for i in range(128):
            var dropped = out_cpu.get(i) == 0.0
            var grad_zero = abs(x_cpu.grad().get(i)) < 1e-6
            if dropped != grad_zero:
                consistent = False
                break
        assert_true(consistent)


def test_dropout2_gradflow_gpu_sum_of_grads_matches_nonzero_count() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(128), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        var grad_sum = Scalar[dtype](0)
        for i in range(128):
            grad_sum += x_cpu.grad().get(i)
        var expected = Scalar[dtype](count_nonzeros(out_cpu)) * Scalar[dtype](2.0)
        assert_true(abs(grad_sum - expected) < 1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CPU / GPU PARITY
# ─────────────────────────────────────────────────────────────────────────────

def test_dropout2_parity_eval_cpu_gpu_match() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.eval()
        var _tmp0 = Tensor[dtype].arange(1.0, 17.0)
        var x_cpu = _tmp0.reshape(4, 4)
        var cpu_out = dropout(x_cpu)
        var gpu_out = dropout(x_cpu.to_gpu()).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-5](gpu_out))


def test_dropout2_parity_scale_value_matches() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # Both CPU and GPU non-zero outputs must be exactly input * scale
        var dropout_cpu = Dropout[dtype](p=0.5)
        var dropout_gpu = Dropout[dtype](p=0.5)
        dropout_cpu.train()
        dropout_gpu.train()
        var x = Tensor[dtype].ones(Shape(200))
        var cpu_out = dropout_cpu(x)
        var gpu_out = dropout_gpu(x.to_gpu()).to_cpu()
        # Both should have scale=2.0 on non-zero elements
        assert_true(all_nonzero_close(cpu_out, 2.0, 1e-5))
        assert_true(all_nonzero_close(gpu_out, 2.0, 1e-5))


def test_dropout2_parity_bwd_grad_scale_matches() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var dropout_cpu = Dropout[dtype](p=0.5)
        var dropout_gpu = Dropout[dtype](p=0.5)
        dropout_cpu.train()
        dropout_gpu.train()

        var x_cpu_leaf = Tensor[dtype].ones(Shape(64), requires_grad=True)
        var out_cpu = dropout_cpu(x_cpu_leaf)
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()

        var x_gpu_leaf = Tensor[dtype].ones(Shape(64), requires_grad=True)
        var out_gpu = dropout_gpu(x_gpu_leaf.to_gpu())
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()

        # Both: non-zero grads must be 2.0, zero grads must be 0.0
        for i in range(64):
            var gc = x_cpu_leaf.grad().get(i)
            var gg = x_gpu_leaf.grad().get(i)
            assert_true(abs(gc) < 1e-6 or abs(gc - 2.0) < 1e-5)
            assert_true(abs(gg) < 1e-6 or abs(gg - 2.0) < 1e-5)
