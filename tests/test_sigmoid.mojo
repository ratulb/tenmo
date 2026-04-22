from tenmo.tensor import Tensor
from std.testing import assert_true
from std.sys import has_accelerator
# ===----------------------------------------------------------------------=== #
# Sigmoid exhaustive tests — prefix: sig_
# Covers: forward, backward, grad flow, 0-D through 4-D, CPU & GPU
# ===----------------------------------------------------------------------=== #


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------
comptime F32 = DType.float32
comptime F64 = DType.float64


# ===----------------------------------------------------------------------=== #
# CPU – Forward pass
# ===----------------------------------------------------------------------=== #

fn test_sig_cpu_scalar_forward() raises:
    """Sigmoid of a 0-D (scalar) tensor on CPU."""
    print("test_sig_cpu_scalar_forward")
    var x = Tensor[F32].scalar(0.0)
    var y = x.sigmoid[track_grad=False]()
    # sigmoid(0) == 0.5
    assert_true(y.all_close(Tensor[F32].scalar(0.5)))


fn test_sig_cpu_1d_forward_known_values() raises:
    """Sigmoid of a 1-D tensor with analytically known values on CPU."""
    print("test_sig_cpu_1d_forward_known_values")
    # sigmoid(0)=0.5  sigmoid(inf)→1  sigmoid(-inf)→0
    # We use large finite magnitudes instead of ±inf
    var x = Tensor[F32].d1([0.0, 100.0, -100.0, 1.0, -1.0])
    var y = x.sigmoid[track_grad=False]()
    var expected = Tensor[F32].d1([0.5, 1.0, 0.0, 0.7310586, 0.2689414])
    assert_true(y.all_close[atol=1e-5](expected))


fn test_sig_cpu_2d_forward() raises:
    """Sigmoid of a 2-D tensor on CPU."""
    print("test_sig_cpu_2d_forward")
    var x = Tensor[F32].d2([[0.0, 1.0], [-1.0, 2.0]])
    var y = x.sigmoid[track_grad=False]()
    var expected = Tensor[F32].d2(
        [[0.5, 0.7310586], [0.2689414, 0.8807970]]
    )
    assert_true(y.all_close[atol=1e-5](expected))


fn test_sig_cpu_3d_forward() raises:
    """Sigmoid of a 3-D tensor on CPU."""
    print("test_sig_cpu_3d_forward")
    var x = Tensor[F32].zeros([2, 3, 4])           # sigmoid(0)=0.5 everywhere
    var y = x.sigmoid[track_grad=False]()
    var expected = Tensor[F32].full([2, 3, 4], 0.5)
    assert_true(y.all_close[atol=1e-6](expected))


fn test_sig_cpu_4d_forward() raises:
    """Sigmoid of a 4-D tensor on CPU."""
    print("test_sig_cpu_4d_forward")
    var x = Tensor[F32].zeros([2, 2, 3, 4])
    var y = x.sigmoid[track_grad=False]()
    var expected = Tensor[F32].full([2, 2, 3, 4], 0.5)
    assert_true(y.all_close[atol=1e-6](expected))


fn test_sig_cpu_f64_forward() raises:
    """Sigmoid forward with float64 dtype on CPU."""
    print("test_sig_cpu_f64_forward")
    var x = Tensor[F64].d1([0.0, 1.0, -1.0])
    var y = x.sigmoid[track_grad=False]()
    var expected = Tensor[F64].d1([0.5, 0.7310585975646973, 0.2689414024353027])
    assert_true(y.all_close[atol=1e-10](expected))


# ===----------------------------------------------------------------------=== #
# CPU – Backward pass
# ===----------------------------------------------------------------------=== #

fn test_sig_cpu_scalar_backward() raises:
    """Backward through sigmoid of a 0-D tensor on CPU."""
    print("test_sig_cpu_scalar_backward")
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    # At x=0: 0.5 * 0.5 = 0.25
    var x = Tensor[F32].scalar(0.0, requires_grad=True)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-6](Tensor[F32].scalar(0.25)))


fn test_sig_cpu_1d_backward() raises:
    """Backward through sigmoid of a 1-D tensor on CPU."""
    print("test_sig_cpu_1d_backward")
    var x = Tensor[F32].d1([0.0, 1.0, -1.0], requires_grad=True)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    # grad = sigmoid(x) * (1 - sigmoid(x))
    var s = Tensor[F32].d1([0.5, 0.7310586, 0.2689414])
    var expected_grad = s * (Tensor[F32].ones_like(s) - s)
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_sig_cpu_2d_backward() raises:
    """Backward through sigmoid of a 2-D tensor on CPU."""
    print("test_sig_cpu_2d_backward")
    var x = Tensor[F32].d2([[0.0, 2.0], [-2.0, 0.0]], requires_grad=True)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    var s = Tensor[F32].d2(
        [[0.5, 0.8807970], [0.1192030, 0.5]]
    )
    var expected_grad = s * (Tensor[F32].ones_like(s) - s)
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_sig_cpu_3d_backward() raises:
    """Backward through sigmoid of a 3-D tensor on CPU (all-zeros input)."""
    print("test_sig_cpu_3d_backward")
    var x = Tensor[F32].zeros([2, 3, 4], requires_grad=True)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    # grad = 0.25 everywhere when x=0
    var expected_grad = Tensor[F32].full([2, 3, 4], 0.25)
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_sig_cpu_4d_backward() raises:
    """Backward through sigmoid of a 4-D tensor on CPU (all-zeros input)."""
    print("test_sig_cpu_4d_backward")
    var x = Tensor[F32].zeros([2, 2, 3, 4], requires_grad=True)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    var expected_grad = Tensor[F32].full([2, 2, 3, 4], 0.25)
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_sig_cpu_f64_backward() raises:
    """Backward through sigmoid with float64 on CPU."""
    print("test_sig_cpu_f64_backward")
    var x = Tensor[F64].d1([0.0], requires_grad=True)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-12](Tensor[F64].d1([0.25])))


# ===----------------------------------------------------------------------=== #
# CPU – Gradient-flow verifications
# ===----------------------------------------------------------------------=== #

fn test_sig_cpu_grad_no_grad_leaf() raises:
    """Sigmoid output has no grad when requires_grad=False on leaf."""
    print("test_sig_cpu_grad_no_grad_leaf")
    var x = Tensor[F32].d1([1.0, 2.0], requires_grad=False)
    var y = x.sigmoid()
    var loss = y.sum()
    loss.backward()
    # x was not tracking gradients; grad should be zero/uninitialised
    assert_true(not x.requires_grad)


fn test_sig_cpu_grad_chained_with_add() raises:
    """Grad flows correctly through sigmoid followed by addition on CPU."""
    print("test_sig_cpu_grad_chained_with_add")
    var x = Tensor[F32].d1([0.0, 1.0], requires_grad=True)
    var y = x.sigmoid()         # sigmoid
    var z = y + y               # elementwise add: z = 2 * sigmoid(x)
    var loss = z.sum()
    loss.backward()
    # d/dx [2 * sigmoid(x)] = 2 * sigmoid'(x)
    var s = Tensor[F32].d1([0.5, 0.7310586])
    var expected = (Tensor[F32].ones_like(s) - s) * s * Tensor[F32].d1([2.0, 2.0])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_sig_cpu_grad_chained_with_mul() raises:
    """Grad flows correctly through sigmoid followed by multiplication on CPU."""
    print("test_sig_cpu_grad_chained_with_mul")
    var x = Tensor[F32].d1([0.0, -1.0], requires_grad=True)
    var y = x.sigmoid()
    var w = Tensor[F32].d1([3.0, 3.0])   # constant weight
    var z = y * w                          # z = 3 * sigmoid(x)
    var loss = z.sum()
    loss.backward()
    # grad = 3 * sigmoid'(x)
    var s = Tensor[F32].d1([0.5, 0.2689414])
    var expected = Tensor[F32].d1([3.0, 3.0]) * s * (Tensor[F32].ones_like(s) - s)
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_sig_cpu_grad_double_sigmoid() raises:
    """Grad flows through two stacked sigmoids on CPU."""
    print("test_sig_cpu_grad_double_sigmoid")
    var x = Tensor[F32].d1([0.0], requires_grad=True)
    var y = x.sigmoid()
    var z = y.sigmoid()        # sigmoid(sigmoid(x))
    var loss = z.sum()
    loss.backward()
    # s1 = sigmoid(0) = 0.5
    # s2 = sigmoid(0.5) ≈ 0.62246
    # grad = s2*(1-s2) * s1*(1-s1) = 0.23500 * 0.25 ≈ 0.05875
    var s1: Float32 = 0.5
    var s2: Float32 = 0.6224593
    var expected_val: Float32 = s2 * (1.0 - s2) * s1 * (1.0 - s1)
    assert_true(
        x.grad().all_close[atol=1e-5](Tensor[F32].d1([expected_val]))
    )


fn test_sig_cpu_grad_track_grad_false_no_backward() raises:
    """When track_grad=False, sigmoid output should not participate in autograd."""
    print("test_sig_cpu_grad_track_grad_false_no_backward")
    var x = Tensor[F32].d1([1.0, 2.0], requires_grad=True)
    var y = x.sigmoid[track_grad=False]()
    # y is detached — it should not carry grad_fn
    assert_true(not y.requires_grad)


fn test_sig_cpu_grad_requires_grad_override() raises:
    """R[r]equires_grad kwarg overrides default grad tracking on CPU."""
    print("test_sig_cpu_grad_requires_grad_override")
    var x = Tensor[F32].d1([0.0, 1.0], requires_grad=True)
    # Explicitly disable grad on the output
    var y = x.sigmoid(requires_grad=False)
    assert_true(not y.requires_grad)


# ===----------------------------------------------------------------------=== #
# GPU – Forward pass
# ===----------------------------------------------------------------------=== #

fn test_sig_gpu_scalar_forward() raises:
    print("test_sig_gpu_scalar_forward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].scalar(0.0).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        assert_true(y.to_cpu().all_close[atol=1e-6](Tensor[dtype].scalar(0.5)))


fn test_sig_gpu_1d_forward() raises:
    print("test_sig_gpu_1d_forward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([0.0, 1.0, -1.0]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].d1([0.5, 0.7310586, 0.2689414])
        assert_true(y.to_cpu().all_close[atol=1e-5](expected))


fn test_sig_gpu_2d_forward() raises:
    print("test_sig_gpu_2d_forward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[0.0, 2.0], [-2.0, 0.0]]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].d2(
            [[0.5, 0.8807970], [0.1192030, 0.5]]
        )
        assert_true(y.to_cpu().all_close[atol=1e-5](expected))


fn test_sig_gpu_3d_forward() raises:
    print("test_sig_gpu_3d_forward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].zeros([2, 3, 4]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].full([2, 3, 4], 0.5)
        assert_true(y.to_cpu().all_close[atol=1e-6](expected))


fn test_sig_gpu_4d_forward() raises:
    print("test_sig_gpu_4d_forward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].zeros([2, 2, 3, 4]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].full([2, 2, 3, 4], 0.5)
        assert_true(y.to_cpu().all_close[atol=1e-6](expected))


fn test_sig_gpu_f64_forward() raises:
    print("test_sig_gpu_f64_forward")
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var x = Tensor[dtype].d1([0.0, 1.0, -1.0]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].d1(
            [0.5, 0.7310585975646973, 0.2689414024353027]
        )
        assert_true(y.to_cpu().all_close[atol=1e-10](expected))


# ===----------------------------------------------------------------------=== #
# GPU – Backward pass
# ===----------------------------------------------------------------------=== #

fn test_sig_gpu_scalar_backward() raises:
    print("test_sig_gpu_scalar_backward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].scalar(0.0, requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        # grad accumulates on the CPU leaf
        assert_true(x_cpu.grad().all_close[atol=1e-6](Tensor[dtype].scalar(0.25)))


fn test_sig_gpu_1d_backward() raises:
    print("test_sig_gpu_1d_backward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var s = Tensor[dtype].d1([0.5, 0.7310586, 0.2689414])
        var expected = s * (Tensor[dtype].ones_like(s) - s)
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


fn test_sig_gpu_2d_backward() raises:
    print("test_sig_gpu_2d_backward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2([[0.0, 2.0], [-2.0, 0.0]], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var s = Tensor[dtype].d2([[0.5, 0.8807970], [0.1192030, 0.5]])
        var expected = s * (Tensor[dtype].ones_like(s) - s)
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


fn test_sig_gpu_3d_backward() raises:
    print("test_sig_gpu_3d_backward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].zeros([2, 3, 4], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var expected = Tensor[dtype].full([2, 3, 4], 0.25)
        assert_true(x_cpu.grad().all_close[atol=1e-6](expected))


fn test_sig_gpu_4d_backward() raises:
    print("test_sig_gpu_4d_backward")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].zeros([2, 2, 3, 4], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var expected = Tensor[dtype].full([2, 2, 3, 4], 0.25)
        assert_true(x_cpu.grad().all_close[atol=1e-6](expected))


fn test_sig_gpu_f64_backward() raises:
    print("test_sig_gpu_f64_backward")
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var x_cpu = Tensor[dtype].d1([0.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-12](Tensor[dtype].d1([0.25])))


# ===----------------------------------------------------------------------=== #
# GPU – Gradient-flow verifications
# ===----------------------------------------------------------------------=== #

fn test_sig_gpu_grad_chained_with_add() raises:
    """Grad flows correctly through sigmoid followed by addition on GPU."""
    print("test_sig_gpu_grad_chained_with_add")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var z = y + y            # 2 * sigmoid(x)
        var loss = z.sum()
        loss.backward()
        var s = Tensor[dtype].d1([0.5, 0.7310586])
        var expected = (Tensor[dtype].ones_like(s) - s) * s * Tensor[dtype].d1([2.0, 2.0])
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


fn test_sig_gpu_grad_chained_with_mul() raises:
    """Grad flows correctly through sigmoid followed by multiplication on GPU."""
    print("test_sig_gpu_grad_chained_with_mul")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0, -1.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var w = Tensor[dtype].d1([3.0, 3.0]).to_gpu()
        var z = y * w
        var loss = z.sum()
        loss.backward()
        var s = Tensor[dtype].d1([0.5, 0.2689414])
        var expected = Tensor[dtype].d1([3.0, 3.0]) * s * (Tensor[dtype].ones_like(s) - s)
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


fn test_sig_gpu_grad_double_sigmoid() raises:
    """Grad flows through two stacked sigmoids on GPU."""
    print("test_sig_gpu_grad_double_sigmoid")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var z = y.sigmoid()
        var loss = z.sum()
        loss.backward()
        var s1: Float32 = 0.5
        var s2: Float32 = 0.6224593
        var expected_val: Float32 = s2 * (1.0 - s2) * s1 * (1.0 - s1)
        assert_true(
            x_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([expected_val]))
        )


fn test_sig_gpu_grad_track_grad_false() raises:
    """T[t]rack_grad=False on GPU: output must not carry requires_grad."""
    print("test_sig_gpu_grad_track_grad_false")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        assert_true(not y.requires_grad)


fn test_sig_gpu_cpu_forward_parity() raises:
    """CPU and GPU sigmoid produce identical results for the same input."""
    print("test_sig_gpu_cpu_forward_parity")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2([[0.5, -0.5], [1.5, -1.5]])
        var x_gpu = x_cpu.to_gpu()
        var y_cpu = x_cpu.sigmoid[track_grad=False]()
        var y_gpu = x_gpu.sigmoid[track_grad=False]()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu.to_cpu()))


fn test_sig_gpu_cpu_backward_parity() raises:
    """CPU and GPU sigmoid produce identical gradients for the same input."""
    print("test_sig_gpu_cpu_backward_parity")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu_ref = Tensor[dtype].d2([[0.5, -0.5], [1.5, -1.5]], requires_grad=True)
        var y_cpu = x_cpu_ref.sigmoid()
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var x_cpu_leaf = Tensor[dtype].d2([[0.5, -0.5], [1.5, -1.5]], requires_grad=True)
        var x_gpu = x_cpu_leaf.to_gpu()
        var y_gpu = x_gpu.sigmoid()
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(x_cpu_ref.grad().all_close[atol=1e-6](x_cpu_leaf.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #

fn main() raises:
    # --- CPU forward ---
    test_sig_cpu_scalar_forward()
    test_sig_cpu_1d_forward_known_values()
    test_sig_cpu_2d_forward()
    test_sig_cpu_3d_forward()
    test_sig_cpu_4d_forward()
    test_sig_cpu_f64_forward()

    # --- CPU backward ---
    test_sig_cpu_scalar_backward()
    test_sig_cpu_1d_backward()
    test_sig_cpu_2d_backward()
    test_sig_cpu_3d_backward()
    test_sig_cpu_4d_backward()
    test_sig_cpu_f64_backward()

    # --- CPU grad-flow ---
    test_sig_cpu_grad_no_grad_leaf()
    test_sig_cpu_grad_chained_with_add()
    test_sig_cpu_grad_chained_with_mul()
    test_sig_cpu_grad_double_sigmoid()
    test_sig_cpu_grad_track_grad_false_no_backward()
    test_sig_cpu_grad_requires_grad_override()

    # --- GPU forward ---
    test_sig_gpu_scalar_forward()
    test_sig_gpu_1d_forward()
    test_sig_gpu_2d_forward()
    test_sig_gpu_3d_forward()
    test_sig_gpu_4d_forward()
    test_sig_gpu_f64_forward()

    # --- GPU backward ---
    test_sig_gpu_scalar_backward()
    test_sig_gpu_1d_backward()
    test_sig_gpu_2d_backward()
    test_sig_gpu_3d_backward()
    test_sig_gpu_4d_backward()
    test_sig_gpu_f64_backward()

    # --- GPU grad-flow ---
    test_sig_gpu_grad_chained_with_add()
    test_sig_gpu_grad_chained_with_mul()
    test_sig_gpu_grad_double_sigmoid()
    test_sig_gpu_grad_track_grad_false()
    test_sig_gpu_cpu_forward_parity()
    test_sig_gpu_cpu_backward_parity()

    print("All sigmoid tests passed.")
