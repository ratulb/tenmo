from tensor import Tensor
from shapes import Shape
from std.testing import assert_true
from sgd import SGD
from std.sys import has_accelerator
from common_utils import s

comptime dtype = DType.float32

fn test_sgd_basic() raises:
    """Test 1: Basic SGD without momentum."""
    print("\n=== Test 1: Basic SGD ===")
    comptime dtype = DType.float32

    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))

    var optimizer = SGD(parameters=params^, lr=0.1)

    # Set gradient
    param.seed_grad(Scalar[dtype](1.0))
    assert_true(params[0][].grad() == Tensor[dtype].ones(2, 2))
    optimizer.step()
    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.9))
    # Expected: 1.0 - 0.1 * 1.0 = 0.9
    optimizer.zero_grad()
    assert_true(param.grad() == Tensor[dtype].full(Shape(2, 2), 0.0))


fn test_sgd_momentum() raises:
    """Test 2: SGD with momentum."""
    print("\n=== Test 2: SGD with Momentum ===")
    comptime dtype = DType.float32

    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))

    var optimizer = SGD(parameters=params^, lr=0.1, momentum=0.9)

    # First step
    param.seed_grad(Scalar[dtype](1.0))
    optimizer.step()
    # v1 = 0.9*0 + 1.0 = 1.0
    # p1 = 1.0 - 0.1*1.0 = 0.9

    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.9))
    optimizer.zero_grad()

    # Second step
    param.seed_grad(Scalar[dtype](1.0))
    optimizer.step()
    # v2 = 0.9*1.0 + 1.0 = 1.9
    # p2 = 0.9 - 0.1*1.9 = 0.71
    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.71))


fn test_sgd_weight_decay() raises:
    """Test 3: SGD with weight decay."""
    print("\n=== Test 3: Weight Decay ===")
    comptime dtype = DType.float32

    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))

    var optimizer = SGD(parameters=params^, lr=0.1, weight_decay=0.01)

    param.seed_grad(Scalar[dtype](1.0))
    optimizer.step()
    # Δw = -η × (∂L/∂w + λw) -> delta part combined
    # effective_grad = 1.0 + 0.01*1.0 = 1.01 -> part within braces
    # p = 1.0 - 0.1*1.01 = 0.899
    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.899))


fn test_sgd_grad_norm_clipping() raises:
    """Test 4: Gradient norm clipping."""
    print("\n=== Test 4: Gradient norm Clipping ===")
    comptime dtype = DType.float32

    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))

    var optimizer = SGD(parameters=params^, lr=0.1, clip_norm=1.0)

    # Set large gradient
    param.seed_grad(Scalar[dtype](10.0))
    optimizer.step()
    # Grad norm before: sqrt(10 * 10)*4)  = 20
    # Clip_norm: 1.0
    # clip_coef = 1.0 / 20 = 0.05
    # effective_grad = 10 * 0.05 = 0.5
    # p = 1.0 - 0.1*0.5 = 0.95
    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.95))


fn test_sgd_value_clipping() raises:
    """Test 5: Gradient value clipping."""
    print("\n=== Test 5: Value Clipping ===")
    comptime dtype = DType.float32

    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))

    var optimizer = SGD(parameters=params^, lr=0.1, clip_value=0.5)

    param.seed_grad(Scalar[dtype](10.0))
    optimizer.step()
    # Grad clipped to 0.5
    # p = 1.0 - 0.1*0.5 = 0.95
    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.95))

# ── CPU Tests ─────────────────────────────────────────────────────────────────

fn test_sgd_cpu_vanilla_single_step() raises:
    print("test_sgd_cpu_vanilla_single_step")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    # Manually set grad
    w.seed_grad(1.0)
    print("w shape:", w.shape().__str__())
    print("w grad shape:", w.gradients()[].shape().__str__())
    print("w grad num_elements:", w.gradients()[].num_elements())
    w.gradients()[].print()

    sgd.step()
    # w = w - lr * grad = [1,2,3] - 0.1*[1,1,1] = [0.9, 1.9, 2.9]
    assert_true(w.all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    print("passed")


fn test_sgd_cpu_vanilla_multiple_steps() raises:
    print("test_sgd_cpu_vanilla_multiple_steps")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    for _ in range(3):
        w.seed_grad(1.0)
        sgd.step()
    # After 3 steps: [1,2,3] - 3*0.1 = [0.7, 1.7, 2.7]
    assert_true(w.all_close(Tensor[dtype].d1([0.7, 1.7, 2.7])))
    print("passed")


fn test_sgd_cpu_vanilla_2d() raises:
    print("test_sgd_cpu_vanilla_2d")
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(2.0)
    sgd.step()
    # w = w - 0.1 * 2 = w - 0.2
    assert_true(w.all_close(Tensor[dtype].d2([[0.8, 1.8], [2.8, 3.8]])))
    print("passed")


fn test_sgd_cpu_zero_grad() raises:
    print("test_sgd_cpu_zero_grad")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step()
    sgd.zero_grad()
    # grad should be zero
    assert_true(w.gradients()[].all_close(Tensor[dtype].zeros(w.shape())))
    print("passed")


fn test_sgd_cpu_weight_decay() raises:
    print("test_sgd_cpu_weight_decay")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, weight_decay=0.1)
    w.seed_grad(1.0)
    sgd.step()
    # g_eff = g + wd*p = 1 + 0.1*p
    # w_new = w - lr * g_eff
    var expected = Tensor[dtype].d1([
        1.0 - 0.1 * (1.0 + 0.1 * 1.0),
        2.0 - 0.1 * (1.0 + 0.1 * 2.0),
        3.0 - 0.1 * (1.0 + 0.1 * 3.0),
    ])
    assert_true(w.all_close(expected))
    print("passed")


fn test_sgd_cpu_momentum_single_step() raises:
    print("test_sgd_cpu_momentum_single_step")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, momentum=0.9)
    w.seed_grad(1.0)
    sgd.step()
    # v = 0.9*0 + 1 = 1, w = w - 0.1*1 = [0.9, 1.9, 2.9]
    assert_true(w.all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    print("passed")


fn test_sgd_cpu_momentum_multiple_steps() raises:
    print("test_sgd_cpu_momentum_multiple_steps")
    var w = Tensor[dtype].d1([1.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, momentum=0.9)
    # Step 1: v=1, w = 1 - 0.1*1 = 0.9
    w.seed_grad(1.0)
    sgd.step()
    # Step 2: v = 0.9*1 + 1 = 1.9, w = 0.9 - 0.1*1.9 = 0.71
    w.seed_grad(1.0)
    sgd.step()
    # Step 3: v = 0.9*1.9 + 1 = 2.71, w = 0.71 - 0.1*2.71 = 0.439
    w.seed_grad(1.0)
    sgd.step()
    assert_true(w.all_close(Tensor[dtype].d1([0.439])))
    print("passed")


fn test_sgd_cpu_clip_value() raises:
    print("test_sgd_cpu_clip_value")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1, clip_value=0.5)
    w.seed_grad(2.0)
    sgd.step()
    # clipped grad = 0.5, w = w - 0.1*0.5 = w - 0.05
    assert_true(w.all_close(Tensor[dtype].d1([0.95, 1.95, 2.95])))
    print("passed")


fn test_sgd_cpu_multiple_parameters() raises:
    print("test_sgd_cpu_multiple_parameters")
    var w1 = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var w2 = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w1))
    params.append(UnsafePointer(to=w2))
    var sgd = SGD(params, lr=0.1)
    w1.seed_grad(1.0)
    w2.seed_grad(2.0)
    sgd.step()
    assert_true(w1.all_close(Tensor[dtype].d1([0.9, 1.9])))
    assert_true(w2.all_close(Tensor[dtype].d1([2.8, 3.8])))
    print("passed")


fn test_sgd_cpu_backward_integration() raises:
    print("test_sgd_cpu_backward_integration")
    # Simple linear: loss = sum(w * x)
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var x = Tensor[dtype].d1([1.0, 1.0, 1.0])
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    var loss = (w * x).sum()
    loss.backward()
    # grad_w = x = [1,1,1]
    sgd.step()
    assert_true(w.all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    print("passed")


fn test_sgd_cpu_set_lr() raises:
    print("test_sgd_cpu_set_lr")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step()
    sgd.set_lr(0.5)
    sgd.zero_grad()
    w.seed_grad(1.0)
    sgd.step()
    # After first step: [0.9, 1.9, 2.9]
    # After second step with lr=0.5: [0.4, 1.4, 2.4]
    assert_true(w.all_close(Tensor[dtype].d1([0.4, 1.4, 2.4])))
    print("passed")

# ── GPU Tests ─────────────────────────────────────────────────────────────────

fn test_sgd_gpu_vanilla_single_step() raises:
    print("test_sgd_gpu_vanilla_single_step")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1)
    w_gpu.seed_grad(1.0)
    sgd.step()
    var result = w_gpu.to_cpu()
    assert_true(result.all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    print("passed")


fn test_sgd_gpu_vanilla_matches_cpu() raises:
    print("test_sgd_gpu_vanilla_matches_cpu")
    var w = Tensor[dtype].rand(4, 8, requires_grad=True)
    var w_gpu = w.to_gpu()
    var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_cpu.append(UnsafePointer(to=w))
    params_gpu.append(UnsafePointer(to=w_gpu))
    var sgd_cpu = SGD(params_cpu, lr=0.01)
    var sgd_gpu = SGD(params_gpu, lr=0.01)
    # Same grad on both
    var grad = Tensor[dtype].rand(4, 8)
    var grad_gpu = grad.to_gpu()
    w.seed_grad(grad)
    w_gpu.seed_grad(grad_gpu)
    sgd_cpu.step()
    sgd_gpu.step()
    assert_true(w.all_close(w_gpu.to_cpu()))
    print("passed")


fn test_sgd_gpu_vanilla_multiple_steps() raises:
    print("test_sgd_gpu_vanilla_multiple_steps")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1)
    for _ in range(3):
        w_gpu.seed_grad(1.0)
        sgd.step()
    var result = w_gpu.to_cpu()
    assert_true(result.all_close(Tensor[dtype].d1([0.7, 1.7, 2.7])))
    print("passed")


fn test_sgd_gpu_weight_decay() raises:
    print("test_sgd_gpu_weight_decay")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1, weight_decay=0.1)
    w_gpu.seed_grad(1.0)
    sgd.step()
    var expected = Tensor[dtype].d1([
        1.0 - 0.1 * (1.0 + 0.1 * 1.0),
        2.0 - 0.1 * (1.0 + 0.1 * 2.0),
        3.0 - 0.1 * (1.0 + 0.1 * 3.0),
    ])
    assert_true(w_gpu.to_cpu().all_close(expected))
    print("passed")


fn test_sgd_gpu_momentum_single_step() raises:
    print("test_sgd_gpu_momentum_single_step")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1, momentum=0.9)
    w_gpu.seed_grad(1.0)
    sgd.step()
    assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    print("passed")


fn test_sgd_gpu_momentum_multiple_steps() raises:
    print("test_sgd_gpu_momentum_multiple_steps")
    var w = Tensor[dtype].d1([1.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1, momentum=0.9)
    for _ in range(3):
        w_gpu.seed_grad(1.0)
        sgd.step()
    assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.439])))
    print("passed")


fn test_sgd_gpu_momentum_matches_cpu() raises:
    print("test_sgd_gpu_momentum_matches_cpu")
    var w = Tensor[dtype].rand(4, 8, requires_grad=True)
    var w_gpu = w.to_gpu()
    var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_cpu.append(UnsafePointer(to=w))
    params_gpu.append(UnsafePointer(to=w_gpu))
    var sgd_cpu = SGD(params_cpu, lr=0.01, momentum=0.9)
    var sgd_gpu = SGD(params_gpu, lr=0.01, momentum=0.9)
    var grad = Tensor[dtype].rand(4, 8)
    var grad_gpu = grad.to_gpu()
    for _ in range(5):
        w.seed_grad(grad)
        w_gpu.seed_grad(grad_gpu)
        sgd_cpu.step()
        sgd_gpu.step()
    assert_true(w.all_close(w_gpu.to_cpu()))
    print("passed")


fn test_sgd_gpu_clip_value() raises:
    print("test_sgd_gpu_clip_value")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1, clip_value=0.5)
    w_gpu.seed_grad(2.0)
    sgd.step()
    assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.95, 1.95, 2.95])))
    print("passed")


fn test_sgd_gpu_clip_value_matches_cpu() raises:
    print("test_sgd_gpu_clip_value_matches_cpu")
    var w = Tensor[dtype].rand(8, 8, requires_grad=True)
    var w_gpu = w.to_gpu()
    var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_cpu.append(UnsafePointer(to=w))
    params_gpu.append(UnsafePointer(to=w_gpu))
    var sgd_cpu = SGD(params_cpu, lr=0.01, clip_value=0.1)
    var sgd_gpu = SGD(params_gpu, lr=0.01, clip_value=0.1)
    var grad = Tensor[dtype].rand(8, 8)
    var grad_gpu = grad.to_gpu()
    w.seed_grad(grad)
    w_gpu.seed_grad(grad_gpu)
    sgd_cpu.step()
    sgd_gpu.step()
    assert_true(w.all_close(w_gpu.to_cpu()))
    print("passed")


fn test_sgd_gpu_clip_norm_matches_cpu() raises:
    print("test_sgd_gpu_clip_norm_matches_cpu")
    var w = Tensor[dtype].rand(8, 8, requires_grad=True)
    var w_gpu = w.to_gpu()
    var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_cpu.append(UnsafePointer(to=w))
    params_gpu.append(UnsafePointer(to=w_gpu))
    var sgd_cpu = SGD(params_cpu, lr=0.01, clip_norm=1.0)
    var sgd_gpu = SGD(params_gpu, lr=0.01, clip_norm=1.0)
    var grad = Tensor[dtype].rand(8, 8)
    var grad_gpu = grad.to_gpu()
    w.seed_grad(grad)
    w_gpu.seed_grad(grad_gpu)
    sgd_cpu.step()
    sgd_gpu.step()
    assert_true(w.all_close(w_gpu.to_cpu()))
    print("passed")


fn test_sgd_gpu_multiple_parameters() raises:
    print("test_sgd_gpu_multiple_parameters")
    var w1 = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var w2 = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var w1_gpu = w1.to_gpu()
    var w2_gpu = w2.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w1_gpu))
    params.append(UnsafePointer(to=w2_gpu))
    var sgd = SGD(params, lr=0.1)
    w1_gpu.seed_grad(1.0)
    w2_gpu.seed_grad(2.0)
    sgd.step()
    assert_true(w1_gpu.to_cpu().all_close(Tensor[dtype].d1([0.9, 1.9])))
    assert_true(w2_gpu.to_cpu().all_close(Tensor[dtype].d1([2.8, 3.8])))
    print("passed")


fn test_sgd_gpu_zero_grad() raises:
    print("test_sgd_gpu_zero_grad")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1)
    w_gpu.seed_grad(1.0)
    sgd.step()
    sgd.zero_grad()
    var grad_after = w_gpu.grad().to_cpu()
    assert_true(grad_after.all_close(Tensor[dtype].zeros(w_gpu.shape())))
    print("passed")


fn test_sgd_gpu_backward_integration() raises:
    print("test_sgd_gpu_backward_integration")
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var w_gpu = w.to_gpu()
    var x = Tensor[dtype].d1([1.0, 1.0, 1.0])
    var x_gpu = x.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w_gpu))
    var sgd = SGD(params, lr=0.1)
    var loss = (w_gpu * x_gpu).sum()
    loss.backward()
    sgd.step()
    # grad_w = x = [1,1,1], w = w - 0.1*1 = [0.9, 1.9, 2.9]
    assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    # grad flows back to CPU w
    assert_true(w.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))
    print("passed")


fn test_sgd_gpu_backward_integration_matches_cpu() raises:
    print("test_sgd_gpu_backward_integration_matches_cpu")
    var w = Tensor[dtype].rand(4, 4, requires_grad=True)
    var x = Tensor[dtype].rand(4, 4)
    var w_gpu = w.to_gpu()
    var x_gpu = x.to_gpu()
    var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_cpu.append(UnsafePointer(to=w))
    params_gpu.append(UnsafePointer(to=w_gpu))
    var sgd_cpu = SGD(params_cpu, lr=0.01)
    var sgd_gpu = SGD(params_gpu, lr=0.01)
    # CPU backward
    var loss_cpu = (w * x).sum()
    loss_cpu.backward()
    sgd_cpu.step()
    var w_cpu_result = w.copy()
    w.zero_grad()
    # GPU backward
    var loss_gpu = (w_gpu * x_gpu).sum()
    loss_gpu.backward()
    sgd_gpu.step()
    assert_true(w_cpu_result.all_close(w_gpu.to_cpu()))
    print("passed")


fn test_sgd_gpu_large_tensor() raises:
    print("test_sgd_gpu_large_tensor")
    var w = Tensor[dtype].rand(128, 256, requires_grad=True)
    var w_gpu = w.to_gpu()
    var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params_cpu.append(UnsafePointer(to=w))
    params_gpu.append(UnsafePointer(to=w_gpu))
    var sgd_cpu = SGD(params_cpu, lr=0.01)
    var sgd_gpu = SGD(params_gpu, lr=0.01)
    var grad = Tensor[dtype].rand(128, 256)
    var grad_gpu = grad.to_gpu()
    w.seed_grad(grad)
    w_gpu.seed_grad(grad_gpu)
    sgd_cpu.step()
    sgd_gpu.step()
    assert_true(w.all_close(w_gpu.to_cpu()))
    print("passed")


fn main() raises:
    test_sgd_basic()
    test_sgd_momentum()
    test_sgd_weight_decay()
    test_sgd_grad_norm_clipping()
    test_sgd_value_clipping()

    # New CPU tests
    test_sgd_cpu_vanilla_single_step()
    test_sgd_cpu_vanilla_multiple_steps()
    test_sgd_cpu_vanilla_2d()
    test_sgd_cpu_zero_grad()
    test_sgd_cpu_weight_decay()
    test_sgd_cpu_momentum_single_step()
    test_sgd_cpu_momentum_multiple_steps()
    test_sgd_cpu_clip_value()
    test_sgd_cpu_multiple_parameters()
    test_sgd_cpu_backward_integration()
    test_sgd_cpu_set_lr()

    comptime if not has_accelerator():
        print("No GPU — skipping GPU SGD tests")
        return

    test_sgd_gpu_vanilla_single_step()
    test_sgd_gpu_vanilla_matches_cpu()
    test_sgd_gpu_vanilla_multiple_steps()
    test_sgd_gpu_weight_decay()
    test_sgd_gpu_momentum_single_step()
    test_sgd_gpu_momentum_multiple_steps()
    test_sgd_gpu_momentum_matches_cpu()
    test_sgd_gpu_clip_value()
    test_sgd_gpu_clip_value_matches_cpu()
    test_sgd_gpu_clip_norm_matches_cpu()
    test_sgd_gpu_multiple_parameters()
    test_sgd_gpu_zero_grad()
    test_sgd_gpu_backward_integration()
    test_sgd_gpu_backward_integration_matches_cpu()
    test_sgd_gpu_large_tensor()

    print("\n=== ALL SGD TESTS PASSED ===")
