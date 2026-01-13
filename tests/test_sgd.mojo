from tenmo import Tensor
from shapes import Shape
from testing import assert_true
from sgd import SGD


fn test_sgd_basic() raises:
    """Test 1: Basic SGD without momentum."""
    print("\n=== Test 1: Basic SGD ===")
    alias dtype = DType.float32

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
    alias dtype = DType.float32

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
    alias dtype = DType.float32

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
    alias dtype = DType.float32

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
    alias dtype = DType.float32

    var param = Tensor[dtype].ones(Shape([2, 2]), requires_grad=True)
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=param))

    var optimizer = SGD(parameters=params^, lr=0.1, clip_value=0.5)

    param.seed_grad(Scalar[dtype](10.0))
    optimizer.step()
    # Grad clipped to 0.5
    # p = 1.0 - 0.1*0.5 = 0.95
    assert_true(param == Tensor[dtype].full(Shape(2, 2), 0.95))


fn main() raises:
    test_sgd_basic()
    test_sgd_momentum()
    test_sgd_weight_decay()
    test_sgd_grad_norm_clipping()
    test_sgd_value_clipping()
