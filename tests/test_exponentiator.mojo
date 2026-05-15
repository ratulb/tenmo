from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.shapes import Shape


# ── CPU Forward Tests ─────────────────────────────────────────────────────────


fn test_exp_cpu_1d_square() raises:
    print("test_exp_cpu_1d_square")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0])
    var result = a ** 2.0
    assert_true(result.all_close(Tensor[dtype].d1([4.0, 9.0, 16.0])))


fn test_exp_cpu_1d_cube() raises:
    print("test_exp_cpu_1d_cube")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0])
    var result = a ** 3.0
    assert_true(result.all_close(Tensor[dtype].d1([8.0, 27.0, 64.0])))


fn test_exp_cpu_1d_fractional() raises:
    print("test_exp_cpu_1d_fractional")
    comptime dtype = DType.float32
    # x ** 0.5 = sqrt(x)
    var a = Tensor[dtype].d1([4.0, 9.0, 16.0])
    var result = a ** 0.5
    assert_true(result.all_close(Tensor[dtype].d1([2.0, 3.0, 4.0])))


fn test_exp_cpu_1d_zero_exponent() raises:
    print("test_exp_cpu_1d_zero_exponent")
    comptime dtype = DType.float32
    # x ** 0 = 1
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0])
    var result = a ** 0.0
    assert_true(result.all_close(Tensor[dtype].ones(Shape(3))))


fn test_exp_cpu_1d_one_exponent() raises:
    print("test_exp_cpu_1d_one_exponent")
    comptime dtype = DType.float32
    # x ** 1 = x
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0])
    var result = a ** 1.0
    assert_true(result.all_close(a))


fn test_exp_cpu_2d_forward() raises:
    print("test_exp_cpu_2d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]])
    var result = a ** 2.0
    assert_true(
        result.all_close(Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]]))
    )


fn test_exp_cpu_3d_forward() raises:
    print("test_exp_cpu_3d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 4.0]]]
    )
    var result = a ** 2.0
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [[[4.0, 9.0], [16.0, 25.0]], [[1.0, 4.0], [9.0, 16.0]]]
            )
        )
    )


fn test_exp_cpu_ones_any_exponent() raises:
    print("test_exp_cpu_ones_any_exponent")
    comptime dtype = DType.float32
    # 1 ** n = 1 for any n
    var a = Tensor[dtype].ones(Shape(4))
    var result = a ** 5.0
    assert_true(result.all_close(Tensor[dtype].ones(Shape(4))))


fn test_exp_cpu_no_grad() raises:
    print("test_exp_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0], requires_grad=False)
    var result = a ** 2.0
    assert_true(not result.requires_grad)


fn test_exp_cpu_requires_grad_propagates() raises:
    print("test_exp_cpu_requires_grad_propagates")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
    var result = a ** 2.0
    assert_true(result.requires_grad)


fn test_exp_cpu_suppress_grad() raises:
    print("test_exp_cpu_suppress_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
    var result = a.__pow__[track_grad=True](2.0, requires_grad=False)
    assert_true(not result.requires_grad)


# ── CPU Backward Tests ────────────────────────────────────────────────────────


fn test_exp_cpu_1d_backward_square() raises:
    print("test_exp_cpu_1d_backward_square")
    comptime dtype = DType.float32
    # ∂(x²)/∂x = 2x
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var result = a ** 2.0
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 6.0, 8.0])))


fn test_exp_cpu_1d_backward_cube() raises:
    print("test_exp_cpu_1d_backward_cube")
    comptime dtype = DType.float32
    # ∂(x³)/∂x = 3x²
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var result = a ** 3.0
    var loss = result.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([12.0, 27.0, 48.0]))
    )


fn test_exp_cpu_1d_backward_fractional() raises:
    print("test_exp_cpu_1d_backward_fractional")
    comptime dtype = DType.float32
    # ∂(x**0.5)/∂x = 0.5 * x**(-0.5) = 0.5/sqrt(x)
    var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
    var result = a ** 0.5
    var loss = result.sum()
    loss.backward()
    # 0.5/sqrt(4)=0.25, 0.5/sqrt(9)≈0.1667, 0.5/sqrt(16)=0.125
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d1([0.25, 0.16666667, 0.125])
        )
    )


fn test_exp_cpu_2d_backward() raises:
    print("test_exp_cpu_2d_backward")
    comptime dtype = DType.float32
    # ∂(x²)/∂x = 2x
    var a = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    var result = a ** 2.0
    var loss = result.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [8.0, 10.0]]))
    )


fn test_exp_cpu_3d_backward() raises:
    print("test_exp_cpu_3d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 4.0]]],
        requires_grad=True,
    )
    var result = a ** 2.0
    var loss = result.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[4.0, 6.0], [8.0, 10.0]], [[2.0, 4.0], [6.0, 8.0]]]
            )
        )
    )


fn test_exp_cpu_backward_chain() raises:
    print("test_exp_cpu_backward_chain")
    comptime dtype = DType.float32
    # (x**2) * 3 → sum → backward
    # grad = 3 * 2x = 6x
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var result = (a ** 2.0) * 3.0
    var loss = result.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([12.0, 18.0, 24.0]))
    )


fn test_exp_cpu_backward_chained_pow() raises:
    print("test_exp_cpu_backward_chained_pow")
    comptime dtype = DType.float32
    # (x**2)**2 = x**4, grad = 4x**3
    var a = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
    var result = (a ** 2.0) ** 2.0
    var loss = result.sum()
    loss.backward()
    # 4 * 2**3 = 32, 4 * 3**3 = 108
    assert_true(a.grad().all_close(Tensor[dtype].d1([32.0, 108.0])))


fn test_exp_cpu_backward_one_exponent() raises:
    print("test_exp_cpu_backward_one_exponent")
    comptime dtype = DType.float32
    # ∂(x**1)/∂x = 1
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var result = a ** 1.0
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


# ── GPU Forward Tests ─────────────────────────────────────────────────────────


fn test_exp_gpu_1d_square() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_square")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var result = a ** 2.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([4.0, 9.0, 16.0]))
        )


fn test_exp_gpu_1d_cube() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_cube")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var result = a ** 3.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([8.0, 27.0, 64.0]))
        )


fn test_exp_gpu_1d_fractional() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_fractional")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 9.0, 16.0]).to_gpu()
        var result = a ** 0.5
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([2.0, 3.0, 4.0]))
        )


fn test_exp_gpu_1d_zero_exponent() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_zero_exponent")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var result = a ** 0.0
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close(Tensor[dtype].ones(Shape(3))))


fn test_exp_gpu_2d_forward() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_2d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]]).to_gpu()
        var result = a ** 2.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]])
            )
        )


fn test_exp_gpu_3d_forward() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_3d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 4.0]]]
        ).to_gpu()
        var result = a ** 2.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[4.0, 9.0], [16.0, 25.0]],
                        [[1.0, 4.0], [9.0, 16.0]],
                    ]
                )
            )
        )


fn test_exp_gpu_ones_any_exponent() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_ones_any_exponent")
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(Shape(4)).to_gpu()
        var result = a ** 5.0
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close(Tensor[dtype].ones(Shape(4))))


# ── GPU Backward Tests ────────────────────────────────────────────────────────


fn test_exp_gpu_1d_backward_square() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_backward_square")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu ** 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([4.0, 6.0, 8.0]))
        )


fn test_exp_gpu_1d_backward_cube() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_backward_cube")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu ** 3.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([12.0, 27.0, 48.0]))
        )


fn test_exp_gpu_1d_backward_fractional() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_1d_backward_fractional")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu ** 0.5
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d1([0.25, 0.16666667, 0.125])
            )
        )


fn test_exp_gpu_2d_backward() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_2d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[2.0, 3.0], [4.0, 5.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu ** 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[4.0, 6.0], [8.0, 10.0]])
            )
        )


fn test_exp_gpu_3d_backward() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_3d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 4.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu ** 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [
                        [[4.0, 6.0], [8.0, 10.0]],
                        [[2.0, 4.0], [6.0, 8.0]],
                    ]
                )
            )
        )


fn test_exp_gpu_backward_chain() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_backward_chain")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = (a_gpu ** 2.0) * 3.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([12.0, 18.0, 24.0]))
        )


fn test_exp_gpu_backward_chained_pow() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_backward_chained_pow")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = (a_gpu ** 2.0) ** 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([32.0, 108.0]))
        )


fn test_exp_gpu_backward_one_exponent() raises:
    comptime if has_accelerator():
        print("test_exp_gpu_backward_one_exponent")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu ** 1.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


# ── CPU/GPU Parity Tests ──────────────────────────────────────────────────────


fn test_exp_parity_1d_forward() raises:
    comptime if has_accelerator():
        print("test_exp_parity_1d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            (a_cpu ** 2.0).all_close((a_gpu ** 2.0).to_cpu())
        )


fn test_exp_parity_2d_forward() raises:
    comptime if has_accelerator():
        print("test_exp_parity_2d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            (a_cpu ** 3.0).all_close((a_gpu ** 3.0).to_cpu())
        )


fn test_exp_parity_fractional_forward() raises:
    comptime if has_accelerator():
        print("test_exp_parity_fractional_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0, 16.0, 25.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            (a_cpu ** 0.5).all_close((a_gpu ** 0.5).to_cpu())
        )


fn test_exp_parity_1d_backward() raises:
    comptime if has_accelerator():
        print("test_exp_parity_1d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True).to_gpu()

        var loss_cpu = (a_cpu ** 2.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu ** 2.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_exp_parity_2d_backward() raises:
    comptime if has_accelerator():
        print("test_exp_parity_2d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[2.0, 3.0], [4.0, 5.0]], requires_grad=True
        )
        var a_gpu = Tensor[dtype].d2(
            [[2.0, 3.0], [4.0, 5.0]], requires_grad=True
        ).to_gpu()

        var loss_cpu = (a_cpu ** 2.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu ** 2.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_exp_parity_chain_backward() raises:
    comptime if has_accelerator():
        print("test_exp_parity_chain_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True).to_gpu()

        var loss_cpu = ((a_cpu ** 2.0) * 3.0).sum()
        loss_cpu.backward()

        var loss_gpu = ((a_gpu ** 2.0) * 3.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_exp_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        print("test_exp_parity_using_zero_grad")
        comptime dtype = DType.float32
        # Use same tensor for both passes — zero_grad between them
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = (a_cpu ** 2.0).sum()
        loss_cpu.backward()
        # Save CPU grad
        var cpu_grad = a_cpu.grad().copy()

        # Clear retained grad before second pass
        a_cpu.zero_grad()

        var loss_gpu = (a_gpu ** 2.0).sum()
        loss_gpu.backward()

        # Now a_cpu.grad() reflects only GPU backward contribution
        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ── Main ──────────────────────────────────────────────────────────────────────


fn main() raises:
    _ = """
    # CPU forward
    test_exp_cpu_1d_square()
    test_exp_cpu_1d_cube()
    test_exp_cpu_1d_fractional()
    test_exp_cpu_1d_zero_exponent()
    test_exp_cpu_1d_one_exponent()
    test_exp_cpu_2d_forward()
    test_exp_cpu_3d_forward()
    test_exp_cpu_ones_any_exponent()
    test_exp_cpu_no_grad()
    test_exp_cpu_requires_grad_propagates()
    test_exp_cpu_suppress_grad()

    # CPU backward
    test_exp_cpu_1d_backward_square()
    test_exp_cpu_1d_backward_cube()
    test_exp_cpu_1d_backward_fractional()
    test_exp_cpu_2d_backward()
    test_exp_cpu_3d_backward()
    test_exp_cpu_backward_chain()
    test_exp_cpu_backward_chained_pow()
    test_exp_cpu_backward_one_exponent()

    # GPU forward
    test_exp_gpu_1d_square()
    test_exp_gpu_1d_cube()
    test_exp_gpu_1d_fractional()
    test_exp_gpu_1d_zero_exponent()
    test_exp_gpu_2d_forward()
    test_exp_gpu_3d_forward()
    test_exp_gpu_ones_any_exponent()

    # GPU backward
    test_exp_gpu_1d_backward_square()
    test_exp_gpu_1d_backward_cube()
    test_exp_gpu_1d_backward_fractional()
    test_exp_gpu_2d_backward()
    test_exp_gpu_3d_backward()
    test_exp_gpu_backward_chain()
    test_exp_gpu_backward_chained_pow()
    test_exp_gpu_backward_one_exponent()

    # Parity
    test_exp_parity_1d_forward()
    test_exp_parity_2d_forward()
    test_exp_parity_fractional_forward()
    test_exp_parity_1d_backward()
    test_exp_parity_2d_backward()
    test_exp_parity_chain_backward()
    test_exp_parity_using_zero_grad()

    print("All exponentiation tests passed!")
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
