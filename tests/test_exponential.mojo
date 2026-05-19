from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.numpy_interop import to_ndarray
from std.python import Python

from tenmo.shapes import Shape
from std.sys import has_accelerator
from std.math import exp as scalar_exp

comptime dtype = DType.float32
comptime tol = Float32(1e-4)


def test_tensor_exponent() raises:
    """
    Tests tensor exp implementation.

    """
    np = Python.import_module("numpy")
    builtins = Python.import_module("builtins")
    tensor = Tensor[dtype].rand(4, 5)
    tensor_exp = tensor.exp()
    ndarray = to_ndarray(tensor)
    ndarray_exp = np.exp(ndarray)
    # builtins.print(ndarray_exp)
    tensor_exp_ndarray = to_ndarray(tensor_exp)
    # builtins.print(tensor_exp_ndarray)
    result = np.allclose(ndarray_exp, tensor_exp_ndarray)
    # builtins.print(result)
    # builtins.print(builtins.type(result))
    assert_true(result, "Tensor exponentiation failed")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def close(a: Tensor[dtype], b: Tensor[dtype]) raises -> Bool:
    return a.all_close[atol=tol](b)


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD — CPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_exp_cpu_scalar() raises:
    var t = Tensor[dtype].scalar(0.0)
    var out = t.exp()
    assert_true(close(out, Tensor[dtype].scalar(1.0)))


def test_exp_cpu_scalar_one() raises:
    var t = Tensor[dtype].scalar(1.0)
    var out = t.exp()
    assert_true(close(out, Tensor[dtype].scalar(scalar_exp(Float32(1.0)))))


def test_exp_cpu_1d_zeros() raises:
    var t = Tensor[dtype].zeros(Shape(5))
    var out = t.exp()
    assert_true(close(out, Tensor[dtype].ones(Shape(5))))


def test_exp_cpu_1d_known_values() raises:
    var t = Tensor[dtype].d1([0.0, 1.0, -1.0, 2.0])
    var out = t.exp()
    var expected = Tensor[dtype].d1(
        [
            1.0,
            scalar_exp(Float32(1.0)),
            scalar_exp(Float32(-1.0)),
            scalar_exp(Float32(2.0)),
        ]
    )
    assert_true(close(out, expected))


def test_exp_cpu_2d_contiguous() raises:
    var t = Tensor[dtype].zeros(Shape(3, 4))
    var out = t.exp()
    assert_true(out.shape() == Shape(3, 4))
    assert_true(close(out, Tensor[dtype].ones(Shape(3, 4))))


def test_exp_cpu_2d_known_values() raises:
    var t = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 2.0]])
    var out = t.exp()
    var expected = Tensor[dtype].d2(
        [
            [1.0, scalar_exp(Float32(1.0))],
            [scalar_exp(Float32(-1.0)), scalar_exp(Float32(2.0))],
        ]
    )
    assert_true(close(out, expected))


def test_exp_cpu_3d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4))
    var out = t.exp()
    assert_true(out.shape() == Shape(2, 3, 4))
    assert_true(close(out, Tensor[dtype].ones(Shape(2, 3, 4))))


def test_exp_cpu_4d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4, 5))
    var out = t.exp()
    assert_true(out.shape() == Shape(2, 3, 4, 5))
    assert_true(close(out, Tensor[dtype].ones(Shape(2, 3, 4, 5))))


def test_exp_cpu_non_contiguous_transposed() raises:
    var t = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 2.0]])
    var t_T = t.transpose()
    var out = t_T.exp()
    # exp of transposed — shape (2,2), values match transposed expected
    var expected = Tensor[dtype].d2(
        [
            [1.0, scalar_exp(Float32(-1.0))],
            [scalar_exp(Float32(1.0)), scalar_exp(Float32(2.0))],
        ]
    )
    assert_true(close(out, expected))


def test_exp_cpu_negative_values() raises:
    var t = Tensor[dtype].d1([-2.0, -1.0, 0.0, 1.0, 2.0])
    var out = t.exp()
    var expected = Tensor[dtype].d1(
        [
            scalar_exp(Float32(-2.0)),
            scalar_exp(Float32(-1.0)),
            1.0,
            scalar_exp(Float32(1.0)),
            scalar_exp(Float32(2.0)),
        ]
    )
    assert_true(close(out, expected))


def test_exp_cpu_large() raises:
    var t = Tensor[dtype].zeros(Shape(64, 128))
    var out = t.exp()
    assert_true(out.shape() == Shape(64, 128))
    assert_true(close(out, Tensor[dtype].ones(Shape(64, 128))))


def test_exp_cpu_no_requires_grad() raises:
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var out = t.exp[track_grad=False]()
    assert_true(not out.requires_grad)
    # assert_true(not out.has_backward_fn())


def test_exp_cpu_requires_grad_propagates() raises:
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var out = t.exp()
    assert_true(out.requires_grad)
    # assert_true(out.has_backward_fn())
    assert_true(out.has_ancestry())


def test_exp_cpu_no_grad_no_ancestry() raises:
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=False)
    var out = t.exp()
    assert_true(not out.requires_grad)
    assert_true(not out.has_ancestry())


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_exp_backward_cpu_scalar() raises:
    var t = Tensor[dtype].scalar(1.0, requires_grad=True)
    var out = t.exp()
    out.backward()
    # grad = exp(1) * 1 = e
    var expected = Tensor[dtype].scalar(scalar_exp(Float32(1.0)))
    assert_true(close(t.grad().as_tensor(), expected))


def test_exp_backward_cpu_zeros() raises:
    var t = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = t.exp()
    out.backward()
    # grad = exp(0) * 1 = 1
    assert_true(close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(4))))


def test_exp_backward_cpu_1d_known() raises:
    var t = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
    var out = t.exp()
    out.backward()
    # grad_input = grad_output * exp(input) = 1 * exp(input)
    var expected = Tensor[dtype].d1(
        [
            1.0,
            scalar_exp(Float32(1.0)),
            scalar_exp(Float32(-1.0)),
        ]
    )
    assert_true(close(t.grad().as_tensor(), expected))


def test_exp_backward_cpu_2d() raises:
    var t = Tensor[dtype].zeros(Shape(3, 4), requires_grad=True)
    var out = t.exp()
    out.backward()
    # grad = exp(0) = 1 everywhere
    assert_true(close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4))))


def test_exp_backward_cpu_3d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4), requires_grad=True)
    var out = t.exp()
    out.backward()
    assert_true(close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4))))


def test_exp_backward_cpu_chain_rule() raises:
    # y = exp(x * 2), dy/dx = 2 * exp(2x)
    var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
    var t2 = t * Scalar[dtype](2)
    var out = t2.exp()
    out.backward()
    var expected = Tensor[dtype].d1(
        [
            2.0 * scalar_exp(Float32(0.0)),
            2.0 * scalar_exp(Float32(2.0)),
        ]
    )
    assert_true(close(t.grad().as_tensor(), expected))


def test_exp_backward_cpu_sum_scalar() raises:
    # loss = sum(exp(x)), dloss/dx_i = exp(x_i)
    var t = Tensor[dtype].d1([0.0, 1.0, 2.0], requires_grad=True)
    var out = t.exp().sum()
    out.backward()
    var expected = Tensor[dtype].d1(
        [
            scalar_exp(Float32(0.0)),
            scalar_exp(Float32(1.0)),
            scalar_exp(Float32(2.0)),
        ]
    )
    assert_true(close(t.grad().as_tensor(), expected))


def test_exp_backward_cpu_large() raises:
    var t = Tensor[dtype].zeros(Shape(32, 32), requires_grad=True)
    var out = t.exp()
    out.backward()
    assert_true(close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(32, 32))))


def test_exp_backward_cpu_double_exp() raises:
    # y = exp(exp(x)), dy/dx = exp(x) * exp(exp(x))
    var t = Tensor[dtype].scalar(0.0, requires_grad=True)
    var out = t.exp().exp()
    out.backward()
    # x=0: exp(0)=1, exp(exp(0))=exp(1)=e, grad = 1 * e = e
    var expected = Tensor[dtype].scalar(scalar_exp(Float32(1.0)))
    assert_true(close(t.grad().as_tensor(), expected))


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_exp_gpu_scalar() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(0.0).to_gpu()
        var out = t.exp()
        assert_true(out.is_on_gpu())
        assert_true(close(out.to_cpu(), Tensor[dtype].scalar(1.0)))


def test_exp_gpu_1d_zeros() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(8)).to_gpu()
        var out = t.exp()
        assert_true(out.is_on_gpu())
        assert_true(close(out.to_cpu(), Tensor[dtype].ones(Shape(8))))


def test_exp_gpu_1d_known_values() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d1([0.0, 1.0, -1.0, 2.0])
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        assert_true(out.is_on_gpu())
        var expected = t_cpu.exp()
        assert_true(close(out.to_cpu(), expected))


def test_exp_gpu_2d_contiguous() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(4, 5)
        var t_gpu = t_cpu.to_gpu()
        var out_gpu = t_gpu.exp()
        var out_cpu = t_cpu.exp()
        assert_true(out_gpu.is_on_gpu())
        assert_true(out_gpu.shape() == Shape(4, 5))
        assert_true(close(out_gpu.to_cpu(), out_cpu))


def test_exp_gpu_3d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(2, 3, 4)
        var t_gpu = t_cpu.to_gpu()
        var out_gpu = t_gpu.exp()
        var out_cpu = t_cpu.exp()
        assert_true(out_gpu.is_on_gpu())
        assert_true(close(out_gpu.to_cpu(), out_cpu))


def test_exp_gpu_4d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(2, 3, 4, 5)
        var t_gpu = t_cpu.to_gpu()
        var out_gpu = t_gpu.exp()
        var out_cpu = t_cpu.exp()
        assert_true(out_gpu.is_on_gpu())
        assert_true(close(out_gpu.to_cpu(), out_cpu))


def test_exp_gpu_negative_values() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d1([-2.0, -1.0, 0.0, 1.0, 2.0])
        var t_gpu = t_cpu.to_gpu()
        var out_gpu = t_gpu.exp()
        var out_cpu = t_cpu.exp()
        assert_true(close(out_gpu.to_cpu(), out_cpu))


def test_exp_gpu_large() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(64, 128)
        var t_gpu = t_cpu.to_gpu()
        var out_gpu = t_gpu.exp()
        var out_cpu = t_cpu.exp()
        assert_true(out_gpu.is_on_gpu())
        assert_true(close(out_gpu.to_cpu(), out_cpu))


def test_exp_gpu_matches_cpu() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(9, 20)
        var t_gpu = t_cpu.to_gpu()
        assert_true(close(t_gpu.exp().to_cpu(), t_cpu.exp()))


def test_exp_gpu_no_requires_grad() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([1.0, 2.0]).to_gpu()
        var out = t.exp[track_grad=False]()
        assert_true(not out.requires_grad)
        # assert_true(not out.has_backward_fn())


def test_exp_gpu_requires_grad_propagates() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var out = t.exp()
        assert_true(out.is_on_gpu())
        assert_true(out.requires_grad)
        # assert_true(out.has_backward_fn())


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_exp_backward_gpu_zeros() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        out.backward()
        # grad = exp(0) = 1
        assert_true(
            close(t_cpu.grad().as_tensor(), Tensor[dtype].ones(Shape(4)))
        )


def test_exp_backward_gpu_1d_known() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        out.backward()
        var expected = Tensor[dtype].d1(
            [
                1.0,
                scalar_exp(Float32(1.0)),
                scalar_exp(Float32(-1.0)),
            ]
        )
        assert_true(close(t_cpu.grad().as_tensor(), expected))


def test_exp_backward_gpu_matches_cpu() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(4, 5, requires_grad=True)
        var t_gpu = t_cpu.to_gpu()

        # CPU backward
        var out_cpu = t_cpu.exp()
        out_cpu.backward()
        var grad_cpu = t_cpu.grad().as_tensor().copy()
        t_cpu.zero_grad()

        # GPU backward
        var out_gpu = t_gpu.exp()
        out_gpu.backward()

        assert_true(close(t_cpu.grad().as_tensor(), grad_cpu))


def test_exp_backward_gpu_2d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].zeros(Shape(3, 4), requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        out.backward()
        assert_true(
            close(t_cpu.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4)))
        )


def test_exp_backward_gpu_3d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].zeros(Shape(2, 3, 4), requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        out.backward()
        assert_true(
            close(t_cpu.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4)))
        )


def test_exp_backward_gpu_chain_rule() raises:
    comptime if has_accelerator():
        # y = exp(x * 2), dy/dx = 2 * exp(2x)
        var t_cpu = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var t2 = t_gpu * Scalar[dtype](2)
        var out = t2.exp()
        out.backward()
        var expected = Tensor[dtype].d1(
            [
                2.0 * scalar_exp(Float32(0.0)),
                2.0 * scalar_exp(Float32(2.0)),
            ]
        )
        assert_true(close(t_cpu.grad().as_tensor(), expected))


def test_exp_backward_gpu_large() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].zeros(Shape(32, 32), requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        out.backward()
        assert_true(
            close(t_cpu.grad().as_tensor(), Tensor[dtype].ones(Shape(32, 32)))
        )


def test_exp_backward_gpu_double_exp() raises:
    comptime if has_accelerator():
        # y = exp(exp(x)), x=0: grad = exp(0)*exp(exp(0)) = 1*e = e
        var t_cpu = Tensor[dtype].scalar(0.0, requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp().exp()
        out.backward()
        var expected = Tensor[dtype].scalar(scalar_exp(Float32(1.0)))
        assert_true(close(t_cpu.grad().as_tensor(), expected))


def test_exp_backward_gpu_scalar() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].scalar(1.0, requires_grad=True)
        var t_gpu = t_cpu.to_gpu()
        var out = t_gpu.exp()
        out.backward()
        var expected = Tensor[dtype].scalar(scalar_exp(Float32(1.0)))
        assert_true(close(t_cpu.grad().as_tensor(), expected))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
