from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shapes import Shape


def test_retain_graph_add() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid = a + a
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Add: retain_graph=False should zero intermediate grad",
    )
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
        "Add: leaf grad correct with retain_graph=False",
    )


def test_retain_graph_add_true() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid = a + a
    var out = mid.sum()
    out.backward(retain_graph=True)
    assert_true(
        mid.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Add: retain_graph=True should preserve intermediate grad",
    )
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
        "Add: leaf grad correct with retain_graph=True",
    )


def test_retain_graph_add_broadcast() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var b = Tensor[dtype].full(Shape(3), 2.0, requires_grad=True)
    var mid = a + b
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "AddBroadcast: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var b2 = Tensor[dtype].full(Shape(3), 2.0, requires_grad=True)
    var mid2 = a2 + b2
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "AddBroadcast: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_add_scalar() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 1.0, requires_grad=True)
    var mid = a + 5.0
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "AddScalar: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 1.0, requires_grad=True)
    var mid2 = a2 + 5.0
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "AddScalar: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_sub() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 5.0, requires_grad=True)
    var mid = a - 3.0
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Sub: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 5.0, requires_grad=True)
    var mid2 = a2 - 3.0
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Sub: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_mul() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var mid = a * a
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Mul: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var mid2 = a2 * a2
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Mul: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_div() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 6.0, requires_grad=True)
    var mid = a / 2.0
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Div: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 6.0, requires_grad=True)
    var mid2 = a2 / 2.0
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Div: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_neg() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid = -a
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Neg: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid2 = -a2
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Neg: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_exp() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(3), 1.0, requires_grad=True)
    var mid = a.exp()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(3))),
        "Exp: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(3), 1.0, requires_grad=True)
    var mid2 = a2.exp()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(3), 1.0)),
        "Exp: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_log() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(3), 2.0, requires_grad=True)
    var mid = a.log()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(3))),
        "Log: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(3), 2.0, requires_grad=True)
    var mid2 = a2.log()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(3), 1.0)),
        "Log: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_sqrt() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(3), 4.0, requires_grad=True)
    var mid = a.sqrt()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(3))),
        "Sqrt: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(3), 4.0, requires_grad=True)
    var mid2 = a2.sqrt()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(3), 1.0)),
        "Sqrt: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_sum() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid = a.sum(axes=[0])
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(3))),
        "Sum: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid2 = a2.sum(axes=[0])
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(3), 1.0)),
        "Sum: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_mean() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 2.0, requires_grad=True)
    var mid = a.mean(axes=[1])
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2))),
        "Mean: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 2.0, requires_grad=True)
    var mid2 = a2.mean(axes=[1])
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(2), 1.0)),
        "Mean: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_matmul() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var b = Tensor[dtype].full(Shape(3, 2), 2.0, requires_grad=True)
    var mid = a.matmul(b)
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 2))),
        "Matmul: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var b2 = Tensor[dtype].full(Shape(3, 2), 2.0, requires_grad=True)
    var mid2 = a2.matmul(b2)
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)),
        "Matmul: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_dot() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var b = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var mid = a.dot(b)
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape())),
        "Dot: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var b2 = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var mid2 = a2.dot(b2)
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(), 1.0)),
        "Dot: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_relu() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid = a.relu()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "ReLU: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid2 = a2.relu()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "ReLU: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_sigmoid() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 0.5, requires_grad=True)
    var mid = a.sigmoid()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Sigmoid: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 0.5, requires_grad=True)
    var mid2 = a2.sigmoid()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Sigmoid: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_power() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var mid = a ** 2
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Power: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var mid2 = a2 ** 2
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Power: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_view_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid = a.into_view()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "View: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid2 = a2.into_view()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "View: retain_graph=True should ALSO zero intermediate grad (view)",
    )
    assert_true(
        a2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "View: leaf grad correct",
    )


def test_retain_graph_reshape_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid = a.reshape(6)
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(6))),
        "Reshape: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid2 = a2.reshape(6)
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(6))),
        "Reshape: retain_graph=True should ALSO zero intermediate grad (view)",
    )
    assert_true(
        a2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "Reshape: leaf grad correct",
    )


def test_retain_graph_transpose_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid = a.transpose()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(3, 2))),
        "Transpose: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid2 = a2.transpose()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(3, 2))),
        "Transpose: retain_graph=True should ALSO zero intermediate grad (view)",
    )
    assert_true(
        a2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "Transpose: leaf grad correct",
    )


def test_retain_graph_permute_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3, 4), 1.0, requires_grad=True)
    var mid = a.permute(axes=[2, 0, 1])
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4, 2, 3))),
        "Permute: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3, 4), 1.0, requires_grad=True)
    var mid2 = a2.permute(axes=[2, 0, 1])
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(4, 2, 3))),
        "Permute: retain_graph=True should ALSO zero intermediate grad (view)",
    )


def test_retain_graph_squeeze_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(1, 3, 1), 1.0, requires_grad=True)
    var mid = a.squeeze()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(3))),
        "Squeeze: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(1, 3, 1), 1.0, requires_grad=True)
    var mid2 = a2.squeeze()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(3))),
        "Squeeze: retain_graph=True should ALSO zero intermediate grad (view)",
    )


def test_retain_graph_unsqueeze_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(3), 1.0, requires_grad=True)
    var mid = a.unsqueeze(0)
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(1, 3))),
        "Unsqueeze: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(3), 1.0, requires_grad=True)
    var mid2 = a2.unsqueeze(0)
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(1, 3))),
        "Unsqueeze: retain_graph=True should ALSO zero intermediate grad (view)",
    )


def test_retain_graph_expand_zero_grad_always() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(3), 1.0, requires_grad=True)
    var mid = a.expand(2, 3)
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "Expand: retain_graph=False should zero intermediate grad (view)",
    )

    var a2 = Tensor[dtype].full(Shape(3), 1.0, requires_grad=True)
    var mid2 = a2.expand(2, 3)
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "Expand: retain_graph=True should ALSO zero intermediate grad (view)",
    )


def test_retain_graph_flatten() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid = a.flatten()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(6))),
        "Flatten: retain_graph=False should zero intermediate grad (non-view)",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid2 = a2.flatten()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)),
        "Flatten: retain_graph=True should preserve intermediate grad (non-view)",
    )


def test_retain_graph_contiguous() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid = a.contiguous()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "Contiguous: retain_graph=False should zero intermediate grad (non-view)",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
    var mid2 = a2.contiguous()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "Contiguous: retain_graph=True should preserve intermediate grad (non-view)",
    )


def test_retain_graph_tanh() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 0.5, requires_grad=True)
    var mid = a.tanh()
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "Tanh: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 0.5, requires_grad=True)
    var mid2 = a2.tanh()
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Tanh: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_softmax() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 0.5, requires_grad=True)
    var mid = a.softmax(axes=[1])
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "Softmax: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 0.5, requires_grad=True)
    var mid2 = a2.softmax(axes=[1])
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "Softmax: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_mul_broadcast() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(2, 3), 2.0, requires_grad=True)
    var b = Tensor[dtype].full(Shape(3), 3.0, requires_grad=True)
    var mid = a * b
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(2, 3))),
        "MulBroadcast: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(2, 3), 2.0, requires_grad=True)
    var b2 = Tensor[dtype].full(Shape(3), 3.0, requires_grad=True)
    var mid2 = a2 * b2
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
        "MulBroadcast: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_mul_scalar() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid = a * 3.0
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(4))),
        "MulScalar: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid2 = a2 * 3.0
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "MulScalar: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_clip() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].linspace(0.0, 5.0, 6, requires_grad=True)
    var mid = a.clip(1.0, 4.0)
    var out = mid.sum()
    out.backward(retain_graph=False)
    assert_true(
        mid.grad().all_close(Tensor[dtype].zeros(Shape(6))),
        "Clip: retain_graph=False should zero intermediate grad",
    )

    var a2 = Tensor[dtype].linspace(0.0, 5.0, 6, requires_grad=True)
    var mid2 = a2.clip(1.0, 4.0)
    var out2 = mid2.sum()
    out2.backward(retain_graph=True)
    assert_true(
        mid2.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)),
        "Clip: retain_graph=True should preserve intermediate grad",
    )


def test_retain_graph_complex_graph() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var b = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var c = a + b
    var d = a * b
    var mid = c + d
    var out = mid.sum()
    out.backward(retain_graph=True)

    assert_true(
        c.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Complex: c grad preserved with retain_graph=True",
    )
    assert_true(
        d.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Complex: d grad preserved with retain_graph=True",
    )
    assert_true(
        mid.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Complex: mid grad preserved with retain_graph=True",
    )
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(4), 4.0)),
        "Complex: a leaf grad correct",
    )
    assert_true(
        b.grad().all_close(Tensor[dtype].full(Shape(4), 3.0)),
        "Complex: b leaf grad correct",
    )


def test_retain_graph_multiple_backward() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var mid = a * a
    var out = mid.sum()

    out.backward(retain_graph=True)
    assert_true(
        mid.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)),
        "Multiple backward: first call preserves mid grad",
    )
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(4), 4.0)),
        "Multiple backward: first call a grad=2*2=4",
    )

    a.zero_grad()
    out.backward(retain_graph=True)
    assert_true(
        mid.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
        "Multiple backward: second call mid grad accumulates (1+1=2)",
    )
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(4), 8.0)),
        "Multiple backward: second call a grad=2*2*2=8",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GPU retain_graph tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_retain_graph_gpu_add() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var mid_gpu = a_gpu + a_gpu
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(4))),
            "GPU Add: retain_graph=False should zero intermediate grad",
        )
        assert_true(
            a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
            "GPU Add: leaf grad correct with retain_graph=False",
        )

        var a2 = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var mid2_gpu = a2_gpu + a2_gpu
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Add: retain_graph=True should preserve intermediate grad",
        )
        assert_true(
            a2.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
            "GPU Add: leaf grad correct with retain_graph=True",
        )


def test_retain_graph_gpu_matmul() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var b = Tensor[dtype].full(Shape(3, 2), 2.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var mid_gpu = a_gpu.matmul(b_gpu)
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(2, 2))),
            "GPU Matmul: retain_graph=False should zero intermediate grad",
        )

        var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var b2 = Tensor[dtype].full(Shape(3, 2), 2.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var b2_gpu = b2.to_gpu()
        var mid2_gpu = a2_gpu.matmul(b2_gpu)
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)),
            "GPU Matmul: retain_graph=True should preserve intermediate grad",
        )


def test_retain_graph_gpu_sum() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var mid_gpu = a_gpu.sum(axes=[0])
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(3))),
            "GPU Sum: retain_graph=False should zero intermediate grad",
        )

        var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var mid2_gpu = a2_gpu.sum(axes=[0])
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(3), 1.0)),
            "GPU Sum: retain_graph=True should preserve intermediate grad",
        )


def test_retain_graph_gpu_view_zero_grad_always() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var mid_gpu = a_gpu.into_view()
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(2, 3))),
            "GPU View: retain_graph=False should zero intermediate grad (view)",
        )

        var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var mid2_gpu = a2_gpu.into_view()
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(2, 3))),
            "GPU View: retain_graph=True should ALSO zero intermediate grad (view)",
        )
        assert_true(
            a2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
            "GPU View: leaf grad correct",
        )


def test_retain_graph_gpu_complex_graph() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var b = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c_gpu = a_gpu + b_gpu
        var d_gpu = a_gpu * b_gpu
        var mid_gpu = c_gpu + d_gpu
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=True)

        assert_true(
            c_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Complex: c grad preserved with retain_graph=True",
        )
        assert_true(
            d_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Complex: d grad preserved with retain_graph=True",
        )
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Complex: mid grad preserved with retain_graph=True",
        )
        assert_true(
            a.grad().all_close(Tensor[dtype].full(Shape(4), 4.0)),
            "GPU Complex: a leaf grad correct",
        )
        assert_true(
            b.grad().all_close(Tensor[dtype].full(Shape(4), 3.0)),
            "GPU Complex: b leaf grad correct",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
