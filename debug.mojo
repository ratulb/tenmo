from tenmo import Tensor
from std.testing import assert_true
from tenmo.common_utils import s, i
from std.sys.defines import get_defined_string

comptime dtype = DType.float32


def main() raises:
    sum_up()
    # test_slice_every_second_row_column1()
    # test_permute_backward()
    # test_add_backward()
    # test_concat_concat_view_with_tensor_ct()
    _ = """comptime BLAS_PATH = get_defined_string[
        "BLAS_PATH", "/lib/x86_64-linux-gnu/libopenblas.so.0"
    ]()
    print("BLAS_PATH: ", BLAS_PATH)"""


def test_slice_every_second_row_column1() raises:
    print("test_slice_every_second_row_column1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(15, requires_grad=True)
    var r = a.reshape(5, 3)
    var v = r[s(None, None, 2), i(1)]  # Select col 1 of rows 0, 2, 4
    var loss = v.sum()
    loss.backward()
    grad = a.grad().copy()
    assert_true(grad.shape() == a.shape())
    assert_true(grad[1] == 1)  # r[0,1]
    assert_true(grad[7] == 1)  # r[2,1]
    assert_true(grad[13] == 1)  # r[4,1]
    assert_true(grad.sum().item() == 3)


def test_permute_backward() raises:
    print("test_permute_backward")
    comptime dtype = DType.float32

    var a = Tensor[dtype].arange(6, requires_grad=True)
    var v = a.view([2, 3])
    print("We are good 1?")
    var p = v.permute([1, 0])  # shape (3, 2), stride [1, 3]

    print("We are good 2?")
    var flat = p.reshape([6])

    print("We are good 3?")
    flat.backward()

    print("We are good 4?")

    var expected = Tensor[dtype].d1([1, 1, 1, 1, 1, 1])

    print("We are good 5?")
    assert_true((a.grad() == expected))

    print("We are good 6?")


def test_add_backward() raises:
    comptime dtype = DType.float32
    A1 = Tensor[dtype].d2([[1, 2, 3]], requires_grad=True)
    AV = A1.into_view()
    AV.backward(3)
    AV.backward()
    AV.backward()
    assert_true(
        (A1.gradients()[] == Tensor[dtype].d2([[5, 5, 5]])),
        "Tensor view backward 4 times grad assertion failed",
    )

    a = Tensor[dtype].d2([[1, 2, 3]], requires_grad=True)
    b = Tensor[dtype].d1([1, 2, 3], requires_grad=True)
    c = a + b
    d = b + a
    e = c + d
    e.backward(26)
    assert_true(
        (a.gradients()[] == Tensor[dtype].d2([[52, 52, 52]])),
        "2D + 1D grad assertion 1 failed",
    )
    assert_true(
        (b.gradients()[] == Tensor[dtype].d1([52, 52, 52])),
        "2D + 1D grad assertion 2 failed",
    )
    ev = e.into_view()
    ev.backward()
    a.gradients()[].print()
    b.gradients()[].print()
    assert_true(
        (a.gradients()[] == Tensor[dtype].d2([[104, 104, 104]])),
        "2D + 1D grad assertion 3 failed",
    )
    assert_true(
        (b.gradients()[] == Tensor[dtype].d1([104, 104, 104])),
        "2D + 1D grad assertion 4 failed",
    )


def test_concat_concat_view_with_tensor_ct() raises:
    """Test concatenating view with regular tensor."""
    comptime dtype = DType.float32

    var base = Tensor[dtype].ones(2, 3, requires_grad=True)
    var view = base.view([2, 3], offset=0)

    var regular = Tensor[dtype].ones(1, 3, requires_grad=True) * 2.0

    var tensors = List[Tensor[dtype]]()
    tensors.append(view)
    tensors.append(regular)

    var result = Tensor[dtype].concat(tensors, axis=0)
    result.print()

    # Shape should be (3, 3)
    assert_true(result.shape()[0] == 3)

    # Values check
    assert_true(result[0, 0] == 1.0)
    assert_true(result[2, 0] == 2.0)

    # Gradient flow
    var loss = result.sum()
    loss.backward()

    assert_true(
        base.grad().all_close[atol=1e-6](Tensor[dtype].ones(base.shape()))
    )
    regular.grad().print()
    assert_true(
        regular.grad().all_close[atol=1e-6](Tensor[dtype].ones(regular.shape()))
    )


def sum_up():
    var a = Tensor[dtype].d2(
        [
            [0.025],
            [0.124],
            [0.050],
            [0.014],
            [0.227],
            [0.033],
            [0.029],
            [0.023],
            [0.075],
            [0.093],
            [0.084],
            [0.072],
            [0.104],
            [0.118],
            [0.108],
            [0.040],
            [0.098],
            [0.052],
            [0.161],
            [0.019],
            [0.059],
            [0.065],
            [0.101],
            [0.036],
            [0.133],
            [0.095],
            [0.089],
            [0.038],
            [0.100],
            [0.103],
            [0.099],
            [0.132],
            [0.119],
            [0.237],
            [0.117],
            [0.402],
            [0.122],
            [0.190],
            [0.185],
            [0.165],
            [0.163],
            [0.155],
            [0.142],
            [0.169],
            [0.111],
            [5.089],
            [0.135],
            [0.054],
            [0.201],
            [0.160],
            [0.196],
            [0.023],
            [0.109],
            [0.626],
            [0.177],
            [0.151],
            [0.008],
            [0.006],
            [0.004],
            [0.004],
            [0.097],
            [0.010],
            [0.006],
            [0.060],
            [0.048],
            [0.074],
            [0.031],
            [0.031],
            [0.038],
            [0.096],
            [0.016],
            [0.014],
            [0.372],
            [0.020],
            [0.018],
            [0.011],
            [0.020],
            [0.014],
            [0.012],
            [0.011],
            [0.013],
            [0.009],
            [0.067],
            [0.048],
            [0.044],
            [0.041],
            [0.007],
            [0.007],
            [0.094],
            [0.170],
            [0.057],
            [0.140],
            [0.141],
            [0.237],
            [0.402],
            [0.155],
            [0.147],
            [0.143],
            [0.234],
            [0.103],
            [0.166],
            [0.269],
            [0.153],
            [0.126],
            [0.129],
            [0.074],
            [0.083],
            [0.084],
            [0.080],
            [0.076],
            [0.061],
            [0.060],
            [0.057],
            [0.009],
            [0.003],
            [0.003],
            [0.009],
            [0.010],
            [0.009],
            [0.008],
            [0.003],
            [0.005],
            [0.007],
            [0.006],
            [0.010],
            [0.010],
            [0.015],
            [0.011],
            [0.008],
            [0.024],
            [0.008],
            [78.054],
            [0.013],
            [0.094],
            [0.076],
            [0.080],
            [0.168],
            [0.113],
            [0.013],
            [0.079],
            [0.044],
            [0.060],
            [0.107],
            [0.045],
            [0.082],
            [0.091],
            [0.094],
            [0.081],
            [0.092],
            [0.044],
            [0.068],
            [0.025],
            [0.041],
            [0.026],
            [0.014],
            [0.012],
            [0.014],
            [0.023],
            [0.016],
            [0.013],
            [0.020],
            [0.017],
            [0.019],
            [0.013],
            [0.052],
            [0.060],
            [0.047],
            [0.056],
            [0.086],
            [0.039],
            [0.016],
            [0.015],
            [0.023],
            [0.086],
            [0.071],
            [0.082],
            [0.005],
            [0.003],
            [0.003],
            [0.003],
            [0.003],
            [0.003],
            [0.002],
            [0.003],
            [0.004],
            [0.004],
            [0.004],
            [0.004],
            [0.004],
            [0.004],
            [0.004],
            [0.008],
            [0.015],
            [0.013],
            [0.013],
            [0.004],
            [0.940],
            [1.883],
            [1.069],
            [22.733],
            [0.006],
            [0.016],
            [0.005],
            [0.004],
            [0.006],
            [0.008],
            [0.008],
            [106.006],
            [0.009],
            [0.005],
            [0.008],
            [0.006],
            [0.005],
            [0.006],
            [0.007],
            [0.007],
            [0.016],
            [0.005],
            [0.034],
            [0.004],
            [0.022],
            [0.004],
            [0.049],
            [0.027],
            [0.023],
            [0.025],
            [0.027],
            [0.044],
            [0.046],
            [0.033],
            [0.034],
            [0.019],
            [0.011],
            [0.015],
            [0.012],
            [0.012],
            [0.012],
            [0.014],
            [0.049],
            [0.034],
            [0.037],
            [0.036],
            [0.036],
            [0.037],
            [0.040],
            [0.018],
            [0.012],
            [0.013],
            [0.101],
            [0.023],
            [0.040],
            [0.049],
            [0.088],
            [0.027],
            [0.020],
            [0.021],
            [0.022],
            [0.027],
            [0.031],
            [0.027],
            [0.026],
            [0.038],
            [0.020],
            [0.019],
            [0.024],
            [0.019],
            [0.024],
        ]
    )

    var s = a.sum()
    print(s.item())
