from tenmo import Tensor
from std.testing import assert_true
from tenmo.common_utils import s, i
from std.sys.defines import get_defined_string

comptime dtype = DType.float32

def main() raises:
    #test_slice_every_second_row_column1()
    #test_permute_backward()
    #test_add_backward()
    test_concat_concat_view_with_tensor_ct()
    _="""comptime BLAS_PATH = get_defined_string[
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
