# run_tensor.mojo  (sits at repo root, outside tenmo/)
from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.reduction_kernel import ProductArg
from tenmo.intarray import IntArray
from std.testing import assert_true


fn main_1() raises:
    # whatever quick test you want
    var t = Tensor[DType.float32].ones(3, 3)
    t.print()


fn main_2() raises:
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a * 2
    var c = a * 3
    var d = b + c
    var loss = d.sum()
    loss.backward()
    # a.grad() == [5.0, 5.0, 5.0]
    a.grad().print()


fn main_3() raises:
    var a = NDBuffer[DType.float32](1.0, 2.0, 3.0)
    a.print()


fn test_prd_cpu_bwd_all_positive_1d() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2, 3, 4], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    # grad_x[i] = product / x[i] = 24 / x[i]
    a.grad().print()
    assert_true(a.grad().all_close[atol=1e-4](Tensor[dtype].d1([12, 8, 6])))


fn test_prd_cpu_bwd_all_positive_1d_orig() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var result = a.buffer.product_cpu(IntArray(0), False)
    var arg = result[1]
    # Check excl_product was stored
    print("excl_product is some:", arg.excl_product.__bool__())
    if arg.excl_product:
        var excl = arg.excl_product.value()
        print("excl[0]:", excl.get(0))
        print("excl[1]:", excl.get(1))
        print("excl[2]:", excl.get(2))


fn main() raises:
    # test_prd_cpu_bwd_all_positive_1d()
    test_prd_cpu_bwd_all_positive_1d()
