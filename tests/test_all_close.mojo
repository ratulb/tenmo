from tenmo.tensor import Tensor
from std.testing import assert_true, assert_false, TestSuite

def test_all_close_nan_default_false() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([Float32(0.0) / Float32(0.0)])
    var b = Tensor[dtype].d1([Float32(0.0) / Float32(0.0)])
    assert_false(a.all_close(b))

def test_all_close_nan_explicit_true() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([Float32(0.0) / Float32(0.0)])
    var b = Tensor[dtype].d1([Float32(0.0) / Float32(0.0)])
    assert_true(a.all_close[treat_nan_equal=True](b))

def test_all_close_inf() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([Float32(1.0) / Float32(0.0)])
    var b = Tensor[dtype].d1([Float32(1.0) / Float32(0.0)])
    assert_true(a.all_close(b))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
