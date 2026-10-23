from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

def test_gather_points_2d() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    var idx = List[List[Int]]()
    var p0 = List[Int](); p0.append(0); p0.append(2)
    var p1 = List[Int](); p1.append(1); p1.append(1)
    var p2 = List[Int](); p2.append(2); p2.append(0)
    idx.append(p0^)
    idx.append(p1^)
    idx.append(p2^)
    var out = t.gather_points(idx)
    assert_true(out.shape()[0] == 3)
    assert_true(out[0] == 3.0)
    assert_true(out[1] == 5.0)
    assert_true(out[2] == 7.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
