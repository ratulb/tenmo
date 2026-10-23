from tenmo.tensor import Tensor
from tenmo.shared.indexhelper import i, s
from std.testing import assert_true, assert_equal, assert_raises, TestSuite

def test_basic_negative_index_read() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert_true(t[-1, -1] == 6.0)
    assert_true(t[-2, 0] == 3.0)
    assert_true(t[-3, -2] == 1.0)


def test_basic_negative_index_write() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].zeros(2, 2)
    t[-1, -1] = 9.0
    t[-2, 0] = 7.0
    assert_true(t[1, 1] == 9.0)
    assert_true(t[0, 0] == 7.0)


def test_slice_negative_indices() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var sl = x[-3:-1]
    assert_true(sl.shape()[0] == 2)
    assert_true(sl[0] == 4.0)
    assert_true(sl[1] == 5.0)


def test_advanced_negative_index_int() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0])
    var v = t[i(-1)]
    assert_true(v.shape().num_elements() == 1)
    assert_true(v.item() == 40.0)


def test_gather_negative_indices() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0])
    var idx_list = List[Int]()
    idx_list.append(-1)
    idx_list.append(-2)
    var g = t.gather(idx_list, axis=0)
    assert_true(g[0] == 40.0)
    assert_true(g[1] == 30.0)



def test_gather_negative_axis() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var idx_list = List[Int]()
    idx_list.append(0)
    var g = t.gather(idx_list, axis=-1)
    assert_true(g[0, 0] == 1.0)


def test_chunk_negative_not_supported_placeholder() raises:
    # chunk with negative indices should normalize like other paths
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var c = t.chunk(i(-1))
    assert_true(c.shape().num_elements() >= 1)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
