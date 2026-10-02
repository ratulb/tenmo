from tenmo.tensor import Tensor
from tenmo.numpy_interop import to_ndarray, from_ndarray

from std.python import Python
from std.testing import assert_true, assert_raises, TestSuite
from std.sys import has_accelerator


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_scalar_tensor_conversion() raises:
    print("test_scalar_tensor_conversion")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(42)
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a))


def test_1d_tensor_random() raises:
    print("test_1d_tensor_random")
    comptime dtype = DType.float32
    var a = Tensor[DType.float32].arange(10)
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a))


def test_2d_tensor_random() raises:
    print("test_2d_tensor_random")
    comptime dtype = DType.float32
    var a = Tensor[DType.float32].d2([[1.5, 2.5], [3.5, 4.5]])
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a))


def test_3d_tensor_random() raises:
    print("test_3d_tensor_random")
    comptime dtype = DType.float32
    var a = Tensor[DType.float32].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a))


def test_4d_tensor_random() raises:
    print("test_4d_tensor_random")
    comptime dtype = DType.float32
    var a = Tensor[DType.float32].d4(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ]
    )
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a))


def test_views_1d_2d() raises:
    print("test_views_1d_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(16)
    var b = a.view([4, 4])
    var v1 = b.view([2, 2], offset=3)
    var nd_v = to_ndarray(v1)
    var v_back = from_ndarray[DType.float32](nd_v)
    assert_true((v_back == v1))
    _ = a


def test_views_3d_4d() raises:
    print("test_views_3d_4d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(64)
    var b = a.view([4, 4, 4])
    var v = b.view([2, 2, 2], offset=10)
    var nd_v = to_ndarray(v)
    var v_back = from_ndarray[DType.float32](nd_v)
    assert_true((v_back == v))
    _ = a


def test_bool_tensor() raises:
    print("test_bool_tensor")
    comptime dtype = DType.float32
    var b = Tensor[DType.bool].full([3, 3], True)
    var nd_b = to_ndarray(b)
    var b_back = from_ndarray[DType.bool](nd_b)
    assert_true((b_back == b))


def test_copy_vs_zero_copy() raises:
    print("test_copy_vs_zero_copy")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(5)

    # copy=True
    var nd_a_copy = to_ndarray(a)
    var a_copy = from_ndarray[DType.float32](nd_a_copy, copy=True)
    assert_true((a_copy == a))

    # copy=False
    var a_zero = from_ndarray[DType.float32](nd_a_copy, copy=False)
    assert_true((a_zero == a))


def test_random_tensor_large() raises:
    print("test_random_tensor_large")
    comptime dtype = DType.float32
    var a = Tensor[DType.float32].arange(120)
    var b = a.view([2, 3, 4, 5])
    var nd_b = to_ndarray(b)
    var b_back = from_ndarray[DType.float32](nd_b)
    assert_true((b_back == b))


def test_1d_tensor_to_numpy_and_back() raises:
    print("test_1d_tensor_to_numpy_and_back")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(10)
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)

    assert_true((a == a_back))


def test_2d_tensor_to_numpy_and_back() raises:
    print("test_2d_tensor_to_numpy_and_back")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)

    assert_true((a == a_back))


def test_tensor_view_to_numpy_and_back() raises:
    print("test_tensor_view_to_numpy_and_back")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(16)
    var b = a.view([4, 4])
    var v = b.view([2, 2], offset=5)  # some arbitrary subview
    var nd_v = to_ndarray(v)
    var v_back = from_ndarray[DType.float32](nd_v)

    assert_true((v_back == v))
    _ = a


def test_bool_tensor_to_numpy_and_back() raises:
    print("test_bool_tensor_to_numpy_and_back")
    comptime dtype = DType.float32
    var b = Tensor[DType.bool].full([3, 3], True)
    var nd_b = to_ndarray(b)
    var b_back = from_ndarray[DType.bool](nd_b)

    assert_true((b_back == b))


def test_copy_vs_zero_copy_behavior() raises:
    print("test_copy_vs_zero_copy_behavior")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(5)

    # Copy
    var nd_a_copy = to_ndarray(a)
    var a_copy = from_ndarray[DType.float32](nd_a_copy, copy=True)
    assert_true((a_copy == a))

    # Zero-copy: requires user to keep ndarray alive
    var a_zero = from_ndarray[DType.float32](nd_a_copy, copy=False)
    assert_true((a_zero == a))


def test_zero_copy_view_aliases_numpy() raises:
    print("test_zero_copy_view_aliases_numpy")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var nd_a = to_ndarray(a)
    var a_zero = from_ndarray[DType.float32](nd_a, copy=False)
    # Views of a borrowed tensor alias the NumPy buffer: a write through
    # the transposed view lands in the NumPy array (caller-managed lifetime).
    var v = a_zero.transpose()
    v[0, 1] = Scalar[dtype](42.0)
    var check = from_ndarray[DType.float32](to_ndarray(a_zero))
    assert_true(check[1, 0] == 42.0 and check[0, 0] == 1.0)
    # Keep-alive: the borrowed buffer aliases nd_a, whose lifetime is
    # caller-managed — a last lexical use must come after the final
    # borrowed access, otherwise the array may be freed early and the
    # round-trip above reads freed memory.
    _ = nd_a


def test_gpu_tensor_to_numpy_contiguous() raises:
    """GPU contiguous tensor → to_ndarray must not read empty CPU buffer."""
    print("test_gpu_tensor_to_numpy_contiguous")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var cpu_t = Tensor[dtype].arange(120)
        var gpu_t = cpu_t.to_gpu()
        var nd = to_ndarray(gpu_t)
        var back = from_ndarray[dtype](nd)
        assert_true(back.all_close(cpu_t))


def test_gpu_tensor_to_numpy_view() raises:
    """GPU non-contiguous view (offset/strides) → to_ndarray."""
    print("test_gpu_tensor_to_numpy_view")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var cpu_t = Tensor[dtype].arange(32)
        var cpu_t_2d = cpu_t.reshape(4, 8)
        var cpu_v = cpu_t_2d.view([2, 3], offset=5)
        var gpu_v = cpu_v.to_gpu()
        var nd = to_ndarray(gpu_v)
        var back = from_ndarray[dtype](nd)
        assert_true(back.all_close(cpu_v))


def test_gpu_bool_tensor_to_numpy() raises:
    """GPU bool tensor (uint8 internal) → to_ndarray."""
    print("test_gpu_bool_tensor_to_numpy")
    comptime if has_accelerator():
        var cpu_t = Tensor[DType.bool].full([3, 4], True)
        var gpu_t = cpu_t.to_gpu()
        var nd = to_ndarray(gpu_t)
        var back = from_ndarray[DType.bool](nd)
        assert_true(back == cpu_t)


def test_gpu_tensor_print() raises:
    """GPU tensor .print() must not crash."""
    print("test_gpu_tensor_print")
    comptime dtype = DType.float32
    var cpu_t = Tensor[dtype].arange(16)
    var cpu_t_2d = cpu_t.reshape(4, 4)
    cpu_t_2d.print()
    comptime if has_accelerator():
        var gpu_t = cpu_t.to_gpu()
        gpu_t.print()


def test_from_ndarray_strided_slice() raises:
    """Non-contiguous slice input: copy path must gather strided values."""
    print("test_from_ndarray_strided_slice")
    var np = Python.import_module("numpy")
    # Every 2nd element of arange(10) -> [0,2,4,6,8]. A flat memcpy from
    # the base pointer would read [0,1,2,3,4] instead.
    var nd = np.arange(10, dtype=np.float32)[::2]
    assert_true(not Bool(py=nd.flags["C_CONTIGUOUS"]))
    var back = from_ndarray[DType.float32](nd, copy=True)
    var expect = Tensor[DType.float32].d1([0.0, 2.0, 4.0, 6.0, 8.0])
    assert_true(back == expect)


def test_from_ndarray_transpose() raises:
    """Transposed (non-contiguous) input: copy path must honor strides."""
    print("test_from_ndarray_transpose")
    var np = Python.import_module("numpy")
    var nd = np.arange(12, dtype=np.float32).reshape(3, 4).T
    assert_true(not Bool(py=nd.flags["C_CONTIGUOUS"]))
    var back = from_ndarray[DType.float32](nd, copy=True)
    var expect = Tensor[DType.float32].d2(
        [[0.0, 4.0, 8.0], [1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0]]
    )
    assert_true(back == expect)


def test_from_ndarray_alias_rejects_strided() raises:
    """Copy=False cannot represent strides: must raise, not alias wrong."""
    print("test_from_ndarray_alias_rejects_strided")
    var np = Python.import_module("numpy")
    var nd = np.arange(10, dtype=np.float32)[::2]
    with assert_raises():
        var aliased = from_ndarray[DType.float32](nd, copy=False)
        _ = aliased
