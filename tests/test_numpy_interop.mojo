from tensors import Tensor
from numpy_interop import to_ndarray, from_ndarray

from testing import assert_true


fn main() raises:
    test_1d_tensor_to_numpy_and_back()
    test_2d_tensor_to_numpy_and_back()
    test_tensor_view_to_numpy_and_back()
    test_bool_tensor_to_numpy_and_back()
    test_copy_vs_zero_copy_behavior()
    test_scalar_tensor_conversion()
    test_1d_tensor_random()
    test_2d_tensor_random()
    test_3d_tensor_random()
    test_4d_tensor_random()
    test_views_1d_2d()
    test_views_3d_4d()
    test_bool_tensor()
    test_copy_vs_zero_copy()
    test_random_tensor_large()


fn test_scalar_tensor_conversion() raises:
    print("test_scalar_tensor_conversion")
    var a = Tensor.scalar(42)
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a).all_true())
    assert_true(a_back.owns_data)


fn test_1d_tensor_random() raises:
    print("test_1d_tensor_random")
    var a = Tensor[DType.float32].arange(10)
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a).all_true())
    assert_true(a_back.owns_data)


fn test_2d_tensor_random() raises:
    print("test_2d_tensor_random")
    var a = Tensor[DType.float32].d2([[1.5, 2.5], [3.5, 4.5]])
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a).all_true())
    assert_true(a_back.owns_data)


fn test_3d_tensor_random() raises:
    print("test_3d_tensor_random")
    var a = Tensor[DType.float32].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a).all_true())
    assert_true(a_back.owns_data)


fn test_4d_tensor_random() raises:
    print("test_4d_tensor_random")
    var a = Tensor[DType.float32].d4(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ]
    )
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)
    assert_true((a_back == a).all_true())
    assert_true(a_back.owns_data)


fn test_views_1d_2d() raises:
    print("test_views_1d_2d")
    var a = Tensor.arange(16)
    var b = a.view([4, 4])
    var v1 = b.view([2, 2], offset=3)
    var nd_v = to_ndarray(v1)
    var v_back = from_ndarray[DType.float32](nd_v)
    assert_true((v_back == v1).all_true())
    assert_true(v_back.owns_data)
    _ = a


fn test_views_3d_4d() raises:
    print("test_views_3d_4d")
    var a = Tensor.arange(64)
    var b = a.view([4, 4, 4])
    var v = b.view([2, 2, 2], offset=10)
    var nd_v = to_ndarray(v)
    var v_back = from_ndarray[DType.float32](nd_v)
    assert_true((v_back == v).all_true())
    assert_true(v_back.owns_data)
    _ = a


fn test_bool_tensor() raises:
    print("test_bool_tensor")
    var b = Tensor[DType.bool].full([3, 3], True)
    var nd_b = to_ndarray(b)
    var b_back = from_ndarray[DType.bool](nd_b)
    assert_true((b_back == b).all_true())
    assert_true(b_back.owns_data)


fn test_copy_vs_zero_copy() raises:
    print("test_copy_vs_zero_copy")
    var a = Tensor.arange(5)

    # copy=True
    var nd_a_copy = to_ndarray(a)
    var a_copy = from_ndarray[DType.float32](nd_a_copy, copy=True)
    assert_true((a_copy == a).all_true())
    assert_true(a_copy.owns_data)

    # copy=False
    var a_zero = from_ndarray[DType.float32](nd_a_copy, copy=False)
    assert_true((a_zero == a).all_true())
    assert_true(a_zero.owns_data)


fn test_random_tensor_large() raises:
    print("test_random_tensor_large")
    var a = Tensor[DType.float32].arange(120)
    var b = a.view([2, 3, 4, 5])
    var nd_b = to_ndarray(b)
    var b_back = from_ndarray[DType.float32](nd_b)
    assert_true((b_back == b).all_true())
    assert_true(b_back.owns_data)


fn test_1d_tensor_to_numpy_and_back() raises:
    print("test_1d_tensor_to_numpy_and_back")
    var a = Tensor.arange(10)
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)

    assert_true((a == a_back).all_true())
    assert_true(a_back.owns_data)  # should own data


fn test_2d_tensor_to_numpy_and_back() raises:
    print("test_2d_tensor_to_numpy_and_back")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var nd_a = to_ndarray(a)
    var a_back = from_ndarray[DType.float32](nd_a)

    assert_true((a == a_back).all_true())
    assert_true(a_back.owns_data)


fn test_tensor_view_to_numpy_and_back() raises:
    print("test_tensor_view_to_numpy_and_back")
    var a = Tensor.arange(16)
    var b = a.view([4, 4])
    var v = b.view([2, 2], offset=5)  # some arbitrary subview
    var nd_v = to_ndarray(v)
    var v_back = from_ndarray[DType.float32](nd_v)

    assert_true((v_back == v).all_true())
    assert_true(v_back.owns_data)
    _ = a


fn test_bool_tensor_to_numpy_and_back() raises:
    print("test_bool_tensor_to_numpy_and_back")
    var b = Tensor[DType.bool].full([3, 3], True)
    var nd_b = to_ndarray(b)
    var b_back = from_ndarray[DType.bool](nd_b)

    assert_true((b_back == b).all_true())
    assert_true(b_back.owns_data)


fn test_copy_vs_zero_copy_behavior() raises:
    print("test_copy_vs_zero_copy_behavior")
    var a = Tensor.arange(5)

    # Copy
    var nd_a_copy = to_ndarray(a)
    var a_copy = from_ndarray[DType.float32](nd_a_copy, copy=True)
    assert_true((a_copy == a).all_true())
    assert_true(a_copy.owns_data)

    # Zero-copy: requires user to keep ndarray alive
    var a_zero = from_ndarray[DType.float32](nd_a_copy, copy=False)
    assert_true((a_zero == a).all_true())
    assert_true(a_zero.owns_data)
