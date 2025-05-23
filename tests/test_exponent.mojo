from tensors import Tensor
from testing import assert_true
from common_utils import to_ndarray
from python import Python


fn test_tensor_exponent() raises:
    """
    Tests tensor exp implementation.

    """
    np = Python.import_module("numpy")
    builtins = Python.import_module("builtins")
    tensor = Tensor[2].rand(4, 5)
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


fn main() raises:
    test_tensor_exponent()
