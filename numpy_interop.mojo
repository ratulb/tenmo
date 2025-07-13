from python import Python, PythonObject
from tensors import Tensor
from memory import UnsafePointer, memcpy


fn numpy_dtype(dtype: DType) raises -> PythonObject:
    np = Python.import_module("numpy")
    if dtype == DType.float32:
        return np.float32
    elif dtype == DType.float64:
        return np.float64
    elif dtype == DType.int8:
        return np.int8
    elif dtype == DType.uint8:
        return np.uint8
    elif dtype == DType.bool:
        return np.bool
    elif dtype == DType.uint16:
        return np.uint16
    elif dtype == DType.uint64:
        return np.uint64
    else:
        return None


fn ndarray_ptr[
    dtype: DType
](ndarray: PythonObject) raises -> UnsafePointer[Scalar[dtype]]:
    return ndarray.__array_interface__["data"][0].unsafe_get_as_pointer[dtype]()


fn to_ndarray[
    axes_sizes: Int, dtype: DType, //
](tensor: Tensor[axes_sizes, dtype]) raises -> PythonObject:
    np = Python.import_module("numpy")
    ndarray = np.zeros(tensor.numels(), dtype=numpy_dtype(tensor.datatype))
    ndarray_ptr = ndarray_ptr[dtype](ndarray)
    buffer_ptr = tensor.unsafe_ptr()
    memcpy(ndarray_ptr, buffer_ptr, tensor.numels())
    return ndarray
