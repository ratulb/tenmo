from python import Python, PythonObject
from tensors import Tensor
from memory import memcpy


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


fn to_ndarray[dtype: DType, //](tensor: Tensor[dtype]) raises -> PythonObject:
    np = Python.import_module("numpy")
    ndarray = np.zeros(tensor.numels(), dtype=numpy_dtype(tensor.dtype))
    ndarray_ptr = ndarray_ptr[dtype](ndarray)
    if tensor.owns_data:
        buffer_ptr = tensor.buffer.data
        memcpy(ndarray_ptr, buffer_ptr, tensor.numels())
    else:
        idx = 0
        for coord in tensor.shape:
            ndarray[idx] = tensor[coord]
            idx += 1
    return ndarray


fn main() raises:
    a = Tensor.arange(10)
    a.print()  # print mojo Tensor
    print()
    result = to_ndarray(a)
    print(result)  # numpy array
    print()
    b = a.view([5], offset=2)
    b.print()  # mojo view
    print()
    from_view = to_ndarray(b)
    print(from_view)  # numpy array
    print()
