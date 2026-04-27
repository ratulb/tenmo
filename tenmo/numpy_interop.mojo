from std.python import Python, PythonObject
from .tensor import Tensor
from std.memory import memcpy
from .shapes import Shape
from .buffers import Buffer
from .ndbuffer import NDBuffer
from .gradbox import Gradbox


fn numpy_dtype(dtype: DType) raises -> PythonObject:
    np = Python.import_module("numpy")
    if dtype == DType.float32:
        return np.float32
    elif dtype == DType.float64:
        return np.float64
    elif dtype == DType.int8:
        return np.int8
    elif dtype == DType.int32:
        return np.int32
    elif dtype == DType.uint8:
        return np.uint8
    elif dtype == DType.bool:
        return np.bool
    elif dtype == DType.uint16:
        return np.uint16
    elif dtype == DType.uint64:
        return np.uint64
    else:
        raise Error("Unsupported dtype for python interop")


fn list_to_tuple(l: List[Int]) raises -> PythonObject:
    py = Python.import_module("builtins")
    var py_list_obj: PythonObject = []
    for elem in l:
        py_list_obj.append(elem)
    py_tuple = py.tuple(py_list_obj)
    return py_tuple


fn ndarray_ptr[
    dtype: DType
](ndarray: PythonObject) raises -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    return ndarray.__array_interface__["data"][0].unsafe_get_as_pointer[dtype]()


fn to_ndarray[dtype: DType, //](tensor: Tensor[dtype]) raises -> PythonObject:
    _="""np = Python.import_module("numpy")
    shape_tuple = list_to_tuple(tensor.shape().tolist())
    ndarray = np.zeros(shape_tuple, dtype=numpy_dtype(tensor.dtype))
    if tensor.is_contiguous():
        dst_ptr = ndarray_ptr[dtype](ndarray)
        buffer_ptr = tensor.data_ptr() + tensor.offset()
        memcpy(dest=dst_ptr, src=buffer_ptr, count=tensor.numels())
    else:
        flat = ndarray.flat
        idx = 0
        for coord in tensor.shape():
            flat[idx] = tensor[coord]
            idx += 1
    return ndarray"""
    return to_ndarray(tensor.buffer)

fn to_ndarray[dtype: DType, //](gradbox: Gradbox[dtype]) raises -> PythonObject:
    return to_ndarray(gradbox.buffer)

fn to_ndarray[dtype: DType, //](ndb: NDBuffer[dtype]) raises -> PythonObject:
    np = Python.import_module("numpy")
    shape_tuple = list_to_tuple(ndb.shape.tolist())
    ndarray = np.zeros(shape_tuple, dtype=numpy_dtype(ndb.dtype))
    if ndb.is_contiguous():
        dst_ptr = ndarray_ptr[dtype](ndarray)
        buffer_ptr = ndb.data_ptr() + ndb.offset
        memcpy(dest=dst_ptr, src=buffer_ptr, count=ndb.numels())
    else:
        flat = ndarray.flat
        idx = 0
        for coord in ndb.shape:
            flat[idx] = ndb[coord]
            idx += 1
    return ndarray




fn from_ndarray[
    dtype: DType
](
    ndarray: PythonObject, requires_grad: Bool = False, copy: Bool = True
) raises -> Tensor[dtype]:
    # Convert Python shape -> Mojo Shape
    var shape_list = ndarray.shape
    var mojo_list = List[Int](capacity=len(shape_list))
    for elem in shape_list:
        var value: Int = Int(py=elem)
        mojo_list.append(value)
    var shape = Shape(mojo_list)

    var numels = shape.product()

    if copy:
        var src_ptr = ndarray_ptr[dtype](ndarray)
        var buffer = Buffer[dtype](numels)
        memcpy(dest=buffer.data, src=src_ptr, count=numels)
        var ndb = NDBuffer[dtype](buffer^, shape)
        var result = Tensor[dtype](ndb^, requires_grad=requires_grad)
        return result^
    else:
        # Wrap external NumPy buffer (lifetime must be managed externally!)
        var data_ptr = ndarray_ptr[dtype](ndarray)
        return Tensor[dtype](
            data_ptr, shape^, requires_grad=requires_grad, copy=False
        )


fn save[dtype: DType, //](t: Tensor[dtype], filename: StaticString) raises:
    var python_obj = to_ndarray(t)
    np = Python.import_module("numpy")
    np.savez(filename, array=python_obj)


fn load[
    dtype: DType
](filename: StaticString, requires_grad: Bool = False) raises -> Tensor[dtype]:
    np = Python.import_module("numpy")
    data = np.load(filename)
    ndarray = data["array"]
    tenmo_tensor = from_ndarray[dtype](ndarray, requires_grad=requires_grad)
    # print(ndarray)
    return tenmo_tensor^


fn as_nested_list[dtype: DType, //](self: Tensor[dtype]) raises -> PythonObject:
    """Convert tensor to a nested Python list retaining shape structure."""
    var ndarray = to_ndarray(self)
    return ndarray.tolist()

fn as_nested_list[dtype: DType, //](self: Gradbox[dtype]) raises -> PythonObject:
    """Convert Gradbox to a nested Python list retaining shape structure."""
    var ndarray = to_ndarray(self)
    return ndarray.tolist()


fn as_nested_list[dtype: DType, //](self: NDBuffer[dtype]) raises -> PythonObject:
    """Convert NDBuffer to a nested Python list retaining shape structure."""
    var ndarray = to_ndarray(self)
    return ndarray.tolist()



fn test_to_ndarray() raises:
    comptime dtype = DType.int32
    o = Tensor[dtype].arange(10)
    a = o.reshape(2, 5)
    py_obj = to_ndarray(a)
    py = Python.import_module("builtins")
    print(py_obj, py_obj.dtype, py.type(py_obj))
    print("Done")
