from std.python import Python, PythonObject
from .tensor import Tensor
from std.memory import memcpy
from .shapes import Shape
from .buffers import Buffer
from .ndbuffer import NDBuffer
from .gradbox import Gradbox
from .net import Sequential
from .named_parameter import NamedParameter
from std.sys import has_accelerator


def numpy_dtype(dtype: DType) raises -> PythonObject:
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


def list_to_tuple(l: List[Int]) raises -> PythonObject:
    py = Python.import_module("builtins")
    var py_list_obj: PythonObject = []
    for elem in l:
        py_list_obj.append(elem)
    py_tuple = py.tuple(py_list_obj)
    return py_tuple


def ndarray_ptr[
    dtype: DType
](ndarray: PythonObject) raises -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    return ndarray.__array_interface__["data"][0].unsafe_get_as_pointer[dtype]()


def to_ndarray[dtype: DType, //](tensor: Tensor[dtype]) raises -> PythonObject:
    return to_ndarray(tensor.buffer)


def to_ndarray[
    dtype: DType, //
](gradbox: Gradbox[dtype]) raises -> PythonObject:
    return to_ndarray(gradbox.buffer())


def to_ndarray[dtype: DType, //](ndb: NDBuffer[dtype]) raises -> PythonObject:
    np = Python.import_module("numpy")
    comptime if has_accelerator():
        if ndb.is_on_gpu():
            var cpu_ndb = ndb.to_cpu()
            var shape_tuple = list_to_tuple(cpu_ndb.shape.tolist())
            var ndarray = np.zeros(shape_tuple, dtype=numpy_dtype(cpu_ndb.dtype))
            if cpu_ndb.is_contiguous():
                var dst_ptr = ndarray_ptr[dtype](ndarray)
                var buffer_ptr = cpu_ndb.data_ptr() + cpu_ndb.offset
                memcpy(dest=dst_ptr, src=buffer_ptr, count=cpu_ndb.numels())
            else:
                var flat = ndarray.flat
                var idx = 0
                for coord in cpu_ndb.shape:
                    flat[idx] = cpu_ndb[coord]
                    idx += 1
            return ndarray
    var shape_tuple = list_to_tuple(ndb.shape.tolist())
    var ndarray = np.zeros(shape_tuple, dtype=numpy_dtype(ndb.dtype))
    if ndb.is_contiguous():
        var dst_ptr = ndarray_ptr[dtype](ndarray)
        var buffer_ptr = ndb.data_ptr() + ndb.offset
        memcpy(dest=dst_ptr, src=buffer_ptr, count=ndb.numels())
    else:
        var flat = ndarray.flat
        var idx = 0
        for coord in ndb.shape:
            flat[idx] = ndb[coord]
            idx += 1
    return ndarray


def from_ndarray[
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


def save[dtype: DType, //](t: Tensor[dtype], filename: StaticString) raises:
    var python_obj = to_ndarray(t)
    np = Python.import_module("numpy")
    np.savez(filename, array=python_obj)


def load[
    dtype: DType
](filename: StaticString, requires_grad: Bool = False) raises -> Tensor[dtype]:
    np = Python.import_module("numpy")
    data = np.load(filename)
    ndarray = data["array"]
    tenmo_tensor = from_ndarray[dtype](ndarray, requires_grad=requires_grad)
    # print(ndarray)
    return tenmo_tensor^


def as_nested_list[
    dtype: DType, //
](self: Tensor[dtype]) raises -> PythonObject:
    """Convert tensor to a nested Python list retaining shape structure."""
    var ndarray = to_ndarray(self)
    return ndarray.tolist()


def as_nested_list[
    dtype: DType, //
](self: Gradbox[dtype]) raises -> PythonObject:
    """Convert Gradbox to a nested Python list retaining shape structure."""
    var ndarray = to_ndarray(self)
    return ndarray.tolist()


def as_nested_list[
    dtype: DType, //
](self: NDBuffer[dtype]) raises -> PythonObject:
    """Convert NDBuffer to a nested Python list retaining shape structure."""
    var ndarray = to_ndarray(self)
    return ndarray.tolist()


def save_checkpoint[dtype: DType, //](
    model: Sequential[dtype], path: String
) raises:
    np = Python.import_module("numpy")
    var state_dict: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        var nd = to_ndarray(tensor_ptr[])
        state_dict[p.name] = nd
    np.save(path, state_dict)


def load_checkpoint[dtype: DType, //](
    mut model: Sequential[dtype], path: String
) raises:
    np = Python.import_module("numpy")
    var state_dict = np.load(path, allow_pickle=True).item()
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if state_dict.__contains__(key):
            var nd = state_dict[key]
            var src_ptr = ndarray_ptr[dtype](nd)
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


def test_to_ndarray() raises:
    comptime dtype = DType.int32
    o = Tensor[dtype].arange(10)
    a = o.reshape(2, 5)
    py_obj = to_ndarray(a)
    py = Python.import_module("builtins")
    print(py_obj, py_obj.dtype, py.type(py_obj))
    print("Done")
