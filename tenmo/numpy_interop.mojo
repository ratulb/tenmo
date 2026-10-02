from std.python import Python, PythonObject
from .tensor import Tensor
from std.memory import unsafe_memcpy
from .shared.shapes import Shape
from .shared.buffers import Buffer
from .ndbuffer import NDBuffer
from .gradbox import Gradbox
from .net import Sequential
from std.sys import has_accelerator


def numpy_dtype(dtype: DType) raises -> PythonObject:
    """Mojo DType → numpy dtype."""
    var np = Python.import_module("numpy")
    if dtype == DType.float16:
        return np.float16
    elif dtype == DType.float32:
        return np.float32
    elif dtype == DType.float64:
        return np.float64
    elif dtype == DType.int8:
        return np.int8
    elif dtype == DType.int16:
        return np.int16
    elif dtype == DType.int32:
        return np.int32
    elif dtype == DType.int64:
        return np.int64
    elif dtype == DType.uint8:
        return np.uint8
    elif dtype == DType.uint16:
        return np.uint16
    elif dtype == DType.uint32:
        return np.uint32
    elif dtype == DType.uint64:
        return np.uint64
    elif dtype == DType.bool:
        return np.bool_
    else:
        raise Error("Unsupported dtype for python interop")


def mojo_dtype(dtype: PythonObject) raises -> DType:
    """NumPy dtype → Mojo DType. Raises on unsupported dtype."""
    var np = Python.import_module("numpy")
    if dtype == np.float16:
        return DType.float16
    elif dtype == np.float32:
        return DType.float32
    elif dtype == np.float64:
        return DType.float64
    elif dtype == np.int8:
        return DType.int8
    elif dtype == np.int16:
        return DType.int16
    elif dtype == np.int32:
        return DType.int32
    elif dtype == np.int64:
        return DType.int64
    elif dtype == np.uint8:
        return DType.uint8
    elif dtype == np.uint16:
        return DType.uint16
    elif dtype == np.uint32:
        return DType.uint32
    elif dtype == np.uint64:
        return DType.uint64
    elif dtype == np.bool_:
        return DType.bool
    else:
        raise Error("Unsupported numpy dtype: " + String(dtype))


def dtype_from_string(s: String) raises -> DType:
    """Parse a dtype string like 'float32', 'int64', 'bool' into a Mojo DType."""
    var lower = s.lower()
    if lower == "float16" or lower == "f2":
        return DType.float16
    elif lower == "float32" or lower == "f4" or lower == "float" or lower == "single":
        return DType.float32
    elif lower == "float64" or lower == "f8" or lower == "double":
        return DType.float64
    elif lower == "int8" or lower == "i1":
        return DType.int8
    elif lower == "int16" or lower == "i2":
        return DType.int16
    elif lower == "int32" or lower == "i4" or lower == "int" or lower == "intc":
        return DType.int32
    elif lower == "int64" or lower == "i8" or lower == "long" or lower == "intp":
        return DType.int64
    elif lower == "uint8" or lower == "u1" or lower == "byte":
        return DType.uint8
    elif lower == "uint16" or lower == "u2":
        return DType.uint16
    elif lower == "uint32" or lower == "u4":
        return DType.uint32
    elif lower == "uint64" or lower == "u8":
        return DType.uint64
    elif lower == "bool" or lower == "bool_" or lower == "?" or lower == "b1":
        return DType.bool
    else:
        raise Error("Unsupported dtype string: " + s)


def list_to_tuple(l: List[Int]) raises -> PythonObject:
    var py = Python.import_module("builtins")
    var py_list_obj: PythonObject = []
    for elem in l:
        py_list_obj.append(elem)
    var py_tuple = py.tuple(py_list_obj)
    return py_tuple


def ndarray_ptr[
    dtype: DType
](ndarray: PythonObject) raises -> Pointer[Scalar[dtype], MutAnyOrigin]:
    return ndarray.__array_interface__["data"][0].unsafe_get_as_pointer[dtype]()


def checked_ndarray_ptr[
    dtype: DType
](
    ndarray: PythonObject, expected: Int, what: String
) raises -> Pointer[Scalar[dtype], MutAnyOrigin]:
    """Restore gate: ndarray_ptr + length.

    Every restore below memcpys `count=live.numels()` (or unsafe_loads one
    scalar) out of a checkpoint array: a short array would OOB-read, a long
    one silently truncate. Mismatches raise here — loudly, naming the key —
    instead of corrupting weights/moments in silence.
    """
    var got = Int(py=ndarray.size)
    if got != expected:
        raise Error(
            "ndarray length mismatch for "
            + what
            + ": expected "
            + String(expected)
            + ", got "
            + String(got)
        )
    return ndarray_ptr[dtype](ndarray)


def to_ndarray[dtype: DType, //](tensor: Tensor[dtype]) raises -> PythonObject:
    return to_ndarray(tensor.buffer)


def to_ndarray[
    dtype: DType, //
](gradbox: Gradbox[dtype]) raises -> PythonObject:
    return to_ndarray(gradbox.buffer())


def to_ndarray[dtype: DType, //](ndb: NDBuffer[dtype]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    comptime if has_accelerator():
        if ndb.is_on_gpu():
            var cpu_ndb = ndb.to_cpu()
            var shape_tuple = list_to_tuple(cpu_ndb.shape.tolist())
            var ndarray = np.zeros(
                shape_tuple, dtype=numpy_dtype(cpu_ndb.dtype)
            )
            if cpu_ndb.is_contiguous():
                var dst_ptr = ndarray_ptr[dtype](ndarray)
                var buffer_ptr = cpu_ndb.data_ptr().unsafe_offset(
                    cpu_ndb.offset
                )
                unsafe_memcpy(
                    dest=dst_ptr, src=buffer_ptr, count=cpu_ndb.numels()
                )
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
        var buffer_ptr = ndb.data_ptr().unsafe_offset(ndb.offset)
        unsafe_memcpy(dest=dst_ptr, src=buffer_ptr, count=ndb.numels())
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
    # Strides/offset live on the NumPy side. `flags["C_CONTIGUOUS"]` is the
    # fast-path signal (the same gate std.python.numpy.from_numpy_array
    # uses); when set, `strides` is the dense default and a flat memcpy is
    # valid. Otherwise the array is a strided view (`strides` tuple holds
    # byte-strides, `ctypes.data` already points at the view's first
    # logical element): memcpy'ing `numels` flat would gather strided
    # gaps as dense data. So the copy path normalizes through
    # `np.ascontiguousarray` (a no-op when already contiguous), while the
    # alias path — which cannot represent strides — refuses loudly.
    var np = Python.import_module("numpy")
    var contiguous = ndarray
    if not Bool(py=ndarray.flags["C_CONTIGUOUS"]):
        if not copy:
            raise Error(
                "from_ndarray: copy=False aliases the NumPy buffer, which "
                "requires a C-contiguous array (got a strided view)"
            )
        contiguous = np.ascontiguousarray(ndarray)
    # Convert Python shape -> Mojo Shape
    var shape_list = contiguous.shape
    var mojo_list = List[Int](capacity=len(shape_list))
    for elem in shape_list:
        var value: Int = Int(py=elem)
        mojo_list.append(value)
    var shape = Shape(mojo_list)

    var numels = shape.product()

    if copy:
        var src_ptr = ndarray_ptr[dtype](contiguous)
        var buffer = Buffer[dtype](numels)
        unsafe_memcpy(
            dest=buffer.data.unsafe_value(), src=src_ptr, count=numels
        )
        var ndb = NDBuffer[dtype](buffer^, shape)
        var result = Tensor[dtype](ndb^, requires_grad=requires_grad)
        return result^
    else:
        # Wrap external NumPy buffer (lifetime must be managed externally!).
        # Tensor view ops (transpose/permute/squeeze/...) alias this borrowed
        # memory — writes through a view mutate the NumPy array.
        var data_ptr = ndarray_ptr[dtype](contiguous)
        return Tensor[dtype](
            data_ptr, shape^, requires_grad=requires_grad, copy=False
        )


def save[dtype: DType, //](t: Tensor[dtype], filename: StaticString) raises:
    var python_obj = to_ndarray(t)
    var np = Python.import_module("numpy")
    np.savez(filename, array=python_obj)


def load[
    dtype: DType
](filename: StaticString, requires_grad: Bool = False) raises -> Tensor[dtype]:
    var np = Python.import_module("numpy")
    var data = np.load(filename)
    var ndarray = data["array"]
    var tenmo_tensor = from_ndarray[dtype](ndarray, requires_grad=requires_grad)
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


def save_checkpoint[
    dtype: DType, //
](model: Sequential[dtype], path: String) raises:
    var np = Python.import_module("numpy")
    var state_dict: PythonObject = {}
    var params = model.named_parameters("")
    for p in params:
        var tensor_ptr = p.tensor_ptr
        var nd = to_ndarray(tensor_ptr[])
        state_dict[p.name] = nd
    np.save(path, state_dict)


def load_checkpoint[
    dtype: DType, //
](mut model: Sequential[dtype], path: String) raises:
    var np = Python.import_module("numpy")
    var state_dict = np.load(path, allow_pickle=True).item()
    var params = model.named_parameters("")
    for p in params:
        var key = p.name
        if state_dict.__contains__(key):
            var nd = state_dict[key]
            var tensor_ptr = p.tensor_ptr
            ref t = tensor_ptr[]
            var src_ptr = checked_ndarray_ptr[dtype](
                nd, t.numels(), "load_checkpoint:" + key
            )
            unsafe_memcpy(
                dest=t.data_ptr().unsafe_mut_cast[True](),
                src=src_ptr,
                count=t.numels(),
            )


def test_to_ndarray() raises:
    comptime dtype = DType.int32
    var o = Tensor[dtype].arange(10)
    var a = o.reshape(2, 5)
    var py_obj = to_ndarray(a)
    var py = Python.import_module("builtins")
    print(py_obj, py_obj.dtype, py.type(py_obj))
    print("Done")
