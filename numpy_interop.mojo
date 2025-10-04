from python import Python, PythonObject
from tensors import Tensor
from memory import memcpy
from shapes import Shape
from buffers import Buffer


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
](ndarray: PythonObject) raises -> UnsafePointer[Scalar[dtype]]:
    return ndarray.__array_interface__["data"][0].unsafe_get_as_pointer[dtype]()


fn to_ndarray[dtype: DType, //](tensor: Tensor[dtype]) raises -> PythonObject:
    np = Python.import_module("numpy")
    shape_tuple = list_to_tuple(tensor.shape.tolist())
    ndarray = np.zeros(shape_tuple, dtype=numpy_dtype(tensor.dtype))
    if tensor.is_dense():
        dst_ptr = ndarray_ptr[dtype](ndarray)
        buffer_ptr = tensor.buffer.data().data
        memcpy(dst_ptr, buffer_ptr, tensor.numels())
    else:
        flat = ndarray.flat
        idx = 0
        for coord in tensor.shape:
            flat[idx] = tensor[coord]
            idx += 1
    return ndarray


fn from_ndarray[
    dtype: DType
](
    ndarray: PythonObject, requires_grad: Bool = False, copy: Bool = True
) raises -> Tensor[dtype]:
    # Convert Python shape -> Mojo Shape
    shape_list = ndarray.shape
    mojo_list = List[Int](capacity=len(shape_list))
    for elem in shape_list:
        mojo_list.append(Int(elem))
    shape = Shape(mojo_list)

    numels = shape.product()

    if copy:
        src_ptr = ndarray_ptr[dtype](ndarray)
        buffer = Buffer[dtype](numels)
        memcpy(buffer.data, src_ptr, numels)
        result = Tensor[dtype](shape, buffer^, requires_grad=requires_grad)
        return result
    else:
        # Wrap external NumPy buffer (⚠️ lifetime must be managed externally)
        data_ptr = ndarray_ptr[dtype](ndarray)
        return Tensor[dtype](shape, data_ptr, requires_grad=requires_grad)


from testing import assert_true


fn main() raises:
    mnist = Python.import_module("mnist_datasets")
    loader = mnist.MNISTLoader(folder="/tmp")

    # Load train dataset
    var train_data: PythonObject = loader.load()
    var images = train_data[0]
    var labels = train_data[1]
    assert_true(len(images) == 60000 and len(labels) == 60000)

    # Load test dataset
    var test_data: PythonObject = loader.load(train=False)
    var test_images = test_data[0]
    var test_labels = test_data[1]
    assert_true(len(test_images) == 10000 and len(test_labels) == 10000)
