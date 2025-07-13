from python import Python, PythonObject
from max.tensor import Tensor, TensorShape, TensorSpec
from memory import UnsafePointer
from max.driver.tensor import Tensor as MT


fn main():
    # var image: MT[DType.float32, 3] = Tensor[DType.float32](TensorShape(40, 80, 3))
    # var image: MT[DType.float32, 2] = Tensor[DType.float32](TensorShape(40, 80))
    try:
        np = Python.import_module("numpy")
        gpt2_weights = np.load("gpt2_weights.npz")
        print(len(gpt2_weights.keys()))
        print(Python.is_type(Python.dict(), gpt2_weights))
        print(Python.type(gpt2_weights))
        print(gpt2_weights.keys())
        print(Python.type(gpt2_weights["model/wpe"]))
        print(gpt2_weights["model/wpe"].shape)
        print(gpt2_weights["model/wpe"][0])
        ptr = numpy_data_pointer[DType.float32](gpt2_weights["model/wpe"])
        print("Here we do go", ptr[0])
        print("Here we do go", ptr[1])
        print("Here we do go", ptr[767])
    except e:
        print(e)


fn numpy_data_pointer[
    dtype: DType
](numpy_array: PythonObject) raises -> UnsafePointer[Scalar[dtype]]:
    return numpy_array.__array_interface__["data"][0].unsafe_get_as_pointer[
        dtype
    ]()
