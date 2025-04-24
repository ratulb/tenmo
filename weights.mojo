from memory import UnsafePointer, memcpy
from python import Python, PythonObject
from tensors import Tensor


struct Weights:
    var npz_file: StaticString
    var gpt2_weights: PythonObject

    fn __init__(out self, npz_file: StaticString = "gpt2_weights.npz"):
        self.npz_file = npz_file
        self.gpt2_weights = None

    fn keys(mut self) raises -> List[String]:
        self.load_weights()
        return Self.keys(self.gpt2_weights)

    fn load_weights(mut self):
        if self.gpt2_weights is None:
            try:
                np = Python.import_module("numpy")
                self.gpt2_weights = np.load(self.npz_file)
            except e:
                print(e)

    fn weights[
        dtype: DType = DType.float32
    ](mut self, key: String) raises -> Tensor:
        self.load_weights()
        np_weights = self.gpt2_weights[key]
        tensor_shape = np_weights.shape
        print("np weghts shape: ", tensor_shape)
        np_weights_ptr = Self.ndarray_ptr[dtype](np_weights)
        #tensor_ptr = tensor.unsafe_ptr()
        #memcpy(tensor_ptr, np_weights_ptr, tensor.num_elements())
        return Tensor(1,1)

    @staticmethod
    fn ndarray_ptr[
        dtype: DType
    ](ndarray: PythonObject) raises -> UnsafePointer[Scalar[dtype]]:
        return ndarray.__array_interface__["data"][0].unsafe_get_as_pointer[
            dtype
        ]()

    @staticmethod
    fn keys(gpt2_weights: PythonObject) -> List[String]:
        list = List[String](capacity=148)
        try:
            keys = gpt2_weights.keys()
            for each_key in keys:
                list.append(each_key.__str__())
        except e:
            print(e)
        return list


fn main() raises:
    print("Weigts")
    weights = Weights()
    print(weights.keys().__str__())
    #tensor = weights.weights('model/wpe')
    #print(tensor.num_elements())

