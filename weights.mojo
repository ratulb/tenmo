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
    print("GPT2 weight explorer")
    #weights = Weights()
    #print(weights.keys().__str__())
    #tensor = weights.weights('model/wpe')
    #print(tensor.num_elements())
    run()

fn run() raises -> None:
    weights = Weights()
    print("Available choices:\n 1) Load keys, 2) Load 10 keys at a time, \n 3) Search for keys with prefix,4) Load numpy weights for key, \n 5) Load Tensor weights from numpy, 6) 'q'(quit)")
    usr_choice = String()
    while True:
        print("Current choice:", usr_choice)
        if usr_choice == "1":
            print("Load keys\n")
            print(weights.keys().__str__())
        elif usr_choice == "2":
            print("Load 10 keys at a time")
        elif usr_choice == "3":
            print("Searching key with prefix")
        elif usr_choice == "4":
            print("numpy weights for key")
        elif usr_choice == "5":
            print("Tensor weight for key")
        else:
            print("Invaild choice or no choice made yet")
        print()
        usr_choice = input("Enter 'q' to quit or press <Enter> to continue: ")
        if usr_choice == "q":
            break

