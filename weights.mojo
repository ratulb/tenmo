from memory import UnsafePointer, memcpy
from python import Python, PythonObject
from tensors import Tensor
from utils import StaticTuple, Variant

alias Variety = Variant[
    Tensor[1, DType.float32],
    Tensor[2, DType.float32],
    Tensor[3, DType.float32],
    Tensor[4, DType.float32],
]

alias Resultant = Optional[Variety]


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

    fn weights(mut self, key: String) raises -> Resultant:
        try:
            self.load_weights()
            np_weights = self.gpt2_weights[key]
            np_tensor_shape = np_weights.shape
            print("Tensor shape: ", np_tensor_shape)
            np_weights_ptr = Self.ndarray_ptr[DType.float32](np_weights)
            dimension = len(np_tensor_shape)
            if dimension == 1:
                print("I am in dimension 1", np_tensor_shape[0])
                axes_dims1 = StaticTuple[Int, 1](np_tensor_shape[0])
                print("I don't come here in dimension 1")
                tensor1 = Tensor[1, DType.float32](axes_dims1)
                tensor_ptr1 = tensor1.unsafe_ptr()
                print("I don't come here in dimension 2", np_weights.size)
                memcpy(tensor_ptr1, np_weights_ptr, np_weights.size)
                print("I don't come here in dimension 3")
                return Optional(Variety(tensor1))
            elif dimension == 2:
                axes_dims2 = StaticTuple[Int, 2](
                    np_tensor_shape[0], np_tensor_shape[1]
                )
                tensor2 = Tensor[2, DType.float32](axes_dims2)
                tensor_ptr2 = tensor2.unsafe_ptr()
                memcpy(tensor_ptr2, np_weights_ptr, np_weights.size)
                return Optional(Variety(tensor2))
            elif dimension == 3:
                axes_dims3 = StaticTuple[Int, 3](
                    np_tensor_shape[0], np_tensor_shape[1], np_tensor_shape[2]
                )
                tensor3 = Tensor[3, DType.float32](axes_dims3)
                tensor_ptr3 = tensor3.unsafe_ptr()
                memcpy(tensor_ptr3, np_weights_ptr, np_weights.size)
                return Optional(Variety(tensor3))
            elif dimension == 4:
                axes_dims4 = StaticTuple[Int, 4](
                    np_tensor_shape[0],
                    np_tensor_shape[1],
                    np_tensor_shape[2],
                    np_tensor_shape[3],
                )
                tensor4 = Tensor[4, DType.float32](axes_dims4)
                tensor_ptr4 = tensor4.unsafe_ptr()
                memcpy(tensor_ptr4, np_weights_ptr, np_weights.size)
                return Optional(Variety(tensor4))
            else:
                print("Unsupported dimension")
                return None
        except e:
            print("Exception here")
            print(e)
            return None

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
    print()
    print("GPT2 weight explorer")
    # weights = Weights()
    # print(weights.keys().__str__())
    # tensor = weights.weights('model/wpe')
    # print(tensor.num_elements())
    run()


fn run() raises -> None:
    weights = Weights()
    print(
        "Available choices:\n 1) Load keys, 2) Load 10 keys at a time, \n 3)"
        " Search for keys with prefix, 4) Load numpy weights for key, \n 5)"
        " Load Tensor weights from numpy, 6) 'q'(quit)"
    )
    usr_choice = String()
    while True:
        print("Current choice:", usr_choice)
        if usr_choice == "1":
            print("Load keys\n")
            print(weights.keys().__str__())
            print()
            print("Total keys: ", len(weights.keys()))
        elif usr_choice == "2":
            print("Load 10 keys at a time")
            print()
            keys = weights.keys()
            for start in range(0, len(keys), 10):
                end = min(start + 10, len(keys))
                batch = keys[start:end]
                s = "keys from " + String(start) + " to " + String(end - 1)
                print(s)
                print()
                # print(batch.__str__())
                for key in batch:
                    print(key[], " ", weights.gpt2_weights[key[]].shape)
                print()
                if (
                    input(
                        "Enter 'q' to break out or press <Enter> to continue: "
                    )
                    == "q"
                ):
                    break

        elif usr_choice == "3":
            print("Searching key with prefix")
        elif usr_choice == "4":
            print("numpy weights for key")
            key_name = input("Enter key name for numpy weights: ")
            try:
                numpy_weights = weights.gpt2_weights[key_name]
                print()
                print("Shape: ", numpy_weights.shape)
                print("Size: ", numpy_weights.size)
                var size: Int = numpy_weights.size
                print("Mojo Int size: ", size)
                print()
                print(numpy_weights)
            except e:
                print("Something went wrong! Wrong key?")
                print(e)
            print()
        elif usr_choice == "5":
            print("Tensor weight for key")
            tensor_key_name = input("Enter key name for tensor weights: ")
            try:
                tensor_weights = weights.weights(tensor_key_name)
                if tensor_weights:
                    print("Something inside!")
                    value = tensor_weights.value()
                    print("***************")
                    d1_tensor = value[Tensor[1, DType.float32]]
                    print("1D tensor numels: ", d1_tensor.numels())
                    print(d1_tensor.__str__())
                    var indices = List[Int]()
                    print()
                    d1_tensor.print_tensor_recursive(indices, 1)
                    print()
                else:
                    print("Got nothing")
                print()
                print("Voild hip hip hurray!")
                print()
            except e:
                print("Something went wrong! Wrong key?")
                print(e)
            print()

        else:
            print("Invaild choice or no choice made yet")
        print()
        usr_choice = input("Enter 'q' to quit or press <Enter> to continue 🔄: ")
        if usr_choice == "q":
            break
