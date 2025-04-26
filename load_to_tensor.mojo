from python import Python
from tensors import Tensor
from utils import StaticTuple
from memory import memcpy


fn main():
    try:
        np = Python.import_module("numpy")
        npz = np.load("gpt2_weights.npz")
        # np_weights = npz['model/h8/attn/c_attn/w'];
        np_weights = npz["model/h8/attn/c_attn/w"]
        # print(np.weights)  # or any key
        print(np_weights.shape)  # or any key
        print(np_weights.size)  # or any key
        print(np_weights.dtype)  # Must be float32
        print(np_weights.ndim)  # Must be float32
        np_ptr = np_weights.__array_interface__["data"][
            0
        ].unsafe_get_as_pointer[DType.float32]()
        # axes_dims = StaticTuple[Int, 1](768)
        axes_dims = StaticTuple[Int, 3](1, 768, 2304)
        print(np_weights[0])
        print(np_ptr[0])
        tensor = Tensor(axes_dims)
        tensor_ptr = tensor.unsafe_ptr()
        memcpy(tensor_ptr, np_ptr, 768)
        print("From tensor value at index: 0 : ", tensor[0])
        print("From tensor - numels: ", tensor.numels())
        print("From tensor - ndm: ", tensor.ndim())
        print("From tensor - datatype: ", tensor.datatype)
        # var indices = List[Int]()
        print()
        # tensor.print_tensor_recursive(indices, 1)
        for i in range(axes_dims[0]):
            for j in range(axes_dims[1]):
                if j >= 1:
                    continue
                for k in range(axes_dims[2]):
                    if k >= 50:
                        continue
                    print(tensor[i, j, k])
    except e:
        print(e)
