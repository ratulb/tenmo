### This file generates "gpt2_weights.npz" file after checkpoint files have been downloaded 
### by running the "fetch_gpt2_weights.mojo" file.
from python import Python
from tensors import Tensor
from utils import StaticTuple
from memory import memcpy


fn main():
    try:
        np = Python.import_module("numpy")
        npz = np.load("gpt2_weights.npz")
        #np_weights = npz['model/h8/attn/c_attn/w'];
        np_weights = npz['model/h8/attn/c_attn/w'];
        #print(np.weights)  # or any key
        print(np_weights.shape)  # or any key
        print(np_weights.size)  # or any key
        print(np_weights.dtype)  # Must be float32
        print(np_weights.ndim)  # Must be float32
        np_ptr = np_weights.__array_interface__["data"][0].unsafe_get_as_pointer[DType.float32]()
        #axes_dims = StaticTuple[Int, 1](768)        
        axes_dims = StaticTuple[Int, 3](1, 3072, 768)        
        print(np_weights[0])
        print(np_ptr[0])
        tensor = Tensor(axes_dims)
        tensor_ptr = tensor.unsafe_ptr()
        memcpy(tensor_ptr, np_ptr, 768)
        print("From tensor: ", tensor[0])
        print("From tensor: ", tensor.numels())
        print("From tensor: ", tensor.ndim())
        print("From tensor: ", tensor.datatype)
        #var indices = List[Int]()
        print()
        #tensor.print_tensor_recursive(indices, 1)
        for i in range(2359296):
           print(tensor[i])

    except e:
        print(e)
