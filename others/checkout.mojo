from strides import Strides
from intlist import IntList
from shapes import Shape
from testing import assert_true, assert_false
from tensors import Tensor
from common_utils import id
from memory import ArcPointer
from collections import BitSet

@fieldwise_init
struct S[ext: Optional[DType] = None](Copyable & Movable):
    var a: Int

fn take_ptr_ref_of_tensor[dtype: DType, //](ptr: UnsafePointer[Tensor[dtype]], tensor: Tensor[dtype]):
    ptr[][7] = 72
    tensor.address()[][1] = 1919
    ptr[].shared_buffer = ptr[].buffer.shared()
fn main() raises:
    a = Tensor.arange(10)
    print("buffer is: ", a.buffer)
    b = a.address()
    take_ptr_ref_of_tensor(b, a)
    b[][4] = 42
    
    print("hellody")
    a.print()
    print("Has the buffer changed: ", a.shared_buffer.value()[])
   
