from shared import TensorLike
from ancestry import Ancestors
from collections import Set
from memory import UnsafePointer
from intlist import IntList

fn trace_ancestry[dtype: DType](node: TensorLike[dtype]) -> Ancestors[dtype]:
    traced = Ancestors[dtype]()
    visited = IntList()
    do_trace_ancestry[dtype](node, visited, traced, is_root=True)
    print("Ancestry - traced")
    traced.print()
    return traced


fn do_trace_ancestry[
    dtype: DType
](
    node: TensorLike[dtype],
    mut visited: IntList,
    mut traced: Ancestors[dtype],
    is_root: Bool = False,
):
    if node.id() in visited:
        return
    visited.append(node.id())
    print("Visited is now: ", visited.__str__())
    var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
    ptr.init_pointee_copy(node)
    traced.append(ptr)
    for ancestor in node.ancestry():
        do_trace_ancestry(
            ancestor[], visited, traced, False
        )


from tensors import Tensor


fn main() raises:
    #var a = Tensor.d1([1, 2], requires_grad=True)
    #var b = Tensor.d3([[[10, 20]]], requires_grad=True)
    print("I am not hellucinating - am I?")

    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    #var a = Tensor.d1([1, 2], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    var c = a + b
    print("IDs:")
    _="""print("a.id:", a.id(), a.address())
    print("b.id:", b.id(), b.address())
    print("c.id:", c.id(), c.address())"""
    print("a.id:", a.id())
    print("b.id:", b.id())
    print("c.id:", c.id())
 
    #s = c.sum(axes=[1], keepdims=False)
    s = c.sum()
    #print("s's id: ", s.id(), s.address())
    print("s's id: ", s.id())
    print("s's parents")
    s.ancestors.print()
    print("c's parents")
    c.ancestors.print()
    s.seed_grad(1.0)
    Tensor.walk_backward(s.into_tensorlike(), verbose=True)
    #result = trace_ancestry(s.into_tensorlike())
    #result.print()
    #print(c.shape)
    _= c^
    #result.free()
