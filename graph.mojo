from shared import TensorLike
from ancestry import Ancestors
from collections import Set
from memory import UnsafePointer


fn trace_ancestry[dtype: DType](node: TensorLike[dtype]) -> Ancestors[dtype]:
    print("Entered trace_ancestry and received node ancestry as shown below:")
    node.ancestry().print()
    traced = Ancestors[dtype]()
    _="""if not node.requires_grad():
        return traced"""
    visited = Set[Int]()

    print("trace_ancestry - ok1")
    do_trace_ancestry[dtype](node, visited, traced, is_root=True)
    print("trace_ancestry - ok2")
    traced.print()
    return traced


fn do_trace_ancestry[
    dtype: DType
](
    node: TensorLike[dtype],
    mut visited: Set[Int],
    mut traced: Ancestors[dtype],
    is_root: Bool = False,
):
    if node.id() in visited:
        return
    visited.add(node.id())

    print("do_trace_ancestry - ok1 -> node.id() -> ", node.id())
    node.ancestry().print()
    for ancestor in node.ancestry():
        do_trace_ancestry(
            ancestor[], visited, traced, False
        )  # Children are not root
    # Only append if NOT the root node
    # if not is_root:
    traced.append(node.address())


from tensors import Tensor


fn main() raises:
    #var a = Tensor.d1([1, 2], requires_grad=True)
    #var b = Tensor.d3([[[10, 20]]], requires_grad=True)
    print("I am not hellucinating - am I?")

    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    var c = a + b
    print("IDs:")
    print("a.id:", a.id())
    print("b.id:", b.id())
    print("c.id:", c.id())
    #s = c.sum(axes=[1], keepdims=False)
    s = c.sum()
    print("s's parents")
    s.ancestors.print()
    print("c's parents")
    c.ancestors.print()
    s.seed_grad(1.0)
    Tensor.walk_backward(s.into_tensorlike(), verbose=True)
    #print(c.shape)
    _= c^
