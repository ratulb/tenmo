from shared import TensorLike
from ancestry import Ancestors
from collections import Set
from memory import UnsafePointer


fn trace_ancestry[dtype: DType](node: TensorLike[dtype]) -> Ancestors[dtype]:
    print("Entered trace_ancestry and received node ancestry as shown below:")
    node.ancestry().print()
    traced = Ancestors[dtype]()
    if not node._requires_grad():
        return traced
    visited = Set[Int]()

    print("trace_ancestry - ok1")
    do_trace_ancestry[dtype](node, visited, traced, is_root=True)
    print("trace_ancestry - ok2")
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
    var a = Tensor.d1([1, 2], requires_grad=True)
    var b = Tensor.d3([[[10, 20]]], requires_grad=True)
    var c = a + b
    print("IDs:")
    print("a.id:", a.id())
    print("b.id:", b.id())
    print("c.id:", c.id())

    print("Ancestors of c:")
    c.ancestors.print()

    c.backward()
    # a.grad[].print()
    c.grad[].print()

    _ = """A = Tensor.d1([1], requires_grad = True)
    B = Tensor.d1([2], requires_grad = True)
    C = Tensor.d1([3], requires_grad = True)
    print("A, B and C's ids: ", A.id(), B.id(), C.id())
    C.ancestors.append(A.into_tensorlike().address())
    C.ancestors.append(B.into_tensorlike().address())
    C.ancestry().print()
    D = C.ancestry()
    D.print()
    C.into_tensorlike().ancestry().print()
    E = C

    #E.into_tensorlike().ancestry().print()
    traced = trace_ancestry(E.into_tensorlike().address())
    print("After tracing - the result is: ")
    traced.print()"""
