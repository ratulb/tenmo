from shared import TensorLike
from ancestry import Ancestors
from collections import Set
from memory import UnsafePointer

fn trace_ancestry[dtype: DType](node: UnsafePointer[TensorLike[dtype]]) -> Ancestors[dtype]:
    print("Tracing receiving ancestry.....")
    node[].ancestry().print()
    print("Received ancestry above")
    traced = Ancestors[dtype]()
    if not node[]._requires_grad():
        return traced
    visited = Set[Int]()
    print("Before going to trace_ancestry1")
    node[].ancestry().print()
    trace_ancestry1[dtype](node[], visited, traced, is_root=True)  # Mark as root
    return traced

fn trace_ancestry1[
    dtype: DType
](
    node: TensorLike[dtype],
    mut visited: Set[Int],
    mut traced: Ancestors[dtype],
    is_root: Bool = False,  # New flag
):
    if node.id() in visited:
        return
    visited.add(node.id())

    for ancestor in node.ancestry():
        trace_ancestry1(ancestor[], visited, traced, is_root=False)  # Children are not root
        print("looping", ancestor[].id())

    # Only append if NOT the root node
    if not is_root:
        traced.append(node.address())

from tensors import Tensor

fn main():
    print("Life is good")
    A = Tensor.d1([1], requires_grad = True)
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
    traced.print()

