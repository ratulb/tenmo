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
    do_trace_ancestry[dtype](
        node,
        visited,
        traced,
        is_root=UnsafePointer(to=Scalar[DType.bool](True)),
    )  # Mark as root
    print("trace_ancestry - ok2")
    return traced


fn do_trace_ancestry[
    dtype: DType
](
    node: TensorLike[dtype],
    mut visited: Set[Int],
    mut traced: Ancestors[dtype],
    is_root: UnsafePointer[Scalar[DType.bool]] = UnsafePointer(
        to=Scalar[DType.bool](False)
    ),
):
    if node.id() in visited:
        return
    visited.add(node.id())

    print("do_trace_ancestry - ok1 -> node.id() -> ", node.id())
    node.ancestry().print()
    for ancestor in node.ancestry():
        is_root[] = Scalar[DType.bool](False)
        do_trace_ancestry(
            ancestor[], visited, traced, is_root
        )  # Children are not root
    # Only append if NOT the root node
    if not is_root[]:
        traced.append(node.address())


from tensors import Tensor


fn main() raises:
    # var a = Tensor.d1([10], requires_grad=True)
    # var b = Tensor.d1([20], requires_grad=True)
    #var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var a = Tensor.d1([1, 2], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    print("Inputs a, b: ", a.id(), b.id())
    var c = a + b
    print("Before calling backward c.id: ", c.id())

    c.backward()

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
