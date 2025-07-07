from shared import Differentiable, TensorLike
from ancestry import Ancestors
from collections import Set

#fn trace_ancestry[dtype: DType](root: Differentiable) -> Ancestors[dtype]:
fn trace_ancestry[dtype: DType](node: TensorLike[dtype]) -> Ancestors[dtype]:
    traced = Ancestors[dtype]()
    if not node._requires_grad():
        return traced
    visited = Set[Int]()
    #node = root.into_tensorlike[dtype]()
    trace_ancestry[dtype](node, visited, traced)
    return traced


fn trace_ancestry[
    dtype: DType
](
    node: TensorLike[dtype],
    mut visited: Set[Int],
    mut traced: Ancestors[dtype],
):
    if node.id() not in visited:
        visited.add(node.id())
        for ancestor in node.ancestry():
            trace_ancestry(ancestor[], visited, traced)
        traced.append(node.address())


fn main():
    print("Life is good")
