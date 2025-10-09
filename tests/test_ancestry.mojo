from testing import assert_true
from tensors import Tensor
from ancestry import Ancestors
from shared import TensorLite


fn test_contains() raises:
    print("test_contains")
    a = Tensor.scalar(10)
    tl1 = TensorLite.of(a)
    tl2 = TensorLite.of(a)
    ancestors = Ancestors[DType.float32].untracked()
    ancestors.append(tl1)
    assert_true(ancestors.__contains__(tl2))


fn validate_ancestor_materialization[
    dtype: DType, //
](parent: Tensor[dtype], container: TensorLite[dtype]) raises:
    assert_true(
        (parent == container.tensor()).all_true(),
        "TensorLite can materialize original parent - assertion failed",
    )


fn test_tensorlite_tossed_around_can_materialize_parent() raises:
    print("test_tensorlite_tossed_around_can_materialize_parent")
    parent = Tensor.rand(5, 3, min=0, max=5)
    descendent = Tensor.arange(5)
    descendent.add_ancestry(parent)
    container = descendent.ancestors.get(0)  # <- TensorLite
    container_copy = container
    validate_ancestor_materialization(parent, container_copy)
    container_transfered = container^
    validate_ancestor_materialization(parent, container_transfered)
    print(
        "test_tensorlite_tossed_around_can_materialize_parent run successfully"
    )


fn test_tensorlite_tossed_around_contains_casted_parent() raises:
    print("test_tensorlite_tossed_around_contains_casted_parent")
    parent = Tensor[DType.int32].rand(5, 4, min=0, max=5)
    parent_casted = parent.float()

    descendent = Tensor.arange(5)
    descendent.add_ancestry(parent_casted)
    container = descendent.ancestors.get(0)  # <- TensorLite
    container_copy = container
    validate_ancestor_materialization(parent_casted, container_copy)
    assert_true(
        (parent == container.tensor().to_dtype[DType.int32]()).all_true(),
        "retrieve and recast assertion 1 failed",
    )
    assert_true(
        (parent == container_copy.tensor().to_dtype[DType.int32]()).all_true(),
        "retrieve and recast assertion 2 failed",
    )
    container_transfered = container^
    validate_ancestor_materialization(parent_casted, container_transfered)
    print(
        "test_tensorlite_tossed_around_contains_casted_parent run successfully"
    )


fn main() raises:
    test_contains()
    test_tensorlite_tossed_around_can_materialize_parent()
    test_tensorlite_tossed_around_contains_casted_parent()
