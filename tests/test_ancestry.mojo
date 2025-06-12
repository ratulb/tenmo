from tensors import Tensor
from testing import assert_true
from tensors import Tensor
from ancestors import Ancestors

fn test_ancestors() raises:
    ancestors = Ancestors.none()
    t1 = Tensor.rand(3)
    addr = t1.address()
    ancestors.append(t1.address())
    assert_true(
        len(ancestors) == 1 and ancestors.capacity == 1 and ancestors.size == 1,
        "Ancestors'length, capacity and size assertion failed",
    )
    assert_true(
        ancestors.get(0) == addr, "Ancestors address get assertion failed"
    )
    t2 = Tensor.arange(0, 10)
    ancestors.append(t2.address())
    assert_true(
        ancestors.get(1) == t2.address() and ancestors.get(0) == addr,
        "Ancestors addresses get assertion failed",
    )
    ancestors.free()
    ancestors = Ancestors(addr, t2.address())
    assert_true(
        len(ancestors) == 2 and ancestors.capacity == 2 and ancestors.size == 2,
        "Ancestors' length, capacity and size assertion failed for variadic addresses",
    )
    ancestors.print()
    iterator = ancestors.__iter__()
    assert_true(iterator.__next__() == addr, "iterator next(0) assertion failed")
    assert_true(iterator.__next__() == t2.address(), "iterator next(1) assertion failed")
    ancestors_copied = ancestors
    ancestors.free()
    assert_true(
        ancestors_copied.get(1) == t2.address() and ancestors_copied.get(0) == addr,
        "Ancestors copied addresses get assertion failed",
    )
    ancestors = Ancestors.with_capacity(2)
    ancestors.append(addr)
    ancestors.append(t2.address())
    first = True
    for address in ancestors:
        if first:
            first = False
            assert_true(address == addr, "with_capacity first assertion failed")
        else:
            assert_true(address == t2.address(), "with_capacity 2nd assertion failed")

fn main() raises:
    test_ancestors()


