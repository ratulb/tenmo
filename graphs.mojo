from shared import TensorLike
from intlist import IntList
from ancestry import Ancestors
from common_utils import log_debug
from os import abort

from tensors import Tensor
from views import TensorView

from testing import assert_true


fn test_scalar_tensor_no_op() raises:
    print("test_scalar_tensor_no_op")
    var a = Tensor.scalar(2.0, requires_grad=True)
    a.backward()
    assert_true(
        a.grad[].item() == 1.0, "Scalar tensor should receive gradient 1.0"
    )


fn test_scalar_addition() raises:
    print("test_scalar_addition")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a + b
    c.backward()
    assert_true(a.grad[].item() == 1.0, "a should get grad 1.0")
    assert_true(b.grad[].item() == 1.0, "b should get grad 1.0")


fn test_scalar_multiplication() raises:
    print("test_scalar_multiplication")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a * b
    c.backward()
    assert_true(a.grad[].item() == 3.0, "a should get grad of b (3.0)")
    assert_true(b.grad[].item() == 2.0, "b should get grad of a (2.0)")


fn test_broadcast_add_2d_1d() raises:
    print("test_broadcast_add_2d_1d")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    var c = a + b
    c.backward(Tensor.full_like(c, 1.0))
    assert_true(
        a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])),
        "Grad for a should be all 1s",
    )
    assert_true(
        b.grad[].all_close(Tensor.d1([2, 2])),
        "Grad for b should be summed across axis 0",
    )


fn test_sum_reduction() raises:
    print("test_sum_reduction")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum()
    s.backward()
    assert_true(s.item() == 10.0, "Sum value should be 10")
    assert_true(
        a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])),
        "Sum should distribute gradient evenly",
    )


fn test_sum_with_axis() raises:
    print("test_sum_with_axis")
    from shapes import Shape

    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[1], keepdims=False)
    s.backward(Tensor.d1([1.0, 1.0]).float())
    assert_true(
        s.shape == Shape.of(2), "Sum shape with keepdims=False should be [2]"
    )
    assert_true(
        a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])),
        "Axis sum should broadcast gradient",
    )


fn test_complex_dag() raises:
    print("test_complex_dag")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a + b
    var d = a * b
    var e = c + d
    e.backward()
    assert_true(a.grad[].item() == 4.0, "a contributes to both branches")
    assert_true(b.grad[].item() == 3.0, "b contributes to both branches")


fn test_same_tensor_twice() raises:
    print("test_same_tensor_twice")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = a + a  # same tensor used twice
    b.backward()
    assert_true(
        a.grad[].item() == 2.0, "a should receive gradient from both uses"
    )


fn test_grad_is_null() raises:
    print("test_grad_is_null")
    from common_utils import is_null

    var a = Tensor.scalar(2.0, requires_grad=False)
    a.backward()  # should be a no-op, not crash
    assert_true(
        is_null(a.grad), "No grad should be computed for requires_grad=False"
    )


fn test_backward_on_freed_node() raises:
    print("test_backward_on_freed_node")
    var a = Tensor.scalar(1.0, requires_grad=True)
    var b = a + 1.0
    print("a.id() and b.id(): ", a.id(), b.id())
    # b.free()  # manually free if you support it
    b.backward()  # should abort or skip safely
    b.print()


fn test_deep_chain() raises:
    print("test_deep_chain")
    var t = Tensor.scalar(1.0, requires_grad=True)
    for _ in range(100):
        t = t + 1.0
    t.backward()
    t.print()
    assert_true(
        t.grad[].item() == 1.0, "Should still compute correctly on long chain"
    )


fn main() raises:
    test_backward_on_freed_node()
    _ = """test_deep_chain()
    test_grad_is_null()
    test_same_tensor_twice()
    test_complex_dag()
    test_sum_with_axis()
    test_sum_reduction()
    test_broadcast_add_2d_1d()
    test_scalar_multiplication()
    test_scalar_addition()
    test_scalar_tensor_no_op()"""


struct Graph[dtype: DType = DType.float32](Copyable & Movable):
    var traced: Ancestors[dtype]
    var freed: Bool

    fn __init__(out self):
        self.traced = Ancestors[dtype]()
        self.freed = False

    fn __moveinit__(out self, owned other: Self):
        self.traced = other.traced
        self.freed = other.freed

    fn __copyinit__(out self, other: Self):
        self.traced = other.traced
        self.freed = other.freed

    fn free_graph(mut self):
        if not self.freed:
            self.traced.free()
            self.freed = True

    fn trace_ancestry(
        mut self,
        node: TensorLike[dtype],
        mut visited: IntList,
    ):
        if node.inner_id() not in visited:
            visited.append(node.inner_id())
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_copy(node)
            self.traced.append(ptr)
            msg = String(
                "Graph -> trace_ancestry -> DAG node inner id: "
                + String(node.inner_id())
                + " => kind:"
                + "Tensor" if ptr[].is_tensor() else "View"
            )
            log_debug(msg)

            for ancestor in node.tensor().ancestors:
                self.trace_ancestry(ancestor[], visited)

    fn walk_backward(
        mut self,
        node: TensorLike[dtype],
        start_grad: Scalar[dtype],
        verbose: Bool = False,
    ):
        with_tensor = Tensor[dtype].full(node.shape(), start_grad)
        self.walk_backward(node, with_tensor, verbose)

    fn walk_backward(
        mut self,
        node: TensorLike[dtype],
        with_tensor: Tensor[dtype],
        verbose: Bool = False,
    ):
        if not node.requires_grad():
            return
        if self.freed:
            self.traced = Ancestors[dtype]()
        visited = IntList()
        node.seed_grad(with_tensor)
        self.trace_ancestry(node, visited)
        print("Printing traced")
        self.traced.print()
        seen_ids = IntList()
        for each in self.traced:
            id = each[].inner_id()
            if id in seen_ids:
                abort("Duplicate TensorLike id in DAG: " + String(id))
            seen_ids.append(id)
            log_debug("About to call grad_fn for: " + String(id))
            ptr = each
            if not ptr.__as_bool__():
                abort("Null pointer found in DAG!")

            if each[].is_view():
                log_debug("→ It's a TensorView")
                v = each[].view()
                if v.base_tensor[].grad_fn:
                    log_debug("  grad_fn present")
                    log_debug(
                        "Calling grad_fn on view id: "
                        + String(each[].inner_id())
                    )
                    try:
                        each[].invoke_grad_fn(verbose)
                    except e:
                        print("grad_fn threw error:", e)
                else:
                    log_debug(
                        "Skipping empty grad_fn on view id: "
                        + String(each[].inner_id())
                    )
            else:
                log_debug("→ It's a Tensor")
                t = each[].tensor()
                if t.grad_fn:
                    log_debug("  grad_fn present")
                    log_debug(
                        "Calling grad_fn on tensor id: "
                        + String(each[].inner_id())
                    )
                    try:
                        # each[].invoke_grad_fn(verbose)
                        t.grad_fn.value()()
                    except e:
                        print("grad_fn threw error:", e)
                else:
                    log_debug(
                        "Skipping empty grad_fn on tensor id: "
                        + String(each[].inner_id()),
                    )
