from shared import Differentiable, TensorLike
from intlist import IntList
from ancestry import Ancestors
from memory import UnsafePointer
from common_utils import log_debug
from tensors import Tensor


fn main():
    print("Way to go buddy! Really!")
    a = Tensor.d2([[2, 3]], requires_grad=True)
    b = Tensor.d1([10])
    c = a + b
    s = c.sum()
    s.ancestors.print()
    graph = Graph()
    graph.walk_backward(s)
    s.free()
    c.free()
    a.free()
    b.free()
    graph.free_graph()
    print("Again")
    a.print()
    graph.walk_backward(s)

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
        if node.id() not in visited:
            visited.append(node.id())
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_copy(node)
            self.traced.append(ptr)
            msg = String(
                "DAG node inner id: "
                + String(node.inner_id())
                + " => kind:"
                + "Tensor" if ptr[].is_tensor() else "View"
            )
            log_debug(msg)

            for ancestor in node.tensor().ancestors:
                self.trace_ancestry(ancestor[], visited)

    fn walk_backward[
        Node: Differentiable, //
    ](
        mut self,
        node: Node,
        start_grad: Scalar[dtype] = 1.0,
        verbose: Bool = False,
    ):
        if self.freed:
            self.traced = Ancestors[dtype]()
        visited = IntList()
        if node.is_tensor():
            var tensor: Tensor[dtype] = rebind[Tensor[dtype]](
                node.into_tensor()
            )
            tensor.seed_grad(start_grad)
            tensor_like: TensorLike[dtype] = tensor.into_tensorlike()
            self.trace_ancestry(tensor_like, visited)
        print("Printing traced")
        self.traced.print()
        for each in self.traced:
            id = each[].inner_id()
            print("About to call grad_fn for", id)
            ptr = each
            if not ptr.__as_bool__():
                print("Null pointer found!")
                continue

            node_ = ptr[]
            print(
                "About to call grad_fn for", node_.inner_id()
            )  # <-- Match this against `traced.print()`

            if each[].is_view():
                print("→ It's a TensorView")
                v = each[].view()
                if v.base_tensor[].grad_fn:
                    # Mojo recommended way for checking Optional
                    print("  grad_fn present")
                    print("Calling grad_fn on view id:", each[].inner_id())
                    try:
                        each[].invoke_grad_fn(verbose)
                    except e:
                        print("grad_fn threw error:", e)
                else:
                    print(
                        "Skipping empty grad_fn on view id:", each[].inner_id()
                    )
            else:
                print("→ It's a Tensor")
                t = each[].tensor()
                if t.grad_fn:
                    print("  grad_fn present")
                    print("Calling grad_fn on tensor id:", each[].inner_id())
                    try:
                        # each[].invoke_grad_fn(verbose)
                        t.grad_fn.value()()
                    except e:
                        print("grad_fn threw error:", e)
                else:
                    print(
                        "Skipping empty grad_fn on tensor id:",
                        each[].inner_id(),
                    )
