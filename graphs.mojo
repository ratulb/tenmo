from shared import TensorLike
from intlist import IntList
from ancestry import Ancestors
from common_utils import log_debug
from os import abort

from tensors import Tensor


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
        #print("Printing traced")
        #self.traced.print()
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
