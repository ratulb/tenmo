from shared import TensorLike, GradStream
from ancestry import Ancestors
from common_utils import log_debug
from os import abort
from tensors import Tensor
from operators import AddTensor, SubtractTensor
from collections import Set


fn main():
    pass


@fieldwise_init
struct Graph[dtype: DType](Copyable & Movable):
    fn trace_ancestry(
        self, node: TensorLike[dtype], mut traced: Ancestors[dtype]
    ):
        if node not in traced:
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_copy(node)
            traced.append(ptr)
            msg = String(
                "Graph -> trace_ancestry -> DAG node inner id: "
                + String(node.inner_id())
                + " => kind:"
                + "Tensor" if ptr[].is_tensor() else "View"
            )
            log_debug(msg)

            for ancestor in node.ancestry():
                self.trace_ancestry(ancestor[], traced)

    fn backward_allocated(
        mut self,
        node: TensorLike[dtype],
        start_grad: Scalar[dtype],
    ):
        with_tensor = Tensor[dtype].full(node.shape(), start_grad)
        self.backward_allocated(node, with_tensor)

    fn backward_allocated(
        mut self,
        node: TensorLike[dtype],
        with_tensor: Tensor[dtype],
    ):
        if not node.requires_grad():
            return
        node.seed_grad(with_tensor)
        traced = Ancestors[dtype].untracked()
        self.trace_ancestry(node, traced)
        log_debug("Printing traced")
        traced.print()
        for each in traced:
            if not each.__as_bool__():
                abort("Null pointer found in DAG!")
            id = each[].inner_id()
            log_debug("About to call backward_allocated_fn for: " + String(id))
            if each[].is_view():
                log_debug("→ It's a TensorView")
                if each[].has_backward_fn():
                    log_debug(
                        "Calling backward_allocated_fn on view id: "
                        + String(each[].inner_id())
                    )
                    for recipient, grad_share, opcode in each[].backward_fn()(
                        each
                    ):
                        recipient.update_grad[AddTensor](
                            grad_share
                        ) if opcode == AddTensor else recipient.update_grad[
                            SubtractTensor
                        ](
                            grad_share
                        )
                else:
                    log_debug(
                        "Skipping empty grad_fn on view id: "
                        + String(each[].inner_id())
                    )
            else:
                log_debug("→ It's a Tensor")
                if each[].has_backward_fn():
                    log_debug(
                        "Calling grad_fn on tensor id: "
                        + String(each[].inner_id())
                    )
                    for _, _, _ in each[].backward_fn()(each):
                        continue
                else:
                    log_debug(
                        "Skipping empty grad_fn on tensor id: "
                        + String(each[].inner_id()),
                    )

    fn backward(self, root: TensorLike[dtype], start_grad: Scalar[dtype] = 1.0):
        if not root.requires_grad():
            return
        seed_tensor = Tensor[dtype].full(root.shape(), start_grad)
        self.backward(root, seed_tensor)

    fn backward(self, root: TensorLike[dtype], seed_tensor: Tensor[dtype]):
        if not root.requires_grad():
            return
        root.seed_grad(seed_tensor)
        tracked = Set[Int]()
        streams = List[GradStream[dtype]]()

        stack = [root]
        while stack:
            stream = stack.pop()
            if stream.inner_id() in tracked:
                continue
            streams.append(GradStream[dtype](stream))
            tracked.add(stream.inner_id())
            for origin in stream.ancestry():
                stack.append(origin[])

        for stream in streams:
            stream.flow()
