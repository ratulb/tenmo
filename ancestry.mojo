from memory import Pointer
from common_utils import log_debug, panic
from tenmo import Tensor
from shapes import Shape
from intarray import IntArray
from strides import Strides
from backpropagation import BackwardFn
from operators import AddTensor, SubtractTensor
from collections import Deque, Set
from gradbox import Gradbox
from time import perf_counter_ns as now, monotonic


struct IDGen:
    @always_inline
    @staticmethod
    fn generate_id() -> Int:
        # Use both perf_counter and monotonic for additional entropy
        perf_time = now()
        mono_time = monotonic()

        # Combine them in a way that preserves uniqueness
        # Use XOR to mix the values
        return perf_time ^ (mono_time << 32)


fn main() raises:
    pass

@fieldwise_init
struct Parents[dtype: DType](Sized & Copyable & Movable):
    var parents: List[Tensor[dtype]]

    fn __len__(self) -> Int:
        return len(self.parents)

    fn __init__(out self):
        self.parents = List[Tensor[dtype]]()

    fn get(self, idx: Int) -> Tensor[dtype]:
        return self.parents[idx]



struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var ancestors: List[Ancestor[dtype]]

    fn __init__(out self):
        self.ancestors = List[Ancestor[dtype]]()

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.ancestors = existing.ancestors.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.ancestors = existing.ancestors^

    @staticmethod
    fn untracked() -> Ancestors[dtype]:
        return Self()

    @always_inline
    fn __del__(deinit self):
        self.ancestors.clear()
        log_debug("Ancestors __del__ called")

    fn get(self, idx: Int) -> Ancestor[dtype]:
        return self.ancestors[idx].copy()

    fn __len__(self) -> Int:
        return len(self.ancestors)

    fn __bool__(self) -> Bool:
        return len(self) > 0

    @always_inline
    fn append(mut self, var ancestor: Ancestor[dtype]):
        self.ancestors.append(ancestor^)

    @no_inline
    fn print(self):
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i).id(), end=" ")
        print()

    fn __iter__(ref self) -> AncestorIterator[self.dtype, origin_of(self)]:
        return AncestorIterator[self.dtype](0, Pointer(to=self))


struct AncestorIterator[dtype: DType, origin: Origin[False]](
    Sized & ImplicitlyCopyable
):
    var index: Int
    var src: Pointer[Ancestors[dtype], origin]

    fn __init__(out self, idx: Int, src: Pointer[Ancestors[dtype], origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Ancestor[dtype]:
        self.index += 1
        return self.src[].get(self.index - 1).copy()

    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


struct Ancestor[dtype: DType](
    Stringable & Representable & Writable & Copyable & Movable
):
    var _tensor: Tensor[dtype]
    var _id: Int

    fn __init__(out self, tensor: Tensor[dtype]):
        self._tensor = tensor.copy()
        self._id = IDGen.generate_id()

    fn __copyinit__(out self, other: Self):
        self._id = other._id
        self._tensor = other._tensor.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self._id = existing._id
        self._tensor = existing._tensor^

    @always_inline
    fn tensor(ref self) -> ref [self._tensor] Tensor[dtype]:
        return self._tensor

    @always_inline
    fn shape(ref self) -> ref [self._tensor.buffer.shape] Shape:
        return self._tensor.buffer.shape

    @always_inline
    fn offset(self) -> Int:
        return self._tensor.offset()

    @always_inline
    fn rank(self) -> Int:
        return self._tensor.rank()

    @always_inline
    fn max_index(self) -> Int:
        return self._tensor.max_index()

    @always_inline
    fn strides(ref self) -> ref [self._tensor.buffer.strides] Strides:
        return self._tensor.buffer.strides

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self._tensor.is_contiguous()

    @always_inline
    fn grad(self) -> Gradbox[dtype]:
        return self._tensor.grad().copy()

    @always_inline
    fn id(self) -> Int:
        return self._id

    @always_inline
    fn has_ancestry(self) -> Bool:
        return self._tensor.has_ancestry()

    @always_inline
    fn ancestry(self) -> Ancestors[dtype]:
        return self._tensor.ancestry().copy()

    @always_inline
    fn requires_grad(self) -> Bool:
        return self._tensor.requires_grad

    @always_inline
    fn has_backward_fn(self) -> Bool:
        return self._tensor.has_backward_fn()

    @always_inline
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self._tensor.backward_fn()

    fn __str__(self) -> String:
        return self._id.__str__()

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn seed_grad(mut self, value: Scalar[dtype]):
        self._tensor.seed_grad(value)

    fn seed_grad(mut self, with_tensor: Tensor[dtype]):
        self._tensor.seed_grad(with_tensor)

    fn update_grad[opcode: Int](self, incoming: Gradbox[dtype]):
        self._tensor.update_grad[opcode](incoming)

    fn init_grad(mut self):
        self._tensor.init_gradbox()

    # fn backward_optimized(mut output, seed_tensor: Tensor[dtype]):
    fn backward(mut output, seed_tensor: Tensor[dtype]):
        if not output.requires_grad():
            return

        output.seed_grad(seed_tensor)

        try:
            # Phase 1: Lightweight topology discovery

            var visited = Set[Int]()
            var topo_ids = List[Int]()
            var fanin = Dict[Int, Int]()
            var id_to_node = Dict[Int, Ancestor[dtype]]()

            # Use ID-based DFS to avoid copying in stack
            var dfs_stack = List[Int]()
            dfs_stack.append(output.id())
            id_to_node[output.id()] = output.copy()  # Copy ONCE

            while len(dfs_stack) > 0:
                var node_id = dfs_stack.pop()

                if node_id in visited:
                    continue

                visited.add(node_id)
                topo_ids.append(node_id)

                ref node = id_to_node[node_id]
                if node.has_ancestry():
                    for parent in node.ancestry():
                        var parent_id = parent.id()
                        fanin[parent_id] = fanin.get(parent_id, 0) + 1

                        if parent_id not in id_to_node:
                            id_to_node[parent_id] = parent.copy()  # Copy ONCE
                            dfs_stack.append(parent_id)

            var ready_queue = Deque[Int]()  # Store IDs, not Ancestors!
            ready_queue.append(output.id())

            while len(ready_queue) > 0:
                var node_id = ready_queue.popleft()  # Just an Int
                ref node = id_to_node[node_id]  # Reference to stored copy

                if node.has_backward_fn():
                    # Execute backward - this is the expensive part
                    for result in node.backward_fn()(node.tensor()):
                        var target_node = result[0].copy()
                        var grad = result[1].copy()
                        var op_code = result[2]

                        var target_id = target_node.id()

                        # Update gradient
                        if target_id in id_to_node:
                            ref target = id_to_node[target_id]
                            if op_code == AddTensor:
                                target.update_grad[AddTensor](grad^)
                            else:
                                target.update_grad[SubtractTensor](grad^)

                        # Schedule when ready
                        if target_id in fanin:
                            fanin[target_id] -= 1
                            if fanin[target_id] == 0:
                                if id_to_node[target_id].has_backward_fn():
                                    ready_queue.append(target_id)

        except e:
            print(e)

    fn backward(mut output: Ancestor[dtype], start_grad: Scalar[dtype] = 1.0):
        if not output.requires_grad():
            return
        var shape = output.shape()
        var seed_tensor = Tensor[dtype].full(shape, start_grad)
        output.backward(seed_tensor)


