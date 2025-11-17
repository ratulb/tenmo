from memory import Pointer
from common_utils import log_debug, panic
from tenmo import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from backpropagation import BackwardFn
from operators import AddTensor, SubtractTensor
from collections import Deque
from gradbox import Gradbox
from time import perf_counter_ns, monotonic


struct IDGen:
    @always_inline
    @staticmethod
    fn generate_id() -> Int:
        # Use both perf_counter and monotonic for additional entropy
        perf_time = perf_counter_ns()
        mono_time = monotonic()

        # Combine them in a way that preserves uniqueness
        # Use XOR to mix the values
        return perf_time ^ (mono_time << 32)


fn main() raises:
    _="""alias dtype = DType.float32
    var W = Tensor[dtype].rand(Shape([20, 30]), requires_grad=True)
    var graph: Optional[ComputationGraph[dtype]] = None

    for epoch in range(100):
        # Forward pass
        var x = Tensor[dtype].rand([30, 30])
        var y = x.matmul(W)
        var loss = y.sum()

        # Backward pass
        if not graph:
            # First iteration - build graph
            graph = Optional(loss.backward_graph())
        else:
            # Subsequent iterations - reuse graph
            loss.seed_grad(Tensor[dtype].ones(loss.shape()))
            graph.value().execute_backward(loss.id())

        # Update weights
        W -= W.grad() * 0.01

        # Zero gradients
        graph.value().zero_grads()"""


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
        self._id = (
            IDGen.generate_id()
        )  # Once it is inside ancestry as parent - we need fixed id

    fn __copyinit__(out self, other: Self):
        self._id = other._id
        self._tensor = other._tensor.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self._id = existing._id
        self._tensor = existing._tensor^

    @always_inline
    fn tensor(self) -> Tensor[dtype]:
        return self._tensor.copy()

    @always_inline
    fn shape(self) -> Shape:
        return self._tensor.shape()

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
    fn strides(self) -> Strides:
        return self._tensor.strides()

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

    fn update_grad[opcode: Int](mut self, incoming: Gradbox[dtype]):
        self._tensor.update_grad[opcode](incoming)

    fn init_grad(mut self):
        self._tensor.init_gradbox()

    fn backward(mut output: Ancestor[dtype], start_grad: Scalar[dtype] = 1.0) -> ComputationGraph[dtype]:
        if not output.requires_grad():
            return ComputationGraph[dtype]()
        shape = output.shape()
        seed_tensor = Tensor[dtype].full(shape, start_grad)
        return output.backward(seed_tensor)


    fn backward(mut self, seed_tensor: Tensor[dtype]) -> ComputationGraph[dtype]:
        """Build and execute backward pass - returns graph for reuse."""
        if not self.requires_grad():
            return ComputationGraph[dtype]()  # Empty graph

        # Seed gradient
        self.seed_grad(seed_tensor)

        # Build graph
        var graph = ComputationGraph[dtype]()
        graph.build(self)

        return graph^


struct ComputationGraph[dtype: DType](Copyable & Movable):
    """Persistent graph structure - build once, execute many times."""

    # Topology
    var topo_order_node_ids: List[Int]
    var initial_fanin_template: List[Int]
    var node_id_to_topo_index: Dict[Int, Int]
    var node_registry: Dict[Int, Ancestor[dtype]]
    var has_graph_been_built: Bool

    fn __init__(out self):
        self.topo_order_node_ids = List[Int]()
        self.initial_fanin_template = List[Int]()
        self.node_id_to_topo_index = Dict[Int, Int]()
        self.node_registry = Dict[Int, Ancestor[dtype]]()
        self.has_graph_been_built = False

    fn build(mut self, output: Ancestor[dtype]):
        """Build graph structure and store all nodes."""
        if self.has_graph_been_built:
            return

        print("[ComputationGraph] Building graph structure...")

        var visited_node_ids = IntList()
        var collected_nodes = List[Ancestor[dtype]]()
        var parent_fanin_counts = Dict[Int, Int]()

        var dfs_stack = List[Ancestor[dtype]]()
        dfs_stack.append(output.copy())

        while len(dfs_stack) > 0:
            var current_node = dfs_stack.pop()
            var current_node_id = current_node.id()

            if current_node_id in visited_node_ids:
                continue

            visited_node_ids.append(current_node_id)
            collected_nodes.append(current_node.copy())

            # Store node in registry
            self.node_registry[current_node_id] = current_node.copy()

            if current_node.has_ancestry():
                for parent in current_node.ancestry():
                    var parent_node_id = parent.id()
                    parent_fanin_counts[parent_node_id] = (
                        parent_fanin_counts.get(parent_node_id, 0) + 1
                    )
                    dfs_stack.append(parent.copy())

        # Store topology
        self.topo_order_node_ids = List[Int](capacity=len(visited_node_ids))
        for i in range(len(visited_node_ids)):
            self.topo_order_node_ids.append(visited_node_ids[i])

        # Store fan-in template
        self.initial_fanin_template = List[Int](capacity=len(visited_node_ids))
        for i in range(len(visited_node_ids)):
            var traced_node_id = visited_node_ids[i]
            var fanin_value = parent_fanin_counts.get(traced_node_id, 0)
            self.initial_fanin_template.append(fanin_value)

        # Build lookup
        for i in range(len(self.topo_order_node_ids)):
            self.node_id_to_topo_index[self.topo_order_node_ids[i]] = i

        self.has_graph_been_built = True
        print("[ComputationGraph] Built graph with", len(self.topo_order_node_ids), "nodes")

    fn execute_backward(mut self, output_id: Int) raises:
        """Execute backward pass using stored graph structure."""
        if not self.has_graph_been_built:
            panic("Graph not built - call build() first")

        print("[ComputationGraph] Executing backward pass")

        var remaining_parent_counts = self.initial_fanin_template.copy()

        if output_id not in self.node_id_to_topo_index:
            panic("Output node not in graph topology")

        var ready_queue = Deque[Ancestor[dtype]]()
        ready_queue.append(self.node_registry[output_id].copy())

        while len(ready_queue) > 0:
            var active_node = ready_queue.popleft()

            if active_node.has_backward_fn():
                for result in active_node.backward_fn()(active_node.tensor().copy()):
                    var target_node = result[0].copy()
                    var incoming_grad = result[1].copy()
                    var op_code = result[2]

                    # Accumulate gradient
                    if op_code == AddTensor:
                        target_node.update_grad[AddTensor](incoming_grad^)
                    else:
                        target_node.update_grad[SubtractTensor](incoming_grad^)

                    var target_node_id = target_node.id()

                    if target_node_id not in self.node_id_to_topo_index:
                        continue

                    var topo_index = self.node_id_to_topo_index[target_node_id]
                    remaining_parent_counts[topo_index] -= 1

                    if remaining_parent_counts[topo_index] == 0:
                        if target_node.has_backward_fn():
                            if target_node_id in self.node_registry:
                                ready_queue.append(self.node_registry[target_node_id].copy())

    fn zero_grads(mut self):
        """Zero out all gradients in the graph."""
        print("[ComputationGraph] Zeroing gradients")
        for ref node in self.node_registry.values():
            if node.requires_grad():
                node._tensor.zero_grad()

    fn reset(mut self):
        """Reset graph - forces rebuild on next use."""
        print("[ComputationGraph] Resetting graph structure")
        self.topo_order_node_ids.clear()
        self.initial_fanin_template.clear()
        self.node_id_to_topo_index.clear()
        self.node_registry.clear()
        self.has_graph_been_built = False

