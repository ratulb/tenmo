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
    pass


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
    var _graph: Optional[ComputationGraph[dtype]]  # Persistent graph

    fn __init__(out self, tensor: Tensor[dtype]):
        self._tensor = tensor.copy()
        self._id = (
            IDGen.generate_id()
        )  # Once it is inside ancestry as parent - we need fixed id
        self._graph = None  # Graph not built yet

    fn __copyinit__(out self, other: Self):
        self._id = other._id
        self._tensor = other._tensor.copy()
        # Don't copy graph - each Ancestor manages its own graph
        self._graph = None

    fn __moveinit__(out self, deinit existing: Self):
        self._id = existing._id
        self._tensor = existing._tensor^
        self._graph = existing._graph^

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

    fn reset_graph(mut self):
        """Reset graph - call when computation graph structure changes."""
        if self._graph:
            self._graph.value().reset()
            self._graph = None

    fn backward(mut output: Ancestor[dtype], start_grad: Scalar[dtype] = 1.0):
        if not output.requires_grad():
            return
        shape = output.shape()
        seed_tensor = Tensor[dtype].full(shape, start_grad)
        output.backward(seed_tensor)

    fn backward(mut self, seed_tensor: Tensor[dtype]):
        """Optimized backward - builds graph once, reuses on subsequent calls.
        """
        try:
            if not self.requires_grad():
                return

            # Seed the gradient for this output node
            self.seed_grad(seed_tensor)

            # Build graph structure on FIRST backward pass only
            if not self._graph:
                print("[Ancestor] First backward - building graph")

                var computation_graph = ComputationGraph[dtype]()

                computation_graph.build(self)
                self._graph = Optional(computation_graph^)
            else:
                print("[Ancestor] Reusing pre-built graph")

            # Execute backward using pre-built graph
            self._backward_with_graph()

        except e:
            print(e)
            panic(e.__str__())

    fn _backward_with_graph(mut self) raises:
        """Execute backward pass using pre-built graph structure."""

        ref computation_graph = self._graph.value()

        # Fresh copy of per-node pending-parent counts
        var remaining_parent_counts = (
            computation_graph.initial_fanin_template.copy()
        )

        # Mapping node_id → Ancestor object that owns the state
        var node_registry = Dict[Int, Ancestor[dtype]]()

        # Tracked node IDs to avoid reprocessing
        var visited_node_ids = IntList()

        # Work stack for DFS traversal
        var dfs_stack = List[Ancestor[dtype]]()

        dfs_stack.append(self.copy())

        # ------------------------------
        # DFS: Collect all nodes reachable from root
        # ------------------------------
        while len(dfs_stack) > 0:
            var current_node = dfs_stack.pop()

            var current_node_id = current_node.id()

            if current_node_id in visited_node_ids:
                continue

            visited_node_ids.append(current_node_id)
            node_registry[current_node_id] = current_node.copy()

            # Trace parents upward
            if current_node.has_ancestry():
                for parent in current_node.ancestry():
                    # parent is an Ancestor
                    dfs_stack.append(parent.copy())

        # ------------------------------
        # Ensure root exists in graph topology
        # ------------------------------

        var root_id = self.id()

        if root_id not in computation_graph.node_id_to_topo_index:
            panic("Root node not in graph topology")

        # ------------------------------
        # Backprop scheduler (BFS with ready queue)
        # ------------------------------

        # Queue of nodes ready for backward execution
        var ready_queue = Deque[Ancestor[dtype]]()

        # Insert root
        ready_queue.append(node_registry[root_id].copy())

        # ------------------------------
        # Execute backward ops in schedule order
        # ------------------------------

        while len(ready_queue) > 0:
            var active_node = ready_queue.popleft()

            if active_node.has_backward_fn():
                # Iterate through (recipient, grad, opcode)
                for result in active_node.backward_fn()(
                    active_node.tensor().copy()
                ):
                    var target_node = result[0].copy()
                    var incoming_grad = result[1].copy()
                    var op_code = result[2]

                    # -----------------------------------
                    # Accumulate gradient into target
                    # -----------------------------------
                    if op_code == AddTensor:
                        target_node.update_grad[AddTensor](incoming_grad^)
                    else:
                        target_node.update_grad[SubtractTensor](incoming_grad^)

                    # -----------------------------------
                    # Update dependency counters
                    # -----------------------------------

                    var target_node_id = target_node.id()

                    # Skip leaf nodes (not in graph)
                    if (
                        target_node_id
                        not in computation_graph.node_id_to_topo_index
                    ):
                        continue

                    var topo_index = computation_graph.node_id_to_topo_index[
                        target_node_id
                    ]

                    # Decrement remaining inputs expected
                    remaining_parent_counts[topo_index] -= 1

                    # When all parent contributions received → schedule it
                    if remaining_parent_counts[topo_index] == 0:
                        if target_node.has_backward_fn():
                            if target_node_id in node_registry:
                                ready_queue.append(
                                    node_registry[target_node_id].copy()
                                )


struct ComputationGraph[dtype: DType](Copyable & Movable):
    """Persistent graph structure - build once, reuse across backward passes."""

    # Node IDs in topologically sorted order
    var topo_order_node_ids: List[Int]
    # Stored initial fan-in count for each node (template for backward use)
    var initial_fanin_template: List[Int]
    # Maps node_id → index in topo_order_node_ids
    var node_id_to_topo_index: Dict[Int, Int]
    # Has the graph been built already?
    var has_graph_been_built: Bool

    fn __init__(out self):
        self.topo_order_node_ids = List[Int]()
        self.initial_fanin_template = List[Int]()
        self.node_id_to_topo_index = Dict[Int, Int]()
        self.has_graph_been_built = False

    fn build(mut self, output: Ancestor[dtype]):
        """Build graph structure once - called on first backward pass."""

        if self.has_graph_been_built:
            return

        print("[ComputationGraph] Building graph structure...")

        # ----------------------------
        # Phase 1: Graph tracing (DFS)
        # ----------------------------

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

            # Count parent fan-in and traverse upwards
            if current_node.has_ancestry():
                for parent in current_node.ancestry():
                    var parent_node_id = parent.id()

                    parent_fanin_counts[parent_node_id] = (
                        parent_fanin_counts.get(parent_node_id, 0) + 1
                    )

                    dfs_stack.append(parent.copy())

        # ----------------------------
        # Phase 2: Store topology
        # ----------------------------

        self.topo_order_node_ids = List[Int](capacity=len(visited_node_ids))

        for i in range(len(visited_node_ids)):
            self.topo_order_node_ids.append(visited_node_ids[i])

        # ----------------------------
        # Phase 3: Store initial fan-in template
        # ----------------------------

        self.initial_fanin_template = List[Int](capacity=len(visited_node_ids))

        for i in range(len(visited_node_ids)):
            var traced_node_id = visited_node_ids[i]
            var fanin_value = parent_fanin_counts.get(traced_node_id, 0)
            self.initial_fanin_template.append(fanin_value)

        # ----------------------------
        # Phase 4: Build lookup: node_id → topo index
        # ----------------------------

        for i in range(len(self.topo_order_node_ids)):
            self.node_id_to_topo_index[self.topo_order_node_ids[i]] = i

        self.has_graph_been_built = True
        print(
            "[ComputationGraph] Built graph with",
            len(self.topo_order_node_ids),
            "nodes",
        )

    fn reset(mut self):
        """Reset graph - call when computation graph structure changes."""
        print("[ComputationGraph] Resetting graph structure")

        self.topo_order_node_ids.clear()
        self.initial_fanin_template.clear()
        self.node_id_to_topo_index.clear()
        self.has_graph_been_built = False
