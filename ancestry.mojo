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
    var _graph: Optional[ComputationGraph[dtype]]  # Persistent graph with topology

    fn __init__(out self, tensor: Tensor[dtype]):
        self._tensor = tensor.copy()
        self._id = IDGen.generate_id()
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
    fn tensor(ref self) -> ref[self._tensor] Tensor[dtype]:
        return self._tensor

    @always_inline
    fn shape(ref self) -> ref[self._tensor.buffer.shape] Shape:
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
    fn strides(ref self) -> ref[self._tensor.buffer.strides] Strides:
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
        var shape = output.shape()
        var seed_tensor = Tensor[dtype].full(shape, start_grad)
        output.backward(seed_tensor)

    fn backward_orig(mut self, seed_tensor: Tensor[dtype]):
        """
        Optimized backward with topology reuse and fresh data.

        First call: Builds topology structure (expensive, once only)
        Subsequent calls: Reuses topology, refreshes node registry (cheap)
        """
        try:
            if not self.requires_grad():
                return

            # Seed the gradient for this output node
            self.seed_grad(seed_tensor)

            # Build topology structure on FIRST backward pass only
            if not self._graph:
                print("[Ancestor] First backward - building topology")
                var computation_graph = ComputationGraph[dtype]()
                computation_graph.build_topology(self)
                self._graph = Optional(computation_graph^)
            else:
                print("[Ancestor] Reusing topology from previous backward")

            var computation_graph = self._graph.take()

            # Refresh node registry with CURRENT tensor data
            computation_graph.refresh_node_registry(self)

            # Execute backward using reused topology + fresh data
            computation_graph.execute_backward(self.id())
             # Put it back
            self._graph = Optional(computation_graph^)

        except e:
            print(e)
            panic(e.__str__())




    fn backward(mut self, seed_tensor: Tensor[dtype]):

        if not self.requires_grad():
            return

        var t_start = perf_counter_ns()
        self.seed_grad(seed_tensor)
        var t_seed = perf_counter_ns()
        print("[Backward] Seed grad:", (t_seed - t_start) / 1e6, "ms")

        if not self._graph:
            var t_build_start = perf_counter_ns()
            var computation_graph = ComputationGraph[dtype]()
            computation_graph.build_topology(self)
            self._graph = Optional(computation_graph^)
            var t_build_end = perf_counter_ns()
            print("[Backward] Build topology:", (t_build_end - t_build_start) / 1e6, "ms")
        else:
            print("[Backward] Reusing existing topology")

        var computation_graph = self._graph.take()

        var t_refresh_start = perf_counter_ns()
        computation_graph.refresh_node_registry(self)
        var t_refresh_end = perf_counter_ns()
        print("[Backward] Refresh registry:", (t_refresh_end - t_refresh_start) / 1e6, "ms")

        var t_exec_start = perf_counter_ns()
        try:
            computation_graph.execute_backward(self.id())
        except e:
            panic(e.__str__())
        var t_exec_end = perf_counter_ns()
        print("[Backward] Execute backward:", (t_exec_end - t_exec_start) / 1e6, "ms")

        self._graph = Optional(computation_graph^)

        var t_total = perf_counter_ns()
        print("[Backward] Total:", (t_total - t_start) / 1e6, "ms")


struct ComputationGraph[dtype: DType](Copyable & Movable):
    """Persistent graph structure - topology built once, registry refreshed each backward."""

    # REUSED: Built once, reused forever
    var topo_order_node_ids: List[Int]
    var initial_fanin_template: List[Int]
    var node_id_to_topo_index: Dict[Int, Int]

    # REFRESHED: Rebuilt each backward with current tensor data
    var node_registry: Dict[Int, Ancestor[dtype]]

    var has_topology_been_built: Bool

    fn __init__(out self):
        self.topo_order_node_ids = List[Int]()
        self.initial_fanin_template = List[Int]()
        self.node_id_to_topo_index = Dict[Int, Int]()
        self.node_registry = Dict[Int, Ancestor[dtype]]()
        self.has_topology_been_built = False

    fn build_topology(mut self, output: Ancestor[dtype]):
        """Build graph topology ONCE - called only on first backward."""
        if self.has_topology_been_built:
            return  # Already built, skip

        print("[ComputationGraph] Building topology structure (once)...")

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
        # Phase 2: Store topology order
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
        # Phase 4: Build lookup: node_id ? topo index
        # ----------------------------
        for i in range(len(self.topo_order_node_ids)):
            self.node_id_to_topo_index[self.topo_order_node_ids[i]] = i

        self.has_topology_been_built = True
        print(
            "[ComputationGraph] Topology built with",
            len(self.topo_order_node_ids),
            "nodes (reusable)"
        )

    fn refresh_node_registry(mut self, output: Ancestor[dtype]):
        """Rebuild node registry with CURRENT tensor data."""
        print("[ComputationGraph] Refreshing node registry with current data...")

        # Clear old registry
        self.node_registry.clear()

        # Rebuild with fresh data
        var visited_node_ids = IntList()
        var dfs_stack = List[Ancestor[dtype]]()
        dfs_stack.append(output.copy())

        while len(dfs_stack) > 0:
            var current_node = dfs_stack.pop()
            var current_node_id = current_node.id()

            if current_node_id in visited_node_ids:
                continue

            visited_node_ids.append(current_node_id)

            # Store fresh copy with current tensor data
            self.node_registry[current_node_id] = current_node.copy()

            if current_node.has_ancestry():
                for parent in current_node.ancestry():
                    dfs_stack.append(parent.copy())

        print("[ComputationGraph] Registry refreshed with", len(self.node_registry), "nodes")

    fn execute_backward(mut self, output_id: Int) raises:
        """Execute backward pass using pre-built topology and fresh node registry."""

        if not self.has_topology_been_built:
            panic("[ComputationGraph] Topology not built - call build_topology first")

        print("[ComputationGraph] Executing backward pass...")

        # Get fresh copy of use counts (cheap array copy)
        var remaining_parent_counts = self.initial_fanin_template.copy()

        # Ensure root exists in topology
        if output_id not in self.node_id_to_topo_index:
            panic("[ComputationGraph] Root node not in graph topology")

        # Initialize ready queue with root
        var ready_queue = Deque[Ancestor[dtype]]()
        ready_queue.append(self.node_registry[output_id].copy())

        # ----------------------------
        # Execute backward ops in schedule order
        # ----------------------------
        while len(ready_queue) > 0:
            var active_node = ready_queue.popleft()

            if active_node.has_backward_fn():
                # Execute backward function
                for result in active_node.backward_fn()(active_node.tensor().copy()):
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

                    # Skip if not in topology (shouldn't happen in well-formed graph)
                    if target_node_id not in self.node_id_to_topo_index:
                        continue

                    var topo_index = self.node_id_to_topo_index[target_node_id]

                    # Decrement remaining inputs expected
                    remaining_parent_counts[topo_index] -= 1

                    # When all parent contributions received ? schedule it
                    if remaining_parent_counts[topo_index] == 0:
                        if target_node_id in self.node_registry:
                            var target_from_registry = self.node_registry.pop(target_node_id)
                            if target_from_registry.has_backward_fn():
                                ready_queue.append(target_from_registry.copy())

        print("[ComputationGraph] Backward pass complete")

    fn reset(mut self):
        """Reset entire graph - forces topology rebuild on next use."""
        print("[ComputationGraph] Resetting entire graph structure")
        self.topo_order_node_ids.clear()
        self.initial_fanin_template.clear()
        self.node_id_to_topo_index.clear()
        self.node_registry.clear()
        self.has_topology_been_built = False


