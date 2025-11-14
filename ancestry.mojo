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

    fn __init__(out self, tensor: Tensor[dtype]):
        self._tensor = tensor.copy()
        self._id = (
            IDGen.generate_id()
        )  # Once it is inside ancestry as parent - we need fixed id

    fn __copyinit__(out self, other: Self):
        self._id = other._id
        self._tensor = other._tensor.copy()

    fn __moveinit__(out self, deinit other: Self):
        self._id = other._id
        self._tensor = other._tensor^

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

    fn backward(mut output: Ancestor[dtype], start_grad: Scalar[dtype] = 1.0):
        if not output.requires_grad():
            return
        shape = output.shape()
        seed_tensor = Tensor[dtype].full(shape, start_grad)
        output.backward(seed_tensor)

    fn backward(mut output: Ancestor[dtype], seed_tensor: Tensor[dtype]):
        try:
            if not output.requires_grad():
                return

            # seed the output grad
            output.seed_grad(seed_tensor)
            traced = IntList()
            streams = List[Ancestor[dtype]]()
            use_count = Dict[
                Int, Int
            ]()  # <-- new: how many children feed each parent?

            # ---- trace phase ----
            stack = [output.copy()]
            while stack:
                node = stack.pop()
                sid = node.id()
                if sid in traced:
                    continue
                streams.append(node.copy())
                traced.append(sid)

                # For each parent, increment use_count
                if node.has_ancestry():
                    for parent in node.ancestry():
                        pid = parent.id()
                        use_count[pid] = use_count.get(pid, 0) + 1
                        stack.append(parent.copy())

            log_debug("\nTraced ancestry: " + traced.__str__() + "\n")

            # Ensure root is schedulable immediately
            rid = output.id()
            if rid not in use_count:
                use_count[rid] = 0

            # Build lookup for fast scheduling
            node_by_id = Dict[Int, Ancestor[dtype]]()
            for s in streams:
                node_by_id[s.id()] = s.copy()

            # ---- backward execution phase ----
            ready = Deque[Ancestor[dtype]]()
            ready.append(node_by_id[rid].copy())

            while ready:
                stream = ready.popleft()

                if stream.has_backward_fn():
                    for result in stream.backward_fn()(stream.tensor().copy()):
                        var recipient = result[0].copy()
                        var grad_share = result[1].copy()
                        var opcode = result[2]
                        # 1) sink grad into recipient (accumulate only!)
                        recipient.update_grad[AddTensor](
                            grad_share^
                        ) if opcode == AddTensor else recipient.update_grad[
                            SubtractTensor
                        ](
                            grad_share^
                        )

                        # 2) decrement parent's fan-in
                        pid = recipient.id()
                        remaining = use_count.get(pid, 0) - 1
                        use_count[pid] = remaining

                        # 3) schedule recipient when all contributions received
                        if remaining == 0 and recipient.has_backward_fn():
                            if pid in node_by_id:
                                ready.append(node_by_id[pid].copy())
                else:
                    # leaf → grads already accumulated via sink
                    pass
        except e:
            print(e)
            panic(e.__str__())
