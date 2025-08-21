from tensors import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from backpropagation import BackwardFn
from os import abort
from ancestry import Ancestors
from operators import AddTensor, SubtractTensor, Noop
from common_utils import compute_output_shape, log_debug
from collections import Deque


fn main() raises:
    a = Tensor.arange(6 * 5 * 10).reshape(6, 5, 10)
    a.print(3, 3)


struct TensorLite[dtype: DType](
    Sized & Stringable & Representable & Writable & Copyable & Movable
):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    var tensor_address: Self.TensorAddress

    fn __init__(out self, tensor_ptr: Self.TensorAddress):
        self.tensor_address = tensor_ptr

    fn __copyinit__(out self, other: Self):
        self.tensor_address = other.tensor_address

    fn __moveinit__(out self, var other: Self):
        self.tensor_address = other.tensor_address

    fn __eq__(self, other: Self) -> Bool:
        return self.tensor_address == other.tensor_address

    @always_inline
    fn shape(self) -> Shape:
        return self.tensor_address[].shape

    @always_inline
    fn gradients(self) -> UnsafePointer[Tensor[dtype]]:
        return self.tensor_address[].gradients()

    @staticmethod
    fn of(tensor: Tensor[dtype]) -> Self:
        return Self(UnsafePointer(to=tensor))

    @always_inline
    fn owns_data(self) -> Bool:
        return self.tensor_address[].owns_data

    @always_inline
    fn inner_id(self) -> Int:
        return Int(self.tensor_address)

    @always_inline
    fn inner_address(self) -> UnsafePointer[Tensor[dtype]]:
        return self.tensor_address

    @always_inline
    fn tensor(self) -> Tensor[dtype]:
        return self.tensor_address[]

    @always_inline
    fn ancestry(self) -> Ancestors[dtype]:
        return self.tensor_address[].ancestors

    @always_inline
    fn requires_grad(self) -> Bool:
        return self.tensor_address[].requires_grad

    @always_inline
    fn has_grad(self) -> Bool:
        return self.tensor_address[].has_grad()

    @always_inline
    fn has_backward_fn(self) -> Bool:
        return self.tensor_address[].has_backward_fn()

    @always_inline
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self.tensor_address[].backward_fn()

    fn __str__(self) -> String:
        return self.inner_id().__str__()

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __len__(self) -> Int:
        return len(self.tensor_address[])

    fn seed_grad(self, value: Scalar[dtype]):
        self.tensor_address[].seed_grad(value)

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        self.tensor_address[].update_grad[opcode](incoming)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        self.tensor_address[].seed_grad(with_tensor)

    fn init_grad(self):
        if (
            not self.tensor_address[].owns_data
        ):  # Currently for tensors requiring grad, we initialize grad upfront
            self.tensor_address[].init_gradbox()

    fn backward(root: Self, start_grad: Scalar[dtype] = 1.0):
        if not root.requires_grad():
            return
        seed_tensor = Tensor[dtype].full(root.shape(), start_grad)
        root.backward(seed_tensor)

        _ = """fn backward(root: Self, seed_tensor: Tensor[dtype]):
        if not root.requires_grad():
            return
        root.seed_grad(seed_tensor)
        traced = IntList.Empty
        streams = List[GradStream[dtype]]()

        stack = [root]
        while stack:
            stream = stack.pop()
            if stream.inner_id() in traced:
                continue
            streams.append(GradStream[dtype](stream))
            traced.append(stream.inner_id())
            for origin in stream.ancestry():
                stack.append(origin[])

        log_debug("Traced ancestry: " + traced.__str__())
        for stream in streams:
            stream.flow()"""

    fn backward(root: TensorLite[dtype], seed_tensor: Tensor[dtype]):
        try:
            if not root.requires_grad():
                return

            # seed the output grad
            root.seed_grad(seed_tensor)

            traced = IntList.Empty
            streams = List[GradStream[dtype]]()
            use_count = Dict[
                Int, Int
            ]()  # <-- new: how many children feed each parent?

            # ---- trace phase ----
            stack = [root]
            while stack:
                node = stack.pop()
                sid = node.inner_id()
                if sid in traced:
                    continue
                streams.append(GradStream[dtype](node))
                traced.append(sid)

                # For each parent, increment use_count
                for origin in node.ancestry():
                    parent = origin[]
                    pid = parent.inner_id()
                    use_count[pid] = use_count.get(pid, 0) + 1
                    stack.append(parent)

            log_debug("Traced ancestry: " + traced.__str__())

            # Ensure root is schedulable immediately
            rid = root.inner_id()
            if rid not in use_count:
                use_count[rid] = 0

            # Build lookup for fast scheduling
            node_by_id = Dict[Int, GradStream[dtype]]()
            for s in streams:
                node_by_id[s.inner_id()] = s

            # ---- backward execution phase ----
            ready = Deque[GradStream[dtype]]()
            ready.append(node_by_id[rid])

            while ready:
                stream = ready.popleft()

                if stream.has_backward_fn():
                    edges = stream.edges()  # collect (recipient, grad, opcode)

                    for recipient, grad_share, opcode in edges:
                        # 1) sink grad into recipient (accumulate only!)
                        gs = GradStream[dtype](
                            recipient, Optional(grad_share), opcode
                        )
                        gs.sink()

                        # 2) decrement parent's fan-in
                        pid = recipient.inner_id()
                        remaining = use_count.get(pid, 0) - 1
                        use_count[pid] = remaining

                        # 3) schedule recipient when all contributions received
                        if remaining == 0 and recipient.has_backward_fn():
                            if pid in node_by_id:
                                ready.append(node_by_id[pid])
                else:
                    # leaf → grads already accumulated via sink
                    pass
        except e:
            abort(e.__str__())


struct GradStream[dtype: DType](Copyable & Movable):
    var recipient: TensorLite[dtype]
    var grad: Optional[
        Tensor[dtype]
    ]  # Gradient to apply (None if recipient is a tensor)
    var opcode: Int

    fn __init__(
        out self,
        recipient: TensorLite[dtype],
        grad: Optional[Tensor[dtype]] = None,
        opcode: Int = Noop,
    ):
        self.recipient = recipient
        self.grad = grad
        self.opcode = opcode

    fn __copyinit__(out self, other: Self):
        self.recipient = other.recipient
        self.grad = other.grad
        self.opcode = other.opcode

    fn __moveinit__(out self, var other: Self):
        self.recipient = other.recipient
        self.grad = other.grad
        self.opcode = other.opcode

    fn has_backward_fn(self) -> Bool:
        return self.recipient.has_backward_fn()

    fn inner_id(self) -> Int:
        return self.recipient.inner_id()

    fn flow(self):
        if self.recipient.has_backward_fn():
            for recipient, grad_share, opcode in self.recipient.backward_fn()(
                self.recipient
            ):
                gradstream = Self(recipient, Optional(grad_share), opcode)
                gradstream.sink()

    fn sink(self):
        grad_share = self.grad.value()
        log_debug("sink(): to id=" + self.recipient.inner_id().__str__())
        if not self.recipient.owns_data() and not self.recipient.has_grad():
            self.recipient.init_grad()
        self.recipient.update_grad[AddTensor](
            grad_share
        ) if self.opcode == AddTensor else self.recipient.update_grad[
            SubtractTensor
        ](
            grad_share
        )

    fn edges(self) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        if not self.recipient.has_backward_fn():
            return []
        log_debug("edges(): firing backward_fn of id=" + self.recipient.inner_id().__str__())

        out = self.recipient.backward_fn()(self.recipient)
        for (ancestor, _, _) in out:
            log_debug(" -> produced edge to id=" + ancestor.inner_id().__str__())
        return out
        #return self.recipient.backward_fn()(self.recipient)
