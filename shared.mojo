from tensors import Tensor
from shapes import Shape
from intlist import IntList
from backpropagation import BackwardFn
from ancestry import Ancestors
from operators import AddTensor, SubtractTensor, Noop
from common_utils import log_debug, panic
from collections import Deque
from common_utils import addr, id


from testing import assert_true


fn main() raises:
    test_tensorlite_basics()


fn test_tensorlite_basics() raises:
    a = Tensor.arange(5, requires_grad=True)
    print("a.id(): ", a.id())
    ptr = UnsafePointer[Tensor[DType.float32]].alloc(1)
    print("ptr to int: ", Int(ptr))
    ptr.init_pointee_copy(a)
    tli = TensorLite[DType.float32](ptr)
    tli.seed_grad(91)
    assert_true(
        a.gradbox[].all_close(
            Tensor.d1([91.0, 91.0, 91.0, 91.0, 91.0]).float()
        ),
        "TensorLite grad setting assertion failed",
    )
    tli.tensor_address[][3] = 8989
    t = tli.tensor()
    print("t.id(): ", t.id())
    t.print()
    a.gradbox[].print()
    print()
    a.print()
    tli.gradients()[].print()


struct TensorLite[dtype: DType](
    Stringable & Representable & Writable & Copyable & Movable
):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    var tensor_address: Self.TensorAddress
    var delegate: Tensor[dtype]

    fn __init__(out self, tensor_address: Self.TensorAddress):
        self.tensor_address = tensor_address
        self.delegate = tensor_address[]
        print("Inside TensorLite: ", self.delegate)

    fn __copyinit__(out self, other: Self):
        self.tensor_address = other.tensor_address.copy()
        self.delegate = other.delegate.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.tensor_address = other.tensor_address
        self.delegate = other.delegate^

    fn __eq__(self, other: Self) -> Bool:
        return self.tensor_address == other.tensor_address

    @always_inline
    fn shape(self) -> Shape:
        return self.delegate.shape

    @always_inline
    fn gradients(self) -> UnsafePointer[Tensor[dtype]]:
        return self.delegate.gradients()

    @always_inline
    fn grad(self) -> Tensor[dtype]:
        return self.delegate.grad()

    @staticmethod
    fn of(tensor: Tensor[dtype]) -> TensorLite[dtype]:
        return Self(addr(tensor))

    @always_inline
    fn inner_id(self) -> Int:
        return Int(self.tensor_address)

    @always_inline
    fn inner_address(self) -> UnsafePointer[Tensor[dtype]]:
        return self.tensor_address

    @always_inline
    fn tensor(self) -> Tensor[dtype]:
        return self.delegate  # Caller needs to clean this up

    fn has_ancestry(self) -> Bool:
        return self.delegate.has_ancestry()

    @always_inline
    fn ancestry(self) -> Ancestors[dtype]:
        return self.delegate.ancestry()

    @always_inline
    fn requires_grad(self) -> Bool:
        return self.delegate.requires_grad

    @always_inline
    fn has_backward_fn(self) -> Bool:
        return self.delegate.has_backward_fn()

    @always_inline
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self.delegate.backward_fn()

    fn __str__(self) -> String:
        return self.inner_id().__str__()

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn seed_grad(self, value: Scalar[dtype]):
        copy = self.delegate
        copy.seed_grad(value)

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        copy = self.delegate
        copy.update_grad[opcode](incoming)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        copy = self.delegate
        copy.seed_grad(with_tensor)

    fn init_grad(self):
        copy = self.delegate
        copy.init_gradbox()

        _ = """fn nullify_ptr(mut self):
        self.tensor_address = UnsafePointer[Tensor[dtype]]()"""

    fn __del__1(deinit self):
        if self.tensor_address.__as_bool__():
            print(
                "TensorLite __del__: cleaning up managed tensor",
                self.delegate.id(),
                self.delegate.has_grad(),
            )
            self.delegate.gradbox = UnsafePointer[Tensor[dtype]]()
            _ = self.delegate^

    fn backward(root: TensorLite[dtype], start_grad: Scalar[dtype] = 1.0):
        if not root.requires_grad():
            return
        shape = root.shape()
        seed_tensor = Tensor[dtype].full(shape, start_grad)
        root.backward(seed_tensor)

    fn backward(root: TensorLite[dtype], seed_tensor: Tensor[dtype]):
        try:
            if not root.requires_grad():
                return

            # seed the output grad
            root.seed_grad(seed_tensor)
            seed_tensor.free()
            traced = IntList()
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
                if node.has_ancestry():
                    for parent in node.ancestry():
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
                            # recipient, Optional(grad_share), opcode
                            recipient,
                            grad_share,
                            opcode,
                        )
                        gs.sink()
                        # gs.free()

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
            panic(e.__str__())


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

    fn __moveinit__(out self, deinit other: Self):
        self.recipient = other.recipient
        self.grad = other.grad
        self.opcode = other.opcode

    fn has_backward_fn(self) -> Bool:
        return self.recipient.has_backward_fn()

    fn inner_id(self) -> Int:
        return self.recipient.inner_id()

    fn sink(self):
        grad_share = self.grad.value()
        log_debug("sink(): to id=" + self.recipient.inner_id().__str__())
        self.recipient.init_grad()
        self.recipient.update_grad[AddTensor](
            grad_share
        ) if self.opcode == AddTensor else self.recipient.update_grad[
            SubtractTensor
        ](
            grad_share
        )

    fn edges(
        self,
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        if not self.recipient.has_backward_fn():
            return []
        log_debug(
            "edges(): firing backward_fn of id="
            + self.recipient.inner_id().__str__()
        )

        out = self.recipient.backward_fn()(self.recipient)
        for ancestor, _, _ in out:
            log_debug(
                " -> produced edge to id=" + ancestor.inner_id().__str__()
            )
        return out
