from tenmo import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from backpropagationcopy import BackwardFn
from os import abort
from copyancestry import Ancestors
from operators import AddTensor, SubtractTensor, Noop
from common_utils import compute_output_shape, log_debug


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

    fn backward(root: Self, seed_tensor: Tensor[dtype]):
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
            stream.flow()


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

    fn flow(self):
        if self.recipient.has_backward_fn():
            for recipient, grad_share, opcode in self.recipient.backward_fn()(
                self.recipient
            ):
                gradstream = Self(recipient, Optional(grad_share), opcode)
                gradstream.sink()

    fn sink(self):
        grad_share = self.grad.value()
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
        return self.recipient.backward_fn()(self.recipient)
