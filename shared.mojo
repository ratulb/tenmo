from tensors import Tensor
from shapes import Shape
from views import TensorView
from intlist import IntList
from strides import Strides
from backpropagation import BackwardFn
from os import abort
from ancestry import Ancestors
from collections import Set
from operators import AddTensor, SubtractTensor, Noop
from common_utils import compute_output_shape


fn main() raises:
    a = Tensor.arange(6 * 5 * 10).reshape(6, 5, 10)
    a.print(3, 3)


struct TensorLike[dtype: DType](
    Sized & Stringable & Representable & Writable & Copyable & Movable
):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    alias ViewAddress = UnsafePointer[TensorView[dtype]]

    var kind: Int
    var tensor_address: Self.TensorAddress
    var view_address: Self.ViewAddress

    fn __init__(out self, tensor_ptr: Self.TensorAddress):
        self.kind = 0
        self.tensor_address = tensor_ptr
        self.view_address = Self.ViewAddress()  # null

    fn __init__(out self, view_ptr: Self.ViewAddress):
        self.kind = 1
        self.tensor_address = Self.TensorAddress()  # null
        self.view_address = view_ptr

    fn __copyinit__(out self, other: Self):
        self.kind = other.kind
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn __moveinit__(out self, var other: Self):
        self.kind = other.kind
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn __eq__(self, other: Self) -> Bool:
        if self.kind != other.kind:
            return False
        return (
            self.tensor_address
            == other.tensor_address if self.kind
            == 0 else self.view_address
            == other.view_address
        )

    fn equal(self, other: Self) -> Bool:
        if self.shape() != other.shape():
            return False
        for indices in self.shape():
            if self[indices] != other[indices]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @staticmethod
    fn from_tensor(tensor: Tensor[dtype]) -> Self:
        return Self(UnsafePointer(to=tensor))

    @staticmethod
    fn from_view(view: TensorView[dtype]) -> Self:
        return Self(UnsafePointer(to=view))

    fn is_view(self) -> Bool:
        return self.kind == 1

    fn is_tensor(self) -> Bool:
        return self.kind == 0

    fn inner_id(self) -> Int:
        return Int(self.tensor_address) if self.kind == 0 else Int(
            self.view_address
        )

    fn rows(self) -> Int:
        return (
            self.tensor_address[].rows() if self.kind
            == 0 else self.view_address[].rows()
        )

    fn cols(self) -> Int:
        return (
            self.tensor_address[].cols() if self.kind
            == 0 else self.view_address[].cols()
        )

    fn tensor(self) -> Tensor[dtype]:
        return self.tensor_address[]

    fn view(self) -> TensorView[dtype]:
        return self.view_address[]

    fn ancestry(self) -> Ancestors[dtype]:
        return (
            self.tensor_address[].ancestors if self.kind
            == 0 else self.view_address[].ancestors
        )

    fn has_grad(self) -> Bool:
        return (
            self.tensor_address[].has_grad() if self.kind
            == 0 else self.view_address[].has_grad()
        )

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        return (
            self.tensor_address[][indices] if self.kind
            == 0 else self.view_address[][indices]
        )

    fn has_backward_fn(self) -> Bool:
        return (
            self.tensor_address[].has_backward_fn() if self.kind
            == 0 else self.view_address[].has_backward_fn()
        )

    fn backward_fn(self) -> BackwardFn[dtype]:
        return (
            self.tensor_address[].backward_fn() if self.kind
            == 0 else self.view_address[].backward_fn()
        )

    fn gradients(self) -> UnsafePointer[Tensor[dtype]]:
        return (
            self.tensor_address[].grad if self.kind
            == 0 else self.view_address[].grad
        )

    fn rank(self) -> Int:
        return self.shape().rank()

    fn base_shape(self) -> Shape:
        if self.kind == 0:
            return self.tensor_address[].shape
        else:
            return self.view_address[].base_tensor[].shape

    fn strides(self) -> Strides:
        return (
            self.tensor_address[].strides if self.kind
            == 0 else self.view_address[].strides
        )

    fn offset(self) -> Int:
        return 0 if self.kind == 0 else self.view_address[].offset

    fn shape(self) -> Shape:
        if self.kind == 0:
            return self.tensor_address[].shape
        else:
            return self.view_address[].shape

    fn seed_grad(self, value: Scalar[dtype]):
        if self.kind == 0:
            self.tensor_address[].seed_grad(value)
        else:
            self.view_address[].seed_grad(value)

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        if self.kind == 0:
            self.tensor_address[].update_grad[opcode](incoming)
        else:
            self.view_address[].update_grad[opcode](incoming)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        if self.kind == 0:
            self.tensor_address[].seed_grad(with_tensor)
        else:
            self.view_address[].seed_grad(with_tensor)

    fn init_grad(self):
        if (
            self.kind == 1
        ):  # Currently for tensors requiring grad, we initialize grad upfront
            self.view_address[].init_grad()

    fn requires_grad(self) -> Bool:
        return (
            self.view_address[]._requires_grad() if self.kind
            == 1 else self.tensor_address[]._requires_grad()
        )

    fn __str__(self) -> String:
        if self.kind == 0:
            t = self.tensor_address[]
            return t.__str__()
        else:
            v = self.view_address[]
            return v.__str__()

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __len__(self) -> Int:
        return len(self.tensor_address[]) if self.kind == 0 else len(
            self.view_address[]
        )

    fn sum_all(self) -> Scalar[dtype]:
        if self.kind == 0:
            return self.tensor_address[].sum_all()
        else:
            return self.view_address[].sum_all()

    fn sum(
        self, normalized_axes: IntList, keepdims: Bool = False
    ) -> Tensor[dtype]:
        """Compute sum along specified axes."""
        shape = self.shape()
        rank = shape.rank()

        out_shape = compute_output_shape(shape, normalized_axes, keepdims)
        out = Tensor[dtype].zeros(out_shape)

        if out_shape == Shape.Void:
            if rank == 0:  # Scalar case
                out[IntList.Empty] = self[IntList.Empty]
            elif rank == len(normalized_axes) and not keepdims:  # Reducing all
                out[IntList.Empty] = self.sum_all()
        else:
            reduced_shape = Shape(shape.axes_spans.select(normalized_axes))
            for out_idx in out_shape:
                var summ = Scalar[dtype](0)
                for red_idx in reduced_shape:
                    full_idx = out_idx.replace(
                        normalized_axes, red_idx
                    ) if keepdims else out_idx.insert(normalized_axes, red_idx)
                    summ += self[full_idx]
                out[out_idx] = summ

        return out

    fn mean(
        self, normalized_axes: IntList, keepdims: Bool = False
    ) -> Tensor[dtype]:
        shape = self.shape()
        # Compute total count of elements being reduced
        count = shape.axes_spans.select(normalized_axes).product()
        # Perform sum and divide by count
        out = self.sum(normalized_axes, keepdims) / Scalar[dtype](count)
        return out

    fn backward(root: Self, start_grad: Scalar[dtype] = 1.0):
        if not root.requires_grad():
            return
        seed_tensor = Tensor[dtype].full(root.shape(), start_grad)
        root.backward(seed_tensor)

    fn backward(root: Self, seed_tensor: Tensor[dtype]):
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

    fn print_tensor_recursive(
        self,
        mut indices: IntList,
        level: Int,
        num_first: Int = 10,
        num_last: Int = 10,
    ):
        if self.rank() == 0:  # Tensor with Shape ()
            print(self[IntList.Empty])
            return
        current_dim = len(indices)
        indent = " " * (level * 2)

        if current_dim >= self.rank():
            print(
                "ERROR: current_dim (",
                current_dim,
                ") >= ndim (",
                self.rank(),
                ")",
            )
            return

        size = self.shape()[current_dim]

        if size < 0 or size > 1_000_000:
            print(
                "ERROR: suspicious size: ",
                size,
                "at dim ",
                current_dim,
                self.shape().__str__(),
            )
            return

        # Base case: last dimension (print actual elements)
        if current_dim == self.rank() - 1:
            print(indent + "[", end="")

            for i in range(size):
                if i < num_first:
                    indices.append(i)
                    print(self[indices], end="")
                    _ = indices.pop()
                    if i != size - 1:
                        print(", ", end="")
                elif i == num_first and size > num_first + num_last:
                    print("..., ", end="")
                elif i >= size - num_last:
                    indices.append(i)
                    print(self[indices], end="")
                    _ = indices.pop()
                    if i != size - 1:
                        print(", ", end="")

            print("]", end="")

        else:
            print(indent + "[")
            for i in range(size):
                if i < num_first:
                    indices.append(i)
                    self.print_tensor_recursive(
                        indices, level + 1, num_first, num_last
                    )
                    _ = indices.pop()
                elif i == num_first and size > num_first + num_last:
                    print(indent + "  ...,")
                elif i >= size - num_last:
                    indices.append(i)
                    self.print_tensor_recursive(
                        indices, level + 1, num_first, num_last
                    )
                    _ = indices.pop()

                # Print comma and newline for all but last element
                if i != size - 1 and (i < num_first or i >= size - num_last):
                    print(",")
                # Special case: last element needs newline before closing bracket
                elif i == size - 1:
                    print()  # Newline before closing bracket

            print(indent + "]", end="")

    fn print(self, num_first: Int = 5, num_last: Int = 1):
        print(
            self.__str__(),
            end="\n",
        )
        empty = IntList()
        self.print_tensor_recursive(
            empty, 1, num_first=num_first, num_last=num_last
        )


struct GradStream[dtype: DType](Copyable & Movable):
    var recipient: TensorLike[dtype]  # Tensor or View
    var grad: Optional[
        Tensor[dtype]
    ]  # Gradient to apply (None if recipient is a tensor)
    var opcode: Int

    fn __init__(
        out self,
        recipient: TensorLike[dtype],
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

    fn is_view(self) -> Bool:
        return self.recipient.is_view()

    fn is_tensor(self) -> Bool:
        return self.recipient.is_tensor()

    fn has_backward_fn(self) -> Bool:
        return self.recipient.has_backward_fn()

    fn flow(self):
        if self.recipient.has_backward_fn():
            for recipient, grad_share, opcode in self.recipient.backward_fn()(
                UnsafePointer(to=self.recipient)
            ):
                gradstream = Self(recipient, Optional(grad_share), opcode)
                gradstream.sink()

    fn sink(self):
        grad_share = self.grad.value()
        if self.recipient.is_view() and not self.recipient.has_grad():
            self.recipient.init_grad()
        self.recipient.update_grad[AddTensor](
            grad_share
        ) if self.opcode == AddTensor else self.recipient.update_grad[
            SubtractTensor
        ](
            grad_share
        )
