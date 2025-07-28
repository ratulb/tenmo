from tensors import Tensor
from shapes import Shape
from views import TensorView
from intlist import IntList
from backpropagation import BackwardFn
from os import abort
from ancestry import Ancestors
from collections import Set
from operators import AddTensor, SubtractTensor, Noop


fn main() raises:
    pass


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

    fn tensor_ptr(self) -> Self.TensorAddress:
        return self.tensor_address

    fn view_ptr(self) -> Self.ViewAddress:
        return self.view_address

    fn inner_id(self) -> Int:
        return Int(self.tensor_address) if self.kind == 0 else Int(
            self.view_address
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
        # Defensive check
        if current_dim >= self.rank():
            # if current_dim > self.rank():
            print(
                "ERROR: current_dim (",
                current_dim,
                ") >= ndim (",
                self.rank(),
                ")",
            )
            return

        size = self.shape()[current_dim]

        # Size sanity check
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
                    print(
                        self[indices],
                        end=", " if (
                            i != num_first - 1 or size > num_first + num_last
                        ) else "",
                    )
                    _ = indices.pop()
                elif i == num_first:
                    if size > num_first + num_last:
                        print("..., ", end="")
                elif i >= size - num_last:
                    indices.append(i)
                    print(self[indices], end=", " if i != size - 1 else "")
                    _ = indices.pop()
                else:
                    # Handles middle region not explicitly caught
                    continue

            print("]", end="\n")

        else:
            print(indent + "[")
            for i in range(size):
                if i < num_first:
                    indices.append(i)
                    self.print_tensor_recursive(indices, level + 1)
                    _ = indices.pop()
                    if i != num_first - 1 or size > num_first + num_last:
                        print(",")
                elif i == num_first:
                    if size > num_first + num_last:
                        print(indent + "  ...,")
                elif i >= size - num_last:
                    indices.append(i)
                    self.print_tensor_recursive(indices, level + 1)
                    _ = indices.pop()
                    if i != size - 1:
                        print(",")
                else:
                    # This path was previously missing, which caused silent looping!
                    continue

                print(indent + "]", end="\n")
                # print("\n")

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(
            self.__str__(),
            end="\n",
        )
        empty = IntList()
        self.print_tensor_recursive(
            empty, 1, num_first=num_first, num_last=num_last
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

    # Note - matmul has not been optimized at all - once everything is place - revisit this
    fn matmul(self, other: Self) -> Tensor[dtype]:
        if not self.rank() == 2:
            abort("TensorLike → matmul: Only supports 2D matmul for now")
        if not other.rank() == 2:
            abort("TensorLike → matmul: Other must be 2D")
        if not self.shape()[1] == other.shape()[0]:
            abort("TensorLike → matmul: Incompatible shapes")

        m, k = self.shape()[0], self.shape()[1]
        n = other.shape()[1]

        requires_grad = self.requires_grad() or other.requires_grad()
        var out = Tensor[dtype](m, n, requires_grad=requires_grad)

        for i in range(m):
            for j in range(n):
                var summ = Scalar[dtype](0)
                for p in range(k):
                    summ += self[IntList(i, p)] * other[IntList(p, j)]
                out[IntList(i, j)] = summ

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
