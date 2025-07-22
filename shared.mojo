from tensors import Tensor
from shapes import Shape
from views import TensorView
from intlist import IntList
from backpropagation import BackwardFn


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

    fn __moveinit__(out self, owned other: Self):
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

    fn is_view(self) -> Bool:
        return self.kind == 1

    fn is_tensor(self) -> Bool:
        return self.kind == 0

    fn tensor_ptr(self) -> Self.TensorAddress:
        return self.tensor_address

    fn view_ptr(self) -> Self.ViewAddress:
        return self.view_address

    fn inner_address(self) -> UnsafePointer[Tensor[dtype]]:
        if self.kind == 0:
            return self.tensor_address
        else:
            return self.view_address[].base_tensor  # base pointer address

    fn inner_id(self) -> Int:
        if self.kind == 0:
            return Tensor.id(self.tensor_address)
        else:
            return TensorView.id(self.view_address)  # View id!

    fn tensor(self) -> Tensor[dtype]:
        return self.tensor_address[]

    fn view(self) -> TensorView[dtype]:
        return self.view_address[]

    fn has_grad(self) -> Bool:
        return (
            self.tensor_address[]
            .has_grad() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .has_grad()
        )

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        return self.tensor_address[][
            indices
        ] if self.is_tensor() else self.view_address[][indices]

    fn has_backward_fn(self) -> Bool:
        return (
            self.tensor_address[]
            .has_backward_fn() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .has_backward_fn()
        )

    fn backward_fn(self) -> BackwardFn[dtype]:
        return (
            self.tensor_address[]
            .backward_fn() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .backward_fn()
        )

    fn gradients(self) -> UnsafePointer[Tensor[dtype]]:
        return (
            self.tensor_address[]
            .gradients() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .gradients()
        )

    fn rank(self) -> Int:
        return self.shape().rank()

    fn shape(self) -> Shape:
        if self.kind == 0:
            return self.tensor_address[].shape
        else:
            return self.view_address[].shape

    fn seed_grad(self, value: Scalar[dtype]):
        if self.is_tensor():
            self.tensor().seed_grad(value)
        else:
            self.view().seed_grad(value)

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        if self.is_tensor():
            self.tensor().update_grad[opcode](incoming)
        else:
            self.view().base_tensor[].update_grad[opcode](incoming)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        if self.is_tensor():
            self.tensor().seed_grad(with_tensor)
        else:
            self.view().seed_grad(with_tensor)

    fn requires_grad(self) -> Bool:
        return (
            self.view()
            ._requires_grad() if self.is_view() else self.tensor()
            ._requires_grad()
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
