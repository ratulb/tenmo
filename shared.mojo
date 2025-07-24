from tensors import Tensor
from shapes import Shape
from views import TensorView
from intlist import IntList
from backpropagation import BackwardFn
from operators import AddTensor, SubtractTensor
from os import abort
from ancestry import Ancestors


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

    fn inner_address(self) -> UnsafePointer[Tensor[dtype]]:
        return (
            self.tensor_address if self.kind
            == 0 else self.view_address[].base_tensor
        )  # base pointer address

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
            self.view_address[].seed_grad(value)

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        if self.is_tensor():
            self.tensor().update_grad[opcode](incoming)
        else:
            self.view().base_tensor[].update_grad[opcode](incoming)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        if self.is_tensor():
            self.tensor().seed_grad(with_tensor)
        else:
            self.view_address[].seed_grad(with_tensor)

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

    # Note - matmul has not been optimized at all - once everything is place - revisit this
    fn matmul(self, other: Self) -> Tensor[dtype]:
        if not self.rank() == 2:
            abort("TesorLike  → matmul: Only supports 2D matmul for now")
        if not other.rank() == 2:
            abort("TesorLike  → matmul: Other must be 2D")
        if not self.shape()[1] == other.shape()[0]:
            abort("TesorLike  → matmul: Incompatible shapes")

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

    fn backward(self, start_grad: Scalar[dtype] = 1.0):
        if not self.requires_grad():
            return
        seed_tensor = Tensor[dtype].full(self.shape(), start_grad)
        self.backward(seed_tensor)

    fn backward(self, seed_tensor: Tensor[dtype]):
        if not self.requires_grad():
            return
        self.seed_grad(seed_tensor)
        visited = IntList.Empty
        topo_order = List[Self]()  # Stores nodes in topological order

        # --- (1) Perform topological sort (DFS-based) ---
        stack = [(self, False)]  # (node, processed)
        while stack:
            node, processed = stack.pop()
            if processed:
                topo_order.append(node)
                continue
            if node.inner_id() in visited:
                continue
            visited.append(node.inner_id())
            stack.append((node, True))  # Mark for post-processing
            # Push Ancestors (dependents) onto the stack
            for ancestor in node.ancestry():
                stack.append((ancestor[], False))

        # --- (2) Process in reverse topological order ---
        visited = IntList.Empty  # Reset for gradient accumulation
        for node in reversed(topo_order):
            if node.has_backward_fn():
                for recipient, grad_share, opcode in node.backward_fn()(
                    node.inner_address()
                ):
                    if opcode == AddTensor:
                        recipient.update_grad[AddTensor](grad_share)
                    elif opcode == SubtractTensor:
                        recipient.update_grad[SubtractTensor](grad_share)
