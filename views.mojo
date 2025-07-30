from tensors import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from os import abort
from shared import TensorLike
from ancestry import Ancestors
from memory import memcpy, memset_zero
from walkback import BackwardFn, MatmulBackward, ViewBackward
from operators import __tensor_op_tensor__


struct TensorView[dtype: DType = DType.float32](
    Sized & Copyable & Movable & Stringable & Representable & Writable
):
    alias Blank: TensorView[dtype] = Self(
        UnsafePointer[Tensor[dtype]](), Shape.Void, Strides(IntList.Empty), 0
    )
    alias Ancestor_of = TensorLike.from_view
    var base_tensor: UnsafePointer[Tensor[dtype]]
    var shape: Shape
    var strides: Strides
    var offset: Int
    var requires_grad: Bool
    var grad: UnsafePointer[Tensor[dtype]]
    var ancestors: Ancestors[dtype]
    var backwardFn: Optional[BackwardFn[dtype]]

    fn __init__(
        out self,
        base_tensor: UnsafePointer[Tensor[dtype]],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Bool = False,
    ):
        self.base_tensor = base_tensor
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.requires_grad = requires_grad
        self.grad = UnsafePointer[Tensor[dtype]]()
        self.ancestors = Ancestors[dtype].untracked()
        self.backwardFn = None

    fn __moveinit__(out self, owned other: Self):
        self.base_tensor = other.base_tensor
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.ancestors = other.ancestors
        self.backwardFn = other.backwardFn

    fn __copyinit__(out self, other: Self):
        self.base_tensor = other.base_tensor
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.ancestors = other.ancestors
        self.backwardFn = other.backwardFn

    fn zero_grad(self):
        if self.requires_grad and self.has_grad():
            memset_zero(self.grad[].data, self.grad[].numels())

    fn init_grad(mut self):
        if self.requires_grad and not self.has_grad():
            gradients = Tensor[dtype](self.shape)
            self.grad = UnsafePointer[Tensor[dtype]].alloc(1)
            self.grad.init_pointee_move(gradients^)
            self.zero_grad()

    fn update_grad[opcode: Int](self, gradients: Tensor[dtype]):
        self.grad[] = __tensor_op_tensor__[dtype, opcode](
            self.grad[], gradients
        )

    fn view(self, shape: List[Int]) -> TensorView[dtype]:
        return self.view(Shape(shape))

    fn view(self, shape: Shape) -> TensorView[dtype]:
        if not self.shape.num_elements() == shape.num_elements():
            abort(
                "TensorView → view: shape"
                + shape.__str__()
                + " is invalid: total number of elements("
                + String(self.shape.num_elements())
                + ") must match"
            )

        strides = Strides.default(shape)
        offset = self.offset

        out = TensorView[dtype](
            base_tensor=self.base_tensor,
            shape=shape,
            strides=strides,
            offset=offset,
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            backward_fn = ViewBackward[dtype](
                shape, strides, offset
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))
        return out

    # Fully custom shape/strides/offset
    fn view(
        self,
        shape: List[Int],
        strides: List[Int],
        offset: Optional[Int] = None,
    ) -> TensorView[dtype]:
        new_offset = offset.value() if offset else self.offset
        return self.view(Shape(shape), Strides(strides), new_offset)

    fn view(
        self, shape: Shape, strides: Strides, offset: Int
    ) -> TensorView[dtype]:
        if not shape.rank() == strides.rank():
            abort(
                "TensorView → view: shape"
                + shape.__str__()
                + " and strides"
                + strides.__str__()
                + " must have the same rank"
            )

        # Basic bounds checking: ensure the view doesn't go out of bounds
        numels = shape.num_elements()
        max_index = 0
        for i in range(shape.rank()):
            max_index += (shape[i] - 1) * strides[i]
        total_offset = offset + max_index
        num_elems = self.shape.num_elements()
        if not total_offset < num_elems:
            abort(
                "TensorView → view: Call exceeds base view's total"
                " number of elements("
                + String(num_elems)
                + ")"
            )
        composite_offset = offset + self.offset
            # (1) Compute bounds of parent view (`self`)
        parent_min = self.offset
        parent_max = self.offset
        for i in range(self.shape.rank()):
            parent_max += (self.shape[i] - 1) * self.strides[i]
        if parent_min > parent_max:
            parent_min, parent_max = parent_max, parent_min

        # (2) Compute bounds of new view (proposed)
        # New view's offset is `self.offset + offset` (base-relative)

        # Compute min/max flat indices in the base tensor
        new_min = composite_offset
        new_max = composite_offset
        for i in range(shape.rank()):
            stride = strides[i]
            dim_size = shape[i]
            if stride >= 0:
                new_max += (dim_size - 1) * stride  # increasing indices
            else:
                new_min += (dim_size - 1) * stride  # decreasing indices (negative stride)

        # (3) Ensure new view fits within parent view's bounds
        if new_min < parent_min or new_max > parent_max:
            abort("TensorView → view: requested view exceeds parent view bounds!")
        return TensorView[dtype](
            base_tensor=self.base_tensor,
            shape=shape,
            strides=strides,
            offset=composite_offset,
            requires_grad=self.requires_grad,
        )

    fn is_contiguous(self) -> Bool:
        # return self.offset == 0 and self.strides.is_contiguous(self.shape)
        return self.strides.is_contiguous(self.shape)

    fn into_tensorlike(self) -> TensorLike[dtype]:
        return TensorLike[dtype](UnsafePointer(to=self))

    # Index calculation: flat offset into underlying tensor's data[]
    fn index_offset(self, indices: IntList) -> Int:
        if not indices.len() == self.shape.rank():
            abort("TensorView → index_offset → rank mismatch")
    # Bounds check before computing flat offset
        for i in range(indices.len()):
            if indices[i] < 0 or indices[i] >= self.shape[i]:
                abort("TensorView → index_offset → index out of bounds for view")
        var flat_idx = self.offset
        for i in range(indices.len()):
            flat_idx += indices[i] * self.strides[i]
        return flat_idx

    # Element access
    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        return self.base_tensor[].data.load[volatile=True](
            self.index_offset(indices)
        )

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        self.base_tensor[].data.store[volatile=True](
            self.index_offset(indices), value
        )

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        return self.base_tensor[].data.load[volatile=True](
            self.index_offset(IntList(indices))
        )

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        self.base_tensor[].data.store[volatile=True](
            self.index_offset(IntList(indices)), value
        )

    fn has_grad(self) -> Bool:
        return self.grad.__as_bool__()

    # Check if it has a backward fn before calling this API
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self.backwardFn.value()

    fn has_backward_fn(self) -> Bool:
        return self.backwardFn is not None

    fn backward(self, start_grad: Scalar[dtype] = 1.0):
        tensor_like = TensorLike.from_view(self)
        tensor_like.backward(start_grad)

    fn backward(self, seed_tensor: Tensor[dtype]):
        tensor_like = TensorLike.from_view(self)
        tensor_like.backward(seed_tensor)

    fn add_ancestry(mut self, *parents: TensorLike[dtype]):
        for parent in parents:
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_copy(parent)
            self.ancestors.append(ptr)

    fn ancestry(self) -> Ancestors[dtype]:
        return self.ancestors

    fn _requires_grad(self) -> Bool:
        return self.requires_grad

    fn is_view(self) -> Bool:
        return True

    fn is_tensor(self) -> Bool:
        return False

    fn into_tensor(self) -> Tensor[dtype]:
        shape = self.shape
        out = Tensor[dtype](shape, requires_grad=self.requires_grad)
        numels = shape.num_elements()

        if self.is_contiguous():
            # Fast path: single memcpy from base tensor
            memcpy[Scalar[dtype]](
                out.data, self.base_tensor[].data + self.offset, numels
            )

        else:
            # Slow path: general indexing using shape
            rank = shape.rank()
            indices = IntList.filled(rank, 0)

            for _ in range(numels):
                # Copy value at current index from view to out
                out[indices] = self[indices]

                # Increment multi-dimensional index (manual shape walker)
                var carry = True
                for dim in reversed(range(rank)):
                    if carry:
                        indices[dim] += 1
                        if indices[dim] >= self.shape[dim]:
                            indices[dim] = 0  # Carry over
                            carry = True
                        else:
                            carry = False
        if self.requires_grad:
            strides = self.strides
            offset = self.offset
            backward_fn = ViewBackward[dtype](
                shape, strides, offset
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn all_close(
        self,
        tensor: Tensor[dtype],
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ) -> Bool:
        if not self.shape.num_elements() + self.offset <= tensor.numels():
            abort(
                "TensorView → all_close(tensor) → TensorView exceeds bounds"
                " of base Tensor: "
                + String(self.shape.num_elements())
                + "(no of elemets in view)  + "
                + String(self.offset)
                + "(view offset) > "
                + String(tensor.numels())
                + "(no of elements in tensor)"
            )

        for idx in self.shape:
            v = self[idx]
            t = tensor[idx]
            diff = abs(v - t)
            tol = atol + rtol * abs(t)

            if diff > tol:
                return False

        return True

    fn seed_grad(mut self, value: Scalar[dtype]):
        if self.requires_grad:
            if not self.has_grad():
                gradients = Tensor[dtype].full(self.shape, value)
                self.grad = UnsafePointer[Tensor[dtype]].alloc(1)
                self.grad.init_pointee_move(gradients^)
            else:
                self.grad[].fill(value)

    fn seed_grad(mut self, with_tensor: Tensor[dtype]):
        if not self.shape == with_tensor.shape:
            abort(
                "TensorView → seed_grad: view shape"
                + self.shape.__str__()
                + " and seed tensor shape"
                + with_tensor.shape.__str__()
                + "does not match"
            )
        if self.requires_grad:
            if not self.has_grad():
                self.grad = UnsafePointer[Tensor[dtype]].alloc(1)
                self.grad.init_pointee_copy(with_tensor)
            else:
                memcpy(self.grad[].data, with_tensor.data, with_tensor.numels())

    fn __str__(self) -> String:
        dims = len(self.shape)
        s = String("[")
        if dims == 1:
            s += "1D View"
        elif dims == 2:
            s += "2D View"
        elif dims == 3:
            s += "3D View"
        elif dims == 4:
            s += "4D View"
        elif dims == 5:
            s += "5D View"
        else:
            s += "View"
        s += self.shape.__str__()
        s += ", strides: " + self.strides.__str__()
        s += ", offset: " + self.offset.__str__()
        s += ", Type: " + dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __len__(self) -> Int:
        return self.numels()

    fn len(self) -> Int:
        return self.numels()

    fn size(self) -> Int:
        return self.numels()

    fn numels(self) -> Int:
        return self.shape.num_elements()

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        tensor_like = self.into_tensorlike()
        tensor_like.print(num_first, num_last)

    # Always use this to print grad to avoid surprises of segmentation fault!
    fn gprint(self, num_first: Int = 10, num_last: Int = 10):
        if not self.requires_grad:
            print("TensorView is non-differentiable")
        elif self.requires_grad and self.grad.__as_bool__() == False:
            print("Requires grad but grad not initialized")
        else:
            self.grad[].print(num_first, num_last)

    # Note - matmul has not been optimized at all - once everything is place - revisit this
    fn matmul(self, other: Self) -> Tensor[dtype]:
        this = TensorLike.from_view(self)
        that = TensorLike.from_view(other)
        out = this.matmul(that)
        if out.requires_grad:
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(this)
            out.add_ancestry(that)

        return out

    fn matmul(self, other: Tensor[dtype]) -> Tensor[dtype]:
        this = TensorLike.from_view(self)
        that = TensorLike.from_tensor(other)
        out = this.matmul(that)
        if out.requires_grad:
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(this)
            out.add_ancestry(that)

        return out


fn main():
    # a = Tensor.arange(16, requires_grad = True)
    a = Tensor.d2(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            [
                4.0,
                5.0,
                6.0,
                7.0,
            ],
            [
                8.0,
                9.0,
                10.0,
                11.0,
            ],
            [
                12.0,
                13.0,
                14.0,
                15.0,
            ],
        ],
        requires_grad=True,
    )
    v1 = a.view([2, 2], [4, 1], offset=5)
    v1.print()
    v2 = v1.view([2, 2], [2, 1], 0)
    v2.print()
    print(v2[3, 3])
    _ = a
    x, y = 10, 100
    print(x, y)
    x, y = y, x

    print(x, y)

