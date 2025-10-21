### Mojo Tensor Gradbox
### Implement tensor library in mojo from first principles
from sys import simd_width_of
from shapes import Shape
from strides import Strides
from common_utils_imports import *
from operators_imports import *
from buffers import Buffer
from validators import Validator

# from tensors import Tensor
from tenmo import Tensor
from layout.int_tuple import IntArray
from algorithm import vectorize


struct Gradbox[dtype: DType = DType.float32](
    Copyable
    & Movable
    & Sized
    & Stringable
    & Representable
    & Writable
    & EqualityComparable
):
    var shape: Shape
    var buffer: Buffer[dtype]

    fn __init__(out self):
        self.shape = Shape.Void()
        self.buffer = Buffer[dtype]()

    fn __init__(out self, shape: Shape):
        self.shape = shape.copy()
        # Take care of gradbox with Shape()
        self.buffer = Buffer[dtype].zeros(1) if shape.rank() == 0 else Buffer[
            dtype
        ].zeros(shape.num_elements())

    fn __init__(out self, var shape: Shape, var buffer: Buffer[dtype]):
        if shape.num_elements() != buffer.size:
            panic(
                "Gradbox __init__: shape numels do not match buffer size: ",
                shape.num_elements().__str__(),
                buffer.size.__str__(),
            )
        self.shape = shape^
        self.buffer = buffer^

    fn __moveinit__(out self, deinit other: Self):
        self.shape = other.shape^
        self.buffer = other.buffer^

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape.copy()
        self.buffer = other.buffer.copy()

    @always_inline
    fn as_tensor(self, requires_grad: Bool = False) -> Tensor[dtype]:
        return Tensor[dtype](
            self.shape.copy(), self.buffer.copy(), requires_grad=requires_grad
        )

    @staticmethod
    fn from_tensor(tensor: Tensor[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](tensor.shape.copy(), tensor.data().copy())

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:  # Gradbox with Shape ()
            panic(
                "Gradbox → __getitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )
        index = self.shape.flatten_index(
            indices, Strides.default(self.shape), 0
        )
        return self.buffer[index]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        if self.rank() == 0 and indices.size() != 0:  # Gradbox with Shape ()
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        index = self.shape.flatten_index(
            indices, Strides.default(self.shape), 0
        )
        return self.buffer[index]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )
        index = self.shape.flatten_index(
            indices, Strides.default(self.shape), 0
        )
        self.buffer[index] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        index = self.shape.flatten_index(
            indices, Strides.default(self.shape), 0
        )
        self.buffer[index] = value

    fn item(self) -> Scalar[dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "Gradbox.item(): Only valid for scalar or singleton gradbox,"
                " got shape: "
                + self.shape.__str__()
            )
        return self[[0]] if self.shape == Shape(1) else self[[]]

    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape()

    fn __eq__(self, other: Gradbox[dtype]) -> Bool:
        return self.eq(other).all_true()

    fn __ne__(self, other: Gradbox[dtype]) -> Bool:
        return self.ne(other).all_true()

    fn __eq__(self, tensor: Tensor[dtype]) -> Bool:
        if self.shape != tensor.shape:
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                self.shape.__str__(),
                ",",
                tensor.shape.__str__(),
            )
        tensor_as_gradbox = Self.from_tensor(tensor)
        return self == tensor_as_gradbox

    fn diff(self, tensor: Tensor[dtype]) -> Tensor[dtype]:
        if self.shape != tensor.shape:
            panic(
                "Gradbox diff(tensor) → dimension mismatch:",
                self.shape.__str__(),
                ",",
                tensor.shape.__str__(),
            )
        tensor_as_gradbox = Self.from_tensor(tensor)
        deviation = self - tensor_as_gradbox
        return deviation.as_tensor()

    fn eq(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __eq__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[Equal](other)
        return out^

    fn ne(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __ne__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[NotEqual](other)
        return out^

    fn float(self) -> Gradbox[DType.float32]:
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Gradbox[DType.float64]:
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Gradbox[NewType]:
        new_buffer = self.buffer.to_dtype[NewType]()
        out = Gradbox[NewType](
            shape=self.shape.copy(),
            buffer=new_buffer^,
        )
        return out^

    fn all_true(self: Gradbox[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == Scalar[DType.bool](True)

        return self.for_all(all_truthy)

    fn for_all[
        simd_width: Int = simd_width_of[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.for_all[simd_width](pred).all_true()

    fn all_close[
        simd_width: Int = simd_width_of[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Tensor[dtype]) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Gradbox → all_close is for floating point data types only",
        ]()
        if self.shape != other.shape:
            panic(
                "Gradbox → all_close expects same shaped Tensor: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        return self.buffer.all_close[simd_width, rtol, atol](other.data())

    fn all_close[
        simd_width: Int = simd_width_of[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Gradbox → all_close is for floating point data types only",
        ]()
        if self.shape != other.shape:
            panic(
                "Gradbox → all_close expects same shaped gradboxes: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        return self.buffer.all_close[simd_width, rtol, atol](other.buffer)

    @always_inline
    fn seed_grad[
        simd_width: Int = simd_width_of[dtype]()
    ](self, with_tensor: Tensor[dtype]):
        """Seed gradient from a tensor."""

        # Shape validation
        if self.shape != with_tensor.shape:
            panic(
                "Gradbox → seed_grad: shapes not equal -> ",
                self.shape.__str__(),
                " ≠ ",
                with_tensor.shape.__str__(),
            )

        @parameter
        fn copy_elems[width: Int](idx: Int):
            var values = with_tensor.buffer.load[width](idx)
            self.buffer.store[width](idx, values)

        vectorize[copy_elems, simd_width](self.buffer.size)

    @always_inline
    fn seed_grad(self, value: Scalar[dtype]):
        self.buffer.fill(value)

    @always_inline
    fn zero_grad(self):
        self.buffer.zero()

    fn reshape(self) -> Gradbox[dtype]:
        if self.numels() != 1:
            panic(
                "Gradbox → reshape: only gradbox with single element can be"
                " reshaped to scalar gradbox"
            )
        return self.reshape(Shape(), validated=True)

    fn reshape(
        self,
        new_shape: Shape,
        validated: Bool = False,
    ) -> Gradbox[dtype]:
        shape = new_shape.copy() if validated else Validator.validate_and_construct_new_shape(
            self.shape.copy(), new_shape.intlist()
        )
        buffer = self.buffer.copy()
        return Gradbox[dtype](shape^, buffer^)

    @always_inline
    fn __iadd__[
        simd_width: Int = simd_width_of[dtype]()
    ](self, incoming: Gradbox[dtype]):
        """Update existing gradients with adding incoming grad values."""
        if self.shape != incoming.shape:
            panic(
                "Gradbox → __iadd__: Shapes not equal -> ",
                self.shape.__str__(),
                " ≠ ",
                incoming.shape.__str__(),
            )

        @parameter
        fn update_gradients[width: Int](idx: Int):
            var incoming_grad = incoming.buffer.load[width](idx)
            var existing_grad = self.buffer.load[width](idx)
            self.buffer.store[width](idx, incoming_grad + existing_grad)

        vectorize[update_gradients, simd_width](self.buffer.size)

    @always_inline
    fn __isub__[
        simd_width: Int = simd_width_of[dtype]()
    ](self, incoming: Gradbox[dtype]):
        """Update existing gradients with subtracting incoming grad values."""
        if self.shape != incoming.shape:
            panic(
                "Gradbox → __isub__: Shapes not equal -> ",
                self.shape.__str__(),
                " ≠ ",
                incoming.shape.__str__(),
            )

        @parameter
        fn update_gradients[width: Int](idx: Int):
            var incoming_grad = incoming.buffer.load[width](idx)
            var existing_grad = self.buffer.load[width](idx)
            self.buffer.store[width](idx, incoming_grad - existing_grad)

        vectorize[update_gradients, simd_width](self.buffer.size)

    @always_inline
    fn __imul__[
        simd_width: Int = simd_width_of[dtype]()
    ](self, incoming: Gradbox[dtype]):
        """Update existing gradients with multiplying incoming grad values."""
        if self.shape != incoming.shape:
            panic(
                "Gradbox → __imul__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + incoming.shape.__str__()
            )

        @parameter
        fn update_gradients[width: Int](idx: Int):
            var incoming_grad = incoming.buffer.load[width](idx)
            var existing_grad = self.buffer.load[width](idx)
            self.buffer.store[width](idx, incoming_grad * existing_grad)

        vectorize[update_gradients, simd_width](self.buffer.size)

    fn __sub__(self, other: Self) -> Gradbox[dtype]:
        if self.shape != other.shape:
            panic(
                "Gradbox → __sub__(other) → dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
            )
        buffer = self.buffer - other.buffer
        return Gradbox[dtype](self.shape.copy(), buffer^)

    fn __rmul__(self, factor: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__mul__(factor)

    fn __mul__(self, factor: Scalar[dtype]) -> Gradbox[dtype]:
        buffer = self.buffer * factor
        return Gradbox[dtype](self.shape.copy(), buffer^)

    @always_inline
    fn __len__(self) -> Int:
        return self.shape[0] if self.shape != Shape() else 0

    @always_inline
    fn len(self) -> Int:
        return self.shape[0] if self.shape != Shape() else 0

    @always_inline
    fn size(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn numels(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn rank(self) -> Int:
        return self.shape.rank()

    fn __str__(self) -> String:
        rank = self.rank()
        s = String("[")
        if rank == 1:
            s += "1D Gradbox"
        elif rank == 2:
            s += "2D Gradbox"
        elif rank == 3:
            s += "3D Gradbox"
        elif rank == 4:
            s += "4D Gradbox"
        elif rank == 5:
            s += "5D Gradbox"
        else:
            s += "Gradbox"
        s += self.shape.__str__()
        s += ", Type: " + dtype.__str__()
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn compare[
        op: Int, simd_width: Int = simd_width_of[dtype]()
    ](this: Gradbox[dtype], that: Gradbox[dtype]) -> Gradbox[DType.bool]:
        out_shape = this.shape[::]
        if op == Equal:
            buffer = this.buffer.eq[simd_width](that.buffer)

        elif op == NotEqual:
            buffer = this.buffer.ne[simd_width](that.buffer)

        elif op == LessThan:
            buffer = this.buffer.lt[simd_width](that.buffer)

        elif op == LessThanEqual:
            buffer = this.buffer.le[simd_width](that.buffer)

        elif op == GreaterThan:
            buffer = this.buffer.gt[simd_width](that.buffer)

        else:  # op == GreaterThanEqual
            buffer = this.buffer.ge[simd_width](that.buffer)

        out = Gradbox[DType.bool](out_shape^, buffer^)
        return out^

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(
            "\n",
            self.__str__(),
            end="\n",
        )
        empty = List[Int]()
        print_gradbox_recursive[dtype](
            UnsafePointer(to=self),
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn __del__(deinit self):
        _ = self.shape^
        _ = self.buffer^
        print("Gradbox freed")


fn main() raises:
    gb = Gradbox(Shape(5, 3))
    seed_tensor = Tensor.full(Shape(5, 3), 1947)
    gb.seed_grad(seed_tensor)  # If we comment this line out everything works.
    gb.print()

    a = gb.as_tensor()
    a.print()
