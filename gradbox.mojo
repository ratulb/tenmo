### Mojo Tensor Gradbox
### Implement tensor library in mojo from first principles
from sys import simdwidthof
from shapes import Shape
from intlist import IntList
from strides import Strides
from common_utils_imports import *
from common_utils import id as identity
from operators_imports import *
from buffers import Buffer
from validators import Validator
#from tensors import Tensor
from tenmo import Tensor


struct Gradbox[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Representable & Writable
):
    var shape: Shape
    var buffer: Buffer[dtype]

    fn __init__(out self, shape: Shape):
        self.shape = shape.copy()
        # Take care of gradbox with Shape()
        self.buffer = Buffer[dtype](1) if shape.rank() == 0 else Buffer[dtype](
            shape.num_elements()
        )

    fn __init__(out self, shape: Shape, var buffer: Buffer[dtype]):
        if shape.num_elements() != buffer.size:
            panic(
                "Gradbox __init__: shape numels do not match buffer size: ",
                shape.num_elements().__str__(),
                buffer.size.__str__(),
            )
        self.shape = shape.copy()
        self.buffer = buffer

    fn __moveinit__(out self, deinit other: Self):
        self.shape = other.shape^
        self.buffer = other.buffer^

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape.copy()
        self.buffer = other.buffer.copy()

    @always_inline
    fn id(self) -> Int:
        return identity(self)

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

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:  # Gradbox with Shape ()
            panic("Gradbox → __getitem__: Scalar gradbox expects empty indices")
        index = self.shape.flatten_index(
            indices, Strides.default(self.shape), 0
        )
        return self.buffer[index]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic("Gradbox → __setitem__: Scalar gradbox expects empty indices")
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

    @always_inline
    fn zero_grad(mut self):
        self.buffer.zero()

    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape()

    fn __eq__(self, scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        return self.compare_scalar[Equal](scalar)

    fn __ne__(self, scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        return self.compare_scalar[NotEqual](scalar)

    fn __lt__(self, scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        return self.compare_scalar[LessThan](scalar)

    fn __le__(self, scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        return self.compare_scalar[LessThanEqual](scalar)

    fn __gt__(self, scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        return self.compare_scalar[GreaterThan](scalar)

    fn __ge__(self, scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        return self.compare_scalar[GreaterThanEqual](scalar)

    fn __eq__(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __eq__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[Equal](other)
        return out

    fn __ne__(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __ne__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[NotEqual](other)
        return out

    fn __lt__(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __lt__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[LessThan](other)
        return out

    fn __le__(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __le__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[LessThanEqual](other)
        return out

    fn __gt__(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __gt__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[GreaterThan](other)
        return out

    fn __ge__(self, other: Gradbox[dtype]) -> Gradbox[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Gradbox __ge__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[GreaterThanEqual](other)
        return out

    fn float(self) -> Gradbox[DType.float32]:
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Gradbox[DType.float64]:
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Gradbox[NewType]:
        new_buffer = self.buffer.to_dtype[NewType]()

        out = Gradbox[NewType](
            shape=self.shape,
            buffer=new_buffer^,
        )

        return out

    fn all_true(self: Gradbox[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == Scalar[DType.bool](True)

        return self.for_all(all_truthy)

    fn for_all[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.for_all[simd_width](pred).all_true()

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
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

    fn address(
        ref self,
    ) -> UnsafePointer[
        Self,
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
        return UnsafePointer(to=self).origin_cast[
            mut = Origin(__origin_of(self)).mut, origin = __origin_of(self)
        ]()

    fn seed_grad(mut self, with_tensor: Tensor[dtype]):
        if self.shape != with_tensor.shape:
            panic(
                "Gradbox → seed_grad: Shapes not equal -> ",
                self.shape.__str__(),
                " ≠ ",
                with_tensor.shape.__str__(),
            )
        self.buffer.zero()
        self.buffer += with_tensor.buffer

    fn seed_grad(mut self, value: Scalar[dtype]):
        with_tensor = Tensor[dtype].full(self.shape, value)
        self.seed_grad(with_tensor)

    @always_inline
    fn load[
        simdwidth: Int = 1
    ](self, row: Int, col: Int) -> SIMD[dtype, simdwidth]:
        constrained[
            simdwidth.is_power_of_two(),
            "Gradbox → load: SIMD width (simdwidth) must be a power of 2",
        ]()

        if self.rank() != 2:
            panic("Gradbox → load: supported only for 2D gradbox")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + simdwidth > self.shape[1]
        ):
            panic("Gradbox → load: Out-of-bounds access")

        strides = Strides.default(self.shape)
        addr = row * strides[0] + col * strides[1]

        return self.buffer.load[simdwidth](addr)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, row: Int, col: Int, value: SIMD[dtype, simdwidth]):
        constrained[
            simdwidth.is_power_of_two(),
            "Gradbox → store: SIMD width (simdwidth) must be a power of 2",
        ]()

        if self.rank() != 2:
            panic("Gradbox → store is supported only for 2D Gradbox")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + simdwidth > self.shape[1]
        ):
            panic("Gradbox → store: out-of-bounds access")

        strides = Strides.default(self.shape)
        addr = row * strides[0] + col * strides[1]
        self.buffer.store[simdwidth](addr, value)

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
        shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            self.shape, new_shape.intlist()
        )
        buffer = self.buffer.copy()
        return Gradbox[dtype](shape, buffer^)

    fn __rtruediv__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        constrained[
            dtype.is_numeric(),
            "Gradbox → __rtruediv__ is for numeric data types only",
        ]()

        buffer = scalar / self.buffer

        return Gradbox[dtype](self.shape, buffer^)

    fn __truediv__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        constrained[
            dtype.is_numeric(),
            "Gradbox → __truediv__ is for numeric data types only",
        ]()

        if scalar == Scalar[dtype](0):
            panic("Gradbox → __truediv__ : canot divide by " + scalar.__str__())

        buffer = self.buffer / scalar

        return Gradbox[dtype](self.shape, buffer^)

    # Element wise division of two tensors
    fn __truediv__(self, other: Self) -> Gradbox[dtype]:
        constrained[
            dtype.is_numeric(),
            "Gradbox → __rtruediv__ is for numeric data types only",
        ]()
        if self.shape != other.shape:
            panic(
                "Gradbox →__truediv__(self * other): dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
            )

        buffer = self.buffer / other.buffer
        return Gradbox[dtype](self.shape, buffer^)

    fn __rmul__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, factor: Scalar[dtype]) -> Gradbox[dtype]:
        buffer = self.buffer * factor
        return Gradbox[dtype](self.shape, buffer^)

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Gradbox[dtype]:
        if not self.shape != other.shape:
            panic(
                "Gradbox → __mul__(self * other) → dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
            )
        buffer = self.buffer * other.buffer
        return Gradbox[dtype](self.shape, buffer^)

    fn __iadd__(mut self, other: Self):
        if self.shape != other.shape:
            panic(
                "Gradbox → __iadd__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        self.buffer += other.buffer

    fn __isub__(mut self, other: Self):
        if self.shape != other.shape:
            panic(
                "Gradbox → __isub__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        self.buffer -= other.buffer

    fn __imul__(mut self, other: Self):
        if self.shape != other.shape:
            panic(
                "Gradbox → __imul__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        self.buffer *= other.buffer

    fn __radd__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        buffer = self.buffer + scalar
        return Gradbox[dtype](self.shape, buffer^)

    fn __add__(self, other: Self) -> Gradbox[dtype]:
        if not self.shape != other.shape:
            panic(
                "Gradbox → __add__(self * other) → dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
            )
        buffer = self.buffer + other.buffer
        return Gradbox[dtype](self.shape, buffer^)

    fn __rsub__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        buffer = scalar - self.buffer
        return Gradbox[dtype](self.shape, buffer^)

    fn __sub__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        buffer = self.buffer - scalar
        return Gradbox[dtype](self.shape, buffer^)

    fn __sub__(self, other: Self) -> Gradbox[dtype]:
        if not self.shape != other.shape:
            panic(
                "Gradbox → __sub__(self * other) → dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
            )
        buffer = self.buffer - other.buffer
        return Gradbox[dtype](self.shape, buffer^)

    fn compare[
        op: Int, simd_width: Int = simdwidthof[dtype]()
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

        out = Gradbox[DType.bool](out_shape, buffer^)
        return out

    fn compare_scalar[
        op: Int, simd_width: Int = simdwidthof[dtype]()
    ](this: Gradbox[dtype], scalar: Scalar[dtype]) -> Gradbox[DType.bool]:
        out_shape = this.shape[::]

        if op == Equal:
            buffer = this.buffer.eq[simd_width](scalar)

        elif op == NotEqual:
            buffer = this.buffer.ne[simd_width](scalar)

        elif op == LessThan:
            buffer = this.buffer.lt[simd_width](scalar)

        elif op == LessThanEqual:
            buffer = this.buffer.le[simd_width](scalar)

        elif op == GreaterThan:
            buffer = this.buffer.gt[simd_width](scalar)

        else:  # GreaterThanEqual
            buffer = this.buffer.ge[simd_width](scalar)

        return Gradbox[DType.bool](out_shape, buffer^)

    fn __iadd__(mut self, scalar: Scalar[dtype]):
        self.buffer += scalar

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
    gb.print()
