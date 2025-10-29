### Mojo Tensor Gradbox
### Implement tensor library in mojo from first principles
from shapes import Shape
from common_utils_imports import *
from operators_imports import *
from buffers import Buffer
from validators import Validator
from common_utils import IntArrayHelper

# from tensors import Tensor
from tenmo import Tensor
from layout.int_tuple import IntArray
from ndbuffer import NDBuffer


struct Gradbox[dtype: DType](
    Copyable
    & Movable
    & Sized
    & Stringable
    & Representable
    & Writable
    & EqualityComparable
):
    var buffer: NDBuffer[dtype]

    fn __init__(out self, shape: Shape):
        buffer = NDBuffer[dtype](shape)
        self.buffer = buffer.share()

    fn __init__(out self, var buffer: NDBuffer[dtype]):
        self.buffer = buffer.share()

        _ = """fn __init__(out self, var shape: Shape, var buffer: Buffer[dtype]):
        if shape.num_elements() != buffer.size:
            panic(
                "Gradbox __init__: shape numels do not match buffer size: ",
                shape.num_elements().__str__(),
                buffer.size.__str__(),
            )
        self.buffer = NDBuffer[dtype](buffer^, shape^)"""

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()

    @always_inline
    fn as_tensor(self, requires_grad: Bool = False) -> Tensor[dtype]:
        _ = """return Tensor[dtype](
            self.shape.copy(), self.buffer.copy(), requires_grad=requires_grad
        )"""
        return Tensor[dtype].scalar(42)

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        return self.buffer[indices]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )

        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        self.buffer[indices] = value

    fn item(self) -> Scalar[dtype]:
        if self.shape() != Shape(1) and self.shape() != Shape():
            panic(
                "Gradbox → item(self): only valid for scalar or singleton"
                " gradbox, got shape: "
                + self.shape().__str__()
            )
        return self.buffer.item()

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.buffer.is_scalar()

    @always_inline
    fn numels(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn __len__(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn rank(self) -> Int:
        return self.buffer.rank()

    @always_inline
    fn shape(self) -> Shape:
        return self.buffer.shape

    fn __eq__(self, other: Gradbox[dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __eq__(other): shape mismatch",
                self.shape().__str__(),
                "≠",
                other.shape().__str__(),
            )
        return (
            self.buffer.compare[Equal](other.buffer).buffer.value().all_true()
        )

    fn __ne__(self, other: Gradbox[dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __ne__(other): shape mismatch",
                self.shape().__str__(),
                "≠",
                other.shape().__str__(),
            )
        return (
            self.buffer.compare[NotEqual](other.buffer)
            .buffer.value()
            .all_true()
        )

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
        s += self.shape().__str__()
        s += ", Type: " + dtype.__str__()
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    fn seed_grad(self, value: Scalar[dtype]):
        self.buffer.fill(value)

    @always_inline
    fn seed_grad(self, with_tensor: Tensor[dtype]):
        self.buffer.fill(with_tensor.buffer)

    @always_inline
    fn zero_grad(self):
        self.buffer.zero()

    fn __sub__(self, other: Self) -> Gradbox[dtype]:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __sub__(other) → dimension mismatch: "
                + self.shape().__str__()
                + "≠"
                + other.shape().__str__(),
            )
        buffer = self.buffer.__sub__(other.buffer)
        return Gradbox[dtype](buffer^)

    @always_inline
    fn __iadd__(self, incoming: Gradbox[dtype]):
        if self.shape() != incoming.shape():
            panic(
                "Gradbox → __iadd__: Shapes not equal -> ",
                self.shape().__str__(),
                " ≠ ",
                incoming.shape().__str__(),
            )

        self.buffer.__iadd__(incoming.buffer)

    @always_inline
    fn __imul__(self, incoming: Gradbox[dtype]):
        if self.shape() != incoming.shape():
            panic(
                "Gradbox → __imul__: Shapes not equal -> ",
                self.shape().__str__(),
                " ≠ ",
                incoming.shape().__str__(),
            )

        self.buffer.__imul__(incoming.buffer)

    @always_inline
    fn __isub__(self, incoming: Gradbox[dtype]):
        if self.shape() != incoming.shape():
            panic(
                "Gradbox → __isub__: Shapes not equal -> ",
                self.shape().__str__(),
                " ≠ ",
                incoming.shape().__str__(),
            )

        self.buffer.__isub__(incoming.buffer)

    @always_inline
    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Gradbox → all_close is for floating point data types only",
        ]()
        if self.shape() != other.shape():
            panic(
                "Gradbox → all_close expects same shaped gradboxes: "
                + self.shape().__str__()
                + ", "
                + other.shape().__str__()
            )

        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

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
            self.shape(), new_shape.intlist()
        )
        buffer = self.buffer.contiguous_buffer()
        nd_buffer = NDBuffer[dtype](buffer^, shape^)

        return Gradbox[dtype](nd_buffer^)

        _ = """fn __eq__(self, tensor: Tensor[dtype]) -> Bool:
        if self.shape != tensor.shape:
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                self.shape.__str__(),
                ",",
                tensor.shape.__str__(),
            )
        tensor_as_gradbox = Self.from_tensor(tensor)
        return self == tensor_as_gradbox

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

        return self.buffer.all_close[simd_width, rtol, atol](other.data())"""

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
        _ = self.buffer^


fn main() raises:
    run = 1
    for _ in range(run):
        test_gradbox_is_shared()
        test_seed_gradbox()
        test_gradbox_in_place_add()
        test_gradbox_reshape()


from testing import assert_true


fn test_gradbox_reshape() raises:
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    reshaped = gradbox.reshape(Shape(3, 2))
    assert_true(reshaped[[2, 1]] == 6 and reshaped[[1, 1]] == 4)
    reshaped.zero_grad()
    assert_true(reshaped[[2, 1]] == 0 and reshaped[[1, 1]] == 0)
    assert_true(gradbox[[1, 2]] == 6 and gradbox[[0, 1]] == 2)


fn test_gradbox_in_place_add() raises:
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)

    buffer2 = Buffer[dtype]([11, 12, 13, 14, 15, 16])
    ndb2 = NDBuffer[dtype](buffer2^, Shape(2, 3))
    gradbox2 = Gradbox[dtype](ndb2^)

    gradbox += gradbox2
    assert_true(
        gradbox.buffer.data() == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(gradbox.buffer.buffer == None)


fn test_gradbox_is_shared() raises:
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared(), "Gradbox buffer is shared - assertion failed"
    )


fn test_seed_gradbox() raises:
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([1, 2, 3, 4, 5, 6])
    )
    gradbox.seed_grad(42)
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([42, 42, 42, 42, 42, 42])
    )
