### Mojo Tensor Gradbox
### Implement tensor library in mojo from first principles
from shapes import Shape
from common_utils_imports import *
from operators import *
from validators import Validator
from tenmo import Tensor
from layout.int_tuple import IntArray
from ndbuffer import NDBuffer
from broadcasthelper import ShapeBroadcaster
from intlist import IntList


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

    fn __init__(out self, shape: Shape, share: Bool = True):
        buffer = NDBuffer[dtype](shape)
        self.buffer = buffer.share() if share else buffer^

    fn __init__(out self, var buffer: NDBuffer[dtype], share: Bool = True):
        self.buffer = buffer.share() if share else buffer^

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()

    @always_inline
    fn as_tensor(self, requires_grad: Bool = False) -> Tensor[dtype]:
        return Tensor[dtype](
            self.buffer.contiguous(), requires_grad=requires_grad
        )

    @staticmethod
    @always_inline
    fn full(
        shape: Shape, scalar: Scalar[dtype], share: Bool = False
    ) -> Gradbox[dtype]:
        return Gradbox[dtype](NDBuffer.full(shape, scalar), share=share)

    @always_inline
    fn unshared(self) -> Gradbox[dtype]:
        return Gradbox[dtype](self.buffer.contiguous(), share=False)

    fn sum(self, axes: IntList, keepdims: Bool) -> Gradbox[dtype]:
        var nd_buffer = self.buffer.sum(reduction_axes=axes, keepdims=keepdims)
        return Gradbox[dtype](nd_buffer^, share=False)

    fn broadcast_to(
        self, target_shape: Shape, share: Bool = False
    ) -> Gradbox[dtype]:
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Gradbox → broadcast_to: shape "
                + self.shape().__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        out = Gradbox[dtype](broadcasted_buffer^, share=share)
        return out^

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(List): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )

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

    fn __mul__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[Multiply](scalar), share=False
        )

    fn __rmul__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__mul__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](self.buffer.scalar_ops[Add](scalar), share=False)

    fn __radd__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__add__(scalar)

    fn __sub__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[Subtract](scalar), share=False
        )

    fn __rsub__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), share=False
        )

    fn __truediv__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        if scalar == Scalar[dtype](0):
            panic("Gradbox → __truediv__(scalar): can not divide by zero")
        return Gradbox[dtype](
            self.buffer.scalar_ops[Divide](scalar), share=False
        )

    fn __rtruediv__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), share=False
        )

    fn __mul__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn __mul__(self, other: Tensor[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn __add__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __sub__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __truediv__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer), share=False
        )

    fn __imul__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Multiply](scalar)

    fn __iadd__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Add](scalar)

    fn __isub__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, incoming: Gradbox[dtype]):
        self.buffer.inplace_ops[Multiply](incoming.buffer)

    @always_inline
    fn __iadd__(self, incoming: Gradbox[dtype]):
        self.buffer.inplace_ops[Add](incoming.buffer)

    @always_inline
    fn __isub__(self, incoming: Gradbox[dtype]):
        self.buffer.inplace_ops[Subtract](incoming.buffer)

    @always_inline
    fn __itruediv__(self, incoming: Gradbox[dtype]):
        self.buffer.inplace_ops[Divide](incoming.buffer)

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

    @always_inline
    fn reshape(self) -> Gradbox[dtype]:
        if self.numels() != 1:
            panic(
                "Gradbox → reshape: only gradbox with single element can be"
                " reshaped to scalar gradbox"
            )
        return self.reshape(Shape(), validated=True)

    @always_inline
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

    fn __eq__(self, tensor: Tensor[dtype]) -> Bool:
        if self.shape() != tensor.shape():
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                self.shape().__str__(),
                ",",
                tensor.shape().__str__(),
            )
        return (
            self.buffer.compare[Equal](tensor.buffer).buffer.value().all_true()
        )

    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Tensor[dtype]) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Gradbox → all_close is for floating point data types only",
        ]()
        if self.shape() != other.shape():
            panic(
                "Gradbox → all_close expects same shaped Tensor: "
                + self.shape().__str__()
                + ", "
                + other.shape().__str__()
            )

        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

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


from buffers import Buffer


fn main() raises:
    run = 1
    for _ in range(run):
        test_gradbox_is_shared()
        test_seed_gradbox()
        test_gradbox_inplace_add()
        test_gradbox_reshape()
        test_gradbox_reverse_subtract()
        test_gradbox_reverse_division()


from testing import assert_true


fn test_gradbox_reverse_division() raises:
    print("test_gradbox_reverse_division")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 / gradbox
    assert_true(
        result.buffer.data()
        == Buffer[dtype]([2.0, 1.0, 0.6666667, 0.5, 0.4, 0.33333334])
    )


fn test_gradbox_reverse_subtract() raises:
    print("test_gradbox_reverse_subtract")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 - gradbox
    assert_true(result.buffer.data() == Buffer[dtype]([1, 0, -1, -2, -3, -4]))


fn test_gradbox_reshape() raises:
    print("test_gradbox_reshape")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    reshaped = gradbox.reshape(Shape(3, 2))
    assert_true(reshaped[[2, 1]] == 6 and reshaped[[1, 1]] == 4)
    reshaped.zero_grad()
    assert_true(reshaped[[2, 1]] == 0 and reshaped[[1, 1]] == 0)
    assert_true(gradbox[[1, 2]] == 6 and gradbox[[0, 1]] == 2)


fn test_gradbox_inplace_add() raises:
    print("test_gradbox_inplace_add")
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
    print("test_gradbox_is_shared")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared(), "Gradbox buffer is shared - assertion failed"
    )


fn test_seed_gradbox() raises:
    print("test_seed_gradbox")
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
