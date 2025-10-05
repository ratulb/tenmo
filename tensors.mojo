### Mojo Tensor
### Implement tensor library in mojo from first principles
from math import iota, exp, floor, log
from random import seed, random_float64
from algorithm import vectorize
from sys import simdwidthof
from utils.numerics import max_finite, min_finite
from os import abort
from memory import memcpy, memset, memset_zero, ArcPointer
from shapes import Shape, ShapeIndexIter
from intlist import IntList
from ancestry import Ancestors
from strides import Strides
from common_utils_imports import *
from common_utils import id as identity
from operators_imports import *
from walkback import *
from forwards import *
from buffers import Buffer
from shared import TensorLite
from validators import Validator
from argminmax import Argmin, Argmax


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Representable & Writable & Absable
):
    alias Row = List[Scalar[dtype]]
    alias Rows = List[Self.Row]
    alias Block = List[Self.Rows]
    alias Blocks = List[Self.Block]

    alias randint = Tensor[DType.int64].randn
    alias randfloat = Tensor[DType.float64].randn
    alias randint32 = Tensor[DType.int32].randn
    alias randfloat32 = Tensor[DType.float32].randn

    var shape: Shape
    var strides: Strides
    var offset: Int
    var _contiguous: Bool
    var buffer: Buffer[dtype]
    var shared_buffer: Optional[ArcPointer[Buffer[dtype]]]
    var requires_grad: Bool
    var gradbox: UnsafePointer[Tensor[dtype]]
    var ancestors: Ancestors[dtype]
    var backwardFn: Optional[BackwardFn[dtype]]
    var owns_data: Bool

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape.copy()
        self.strides = Strides.default(shape)
        self.offset = 0
        self.requires_grad = requires_grad
        self.backwardFn = None
        self.gradbox = UnsafePointer[Tensor[dtype]]()
        self.ancestors = Ancestors[dtype].untracked()
        # Take care of Tensor with Shape.Void
        self.buffer = Buffer[dtype](1) if shape.rank() == 0 else Buffer[dtype](
            shape.num_elements()
        )
        self.shared_buffer = None
        self.owns_data = True
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn __init__(
        out self,
        shape: Shape,
        ptr: UnsafePointer[Scalar[dtype]],
        requires_grad: Bool = False,
        *,
        copy: Bool = True,
    ):
        Shape.validate(shape)
        self.shape = shape.copy()
        self.strides = Strides.default(shape)
        self.offset = 0
        self.requires_grad = requires_grad
        self.backwardFn = None
        self.gradbox = UnsafePointer[Tensor[dtype]]()
        self.ancestors = Ancestors[dtype].untracked()
        self.buffer = Buffer[dtype](shape.num_elements(), ptr, copy=copy)
        self.shared_buffer = None
        self.owns_data = True
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn __init__(
        out self,
        shape: Shape,
        var buffer: Buffer[dtype],
        requires_grad: Bool = False,
        *,
        strides: Optional[Strides] = None,
        offset: Int = 0,
        var shared_buffer: Optional[ArcPointer[Buffer[dtype]]] = None,
        owns_data: Bool = True,
    ):
        Shape.validate(shape)
        self.shape = shape.copy()
        self.strides = strides.value() if strides else Strides.default(shape)
        self.offset = offset
        self.requires_grad = requires_grad
        self.backwardFn = None
        self.gradbox = UnsafePointer[Tensor[dtype]]()
        self.ancestors = Ancestors[dtype].untracked()
        self.buffer = buffer^
        self.shared_buffer = shared_buffer^
        self.owns_data = owns_data
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn build_view(
        mut self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        if self.owns_data:
            self.shared_buffer = Optional(self.buffer.shared())
            self.buffer = Buffer[dtype].Empty
            self.owns_data = False

        return Tensor[dtype](
            shape,
            Buffer[dtype].Empty,
            requires_grad,
            strides=strides,
            offset=offset,
            shared_buffer=self.shared_buffer.copy(),
            owns_data=False,
        )

    fn __moveinit__(out self, deinit other: Self):
        self.shape = other.shape^
        self.strides = other.strides^
        self.offset = other.offset
        self._contiguous = other._contiguous
        self.buffer = other.buffer^
        self.shared_buffer = other.shared_buffer^
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox
        self.ancestors = other.ancestors^
        self.backwardFn = other.backwardFn^
        self.owns_data = other.owns_data

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape.copy()
        self.strides = other.strides.copy()
        self.offset = other.offset
        self._contiguous = other._contiguous
        self.buffer = other.buffer.copy()
        self.shared_buffer = other.shared_buffer.copy()
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox
        self.ancestors = other.ancestors.copy()
        self.backwardFn = other.backwardFn.copy()
        self.owns_data = other.owns_data

    @always_inline
    fn data(
        ref self,
    ) -> ref [__origin_of(self.buffer, self.shared_buffer.value()[])] Buffer[
        dtype
    ]:
        if not self.shared_buffer:
            return self.buffer
        return self.shared_buffer.value()[]

    fn id(self) -> Int:
        return identity(self)

    fn init_gradbox(mut self):
        if self.requires_grad and not self.gradbox.__as_bool__():
            gradbox = Tensor[dtype](self.shape)
            self.gradbox = UnsafePointer[Tensor[dtype]].alloc(1)
            self.gradbox.init_pointee_move(gradbox^)
            self.zero_grad()

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.strides.is_contiguous(self.shape)

    @always_inline
    fn is_leaf(self) -> Bool:
        return self.requires_grad and not self.has_backward_fn()

    @always_inline
    fn __len__(self) -> Int:
        return self.shape[0]

    @always_inline
    fn len(self) -> Int:
        return self.shape[0]

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
    fn max_index(self) -> Int:
        var max_index = self.offset
        for i in range(self.shape.rank()):
            max_index += (self.shape[i] - 1) * abs(self.strides[i])
        return max_index

    @always_inline
    fn flatten_index(self, indices: List[Int]) -> Int:
        return self.flatten_index(IntList.new(indices))

    @always_inline
    fn flatten_index(self, indices: VariadicList[Int]) -> Int:
        list = variadiclist_as_intlist(indices)
        return self.flatten_index(list)

    @always_inline
    fn flatten_index(self, indices: IntList) -> Int:
        # 1. Rank check
        if len(indices) != self.rank():
            panic(
                "Tensor → flatten_index: number of indices does not match",
                " tensor rank",
                ": indices →",
                indices.__str__(),
                "rank →",
                self.rank().__str__(),
            )

        flat = self.offset  # absolute base offset (0 for owning tensors)

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = self.shape[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic(
                    "Tensor → flatten_index: index out of bounds: axis",
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * self.strides[dim_idx]

        return flat

    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        return self.__getitem__(IntList.new(indices))

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic("Tensor → __getitem__: Scalar tensor expects no indices")
        index = self.flatten_index(indices)
        return self.buffer[
            index
        ] if self.owns_data else self.shared_buffer.value()[][index]

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.rank() == 0:  # Tensor with Shape ()
            panic(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.flatten_index(indices)
        return self.buffer[
            index
        ] if self.owns_data else self.shared_buffer.value()[][index]

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.rank() == 0:  # Tensor with Shape ()
            panic(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(IntList())"
            )
        index = self.flatten_index(indices)
        if self.owns_data:
            self.buffer[index] = value
        else:
            ref buffer = self.shared_buffer.value()[]
            buffer[index] = value

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        self.__setitem__(IntList.new(indices), value)

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic("Tensor → __setitem__: Scalar tensor expects no indices")
        index = self.flatten_index(indices)
        if self.owns_data:
            self.buffer[index] = value
        else:
            ref buffer = self.shared_buffer.value()[]
            buffer[index] = value

    fn item(self) -> Scalar[dtype]:
        # if self.shape != Shape(1) and self.rank() != 0:  # Tensor with Shape ()
        if self.shape != Shape.Unit and self.shape != Shape.Void:
            panic(
                "Tensor.item(): Only valid for scalar or singleton tensors, got"
                " shape: "
                + self.shape.__str__()
            )
        return self[0] if self.shape == Shape(1) else self[IntList.Empty]

    fn __str__(self) -> String:
        rank = self.rank()
        s = String("[")
        if rank == 1:
            s += "1D Tensor"
        elif rank == 2:
            s += "2D Tensor"
        elif rank == 3:
            s += "3D Tensor"
        elif rank == 4:
            s += "4D Tensor"
        elif rank == 5:
            s += "5D Tensor"
        else:
            s += "Tensor"
        s += self.shape.__str__()
        if not self.owns_data:
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

    # Check if it has a backward fn before calling this API
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self.backwardFn.value()

    fn has_backward_fn(self) -> Bool:
        return self.backwardFn is not None

    fn has_grad(self) -> Bool:
        return self.gradbox.__as_bool__()

    fn zero_grad(self):
        if self.requires_grad and self.has_grad():
            self.gradbox[].buffer.zero()

    fn gradients(self) -> UnsafePointer[Tensor[dtype]]:
        if self.requires_grad and self.has_grad():
            return self.gradbox
        else:
            return UnsafePointer[Tensor[dtype]]()

    fn grad(self) -> Tensor[dtype]:
        if not self.requires_grad:
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad"
            )
        if not self.has_grad():
            panic("Tensor → grad(self): grad not initialized")
        return self.gradbox[]

    fn grad_is_zero(self) -> Bool:
        if not self.requires_grad:
            panic(
                "Tensor → grad_is_zero: checking grad on a",
                "that does have grad",
            )

        fn all_zero(val: Scalar[dtype]) -> Bool:
            return val == Scalar[dtype](0)

        return self.has_grad() and self.gradbox[].for_all(all_zero)

    fn rows(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → rows: tensor rank is not 2")
        return self.shape[0]

    fn cols(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → cols: tensor rank is not 2")
        return self.shape[1]

    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape.Void  # Shape()

    fn __eq__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return self.compare_scalar[Equal](scalar)

    fn __ne__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return self.compare_scalar[NotEqual](scalar)

    fn __lt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return self.compare_scalar[LessThan](scalar)

    fn __le__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return self.compare_scalar[LessThanEqual](scalar)

    fn __gt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return self.compare_scalar[GreaterThan](scalar)

    fn __ge__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return self.compare_scalar[GreaterThanEqual](scalar)

    fn __eq__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __eq__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[Equal](other)
        return out

    fn __ne__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __ne__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[NotEqual](other)
        return out

    fn __lt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __lt__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[LessThan](other)
        return out

    fn __le__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __le__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[LessThanEqual](other)
        return out

    fn __gt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __gt__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[GreaterThan](other)
        return out

    fn __ge__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __ge__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        out = self.compare[GreaterThanEqual](other)
        return out

    fn float(self) -> Tensor[DType.float32]:
        if self.dtype == DType.float32:
            return rebind[Tensor[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Tensor[DType.float64]:
        if self.dtype == DType.float64:
            return rebind[Tensor[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Tensor[NewType]:
        var new_buffer: Buffer[NewType]
        if self.is_contiguous():
            if self.owns_data:
                new_buffer = self.buffer.to_dtype[NewType]()
            else:
                offset = self.offset
                numels = self.numels()
                new_buffer = self.data()[offset : offset + numels].to_dtype[
                    NewType
                ]()
        else:
            new_buffer = Buffer[NewType](self.numels())
            idx = 0
            for coord in self.shape:
                new_buffer[idx] = self[coord].cast[NewType]()
                idx += 1

        out = Tensor[NewType](
            shape=self.shape,
            buffer=new_buffer^,
            requires_grad=self.requires_grad,
        )

        return out

    fn add_ancestry(mut self, *parents: TensorLite[dtype]):
        for parent in parents:
            parent_ptr = UnsafePointer[Tensor[dtype]].alloc(1)
            parent_ptr.init_pointee_move(parent.tensor())
            ptr_shield = TensorLite[dtype](parent_ptr)
            shield_ptr = UnsafePointer[TensorLite[dtype]].alloc(1)
            shield_ptr.init_pointee_move(ptr_shield)
            self.ancestors.append(shield_ptr)

    fn ancestry(self) -> Ancestors[dtype]:
        return self.ancestors

    @always_inline
    fn broadcastable(self, to: Tensor[dtype]) -> Bool:
        return self.shape.broadcastable(to.shape)

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == Scalar[DType.bool](True)

        return self.for_all(all_truthy)

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        fn any_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == Scalar[DType.bool](True)

        return self.any(any_truthy)

    fn for_all[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.data().for_all[simd_width](pred).all_true()

    fn any[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.data().any[simd_width](pred)

    fn log[
        simd_width: Int = simdwidthof[dtype](),
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        shape = self.shape
        offset = self.offset
        numels = shape.num_elements()
        var out: Tensor[dtype]
        if self.is_contiguous():
            var buffer: Buffer[dtype]
            if self.owns_data:
                buffer = self.buffer.log()
            else:
                buffer = self.data()[offset : offset + numels].log()
            out = Tensor[dtype](shape, buffer^, requires_grad=grad_required)
        else:
            out = Tensor[dtype](shape, requires_grad=grad_required)
            for idx, value in self:
                out[idx] = log(value)

        return out

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self,) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Tensor → all_close is for floating point data types only",
        ]()
        if self.shape != other.shape:
            panic(
                "Tensor → all_close expects same shaped tensors: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        if self.owns_data and other.owns_data:
            return self.buffer.all_close[simd_width, rtol, atol](other.buffer)
        else:
            for coord, value in self:
                other_value = other[coord]
                if abs(value - other_value).gt(atol + rtol * abs(other_value)):
                    return False
            return True

    fn count(self, key: Scalar[dtype]) -> Int:
        if self.owns_data:
            return self.buffer.count(key)
        else:
            count = 0
            for coord in self.shape:
                if key == self[coord]:
                    count += 1
            return count

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
        if not self.requires_grad:
            return
        if self.shape != with_tensor.shape:
            panic(
                "Tensor → seed_grad: Shapes not equal -> ",
                self.shape.__str__(),
                " ≠ ",
                with_tensor.shape.__str__(),
            )
        if not self.has_grad():
            self.requires_grad_()
        if with_tensor.owns_data:
            self.gradbox[].data() += with_tensor.data()
        else:
            for coord in self.shape:
                self.gradbox[][coord] = with_tensor[coord]

    fn seed_grad(mut self, value: Scalar[dtype]):
        if self.has_grad():
            with_tensor = Tensor[dtype].full(self.shape, value)
            self.seed_grad(with_tensor)

    @always_inline
    fn fill(self, value: Scalar[dtype]):
        if self.is_contiguous():
            if self.owns_data:
                self.buffer.fill(value)
            else:
                offset = self.offset
                numels = self.numels()
                ref buffer = self.shared_buffer.value()[]
                for i in range(numels):
                    buffer[offset + i] = value
        else:
            for coord in self.shape:
                self[coord] = value

    @staticmethod
    fn full_like(
        like: Tensor[dtype], value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        shape = like.shape
        return Tensor[dtype].full(shape, value, requires_grad=requires_grad)

    @staticmethod
    fn full(
        shape: List[Int], value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        return Self.full(Shape(shape), value, requires_grad)

    @staticmethod
    fn full(
        shape: Shape, value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        tensor.fill(value)
        return tensor

    @staticmethod
    fn randn(
        shape: List[Int],
        low: Scalar[dtype] = 0,
        high: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → randint: is supported only for integral type",
        ]()

        return Self.rand(Shape(shape), low, high, init_seed, requires_grad)

    @staticmethod
    fn rand(
        shape: List[Int],
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        return Self.rand(Shape(shape), min, max, init_seed, requires_grad)

    @staticmethod
    fn rand(
        *axes_spans: Int,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        return Self.rand(Shape(axes_spans), min, max, init_seed, requires_grad)

    @staticmethod
    fn rand(
        shape: Shape,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        for i in range(numels):
            buffer[i] = random_float64(
                min.cast[DType.float64](), max.cast[DType.float64]()
            ).cast[dtype]()

        return Tensor[dtype](shape, buffer^, requires_grad)

    @staticmethod
    fn arange(
        *args: Scalar[dtype],
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        start: Scalar[dtype] = 0
        end: Scalar[dtype] = max_finite[dtype]()
        step: Scalar[dtype] = 1

        n = len(args)
        if n == 1:
            end = args[0]
        elif n == 2:
            start = args[0]
            end = args[1]
        elif n == 3:
            start = args[0]
            end = args[1]
            step = args[2]
        else:
            panic(
                "Tensor.arange expects 1 to 3 arguments:\n"
                + "- arange(end)\n"
                + "- arange(start, end)\n"
                + "- arange(start, end, step)\n"
                + "Got: "
                + String(len(args))
                + " argument(s)"
            )

        if step == 0:
            panic("step can not be zero")
        if (step > 0 and start >= end) or (step < 0 and start <= end):
            panic("Invalid range for the given step")
        delta = end - start
        size = floor(delta / step)
        if size <= 0:
            panic("Error: computed arange size is zero")
        count = size.__int__()
        buffer = Buffer[dtype](count)

        var value = start
        for i in range(count):
            buffer[i] = value
            value += step

        tensor = Tensor[dtype](
            Shape([count]), buffer^, requires_grad=requires_grad
        )
        return tensor

    @staticmethod
    fn zeros(
        axes_spans: List[Int], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        return Self.zeros(Shape(axes_spans), requires_grad)

    @staticmethod
    fn zeros(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        shape = Shape(axes_spans)
        return Self.zeros(shape, requires_grad)

    @staticmethod
    fn zeros_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        buffer = Buffer[dtype].full(
            Scalar[dtype](0), tensor.shape.num_elements()
        )
        out = Tensor[dtype](
            shape=tensor.shape, buffer=buffer, requires_grad=requires_grad
        )
        return out

    @staticmethod
    fn ones_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype].full(tensor.shape, 1, requires_grad=requires_grad)
        return out

    @staticmethod
    fn zeros(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        buffer = Buffer[dtype].full(Scalar[dtype](0), shape.num_elements())
        out = Tensor[dtype](
            shape=shape, buffer=buffer, requires_grad=requires_grad
        )
        return out

    fn onehot(self: Tensor[DType.int64], num_classes: Int) -> Tensor[dtype]:
        """Convert tensor of class indices to one-hot encoding.
        Args:
            self: Tensor of shape (...,) containing class indices.
            num_classes: Number of classes.
        Returns: Tensor of shape (..., num_classes).
        """
        shape = self.shape
        result = Tensor[dtype](shape + [num_classes])

        result.fill(Scalar[dtype](0))

        # Set appropriate positions to 1.0
        for idx in self.shape:
            var class_idx = self[idx].__int__()
            if class_idx < 0 or class_idx >= num_classes:
                panic(
                    "Tensor → onehot: invalid class at coordinate: ",
                    idx.__str__(),
                )
            if class_idx >= 0 and class_idx < num_classes:
                var one_hot_idx = idx + [class_idx]
                result[one_hot_idx] = Scalar[dtype](1)

        return result

    @staticmethod
    fn d1(row: Self.Row, requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d1")
        # Attention! Tensor([])
        if len(row) == 0:
            return Tensor[dtype].scalar(
                min_finite[dtype](), requires_grad=requires_grad
            )
        numels = len(row)
        shape = Shape(IntList(numels))
        buffer = Buffer[dtype](numels)
        memcpy(buffer.data, row._data, numels)
        return Tensor[dtype](shape, buffer^, requires_grad)

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntList(len(rows), len(rows[0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                panic("Tensor → d2 → not all rows equal in length")
            flattened.extend(row)
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(buffer.data, flattened._data, numels)
        return Tensor[dtype](shape, buffer^, requires_grad)

    @staticmethod
    fn d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d3")
        dims = IntList(len(blocks), len(blocks[0]), len(blocks[0][0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                panic("Tensor → d3 → not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    panic("Tensor → d3 → not all rows equal in length")

                flattened.extend(row)
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(buffer.data, flattened._data, numels)
        return Tensor[dtype](shape, buffer^, requires_grad)

    @staticmethod
    fn d4(
        blockgrid: List[Self.Block], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d4")
        dims = IntList(
            len(blockgrid),
            len(blockgrid[0]),
            len(blockgrid[0][0]),
            len(blockgrid[0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blockgrid:
            if len(block) != dims[1]:
                panic(
                    "Tensor → d4 → not all blocks are of equal length in the"
                    " blockgrid"
                )
            for matrix in block:
                if len(matrix) != dims[2]:
                    panic(
                        "Tensor → d4 → not all matrices are of equal length"
                        " in block"
                    )
                for row in matrix:
                    if len(row) != dims[3]:
                        panic(
                            "Tensor → d4 not all rows are of equal length in"
                            " matrix"
                        )
                    flattened.extend(row)
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(buffer.data, flattened._data, numels)
        return Tensor[dtype](shape, buffer^, requires_grad)

    @staticmethod
    fn d5(
        blockhive: List[Self.Blocks], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d5")
        dims = IntList(
            len(blockhive),
            len(blockhive[0]),
            len(blockhive[0][0]),
            len(blockhive[0][0][0]),
            len(blockhive[0][0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for blocks in blockhive:
            if len(blocks) != dims[1]:
                panic(
                    "Tensor → d5 → not all blocks are of equal length in the"
                    " input"
                )
            for block in blocks:
                if len(block) != dims[2]:
                    panic("Tensor → d5 → unequal block length")
                for matrix in block:
                    if len(matrix) != dims[3]:
                        panic(
                            "Tensor → d5 not all matrices are of equal length"
                            " in block"
                        )
                    for row in matrix:
                        if len(row) != dims[4]:
                            panic(
                                "Tensor → d5 not all rows are of equal length"
                                " in matrix"
                            )
                        flattened.extend(row)
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(buffer.data, flattened._data, numels)
        return Tensor[dtype](shape, buffer^, requires_grad)

    @staticmethod
    fn of(*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "of(*elems)")
        shape = Shape(IntList(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of(
        elems: Self.Row,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "of(elems)")
        shape = Shape(IntList(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of[
        row_size: Int
    ](*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(
            dtype, requires_grad, "of[row_size]"
        )

        if not (row_size >= 1 and row_size <= len(elems)):
            panic(
                (
                    "Tensor → of[row_size] → invalid row size or not enough"
                    " elements"
                ),
            )
        num_rows = len(elems) // row_size
        axes_spans = variadic1or2(num_rows, row_size)
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(num_rows):
            for j in range(row_size):
                tensor[i, j] = elems[i * row_size + j]
        return tensor

    @staticmethod
    fn scalar(val: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        result = Tensor[dtype](Shape(True), requires_grad=requires_grad)
        result[IntList()] = val
        return result

    @staticmethod
    fn ones(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        return Self.ones(Shape(axes_spans), requires_grad)

    @staticmethod
    fn ones(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        var value: SIMD[dtype, 1]

        @parameter
        if dtype.is_floating_point():
            value = SIMD[dtype, 1](1.0)
        else:
            value = SIMD[dtype, 1](1)
        for i in range(numels):
            buffer[i] = value
        return Tensor[dtype](shape, buffer^, requires_grad)

    fn sum_all(self) -> Scalar[dtype]:
        if self.is_contiguous():
            if self.owns_data:
                return self.buffer.sum()
            else:
                return self.shared_buffer.value()[].sum(
                    self.offset, self.max_index() + 1
                )
        else:
            summ = Scalar[dtype](0)
            for _, value in self:
                summ += value
            return summ

    fn broadcast_to(self, target_shape: Shape) -> Tensor[dtype]:
        if not self.shape.broadcastable(target_shape):
            panic(
                "Tensor → broadcast_to: shape "
                + self.shape.__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        mask = self.shape.broadcast_mask(target_shape)
        out = Tensor[dtype](target_shape, requires_grad=self.requires_grad)

        for idx in target_shape:
            src_idx = self.shape.translate_index(idx, mask, target_shape)
            out[idx] = self[src_idx]

        return out

    @always_inline
    fn broadcast_mask(self, broadcast_shape: Shape) -> IntList:
        return self.shape.broadcast_mask(broadcast_shape)

    @always_inline
    fn translate_index(
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        return self.shape.translate_index(indices, mask, broadcast_shape)

    @always_inline
    fn load[
        simdwidth: Int = 1
    ](self, row: Int, col: Int) -> SIMD[dtype, simdwidth]:
        constrained[
            simdwidth.is_power_of_two(),
            "Tensor → load: SIMD width (simdwidth) must be a power of 2",
        ]()

        if self.rank() != 2:
            panic("Tensor → load: supported only for 2D tensors")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + simdwidth > self.shape[1]
        ):
            panic("Tensor → load: Out-of-bounds access")
        if not self.owns_data and self.strides[1] != 1 and simdwidth > 1:
            panic(
                "Tensor → SIMD load attempted on non-contiguous Tensor - only"
                " single-element loads are permitted for non-contiguous tensor"
            )

        addr = row * self.strides[0] + col * self.strides[1] + self.offset
        if self.owns_data:
            return self.buffer.load[simdwidth](addr)
        else:
            return self.shared_buffer.value()[].load[simdwidth](addr)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, row: Int, col: Int, value: SIMD[dtype, simdwidth]):
        constrained[
            simdwidth.is_power_of_two(),
            "Tensor → store: SIMD width (simdwidth) must be a power of 2",
        ]()

        if self.rank() != 2:
            panic("Tensor → store is supported only for 2D tensors")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + simdwidth > self.shape[1]
        ):
            panic("Tensor → store: out-of-bounds access")

        if not self.owns_data and self.strides[1] != 1 and simdwidth > 1:
            panic(
                "Tensor → SIMD store attempted on non-contiguous Tensor - only"
                " single-element stores are permitted for non-contiguous tensor"
            )

        addr = row * self.strides[0] + col * self.strides[1] + self.offset

        if self.owns_data:
            self.buffer.store[simdwidth](addr, value)
        else:
            self.shared_buffer.value()[].store[simdwidth](addr, value)

    fn into_view(
        mut self, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        if not self.owns_data:
            panic("Tensor → into_view: not allowed on non-owning tensor")
        shape, strides = self.shape, self.strides
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        out = self.build_view(shape, strides, 0, grad_required)

        if grad_required:
            backward_fn = ViewBackward[dtype](
                shape, strides, 0
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[dtype]:
        return View[dtype].forward[track_grad](
            self, shape, strides, offset, requires_grad, validated
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: List[Int],
        strides: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        view_shape, view_strides = Shape(shape), Strides(strides)
        return View[dtype].forward[track_grad](
            self, view_shape, view_strides, offset, requires_grad, False
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        *shape_dims: Int,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = Shape(shape_dims)
        strides = Strides.default(shape)
        return View[dtype].forward[track_grad](
            self, shape, strides, offset, requires_grad, False
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        view_shape = Shape(shape)
        strides = Strides.default(view_shape)
        return View[dtype].forward[track_grad](
            self, view_shape, strides, offset, requires_grad, False
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: Shape,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        _ = """return self.view[track_grad](
            shape=shape,
            strides=Strides.default(shape),
            offset=offset,
            requires_grad=requires_grad,
            validated=False,
        )"""

        return View[dtype].forward[track_grad](
            self, shape, Strides.default(shape), offset, requires_grad, False
        )

    fn flatten[
        track_grad: Bool = True
    ](
        self,
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Flatten[dtype].forward[track_grad](
            self, start_dim, end_dim, requires_grad
        )

    fn repeat[
        track_grad: Bool = True
    ](self, repeat: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return self.repeat[track_grad](IntList.new(repeat), requires_grad)

    fn repeat[
        track_grad: Bool = True
    ](self, repeat: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Repeat.forward[track_grad](self, repeat, requires_grad)

    fn tile[
        track_grad: Bool = True
    ](self, repeat: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return self.tile[track_grad](IntList.new(repeat), requires_grad)

    fn tile[
        track_grad: Bool = True
    ](self, repeat: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Tile.forward[track_grad](self, repeat, requires_grad)

    fn slice[
        track_grad: Bool = True
    ](mut self, start: Int, end: Int, step: Int = 1, axis: Int = 0) -> Tensor[
        dtype
    ]:
        """
        Slice the tensor along a single axis and return a view.

        Args:
            start: Starting index (inclusive).
            end: Ending index (exclusive).
            step: Step size.
            axis: Axis along which to slice (default 0).

        Returns:
            Tensor[dtype]: A view of the sliced tensor.
        """
        # Call Validator to compute everything
        var shape, strides, offset = (
            Validator.validate_and_compute_slice_metadata(
                self.shape, self.strides, axis, start, end, step
            )
        )

        # Return view
        return View[dtype].forward[track_grad](
            self,
            shape,
            strides,
            offset,
            self.requires_grad,
            True,
        )

    fn slice[
        track_grad: Bool = True
    ](
        mut self,
        axes: List[Int],
        starts: List[Int],
        ends: List[Int],
        steps: List[Int] = [],
    ) -> Tensor[dtype]:
        # Default step = 1 if not provided
        jumps = IntList(steps)
        if len(steps) == 0:
            jumps = IntList.filled(len(axes), 1)
        elif len(steps) != len(axes):
            panic("Tensor → slice: length of steps must match axes length")

        # Call Validator
        var shape, strides, offset = (
            Validator.validate_and_compute_slice_metadata_multi(
                self.shape,
                self.strides,
                IntList(axes),
                IntList(starts),
                IntList(ends),
                jumps,
            )
        )

        # Return view
        return View[dtype].forward[track_grad](
            self,
            shape,
            strides,
            offset,
            self.requires_grad,
            True,
        )

    fn __getitem__(mut self, *slices: Slice) -> Tensor[dtype]:
        # Delegate shape/strides/offset computation
        shape, strides, offset = Validator.validate_and_compute_view_metadata(
            self.shape,
            self.strides,
            slices,
        )
        return View[dtype].forward[track_grad=True](
            self, shape, strides, offset, self.requires_grad, True
        )

    fn set(mut self, mut tensor: Tensor[dtype], *indices: Idx):
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape, self.strides, indices
            )
        )
        if len(shape) == 0:
            if not tensor.numels() == 1:
                panic(
                    (
                        "Tensor → set: expected single element tensor. Received"
                        " tensor with"
                    ),
                    tensor.numels().__str__(),
                    "elements tensor",
                )

            else:
                elem = (
                    tensor.item() if tensor.shape
                    == Shape.Void else (
                        tensor.squeeze[track_grad=False](
                            [], requires_grad=False
                        )
                    )[IntList.Empty]
                )
                if self.owns_data:
                    self.buffer[offset] = elem
                else:
                    self.shared_buffer.value()[][offset] = elem
        else:
            if not tensor.shape.broadcastable(shape):
                panic(
                    "Tensor → set: input tensor not broadcastable to shape",
                    shape.__str__(),
                )
            else:
                sliced = self.view(shape, strides, offset, False)
                if tensor.shape == shape:
                    for idx in shape:
                        sliced[idx] = tensor[idx]
                else:
                    mask = tensor.shape.broadcast_mask(shape)
                    for idx in shape:
                        tensor_idx = tensor.shape.translate_index(
                            idx, mask, shape
                        )
                        sliced[idx] = tensor[tensor_idx]

    fn set(mut self, value: Scalar[dtype], *indices: Idx):
        # Compute view metadata
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape, self.strides, indices
            )
        )
        if len(shape) == 0:
            if self.owns_data:
                self.buffer[offset] = value
            else:
                self.shared_buffer.value()[][offset] = value
        else:
            sliced = self.view(shape, strides, offset, False)
            for idx in shape:
                sliced[idx] = value

    fn __getitem__(mut self, *indices: Idx) -> Tensor[dtype]:
        # Compute view metadata
        view_shape, view_strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape, self.strides, indices
            )
        )

        # Handle scalar (rank-0) case
        is_scalar = len(view_shape) == 0
        shape = Shape.Void if is_scalar else view_shape
        strides = Strides.Zero if is_scalar else view_strides
        return View[dtype].forward[track_grad=True](
            self,
            shape,
            strides,
            offset,
            self.requires_grad,
            True,
        )

    fn contiguous[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return Contiguous[dtype].forward[track_grad](self, requires_grad)

    fn reshape[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        if self.numels() != 1:
            panic(
                "Tensor → reshape: only tensor with single element can be"
                " reshaped to scalar tensor"
            )
        return self.reshape[track_grad](
            Shape(True), requires_grad=requires_grad, validated=True
        )

    fn reshape[
        track_grad: Bool = True
    ](self, *newdims: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        if len(newdims) == 1 and newdims[0] == 0:
            return self.reshape[track_grad](requires_grad=requires_grad)
        shape = Validator.validate_and_construct_new_shape(
            self.shape, IntList(newdims)
        )
        return self.reshape[track_grad](
            shape, requires_grad=requires_grad, validated=True
        )

    fn reshape[
        track_grad: Bool = True
    ](self, shape: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        new_shape = Validator.validate_and_construct_new_shape(
            self.shape, IntList.new(shape)
        )
        return self.reshape[track_grad](
            new_shape, requires_grad=requires_grad, validated=True
        )

    fn reshape[
        track_grad: Bool = True
    ](
        self,
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[dtype]:
        return Reshape[dtype].forward[track_grad](
            self, new_shape, requires_grad, validated
        )

    fn transpose[
        track_grad: Bool = True
    ](mut self, *axes: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return self.transpose[track_grad](IntList(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](
        mut self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return self.transpose[track_grad](IntList.new(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](mut self, axes: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Transpose.forward[track_grad](self, axes, requires_grad)

    fn broadcast_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        if self.shape.rank() == 0 or other.shape.rank() == 0:
            return self.broadcast_scalar_op(other, op)
        else:
            result = self.broadcast_tensor_op(other, op)
            return result

    fn broadcast_scalar_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        # Decide result shape
        result_shape = other.shape if self.shape.rank() == 0 else self.shape
        result = Tensor[dtype](result_shape, requires_grad=False)

        for coord in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[coord]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[coord]
            )
            result[coord] = op(self_val, other_val)

        return result

    fn broadcast_tensor_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        result_shape = Shape.broadcast_shape(self.shape, other.shape)
        mask1 = self.broadcast_mask(result_shape)
        mask2 = other.broadcast_mask(result_shape)
        result = Tensor[dtype](result_shape, requires_grad=False)
        for indices in result_shape:
            self_indices = self.translate_index(indices, mask1, result_shape)
            other_indices = other.translate_index(indices, mask2, result_shape)
            result[indices] = op(self[self_indices], other[other_indices])
        return result

    fn backward_contribution(
        self,
        other: Tensor[dtype],
        upstream_grad: Tensor[dtype],
        do_multiply: Bool,
    ) -> Tensor[dtype]:
        var grad_contrib: Tensor[dtype]
        if upstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                self.shape, upstream_grad.item(), requires_grad=False
            )
        else:
            grad_contrib = (
                upstream_grad * other if do_multiply else upstream_grad
            )
            if grad_contrib.shape != self.shape:
                axes = self.broadcast_mask(grad_contrib.shape).indices_of(1)
                grad_contrib = grad_contrib.sum[track_grad=False](
                    axes=axes, keepdims=True
                )
            if grad_contrib.shape != self.shape:
                grad_contrib = grad_contrib.reshape[track_grad=False](
                    self.shape
                )
            grad_contrib.requires_grad = False

        return grad_contrib

    fn sum[
        track_grad: Bool = True
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.sum[track_grad](IntList.new(axes), keepdims, requires_grad)

    fn sum[
        track_grad: Bool = True
    ](
        self,
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Summer[dtype].forward[track_grad](
            self, axes, keepdims, requires_grad
        )

    fn mean[
        track_grad: Bool = True
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.mean[track_grad](IntList.new(axes), keepdims, requires_grad)

    fn mean[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Mean[dtype].forward[track_grad](
            self, axes, keepdims, requires_grad
        )

    fn __rtruediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return DivideScalar[dtype].forward[True](self, scalar)

    fn __truediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return DivideByScalar[dtype].forward[True](self, scalar)

    # Element wise division of two tensors
    fn __truediv__(self, other: Self) -> Tensor[dtype]:
        return Divider[dtype].forward[True](self, other)

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, factor: Scalar[dtype]) -> Tensor[dtype]:
        return MultiplyScalar[dtype].forward[True](self, factor)

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Tensor[dtype]:
        return Multiplicator[dtype].forward[True](self, other)

    fn update_grad[opcode: Int](mut self, incoming: Tensor[dtype]):
        if opcode == MulTensor:
            self.gradbox[].__imul__(incoming)
        if opcode == AddTensor:
            self.gradbox[].__iadd__(incoming)
        if opcode == SubtractTensor:
            self.gradbox[].__isub__(incoming)
        if opcode == ZeroGrad:
            self.zero_grad()

    fn __iadd__(mut self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __iadd__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            panic(
                "Tensor → __iadd__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        if self.owns_data and other.owns_data:
            self.buffer += other.buffer
        else:
            for coord in self.shape:
                self[coord] += other[coord]

    fn __isub__(mut self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __isub__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            panic(
                "Tensor → __isub__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        if self.owns_data and other.owns_data:
            self.buffer -= other.buffer
        else:
            for coord in self.shape:
                self[coord] -= other[coord]

    fn __imul__(mut self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __imul__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            panic(
                "Tensor → __imul__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        if self.owns_data and other.owns_data:
            self.buffer *= other.buffer
        else:
            for coord in self.shape:
                self[coord] *= other[coord]

    fn exp(self) -> Tensor[dtype]:
        if self.owns_data:
            return Tensor[dtype](
                self.shape, self.data().exp(), self.requires_grad
            )
        else:
            tensor = Tensor[dtype](self.shape, self.requires_grad)
            for idx, value in self:
                tensor[idx] = exp(value)
            return tensor

    fn __neg__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __neg__ is for numeric data types only",
        ]()
        if self.owns_data:
            return Tensor(self.shape, -self.data(), self.requires_grad)
        else:
            tensor = Tensor[dtype](self.shape, self.requires_grad)
            for idx, value in self:
                tensor[idx] = -value
            return tensor

    fn __invert__(self: Tensor[DType.bool]) -> Tensor[DType.bool]:
        if self.owns_data:
            return Tensor(self.shape, ~self.data(), self.requires_grad)
        else:
            tensor = Tensor[DType.bool](self.shape, self.requires_grad)
            for idx, value in self:
                tensor[idx] = ~value
            return tensor

    fn __abs__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __abs__ is for numeric data types only",
        ]()
        if self.owns_data:
            return Tensor(self.shape, abs(self.data()), self.requires_grad)
        else:
            tensor = Tensor[dtype](self.shape, self.requires_grad)
            for idx, value in self:
                tensor[idx] = abs(value)
            return tensor

    fn __radd__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return AddScalar[dtype].forward[True](self, scalar)

    fn __add__(self, other: Self) -> Tensor[dtype]:
        return Adder[dtype].forward[True](self, other)

    fn __pow__[
        track_grad: Bool = True
    ](self, exponent: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __pow__ is for numeric data types only",
        ]()

        return Exponentiator[dtype].forward[track_grad](self, exponent)

    fn __rsub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractFromScalar[dtype].forward[True](self, scalar)

    fn __sub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractScalar[dtype].forward[True](self, scalar)

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        return Subtractor[dtype].forward[True](self, other)

    fn dot[
        track_grad: Bool = True
    ](self, other: Self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return Dot[dtype].forward[track_grad](self, other, requires_grad)

    fn compare[
        op: Int, simd_width: Int = simdwidthof[dtype]()
    ](this: Tensor[dtype], that: Tensor[dtype]) -> Tensor[DType.bool]:
        out_shape = this.shape[::]
        if this.is_contiguous() and that.is_contiguous():
            if this.owns_data and that.owns_data:
                this_buffer = this.data()
                that_buffer = that.data()

            elif this.owns_data and not that.owns_data:
                this_buffer = this.data()
                that_buffer = that.data()[
                    that.offset : that.offset + that.numels()
                ]
            elif not this.owns_data and that.owns_data:
                this_buffer = this.data()[
                    this.offset : this.offset + this.numels()
                ]
                that_buffer = that.data()
            else:
                this_buffer = this.data()[
                    this.offset : this.offset + this.numels()
                ]
                that_buffer = that.data()[
                    that.offset : that.offset + that.numels()
                ]

            if op == Equal:
                buffer = this_buffer.eq[simd_width](that_buffer)

            elif op == NotEqual:
                buffer = this_buffer.ne[simd_width](that_buffer)

            elif op == LessThan:
                buffer = this_buffer.lt[simd_width](that_buffer)

            elif op == LessThanEqual:
                buffer = this_buffer.le[simd_width](that_buffer)

            elif op == GreaterThan:
                buffer = this_buffer.gt[simd_width](that_buffer)

            else:  # op == GreaterThanEqual
                buffer = this_buffer.ge[simd_width](that_buffer)

            out = Tensor[DType.bool](out_shape, buffer^, requires_grad=False)
            return out

        else:
            out = Tensor[DType.bool].full(out_shape, Scalar[DType.bool](False))

            for idx, value in this:
                if op == Equal:
                    out[idx] = that[idx] == value

                elif op == NotEqual:
                    out[idx] = that[idx] != value

                elif op == LessThan:
                    out[idx] = that[idx] < value

                elif op == LessThanEqual:
                    out[idx] = that[idx] <= value

                elif op == GreaterThan:
                    out[idx] = that[idx] > value

                else:  # op == GreaterThanEqual
                    out[idx] = that[idx] >= value
            return out

    fn compare_scalar[
        op: Int, simd_width: Int = simdwidthof[dtype]()
    ](this: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        out_shape = this.shape[::]

        if this.is_contiguous():
            if this.owns_data:
                this_buffer = this.data()
            else:
                this_buffer = this.data()[
                    this.offset : this.offset + this.numels()
                ]

            if op == Equal:
                buffer = this_buffer.eq[simd_width](scalar)

            elif op == NotEqual:
                buffer = this_buffer.ne[simd_width](scalar)

            elif op == LessThan:
                buffer = this_buffer.lt[simd_width](scalar)

            elif op == LessThanEqual:
                buffer = this_buffer.le[simd_width](scalar)

            elif op == GreaterThan:
                buffer = this_buffer.gt[simd_width](scalar)

            else:  # GreaterThanEqual
                buffer = this_buffer.ge[simd_width](scalar)

            out = Tensor[DType.bool](out_shape, buffer^, requires_grad=False)
            return out

        else:
            out = Tensor[DType.bool].full(out_shape, Scalar[DType.bool](False))

            for idx, value in this:
                if op == Equal:
                    out[idx] = value == scalar

                elif op == NotEqual:
                    out[idx] = value != scalar

                elif op == LessThan:
                    out[idx] = value < scalar

                elif op == LessThanEqual:
                    out[idx] = value <= scalar

                elif op == GreaterThan:
                    out[idx] = value > scalar

                else:  # op GreaterThanEqual
                    out[idx] = value >= scalar
            return out

    fn vector_matrix_mm[
        track_grad: Bool = True
    ](
        A: Tensor[dtype], mut B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        return VectorMatrixMM[dtype].forward[track_grad](A, B, requires_grad)

    fn matrix_vector_mm[
        track_grad: Bool = True
    ](
        mut A: Tensor[dtype], B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        return MatrixVectorMM[dtype].forward[track_grad](A, B, requires_grad)

    fn __iadd__(mut self, scalar: Scalar[dtype]):
        if self.is_leaf():
            panic(
                "Tensor → Cannot perform in-place operation on a leaf tensor"
                " requiring grad."
            )

        if self.owns_data:
            self.buffer += scalar
        else:
            for coord in self.shape:
                self[coord] += scalar

    fn permute[
        track_grad: Bool = True
    ](
        mut self, axes: List[Int], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return Permute[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    fn permute[
        track_grad: Bool = True
    ](mut self, axes: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Permute[dtype].forward[track_grad](self, axes, requires_grad)

    fn unsqueeze[
        track_grad: Bool = True
    ](mut self, axis: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Unsqueeze[dtype].forward[track_grad](
            self, IntList(axis), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](
        mut self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        """Unsqueeze multiple axes by inserting dimensions of size 1."""
        return Unsqueeze[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](mut self, axes: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        """Unsqueeze multiple axes by inserting dimensions of size 1."""
        return Unsqueeze[dtype].forward[track_grad](self, axes, requires_grad)

    fn softmax[
        track_grad: Bool = True
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Softmax[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    fn softmax[
        track_grad: Bool = True
    ](
        self,
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        return Softmax[dtype].forward[track_grad](self, axes, requires_grad)

    fn max(
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return MinMax[dtype].forward[True](
            self, IntList.new(axes), keepdims, requires_grad
        )

    fn max(
        self,
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return MinMax[dtype].forward[True](self, axes, keepdims, requires_grad)

    fn min(
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return MinMax[dtype].forward[False](
            self, IntList.new(axes), keepdims, requires_grad
        )

    fn min(
        self,
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return MinMax[dtype].forward[False](self, axes, keepdims, requires_grad)

    fn relu(
        self,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return ReLU[dtype].forward(self, requires_grad)

    fn shuffle[
        track_grad: Bool = True
    ](
        self,
        perm: List[Int] = [],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Shuffle[dtype].forward[track_grad](
            self, perm, axis, requires_grad
        )

    fn argmax(self, axis: Int = 0) -> Tensor[DType.int32]:
        return Argmax[dtype].argmax(tensor=self, axis=axis)

    fn argmin(self, axis: Int = 0) -> Tensor[DType.int32]:
        return Argmin[dtype].argmin(tensor=self, axis=axis)

    fn expand[
        track_grad: Bool = True
    ](
        mut self: Tensor[dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Expand[dtype].forward[track_grad](self, target, requires_grad)

    fn expand[
        track_grad: Bool = True
    ](
        mut self: Tensor[dtype],
        *target_dims: Int,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Expand[dtype].forward[track_grad](
            self, Shape(target_dims), requires_grad
        )

    # Squeeze specified axes or all dims of size 1 if no axes provided
    fn squeeze[
        track_grad: Bool = True
    ](
        mut self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        """
        Squeeze dimensions of size 1.

        Args:
            axes: Optional list of axes to squeeze. If None, squeeze all dims of size 1.
            requires_grad: Optional override for gradient requirement.

        Returns:
            Tensor with specified dimensions squeezed.
        """
        return Squeeze[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    # Squeeze single axis if provided, otherwise squeeze all dims of size 1
    fn squeeze[
        track_grad: Bool = True
    ](
        mut self,
        axis: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Squeeze[dtype].forward[track_grad](
            self,
            IntList(axis.value()) if axis else IntList(),
            requires_grad,
        )

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(
            "\n",
            self.__str__(),
            end="\n",
        )
        empty = IntList()
        print_tensor_recursive(
            UnsafePointer(to=self),
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    # Always use this to print grad to avoid surprises of segmentation fault!
    fn gprint(self, num_first: Int = 10, num_last: Int = 10):
        if not self.requires_grad:
            print("Tensor is non-differentiable")
        elif self.requires_grad and not self.has_grad():
            print("Requires grad but grad not initialized")
        else:
            self.gradients()[].print(num_first, num_last)

    fn free(deinit self):
        print("Tensor__del__ → deleting tensor with id: " + self.id().__str__())
        if self.owns_data:
            if self.has_grad():
                self.gradbox.destroy_pointee()
                self.gradbox.free()
                log_debug("Tensor__del__ → freed grad")
                print("Tensor__del__ → freed grad")

    fn mse(self, target: Tensor[dtype]) -> Tensor[dtype]:
        return ((self - target) ** 2).mean()

    fn backward(self, start_grad: Scalar[dtype] = 1.0):
        TensorLite.of(self).backward(start_grad)

    fn backward(self, seed_tensor: Tensor[dtype]):
        TensorLite.of(self).backward(seed_tensor)

    fn requires_grad_(mut self, requires_grad: Bool = True):
        self.requires_grad = requires_grad
        self.init_gradbox()

    fn matmul[
        track_grad: Bool = True, simd_width: Int = simdwidthof[dtype]()
    ](mut A: Tensor[dtype], mut B: Tensor[dtype]) -> Tensor[dtype]:
        return Matmul[dtype].forward[track_grad](A, B)

    @staticmethod
    fn matmul_2d[
        track_grad: Bool = True, simd_width: Int = simdwidthof[dtype]()
    ](
        A_ptr: UnsafePointer[Tensor[dtype]],
        B_ptr: UnsafePointer[Tensor[dtype]],
        C_ptr: UnsafePointer[Tensor[dtype]] = UnsafePointer[Tensor[dtype]](),
        requires_grad: Bool = True,
    ) -> Tensor[dtype]:
        return Matmul_2d[dtype].forward[track_grad](
            A_ptr, B_ptr, C_ptr, requires_grad
        )

    fn matmul_nd[
        track_grad: Bool = True
    ](
        mut A: Tensor[dtype], mut B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        return Matmul_nd[dtype].forward[track_grad](A, B, requires_grad)

    @staticmethod
    fn broadcasted_indices(
        target_indices: IntList, target_shape: Shape, source_shape: Shape
    ) -> IntList:
        """Get coordinates for source tensor given target coordinates."""
        var source_indices = IntList.with_capacity(len(source_shape))

        for i in range(len(source_shape)):
            target_idx = len(target_shape) - len(source_shape) + i
            if source_shape[i] == 1:
                source_indices.append(0)  # Broadcasted dimension → use 0
            else:
                source_indices.append(
                    target_indices[target_idx]
                )  # Normal dimension

        return source_indices

    @staticmethod
    fn sum_over_broadcasted_axes(
        batch_grad: Tensor[dtype], recipient_shape: Shape
    ) -> Tensor[dtype]:
        """Sum over dimensions that were broadcasted in the forward pass."""
        result = batch_grad
        current_shape = batch_grad.shape

        # Sum over extra leading dimensions
        while len(current_shape) > len(recipient_shape):
            result = result.sum(axes=[0], keepdims=False)
            current_shape = result.shape

        # Sum over mismatched dimensions
        for i in range(len(recipient_shape)):
            if current_shape[i] != recipient_shape[i] and current_shape[i] > 1:
                result = result.sum(axes=[i], keepdims=True)
                current_shape = result.shape
        return result

    fn __iter__(ref self) -> ElemIterator[dtype, __origin_of(self)]:
        return ElemIterator[dtype, __origin_of(self)](Pointer(to=self))

    fn element_at(self, index: Int) -> Scalar[dtype]:
        idx = index + self.max_index() if index < 0 else index
        if idx < 0 or idx > self.max_index():
            panic(
                "Tensor → element_at: index out of bounds.",
                "Tensor max index",
                self.max_index().__str__(),
                ", provided index",
                index.__str__(),
            )
        return self.data()[idx]


struct ElemIterator[dtype: DType, origin: ImmutableOrigin](Copyable & Movable):
    var src: Pointer[Tensor[dtype], origin]
    var index_itr: ShapeIndexIter[ImmutableAnyOrigin]

    fn __init__(out self, src: Pointer[Tensor[dtype], origin]):
        self.src = src
        self.index_itr = rebind[ShapeIndexIter[ImmutableAnyOrigin]](
            src[].shape.__iter__()
        )

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> (IntList, Scalar[dtype]):
        next = self.index_itr.__next__()
        return next, self.src[][next]

    fn __len__(self) -> Int:
        return self.index_itr.__len__()

    fn __has_next__(self) -> Bool:
        return self.index_itr.__has_next__()


fn main() raises:
    # test_element_at()
    test_fill()


from testing import assert_true


fn test_fill() raises:
    a = Tensor.zeros(10)
    a.fill(42)
    v = a.view(shape=[3], offset=2)
    v.fill(99)
    assert_true(
        (v == Tensor.d1([99, 99, 99])).all_true(), "view fill assertion failed"
    )
    assert_true(
        (a == Tensor.d1([42, 42, 99, 99, 99, 42, 42, 42, 42, 42])).all_true(),
        "view fill propagation1 to parent failed",
    )
    v1 = a.view(shape=[2, 5])
    v2 = v1[il(1), s(2, None, 2)]
    v2.fill(101)

    assert_true(
        (a == Tensor.d1([42, 42, 99, 99, 99, 42, 42, 101, 42, 101])).all_true(),
        "view fill propagation2 to parent failed",
    )
    assert_true(
        (v.sum_all() == 3 * 99) and (v2.sum_all() == 2 * 101),
        "fill sum_all assertion failed for views",
    )
    b = Tensor.d1([1919, 1919])
    v2.set(b, s())
    assert_true(
        (
            a == Tensor.d1([42, 42, 99, 99, 99, 42, 42, 1919, 42, 1919])
        ).all_true(),
        "view set propagation to parent failed",
    )


fn test_element_at() raises:
    a = Tensor.arange(10)
    a.print()
    print()
    v = a[s(2, 8, 2)]
    assert_true(
        v.max_index() == 6 and v.element_at(-4) == 2,
        "max_index and element_at assertion failed",
    )
