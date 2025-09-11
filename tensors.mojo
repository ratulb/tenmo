### Mojo Tensor
### Implement tensor library in mojo from first principles
from math import iota, exp, floor, log
from random import seed, random_float64
from algorithm import vectorize
from sys import simdwidthof
from utils.numerics import max_finite, min_finite
from os import abort
from memory import memcpy, memset, memset_zero
from shapes import Shape
from intlist import IntList
from ancestry import Ancestors
from strides import Strides
from common_utils_imports import *
from operators_imports import *
from walkback import *
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
    var shape: Shape
    var strides: Strides
    var offset: Int
    var _contiguous: Bool
    var buffer: Buffer[dtype]
    var requires_grad: Bool
    var gradbox: UnsafePointer[Tensor[dtype]]
    var ancestors: Ancestors[dtype]
    var base: UnsafePointer[Tensor[dtype]]  # Only allocated on need basis
    var backwardFn: Optional[BackwardFn[dtype]]
    var owns_data: Bool

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(
        out self,
        shape: Shape,
        base: UnsafePointer[Tensor[dtype]],
        strides: Optional[Strides] = None,
        offset: Int = 0,
        requires_grad: Bool = False,
    ):
        self.shape = shape
        self.base = base
        self.strides = strides.value() if strides else Strides.default(shape)
        self.offset = offset
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Tensor[dtype]]()
        self.backwardFn = None
        self.ancestors = Ancestors[dtype].untracked()
        self.buffer = Buffer[dtype].Empty
        self.owns_data = False
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn __init__(
        out self,
        shape: Shape,
        buffer: Buffer[dtype],
        requires_grad: Bool = False,
    ):
        Shape.validate(shape)
        self.shape = shape
        self.strides = Strides.default(shape)
        self.offset = 0
        self.requires_grad = requires_grad
        self.backwardFn = None
        self.gradbox = UnsafePointer[Tensor[dtype]]()
        self.ancestors = Ancestors[dtype].untracked()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.buffer = buffer
        self.owns_data = True
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape
        self.strides = Strides.default(shape)
        self.offset = 0
        self.requires_grad = requires_grad
        self.base = UnsafePointer[Tensor[dtype]]()
        self.backwardFn = None
        self.gradbox = UnsafePointer[Tensor[dtype]]()
        self.ancestors = Ancestors[dtype].untracked()
        # Take care of Tensor with Shape.Void
        self.buffer = Buffer[dtype](1) if shape.rank() == 0 else Buffer[dtype](
            shape.num_elements()
        )
        self.owns_data = True
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn __moveinit__(out self, deinit other: Self):
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self._contiguous = other._contiguous
        self.buffer = other.buffer
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox
        self.ancestors = other.ancestors
        self.base = other.base
        self.backwardFn = other.backwardFn
        self.owns_data = other.owns_data

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self._contiguous = other._contiguous
        self.buffer = other.buffer
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox
        self.ancestors = other.ancestors
        self.base = other.base
        self.backwardFn = other.backwardFn
        self.owns_data = other.owns_data

    fn id(self) -> Int:
        return Int(UnsafePointer(to=self))

    fn init_gradbox(mut self):
        if self.requires_grad and not self.gradbox.__as_bool__():
            gradbox = Tensor[dtype](self.shape)
            self.gradbox = UnsafePointer[Tensor[dtype]].alloc(1)
            self.gradbox.init_pointee_move(gradbox^)
            self.zero_grad()

    fn is_contiguous(self) -> Bool:
        if self.shape.rank() == 0:
            return True  # scalar is trivially contiguous
        var expected_stride = 1
        for i in reversed(range(self.shape.rank())):
            if self.shape[i] > 1 and self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]

        return True

    fn is_tensor(self) -> Bool:
        return self.owns_data

    @always_inline
    fn is_leaf(self) -> Bool:
        return self.requires_grad and not self.has_backward_fn()

    fn is_view(self) -> Bool:
        return not self.owns_data

    fn __len__(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn len(self) -> Int:
        return self.shape.num_elements()

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
        var max_index = 0
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
        return self.buffer.load(
            index
        ) if self.owns_data else self.base_address()[].buffer.load(index)

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.rank() == 0:  # Tensor with Shape ()
            panic(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.flatten_index(indices)
        return self.buffer[
            index
        ] if self.owns_data else self.base_address()[].buffer[index]

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.rank() == 0:  # Tensor with Shape ()
            panic(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(IntList())"
            )
        index = self.flatten_index(indices)
        self.buffer.store(
            index, value
        ) if self.owns_data else self.base_address()[].buffer.store(
            index, value
        )

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        self.__setitem__(IntList.new(indices), value)

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic("Tensor → __setitem__: Scalar tensor expects no indices")
        index = self.flatten_index(indices)
        self.buffer.store(
            index, value
        ) if self.owns_data else self.base_address()[].buffer.store(
            index, value
        )

    fn item(self) -> Scalar[dtype]:
        if (
            self.shape != Shape.Unit and self.rank() != 0
        ):  # Tensor with Shape ()
            panic(
                "Tensor.item(): Only valid for scalar or singleton tensors, got"
                " shape: "
                + self.shape.__str__()
            )
        return (
            self[0] if self.shape == Shape.Unit else self[IntList.Empty]
        ) if self.owns_data else (
            self.base[][0] if self.shape
            == Shape.Unit else self.base[][IntList.Empty]
        )

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

    fn grad_is_zero(self) -> Bool:
        if not self.requires_grad:
            target = "Tensor" if self.owns_data else "View"
            panic(
                target,
                "→ grad_is_zero: checking grad on a",
                target.lower(),
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
        return self.numels() == 1 and self.shape == Shape.Void

    fn __eq__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Comparator.compare_scalar[Equal](self, scalar)

    fn __ne__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Comparator.compare_scalar[NotEqual](self, scalar)

    fn __lt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Comparator.compare_scalar[LessThan](self, scalar)

    fn __le__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Comparator.compare_scalar[LessThanEqual](self, scalar)

    fn __gt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Comparator.compare_scalar[GreaterThan](self, scalar)

    fn __ge__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Comparator.compare_scalar[GreaterThanEqual](self, scalar)

    fn __eq__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __eq__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[Equal](self, other)

    fn __ne__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __ne__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[NotEqual](self, other)

    fn __lt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __lt__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[LessThan](self, other)

    fn __le__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __le__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[LessThanEqual](self, other)

    fn __gt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __gt__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[GreaterThan](self, other)

    fn __ge__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __ge__ → dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[GreaterThanEqual](self, other)

    fn float(self) -> Tensor[DType.float32]:
        if dtype == DType.float32:
            return rebind[Tensor[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Tensor[DType.float64]:
        if dtype == DType.float64:
            return rebind[Tensor[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Tensor[NewType]:
        new_buffer = self.buffer.to_dtype[NewType]()
        out = Tensor[NewType](
            shape=self.shape,
            buffer=new_buffer,
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
        return self.buffer.for_all[simd_width](pred).all_true()

    fn any[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.any[simd_width](pred)

    fn log[
        simd_width: Int = simdwidthof[dtype](),
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        shape = self.shape
        offset = self.offset
        numels = shape.num_elements()
        out = Tensor[dtype](shape, requires_grad=grad_required)
        if self.is_contiguous():
            if self.owns_data:
                buffer = self.buffer.log()
                memcpy(out.buffer.data, buffer.data, numels)
            else:
                buffer = Buffer[dtype](numels)
                memcpy(buffer.data, self.base[].buffer.data + offset, numels)
                buffer = buffer.log()
                memcpy(out.buffer.data, buffer.data, numels)
        else:
            for indices in self.shape:
                out[indices] = log(self[indices])

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
            for indices in self.shape:
                value1 = self[indices]
                value2 = other[indices]
                if abs(value1 - value2).gt(atol + rtol * abs(value2)):
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

    @always_inline
    fn data_buffer(self) -> Buffer[dtype]:
        return self.buffer if self.owns_data else self.base_address()[].buffer

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

    @always_inline
    fn base_address(
        ref self,
    ) -> UnsafePointer[
        Tensor[dtype],
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
        if self.owns_data and not self.base.__as_bool__():
            panic(
                "Tensor → base_address: called on owning tensor. Valid only for"
                " view type tensor"
            )
        return self.base.origin_cast[
            mut = Origin(__origin_of(self)).mut, origin = __origin_of(self)
        ]()

    fn seed_grad(mut self, with_tensor: Tensor[dtype]):
        if not self.requires_grad:
            return
        if self.shape != with_tensor.shape:
            target = "Tensor" if self.owns_data else "View"
            panic(
                target,
                "→ seed_grad: Shapes not equal -> ",
                self.shape.__str__(),
                " ≠ ",
                with_tensor.shape.__str__(),
            )
        if not self.has_grad():
            self.requires_grad_()
        if with_tensor.owns_data:
            self.gradbox[].buffer += with_tensor.buffer
        else:
            for indices in self.shape:
                self.gradbox[][indices] = with_tensor[indices]

    fn seed_grad(mut self, value: Scalar[dtype]):
        if self.has_grad():
            with_tensor = Tensor[dtype].full(self.shape, value)
            self.seed_grad(with_tensor)

    @always_inline
    fn fill(self, value: Scalar[dtype]):
        self.buffer.fill(value)

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
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(tensor.numels()):  # To be vectorized
            tensor.buffer.store(
                i,
                random_float64(
                    min.cast[DType.float64](), max.cast[DType.float64]()
                ).cast[dtype](),
            )
        return tensor

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
        shape = Shape(IntList(len(row)))
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, row._data, len(row))
        return tensor

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
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened._data, tensor.numels())
        return tensor

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
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened._data, tensor.numels())
        return tensor

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
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened._data, tensor.numels())
        return tensor

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
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened._data, tensor.numels())
        return tensor

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
        result = Tensor[dtype](Shape.Void, requires_grad=requires_grad)
        result[IntList.Empty] = val
        return result

    @staticmethod
    fn ones(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        return Self.ones(Shape(axes_spans), requires_grad)

    @staticmethod
    fn ones(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        var value: SIMD[dtype, 1]

        @parameter
        if dtype.is_floating_point():
            value = SIMD[dtype, 1](1.0)
        else:
            value = SIMD[dtype, 1](1)
        for i in range(tensor.numels()):
            tensor.buffer.store(i, value)
        return tensor

    fn sum_all(self) -> Scalar[dtype]:
        if self.owns_data:
            return self.buffer.sum()
        else:
            if self._contiguous:
                return self.base_address()[].buffer.sum(
                    self.offset, self.max_index() + self.offset + 1
                )
            else:
                summ = Scalar[dtype](0)
                for indices in self.shape:
                    summ += self[indices]
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
            # panic("Tensor → load: supported only for 2D tensors")
            # panic("Tensor → load: supported only for 2D tensors")
            pass

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
            return self.base_address()[].buffer.load[simdwidth](addr)

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

        self.buffer.store[simdwidth](
            addr, value
        ) if self.owns_data else self.base_address()[].buffer.store[simdwidth](
            addr, value
        )

    fn into_view(self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        if not self.owns_data:
            panic("Tensor → into_view: not allowed on non-owning tensor")
        shape, strides = self.shape, self.strides
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        out = Tensor[dtype](shape, self.address(), strides, 0, grad_required)

        if grad_required:
            backward_fn = ViewBackward[dtype](
                shape, strides, 0
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out

    fn view(
        ref self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        # Validate parameters and compute absolute bounds
        var _abs_min, _abs_max, abs_offset = Validator.validate_view_params(
            self, shape, strides, offset
        )

        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        base_addr = self.address() if self.owns_data else self.base.copy()

        out = Tensor[dtype](
            shape, base_addr, strides, abs_offset, grad_required
        )

        if grad_required:
            backward_fn = ViewBackward[dtype](
                shape, strides, abs_offset
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out

    fn view(
        self,
        shape: List[Int],
        strides: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.view(
            shape=Shape(shape),
            strides=Strides(strides),
            offset=offset,
            requires_grad=requires_grad,
        )

    fn view(
        self,
        *shape_dims: Int,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.view(Shape(shape_dims), offset, requires_grad)

    fn view(
        self,
        shape: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.view(
            shape=Shape(shape), offset=offset, requires_grad=requires_grad
        )

    fn view(
        self,
        shape: Shape,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.view(
            shape=shape,
            strides=Strides.default(shape),
            offset=offset,
            requires_grad=requires_grad,
        )

    fn __getitem__(self, *slices: Slice) -> Tensor[dtype]:
        # Delegate shape/strides/offset computation
        shape, strides, offset = Validator.validate_and_compute_view_metadata(
            self.shape,
            self.strides,
            slices,
        )
        return self.view(
            shape=shape,
            strides=strides,
            offset=offset,
            requires_grad=self.requires_grad,
        )

    fn set(self, tensor: Tensor[dtype], *indices: Idx):
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
                        tensor.squeeze([], requires_grad=False)
                    )[IntList.Empty]
                )
                self.buffer.store(
                    offset, elem
                ) if self.owns_data else self.base_address()[].buffer.store(
                    offset, elem
                )
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

    fn set(self, value: Scalar[dtype], *indices: Idx):
        # Compute view metadata
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape, self.strides, indices
            )
        )
        if len(shape) == 0:
            self.buffer.store(
                offset, value
            ) if self.owns_data else self.base_address()[].buffer.store(
                offset, value
            )
        else:
            sliced = self.view(shape, strides, offset, False)
            for idx in shape:
                sliced[idx] = value

    fn __getitem__(self, *indices: Idx) -> Tensor[dtype]:
        # Compute view metadata
        view_shape, view_strides, view_offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape, self.strides, indices
            )
        )

        # Handle scalar (rank-0) case
        is_scalar = len(view_shape) == 0
        shape = Shape.Void if is_scalar else view_shape
        strides = Strides.Zero if is_scalar else view_strides
        return self.view(
            shape=shape,
            strides=strides,
            offset=view_offset,
            requires_grad=self.requires_grad,
        )

    fn contiguous(self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        shape = self.shape
        offset = self.offset
        numels = self.numels()
        out = Tensor[dtype].zeros(shape, grad_required)
        if self.is_contiguous():
            if self.owns_data:
                memcpy(out.buffer.data, self.buffer.data, numels)
            else:
                memcpy(
                    out.buffer.data, self.base[].buffer.data + offset, numels
                )
        else:
            for indices in shape:
                out[indices] = self[indices]

        if grad_required:
            strides = self.strides
            backward_fn = ViewBackward[dtype](
                shape, strides, offset * 2
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out

    fn reshape[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        if self.numels() != 1:
            panic(
                "Tensor → reshape: only tensor with single element can be"
                " reshaped to scalar tensor"
            )
        return self.reshape[track_grad](
            Shape.Void, requires_grad=requires_grad, validated=True
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

    fn transpose(
        self, *axes: Int, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return self.transpose(IntList(axes), requires_grad)

    fn transpose(
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return self.transpose(IntList.new(axes))

    fn transpose(
        self, axes: IntList, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        shape = self.shape
        normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntList.range_list(shape.rank()).reversed()
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)

        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )

        base_addr = self.address() if self.owns_data else self.base.copy()
        out = Tensor[dtype](
            new_shape, base_addr, new_strides, self.offset, grad_required
        )

        if grad_required:
            out.requires_grad_(True)
            backward_fn = TransposeBackward[dtype](
                normalized_axes
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

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

        for indices in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[indices]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[indices]
            )
            result[indices] = op(self_val, other_val)

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
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)
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
            for indices in self.shape:
                self[indices] += other[indices]

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
            for indices in self.shape:
                self[indices] -= other[indices]

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
            for indices in self.shape:
                self[indices] *= other[indices]

    fn exp(self) -> Tensor[dtype]:
        if self.owns_data:
            return Tensor[dtype](
                self.shape, self.buffer.exp(), self.requires_grad
            )
        else:
            buffer = Buffer[dtype](self.numels())
            tensor = Tensor[dtype](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                tensor[indices] = exp(self[indices])
            return tensor

    fn __neg__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __neg__ is for numeric data types only",
        ]()
        if self.owns_data:
            return Tensor(self.shape, -self.buffer, self.requires_grad)
        else:
            buffer = Buffer[dtype](self.numels())
            tensor = Tensor[dtype](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                tensor[indices] = -self[indices]
            return tensor

    fn __invert__(self: Tensor[DType.bool]) -> Tensor[DType.bool]:
        if self.owns_data:
            return Tensor(self.shape, ~self.buffer, self.requires_grad)
        else:
            buffer = Buffer[DType.bool](self.numels())
            tensor = Tensor[DType.bool](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                tensor[indices] = ~self[indices]
            return tensor

    fn __abs__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __abs__ is for numeric data types only",
        ]()
        if self.owns_data:
            return Tensor(self.shape, abs(self.buffer), self.requires_grad)
        else:
            buffer = Buffer[dtype](self.numels())
            tensor = Tensor[dtype](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                tensor[indices] = abs(self[indices])
            return tensor

    fn __radd__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return AddScalar[dtype].forward[True](self, scalar)

    fn __add__(self, other: Self) -> Tensor[dtype]:
        return Adder[dtype].forward[True](self, other)

    fn __pow__(self, exponent: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __pow__ is for numeric data types only",
        ]()

        var out: Tensor[dtype]
        if self.owns_data:
            out = Tensor[dtype](
                self.shape, self.buffer**exponent, self.requires_grad
            )
        else:
            buffer = Buffer[dtype](self.numels())
            out = Tensor[dtype](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                out[indices] = self[indices] ** exponent

        if self.requires_grad:
            backward_fn = ExponientionBackward[dtype](
                exponent
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __rsub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractFromScalar[dtype].forward[True](self, scalar)

    fn __sub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractScalar[dtype].forward[True](self, scalar)

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        return Subtractor[dtype].forward[True](self, other)

    fn dot(self, other: Self, track_grad: Bool = True) -> Tensor[dtype]:
        rank1 = self.rank()
        rank2 = other.rank()
        if not rank1 == rank2 and not rank1 <= 1:
            panic("Tensor → dot: not supported for rank > 1")
        numels1 = self.numels()
        numels2 = other.numels()
        if not numels1 == numels2:
            panic(
                "Tensor → dot: size does not match",
                numels1.__str__(),
                numels2.__str__(),
            )
        var out: Tensor[dtype]
        requires_grad = (
            self.requires_grad or other.requires_grad
        ) and track_grad
        if self.owns_data and other.owns_data:
            scalar = self.buffer.dot(other.buffer)
            out = Tensor[dtype].scalar(scalar, requires_grad)
        else:
            scalar = Scalar[dtype](0)
            for idx in self.shape:
                scalar += self[idx] * other[idx]
            out = Tensor[dtype].scalar(scalar, requires_grad)

        if requires_grad:
            backward_fn = DotBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(
                TensorLite[dtype].of(self), TensorLite[dtype].of(other)
            )

        return out

    fn vector_matrix_mm(
        A: Tensor[dtype], B: Tensor[dtype], track_grad: Bool = True
    ) -> Tensor[dtype]:
        # A: (n,)(or batched: batch_A..., n)
        # B: (..., n, m)  (rank >= 2)
        if A.rank() != 1:
            panic("vector_matrix_mm: A must be rank-1 (vector)")
        if B.rank() < 2:
            panic("vector_matrix_mm: B must be rank>=2 (matrix or higher)")

        # n = contraction dim
        n = A.shape[0]
        if B.shape[-2] != n:
            panic(
                "vector_matrix_mm: incompatible shapes (A.shape[0] !="
                " B.shape[-2])"
            )

        # --- Lift A to (..., 1, n) so it matches matmul_nd's A shape of (..., m, k)
        # Start as (1, n)
        A_lifted = A.reshape(1, -1, requires_grad=False)  # shape (1, n)

        # Determine target batch_shape from B (all dims except last two)
        batch_shape = B.shape[0:-2]  # can be empty
        var A_expanded: Tensor[dtype]
        # Expand A_lifted to broadcast over B's batch dims:
        # - If batch_shape is empty, A_expanded stays (1, n)
        # - Otherwise we want shape batch_shape + [1, n]
        if len(batch_shape) > 0:
            # prepend required number of leading dims = len(batch_shape)
            A_padded = A_lifted
            intermediates = [A_padded]
            for _ in range(len(batch_shape)):
                current = intermediates[-1]
                unsqueezed = current.unsqueeze(0, requires_grad=False)
                intermediates.append(unsqueezed)
                # A_expanded = A_expanded.unsqueeze(
                #   0, requires_grad = False
                # )  # add leading dims to the front
            # Now A_expanded.shape == (1,...,1, n) ; expand to batch_shape + [1, n]
            A_last_padded = intermediates[-1].contiguous()
            A_expanded = A_last_padded.expand(
                batch_shape + [1, n], requires_grad=False
            )
        else:
            A_expanded = A_lifted  # shape (1,n)

        # --- Call matmul_nd (handles batching/broadcasting across batch_shape)
        # Note: matmul_nd expects A.shape = batch + [m, k], B.shape = batch + [k, n_out]
        # For us m == 1 and k == n
        C = Self.matmul_nd(
            A_expanded, B, track_grad=False
        )  # shape: batch_shape + [1, m_out]

        # --- Squeeze out the intermediary m==1 dimension to match PyTorch-style (batch, m_out) -> if no batch, just (m_out,)
        # m_out == B.shape[-1]
        if len(batch_shape) == 0:
            out = C.reshape([C.shape[1]], requires_grad=False)  # (m_out,)
        else:
            # remove the singular second-last dim (axis = -2)
            # we can reshape: batch_shape + [B.shape[-1]]
            out = C.reshape(batch_shape + [B.shape[-1]], requires_grad=False)

        # --- Attach autograd wrapper that routes backward to VectorMatrixMMBackward
        requires_grad = (A.requires_grad or B.requires_grad) and track_grad
        if requires_grad:
            out.requires_grad_()
            out.backwardFn = Optional(
                VectorMatrixMMBackward[dtype]().into_backward_fn()
            )
            out.add_ancestry(TensorLite[dtype].of(A), TensorLite[dtype].of(B))

        return out

    fn matrix_vector_mm(
        A: Tensor[dtype], B: Tensor[dtype], track_grad: Bool = True
    ) -> Tensor[dtype]:
        # --------------------------
        # Shapes
        # --------------------------
        # A: batch_shape + [n, m]
        # B: shape [m] or batch_shape + [m]
        # result: batch_shape + [n]
        var a_shape = A.shape
        var b_shape = B.shape

        var batch_shape = a_shape[0:-2]  # may be empty
        var n = a_shape[-2]
        var m = a_shape[-1]

        # --------------------------
        # Lift B to batch shape if needed
        # --------------------------
        var B_lifted: Tensor[dtype]
        if len(b_shape) == 1:
            # B is 1D vector -> reshape to [1, m] then expand to batch_shape + [1, m]
            var B_reshaped = B.reshape([1, m], requires_grad=False)
            if len(batch_shape) > 0:
                B_lifted = B_reshaped.expand(
                    batch_shape + [1, m], requires_grad=False
                )
            else:
                B_lifted = B_reshaped
        else:
            # B already has batch dimensions
            B_lifted = B

        # --------------------------
        # Compute result: batch matrix-vector multiplication
        # result = A @ B_lifted^T ? Actually, B_lifted is (batch_shape + [1, m]),
        # A: (batch_shape + [n, m]), so matmul_nd(A, B_lifted.T)
        # --------------------------
        B_T = (
            B_lifted.contiguous()
            .transpose(axes=[-1, -2], requires_grad=False)
            .contiguous()
        )  # shape: batch_shape + [m,1]

        out = A.matmul_nd(B_T)  # shape: batch_shape + [n,1]
        out = out.reshape(batch_shape + [n], requires_grad=False)
        requires_grad = (A.requires_grad or B.requires_grad) and track_grad
        if requires_grad:
            out.requires_grad_()

            backward_fn = MatrixVectorMMBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(A), TensorLite[dtype].of(B))

        return out

    fn __iadd__(mut self, scalar: Scalar[dtype]):
        if self.is_leaf():
            panic(
                "Tensor → Cannot perform in-place operation on a leaf tensor"
                " requiring grad."
            )

        if self.owns_data:
            self.buffer += scalar
        else:
            for indices in self.shape:
                self[indices] += scalar

    fn permute(self, axes: List[Int]) -> Tensor[dtype]:
        return self.permute(IntList.new(axes))

    fn permute(self, axes: IntList) -> Tensor[dtype]:
        if len(axes) != self.shape.rank():
            panic("Tensor → permute: number of axes must match tensor rank")

        # Check for valid permutation
        seen = IntList.with_capacity(len(axes))
        for axis in axes:
            if axis < 0 or axis >= self.shape.rank():
                panic("Tensor → permute: invalid axis index")
            if axis in seen:
                panic("Tensor → permute: duplicate axis in permutation")
            seen.append(axis)

        # Create new shape and strides
        new_shape = IntList.with_capacity(len(axes))
        new_strides = IntList.with_capacity(len(axes))
        for axis in axes:
            new_shape.append(self.shape[axis])
            new_strides.append(self.strides[axis])

        # Return new view with same base but reordered axes
        out = self.view(
            shape=Shape(new_shape),
            strides=Strides(new_strides),
            offset=self.offset,  # Permute doesn't change offset
            requires_grad=False,
        )
        out.requires_grad_(self.requires_grad)
        if self.requires_grad:
            permutation = axes.copy()
            backward_fn = PermuteBackward[dtype](permutation).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn unsqueeze(
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        """Unsqueeze multiple axes by inserting dimensions of size 1."""
        return Unsqueeze[dtype].unsqueeze(
            self, IntList.new(axes), requires_grad
        )

    fn unsqueeze(
        self, axes: IntList, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        """Unsqueeze multiple axes by inserting dimensions of size 1."""
        return Unsqueeze[dtype].unsqueeze(self, axes, requires_grad)

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

    fn shuffle(
        self,
        axis: Int = 0,
        perm: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.shuffle(axis, IntList.new(perm), requires_grad)

    fn shuffle(
        self,
        axis: Int = 0,
        perm: Optional[IntList] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Shuffle[dtype].forward(self, axis, perm, requires_grad)

    fn argmax(self, axis: Int = 0) -> Tensor[DType.int32]:
        return Argmax[dtype].argmax(tensor=self, axis=axis)

    fn argmin(self, axis: Int = 0) -> Tensor[DType.int32]:
        return Argmin[dtype].argmin(tensor=self, axis=axis)

    fn unsqueeze(
        self, axis: Int, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return Unsqueeze[dtype].unsqueeze(self, IntList(axis), requires_grad)

    fn expand(
        self: Tensor[dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Expand[dtype].forward(self, target, requires_grad)

    fn expand(
        self: Tensor[dtype],
        *target_dims: Int,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Expand[dtype].forward(self, Shape(target_dims), requires_grad)

    # Squeeze specified axes or all dims of size 1 if no axes provided
    fn squeeze(
        self,
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
        return Squeeze[dtype].squeeze(self, IntList.new(axes), requires_grad)

    # Squeeze single axis if provided, otherwise squeeze all dims of size 1
    fn squeeze(
        self,
        axis: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Squeeze[dtype].squeeze(
            self,
            IntList(axis.value()) if axis else IntList.Empty,
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
        # fn __del__(owned self):
        log_debug(
            "Tensor__del__ → deleting tensor with id: " + self.id().__str__()
        )
        if self.owns_data:
            self.buffer.free()
        if self.has_grad():
            self.gradbox[].free()
            self.gradbox.destroy_pointee()
            self.gradbox.free()
            log_debug("Tensor__del__ → freed grad")
        self.shape.free()
        self.strides.free()
        self.ancestors.free()
        _ = self^

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
        simd_width: Int = simdwidthof[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        rank_a = A.rank()
        rank_b = B.rank()
        requires_grad = A.requires_grad or B.requires_grad

        if rank_a <= 1 and rank_b <= 1:
            C = A.dot(B, track_grad=False)
            if requires_grad:
                C.requires_grad_()
                backward_fn = DotBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(backward_fn)
                C.add_ancestry(TensorLite[dtype].of(A), TensorLite[dtype].of(B))
            return C

        elif rank_a == 1 and rank_b >= 2:
            C = A.vector_matrix_mm(B, track_grad=False)
            if requires_grad:
                C.requires_grad_()
                backward_fn = VectorMatrixMMBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(backward_fn)
                C.add_ancestry(TensorLite[dtype].of(A), TensorLite[dtype].of(B))
            return C

        elif rank_a >= 2 and rank_b == 1:
            C = A.matrix_vector_mm(B, track_grad=False)
            if requires_grad:
                C.requires_grad_()
                backward_fn = MatrixVectorMMBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(backward_fn)
                C.add_ancestry(TensorLite[dtype].of(A), TensorLite[dtype].of(B))
            return C

        else:
            C = A.matmul_nd(B, track_grad=False)
            if requires_grad:
                C.requires_grad_()
                if rank_a == 2 and rank_b == 2:
                    mbfn = MatmulBackward[dtype]().into_backward_fn()
                    C.backwardFn = Optional(mbfn)
                else:
                    bmbfn = BatchedMatmulBackward[dtype]().into_backward_fn()
                    C.backwardFn = Optional(bmbfn)

                C.add_ancestry(TensorLite[dtype].of(A), TensorLite[dtype].of(B))
            return C

    @staticmethod
    fn matmul_2d[
        simd_width: Int = simdwidthof[dtype]()
    ](
        A_ptr: UnsafePointer[Tensor[dtype]],
        B_ptr: UnsafePointer[Tensor[dtype]],
        C_ptr: UnsafePointer[Tensor[dtype]] = UnsafePointer[Tensor[dtype]](),
        track_grad: Bool = True,
    ) -> Tensor[dtype]:
        A = A_ptr[]
        B = B_ptr[]

        Shape.validate_matrix_shapes_2d(A.shape, B.shape)

        rows_a = A.shape[0]
        cols_a = A.shape[1]
        cols_b = B.shape[1]
        packed = B.is_contiguous()

        C = C_ptr[] if C_ptr.__as_bool__() else Tensor[dtype].zeros(
            rows_a, cols_b
        )
        for i in range(0, rows_a):
            for j in range(0, cols_b, simd_width):
                mbatch = min(simd_width, cols_b - j)
                var accum = SIMD[dtype, simd_width](0)

                for k in range(0, cols_a):
                    scalar_a = A.load(i, k)
                    if packed and mbatch == simd_width:
                        simd_vector = B.load[simd_width](k, j)
                        accum += simd_vector * scalar_a
                    else:
                        # mbatch < simd_width or scattered B cols
                        for step in range(0, mbatch):
                            scalar_b = B.load(k, j + step)
                            accum[step] += scalar_a * scalar_b

                if mbatch == simd_width:
                    C.store[simd_width](i, j, accum)
                else:
                    for step in range(0, mbatch):
                        C.store(i, j + step, accum[step])

        requires_grad = A.requires_grad or B.requires_grad
        if requires_grad and track_grad:
            C.requires_grad = True
            C.init_gradbox()
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            C.backwardFn = Optional(backward_fn)
            C.add_ancestry(TensorLite.of(A), TensorLite.of(B))
        return C

    fn matmul_nd(
        A: Tensor[dtype], B: Tensor[dtype], track_grad: Bool = True
    ) -> Tensor[dtype]:
        Shape.validate_matrix_shapes_nd(A.shape, B.shape)
        # shapes: batch + [m, k], batch + [k, n]
        batch_shape = Shape.broadcast_shape(
            A.shape[0:-2], B.shape[0:-2]
        )  # all dims except last 2

        m = A.shape[-2]
        n = B.shape[-1]

        batch_dims_a = A.shape[:-2]
        batch_dims_b = B.shape[:-2]

        out_shape = batch_shape + [m, n]
        C = Tensor[dtype].zeros(out_shape)

        for indices in batch_shape:
            # select batch slices
            A_indices = Self.broadcasted_indices(
                indices, batch_shape, batch_dims_a
            )
            B_indices = Self.broadcasted_indices(
                indices, batch_shape, batch_dims_b
            )

            A_slice = A[il(A_indices), s(), s()]
            B_slice = B[il(B_indices), s(), s()]
            C_slice = C[il(indices), s(), s()]

            _ = Self.matmul_2d(
                UnsafePointer(to=A_slice),
                UnsafePointer(to=B_slice),
                UnsafePointer(to=C_slice),
                track_grad=False,
            )

        requires_grad = (A.requires_grad or B.requires_grad) and track_grad
        if requires_grad:
            C.requires_grad_()

            two_dim = A.rank() == 2 and B.rank() == 2

            if two_dim:
                mbfn = MatmulBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(mbfn)
            else:
                bmbfn = BatchedMatmulBackward[dtype]().into_backward_fn()
                C.backwardFn = Optional(bmbfn)

            C.add_ancestry(TensorLite.of(A), TensorLite.of(B))

        return C

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


fn main() raises:
    pass

from testing import assert_true


