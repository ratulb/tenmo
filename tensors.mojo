### Mojo Tensor
### Implement tensor library in mojo from first principles
from math import iota, exp, floor
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
        # Take of Tensor with Shape.Void
        self.buffer = Buffer[dtype](1) if shape.rank() == 0 else Buffer[dtype](
            shape.num_elements()
        )
        self.owns_data = True
        self._contiguous = False
        self._contiguous = self.is_contiguous()
        self.init_gradbox()

    fn __moveinit__(out self, var other: Self):
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
        # if not self.owns_data:
        # return False
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

    fn flatten_index(self, indices: List[Int]) -> Int:
        return self.flatten_index(IntList.new(indices))

    fn flatten_index(self, indices: VariadicList[Int]) -> Int:
        list = variadiclist_as_intlist(indices)
        return self.flatten_index(list)

    fn flatten_index(self, indices: IntList) -> Int:
        # 1. Rank check
        if len(indices) != self.rank():
            panic(
                "Tensor → flatten_index: number of indices does not match"
                " tensor rank"
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
            abort("Tensor → __getitem__: Scalar tensor expects no indices")
        index = self.flatten_index(indices)
        return self.buffer.load(
            index
        ) if self.owns_data else self.base_address()[].buffer.load(index)

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.rank() == 0:  # Tensor with Shape ()
            abort(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.flatten_index(indices)
        return self.buffer[
            index
        ] if self.owns_data else self.base_address()[].buffer[index]

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.rank() == 0:  # Tensor with Shape ()
            abort(
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
            abort("Tensor → __setitem__: Scalar tensor expects no indices")
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
            abort(
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
            abort("Tensor → rows: tensor rank is not 2")
        return self.shape[0]

    fn cols(self) -> Int:
        if not self.rank() == 2:
            abort("Tensor → cols: tensor rank is not 2")
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
                "Tensor __eq__ → Dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[Equal](self, other)

    fn __ne__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __ne__ → Dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[NotEqual](self, other)

    fn __lt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __lt__ → Dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[LessThan](self, other)

    fn __le__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __le__ → Dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[LessThanEqual](self, other)

    fn __gt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __gt__ → Dimension mismatch:",
                self.shape.__str__(),
                ",",
                other.shape.__str__(),
            )
        return Comparator.compare[GreaterThan](self, other)

    fn __ge__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            panic(
                "Tensor __ge__ → Dimension mismatch:",
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
            var ptr = UnsafePointer[TensorLite[dtype]].alloc(1)
            ptr.init_pointee_copy(parent)
            self.ancestors.append(ptr)

    fn ancestry(self) -> Ancestors[dtype]:
        return self.ancestors

    @always_inline
    fn broadcastable(self, to: Tensor[dtype]) -> Bool:
        return self.shape.broadcastable(to.shape)

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.for_all(all_truthy)

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        fn any_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.any(any_truthy)

    fn for_all[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.for_all[simd_width](pred).all_true()

    fn any[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.any[simd_width](pred)

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
    ](
        self,
        other: Self,
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Tensor → all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            abort(
                "Tensor → all_close expects same shaped tensors: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        return self.buffer.all_close[simd_width](other.buffer, rtol, atol)

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
            abort(
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
            self.init_gradbox()
        self.gradbox[].buffer += with_tensor.buffer

    fn seed_grad(mut self, value: Scalar[dtype]):
        if self.has_grad():
            with_tensor = Tensor[dtype].full(self.shape, value)
            self.seed_grad(with_tensor)

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
        shape: Shape, value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        tensor.fill(value)
        return tensor

    @staticmethod
    fn rand(
        *axes_spans: Int,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        shape = Shape(axes_spans)
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
            abort("step can not be zero")
        if (step > 0 and start >= end) or (step < 0 and start <= end):
            abort("Invalid range for the given step")
        delta = end - start
        size = floor(delta / step)
        if size <= 0:
            abort("Error: computed arange size is zero")
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
        memcpy(tensor.buffer.data, row.data, len(row))
        return tensor

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntList(len(rows), len(rows[0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                abort("Tensor → d2 → not all rows equal in length")
            flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened.data, tensor.numels())
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
                abort("Tensor → d3 → not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    abort("Tensor → d3 → not all rows equal in length")

                flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened.data, tensor.numels())
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
                abort(
                    "Tensor → d4 → not all blocks are of equal length in the"
                    " blockgrid"
                )
            for matrix in block:
                if len(matrix) != dims[2]:
                    abort(
                        "Tensor → d4 → not all matrices are of equal length"
                        " in block"
                    )
                for row in matrix:
                    if len(row) != dims[3]:
                        abort(
                            "Tensor → d4 not all rows are of equal length in"
                            " matrix"
                        )
                    flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened.data, tensor.numels())
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
                abort(
                    "Tensor → d5 → not all blocks are of equal length in the"
                    " input"
                )
            for block in blocks:
                if len(block) != dims[2]:
                    abort("Tensor → d5 → unequal block length")
                for matrix in block:
                    if len(matrix) != dims[3]:
                        abort(
                            "Tensor → d5 not all matrices are of equal length"
                            " in block"
                        )
                    for row in matrix:
                        if len(row) != dims[4]:
                            abort(
                                "Tensor → d5 not all rows are of equal length"
                                " in matrix"
                            )
                        flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.buffer.data, flattened.data, tensor.numels())
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
            abort(
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
            abort(
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
            abort("Tensor → load: supported only for 2D tensors")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + simdwidth > self.shape[1]
        ):
            abort("Tensor → load: Out-of-bounds access")
        if not self.owns_data and self.strides[1] != 1 and simdwidth > 1:
            abort(
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
            abort("Tensor → store is supported only for 2D tensors")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + simdwidth > self.shape[1]
        ):
            abort("Tensor → store: out-of-bounds access")

        if not self.owns_data and self.strides[1] != 1 and simdwidth > 1:
            abort(
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
            abort("Tensor → into_view: not allowed on non-owning tensor")
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
        # Calculate logical bounds of new view (relative to parent)
        var min_index = offset
        var max_index = offset

        for i in range(shape.rank()):
            stride = strides[i]
            if stride == 0:
                abort("Tensor → view: stride cannot be 0 in a view")
            extent = (shape[i] - 1) * stride
            if extent >= 0:
                max_index += extent
            else:
                min_index += extent  # negative stride

        # Convert to absolute coordinates (relative to base tensor)
        abs_min = self.offset + min_index
        abs_max = self.offset + max_index
        abs_offset = self.offset + offset

        # Normalize bounds (account for negative strides)
        lo = min(abs_min, abs_max)
        hi = max(abs_min, abs_max)

        # Bounds checking - PyTorch style
        if self.owns_data:
            # For root tensor, check against storage size
            if lo < 0 or hi >= self.numels():
                abort("Tensor → view: exceeds tensor's memory bounds")
        else:
            # For views, check logical range is contained in parent's logical range
            parent_lo = self.offset
            parent_hi = self.offset + self.max_index()
            if lo < parent_lo or hi > parent_hi:
                abort("Tensor → view: exceeds parent tensor's memory bounds")

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

    fn contiguous(self) -> Tensor[dtype]:
        if self.owns_data and self._contiguous:
            return self
        shape = self.shape
        out = Tensor[dtype](shape, requires_grad=self.requires_grad)
        numels = shape.num_elements()

        if self.is_contiguous():  # View is contiguous
            # Fast path: single memcpy
            src_data = self.base_address()[].buffer.data + self.offset
            memcpy(out.buffer.data, src_data, numels)
        else:
            # Slow path: iterate and copy
            if shape.rank() == 0:
                out[IntList.Empty] = self[IntList.Empty]  # Handle 0D tensors
            else:
                indices = IntList(0) * shape.rank()  # Initialize to zeros
                for _ in range(numels):
                    out[indices] = self[indices]
                    # Increment multi-dimensional index
                    for dim in reversed(range(shape.rank())):
                        indices[dim] += 1
                        if indices[dim] < shape[dim]:
                            break
                        indices[dim] = 0  # Carry to next dimension
        return out

    fn reshape(self) -> Tensor[dtype]:
        if self.numels() != 1:
            abort(
                "Only tensor with single element can be reshaped to scalar"
                " tensor"
            )
        return self.reshape(Shape.Void)

    fn reshape(self, *newdims: Int) -> Tensor[dtype]:
        if len(newdims) == 1 and newdims[0] == 0:
            return self.reshape()
        shape = Validator.validate_new_shape(
            self.shape.intlist(), IntList(newdims)
        )
        return self.reshape(shape, validated=True)

    fn reshape(self, shape: List[Int]) -> Tensor[dtype]:
        new_shape = Validator.validate_new_shape(
            self.shape.intlist(), IntList.new(shape)
        )
        return self.reshape(new_shape, validated=True)

    fn reshape(
        self, new_shape: Shape, validated: Bool = False
    ) -> Tensor[dtype]:
        shape = new_shape if validated else Validator.validate_new_shape(
            self.shape.intlist(), new_shape.intlist()
        )
        if self.numels() != shape.num_elements():
            abort(
                "Tensor with "
                + String(self.numels())
                + " element(s) can't be converted to a tensor containing "
                + String(shape.num_elements())
                + " element(s)"
            )

        requires_grad = self.requires_grad
        buffer = self.buffer if self.owns_data else self.base_address()[].buffer
        out = Tensor[dtype](shape, buffer, requires_grad)

        if requires_grad:
            backward_fn = ReshapeBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out

    fn transpose(
        self, *axes: Int, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        transpose_axes = IntList.with_capacity(len(axes))
        for axis in transpose_axes:
            transpose_axes.append(axis)
        return self.transpose(transpose_axes, requires_grad)

    fn transpose(
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return self.transpose(IntList.new(axes))

    fn transpose(
        self, axes: IntList, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        shape = self.shape
        var normalized_axes = Validator.validate_axes(
            axes if len(axes)
            > 0 else IntList.range_list(shape.rank()).reversed(),
            shape,
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)
        out = self.view(
            new_shape,
            new_strides,
            offset=0,
            requires_grad=False,
        )
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
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
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

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
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)
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

    fn sum(self, axes: List[Int] = [], keepdims: Bool = False) -> Tensor[dtype]:
        return self.sum(IntList.new(axes), keepdims)

    fn sum(
        self,
        axes: IntList,
        keepdims: Bool = False,
        track_grad: Bool = True,
    ) -> Tensor[dtype]:
        shape = self.shape
        rank = shape.rank()
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        out_shape = compute_output_shape(shape, normalized_axes, keepdims)
        out = Tensor[dtype].zeros(
            out_shape, requires_grad=self.requires_grad and track_grad
        )

        if out_shape == Shape.Void:
            if rank == 0:  # Scalar case
                out[IntList.Empty] = self[IntList.Empty]
            elif rank == len(normalized_axes) and not keepdims:  # Reducing all
                out[IntList.Empty] = self.sum_all()
        else:
            reduced_shape = Shape(shape.axes_spans.select(normalized_axes))
            for out_idx in out_shape:
                var summ = Scalar[dtype](0)
                for red_idx in reduced_shape:
                    full_idx = out_idx.replace(
                        normalized_axes, red_idx
                    ) if keepdims else out_idx.insert(normalized_axes, red_idx)
                    summ += self[full_idx]
                out[out_idx] = summ

        if self.requires_grad and track_grad:
            out.requires_grad_(True)
            backward_fn = SumBackward[dtype](
                normalized_axes.copy(), keepdims
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn mean(
        self, axes: List[Int] = [], keepdims: Bool = False
    ) -> Tensor[dtype]:
        return self.mean(IntList.new(axes), keepdims)

    fn mean(
        self: Tensor[dtype], axes: IntList, keepdims: Bool = False
    ) -> Tensor[dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            self.shape, axes
        )
        count = self.shape.axes_spans.select(normalized_axes).product()
        out = self.sum(normalized_axes, keepdims, track_grad=False) / Scalar[
            dtype
        ](count)

        if self.requires_grad:
            out.requires_grad_(True)
            backward_fn = MeanBackward[dtype](
                normalized_axes.copy(), keepdims
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __rtruediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __rtruediv__ is for numeric data types only",
        ]()

        buffer = scalar / (
            self.buffer if self.owns_data else self.base_address()[].buffer
        )
        out = Tensor[dtype](self.shape, buffer, self.requires_grad)

        if self.requires_grad:
            backward_fn = RightTrueDivBackwardScalar[dtype](
                scalar
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __truediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __truediv__ is for numeric data types only",
        ]()

        if scalar == Scalar[dtype](0):
            abort("Tensor → __truediv__ : canot divide by " + scalar.__str__())

        buffer = (
            self.buffer if self.owns_data else self.base_address()[].buffer
        ) / scalar
        out = Tensor[dtype](self.shape, buffer, self.requires_grad)

        if self.requires_grad:
            backward_fn = TrueDivBackwardScalar[dtype](
                scalar
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, factor: Scalar[dtype]) -> Tensor[dtype]:
        buffer = (
            self.buffer if self.owns_data else self.base_address()[].buffer
        ) * factor
        requires_grad = self.requires_grad
        out = Tensor[dtype](self.shape, buffer, requires_grad=requires_grad)

        if requires_grad:
            backward_fn = MulBackwardScalar[dtype](factor).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__mul__(self * other) → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_mul_operation(other)
        buffer = (
            self.buffer if self.owns_data else self.base_address()[].buffer
        ) * (other.buffer if other.owns_data else other.base_address()[].buffer)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor[dtype](self.shape, buffer, requires_grad=requires_grad)

        if requires_grad:
            backward_fn = MultiplyBackward[dtype]().into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self), TensorLite.of(other))

        return out

    fn broadcast_mul_operation(
        self: Self,
        other: Self,
    ) -> Tensor[dtype]:
        out = self.broadcast_op(other, scalar_ops[dtype, Multiply])
        requires_grad = self.requires_grad or other.requires_grad
        if requires_grad:
            backward_fn = BroadcastBackward[
                dtype, AddTensor, AddTensor, True
            ]().into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self), TensorLite.of(other))

        return out

    fn update_grad[opcode: Int](mut self, incoming: Tensor[dtype]):
        if opcode == MulTensor:
            self.gradbox[].__imul__(incoming)
        if opcode == AddTensor:
            self.gradbox[].__iadd__(incoming)
        if opcode == SubtractTensor:
            self.gradbox[].__isub__(incoming)

    fn __iadd__(mut self, other: Self):
        if self.is_leaf():
            abort(
                "Tensor → __iadd__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            abort(
                "Tensor → __iadd__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        if self.owns_data:
            self.buffer.__iadd__(
                other.buffer if other.owns_data else other.base_address()[].buffer
            )
        else:
            self.base_address()[].buffer.__iadd__(
                other.buffer if other.owns_data else other.base_address()[].buffer
            )

    fn __isub__(mut self, other: Self):
        if self.is_leaf():
            abort(
                "Tensor → __isub__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            abort(
                "Tensor → __isub__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        if self.owns_data:
            self.buffer -= (
                other.buffer if other.owns_data else other.base_address()[].buffer
            )
        else:
            self.base_address()[].buffer -= (
                other.buffer if other.owns_data else other.base_address()[].buffer
            )

    fn __imul__(mut self, other: Self):
        if self.is_leaf():
            abort(
                "Tensor → __imul__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            abort(
                "Tensor → __imul__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        if self.owns_data:
            self.buffer.__imul__(
                other.buffer if other.owns_data else other.base_address()[].buffer
            )
        else:
            self.base_address()[].buffer.__imul__(
                other.buffer if other.owns_data else other.base_address()[].buffer
            )

    fn exp(self) -> Tensor[dtype]:
        return Tensor[dtype](
            self.shape,
            (
                self.buffer if self.owns_data else self.base_address()[].buffer
            ).exp(),
            self.requires_grad,
        )

    fn __neg__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __neg__ is for numeric data types only",
        ]()

        return Tensor[dtype](
            self.shape,
            -(self.buffer if self.owns_data else self.base_address()[].buffer),
            self.requires_grad,
        )

    fn __invert__(self: Tensor[DType.bool]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.shape,
            ~(self.buffer if self.owns_data else self.base_address()[].buffer),
        )

    fn __abs__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __abs__ is for numeric data types only",
        ]()

        return Tensor[dtype](
            self.shape,
            abs(
                self.buffer if self.owns_data else self.base_address()[].buffer
            ),
            self.requires_grad,
        )

    fn __radd__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        out = Tensor[dtype](
            self.shape,
            (self.buffer if self.owns_data else self.base_address()[].buffer)
            + scalar,
            self.requires_grad,
        )

        if self.requires_grad:
            backward_fn = AddBackwardScalar[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out

    fn __pow__(self, exponent: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __pow__ is for numeric data types only",
        ]()

        out = Tensor[dtype](
            self.shape,
            (self.buffer if self.owns_data else self.base_address()[].buffer)
            ** exponent,
            self.requires_grad,
        )

        if self.requires_grad:
            backward_fn = ExponientionBackward[dtype](
                exponent
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __rsub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        out = Tensor[dtype](
            self.shape,
            scalar
            - (self.buffer if self.owns_data else self.base_address()[].buffer),
            self.requires_grad,
        )
        if self.requires_grad:
            backward_fn = SubLeftRightBackwardScalar[dtype](
                True
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __sub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        out = Tensor[dtype](
            self.shape,
            (self.buffer if self.owns_data else self.base_address()[].buffer)
            - scalar,
            self.requires_grad,
        )

        if self.requires_grad:
            backward_fn = SubLeftRightBackwardScalar[dtype](
                False
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__sub__ → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_add_subtract_operation[
                Subtract, AddTensor, SubtractTensor
            ](other)

        lhs_buffer = (
            self.buffer if self.owns_data else self.base_address()[].buffer
        )
        rhs_buffer = (
            other.buffer if other.owns_data else other.base_address()[].buffer
        )

        buffer = lhs_buffer - rhs_buffer
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor[dtype](self.shape, buffer, requires_grad)

        if requires_grad:
            sub_backward = SubBackward[dtype]()
            if self.requires_grad:
                out.add_ancestry(TensorLite.of(self))
                sub_backward.negate(False)
            if other.requires_grad:
                out.add_ancestry(TensorLite.of(other))
                sub_backward.negate(True)
            backward_fn = sub_backward.into_backward_fn()
            out.backwardFn = Optional(backward_fn)

        return out

    fn __add__(self, other: Self) -> Tensor[dtype]:
        if self.address() == other.address():
            return self.__mul__(2)
        if not self.broadcastable(other):
            abort(
                "__add__ → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_add_subtract_operation[
                Add, AddTensor, AddTensor
            ](
                other,
            )

        var buffer: Buffer[dtype]

        if self.owns_data and other.owns_data:
            buffer = self.buffer + other.buffer
        elif self.owns_data and not other.owns_data:
            buffer = self.buffer + other.base_address()[].buffer
        elif not self.owns_data and not other.owns_data:
            buffer = (
                self.base_address()[].buffer + other.base_address()[].buffer
            )
        else:
            buffer = self.base_address()[].buffer + other.buffer

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor[dtype](self.shape, buffer, requires_grad)

        if requires_grad:
            backward_fn = AddBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            if self.requires_grad:
                out.add_ancestry(TensorLite[dtype].of(self))
            if other.requires_grad:
                out.add_ancestry(TensorLite[dtype].of(other))

        return out

    fn __iadd__(mut self, scalar: Scalar[dtype]):
        if self.is_leaf():
            abort(
                "Cannot perform in-place operation on a leaf tensor requiring"
                " grad."
            )

        if self.owns_data:
            self.buffer += scalar
        else:
            self.base_address()[].buffer += scalar

    fn broadcast_add_subtract_operation[
        Element_Wise_Op: Int, Tensor_Op_First: Int, Tensor_Op_Second: Int
    ](self, other: Self) -> Tensor[dtype]:
        var out = self.broadcast_op(other, scalar_ops[dtype, Element_Wise_Op])
        if self.requires_grad or other.requires_grad:
            backward_fn = BroadcastBackward[
                dtype, Tensor_Op_First, Tensor_Op_Second, False
            ]().into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self), TensorLite.of(other))
            pass

        return out

    fn permute(self, axes: List[Int]) -> Tensor[dtype]:
        return self.permute(IntList.new(axes))

    fn permute(self, axes: IntList) -> Tensor[dtype]:
        if len(axes) != self.shape.rank():
            abort("Tensor → permute: number of axes must match tensor rank")

        # Check for valid permutation
        seen = IntList.with_capacity(len(axes))
        for axis in axes:
            if axis < 0 or axis >= self.shape.rank():
                abort("Tensor → permute: invalid axis index")
            if axis in seen:
                abort("Tensor → permute: duplicate axis in permutation")
            seen.append(axis)

        seen.free()

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
        )
        if self.requires_grad:
            permutation = axes.copy()
            backward_fn = PermuteBackward[dtype](permutation).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out

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

        if current_dim >= self.rank():
            print(
                "ERROR: current_dim (",
                current_dim,
                ") >= ndim (",
                self.rank(),
                ")",
            )
            return

        size = self.shape[current_dim]

        if size < 0 or size > 1_000_000:
            print(
                "ERROR: suspicious size: ",
                size,
                "at dim ",
                current_dim,
                self.shape.__str__(),
            )
            return

        # Base case: last dimension (print actual elements)
        if current_dim == self.rank() - 1:
            print(indent + "[", end="")

            for i in range(size):
                if i < num_first:
                    indices.append(i)
                    print(self[indices], end="")
                    _ = indices.pop()
                    if i != size - 1:
                        print(", ", end="")
                elif i == num_first and size > num_first + num_last:
                    print("..., ", end="")
                elif i >= size - num_last:
                    indices.append(i)
                    print(self[indices], end="")
                    _ = indices.pop()
                    if i != size - 1:
                        print(", ", end="")

            print("]", end="")

        else:
            print(indent + "[")
            for i in range(size):
                if i < num_first:
                    indices.append(i)
                    self.print_tensor_recursive(
                        indices, level + 1, num_first, num_last
                    )
                    _ = indices.pop()
                elif i == num_first and size > num_first + num_last:
                    print(indent + "  ...,")
                elif i >= size - num_last:
                    indices.append(i)
                    self.print_tensor_recursive(
                        indices, level + 1, num_first, num_last
                    )
                    _ = indices.pop()

                # Print comma and newline for all but last element
                if i != size - 1 and (i < num_first or i >= size - num_last):
                    print(",")
                # Special case: last element needs newline before closing bracket
                elif i == size - 1:
                    print()  # Newline before closing bracket

            print(indent + "]", end="")

    fn print(self, num_first: Int = 5, num_last: Int = 1):
        print(
            self.__str__(),
            end="\n",
        )
        empty = IntList()
        self.print_tensor_recursive(
            empty, 1, num_first=num_first, num_last=num_last
        )

    # Always use this to print grad to avoid surprises of segmentation fault!
    fn gprint(self, num_first: Int = 10, num_last: Int = 10):
        if not self.requires_grad:
            print("Tensor is non-differentiable")
        elif self.requires_grad and not self.has_grad():
            print("Requires grad but grad not initialized")
        else:
            self.gradients()[].print(num_first, num_last)

    fn free(owned self):
        # fn __del__(owned self):
        log_debug(
            "Tensor__del__ → deleting tensor with id: " + self.id().__str__()
        )
        log_debug("Tensor owns data? " + self.owns_data.__str__())
        self.buffer.free()
        self.shape.free()
        self.strides.free()
        self.ancestors.free()
        if self.has_grad():
            for i in range(self.numels()):
                (self.gradbox + i).destroy_pointee()
            self.gradbox.free()
            log_debug("Tensor__del__ → freed grad")
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
    ](A: Tensor[dtype], B: UnsafePointer[Tensor[dtype]]) -> Tensor[dtype]:
        return A.matmul[simd_width](B[])

    fn matmul[
        simd_width: Int = simdwidthof[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        Validator.validate_matrix_shapes(A, B)
        rows_a = A.shape[0]
        cols_a = A.shape[1]
        cols_b = B.shape[1]
        packed = B.is_contiguous()

        C = Tensor[dtype].zeros(rows_a, cols_b)
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
        if requires_grad:
            C.requires_grad = True
            C.init_gradbox()
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            C.backwardFn = Optional(backward_fn)
            C.add_ancestry(TensorLite.of(A))
            C.add_ancestry(TensorLite.of(B))
        return C

    fn _matmul[
        simd_width: Int = simdwidthof[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        Validator.validate_matrix_shapes(A, B)

        m = A.rows()  # rows of A / C
        kN = A.cols()  # cols of A == rows of B
        p = B.cols()  # cols of B / C

        # We assume C is freshly allocated -> contiguous in last dim.
        C = Tensor[dtype].zeros(m, p)

        # We only care if B is contiguous **in its last dimension** (columns),
        # because we vectorize over j. No need to care about A's contiguity.
        b_lastdim_contig = B.is_contiguous()

        for i in range(m):
            for k in range(kN):
                a_val = A.load(i, k)
                # print("a_val: ", a_val, i, k)
                if b_lastdim_contig:
                    # -------- FAST PATH: B[k, j..] is contiguous => true SIMD loads
                    @parameter
                    fn muladd_fast[width: Int](j0: Int):
                        # base offsets for this block
                        b_off = B.offset + k * B.strides[0] + j0 * B.strides[1]
                        c_off = C.offset + i * C.strides[0] + j0 * C.strides[1]

                        # vector load from B, broadcast a, fused update into C
                        vb = B.buffer.load[simdwidth=width](b_off)
                        va = SIMD[dtype, width](a_val)
                        # print(b_off, c_off, width, vb, va)
                        # print(va)
                        vc = C.buffer.load[simdwidth=width](c_off)
                        C.buffer.store[simdwidth=width](c_off, vc + va * vb)

                    vectorize[muladd_fast, simd_width](p)

                else:
                    # -------- SLOW PATH: arbitrary B layout => gather per lane
                    # Still chunk by `width` with vectorize (for unrolling + tail)
                    @parameter
                    fn muladd_gather[width: Int](j0: Int):
                        # Update C block lane-by-lane using scalar B loads.
                        # This keeps correctness without forcing a copy.
                        for lane in range(width):
                            j = j0 + lane
                            if j >= p:
                                break
                            b_elt = B.load(k, j)  # stride-aware scalar
                            c_old = C.load(
                                i, j
                            )  # stride-aware scalar (contiguous by design)
                            C.store(i, j, c_old + a_val * b_elt)

                    vectorize[muladd_gather, simd_width](p)

        requires_grad = A.requires_grad or B.requires_grad
        if requires_grad:
            C.requires_grad = True
            C.init_gradbox()
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            C.backwardFn = Optional(backward_fn)
            C.add_ancestry(TensorLite.of(A), TensorLite.of(B))

        return C


from testing import assert_true


fn main() raises:
    pass
