### Mojo Tensor
### Implement tensor library in mojo from first principles
from math import exp, floor
from random import seed, random_float64
from sys import simd_width_of
from utils.numerics import max_finite, min_finite
from memory import memcpy, memset, memset_zero, ArcPointer
from shapes import Shape, ShapeIndexIterator
from intlist import IntList
from ancestry import Ancestors, Ancestor
from strides import Strides
from common_utils_imports import *
from common_utils import id as identity, IntArrayHelper
from operators_imports import *

# from walkback import *
from backpropagation import BackwardFn
from forwards import (
    AddScalar,
    Adder,
    Reshape,
    MultiplyScalar,
    Multiplicator,
    SubtractFromScalar,
    SubtractScalar,
    Subtractor,
    DivideByScalar,
    DivideScalar,
    Divider,
    Exponentiator,
    Summer,
    Mean,
    View,
    Contiguous,
    Transpose,
    Dot,
    Expand,
    Flatten,
    Squeeze,
    Unsqueeze,
    Permute,
    Argmax,
    Argmin,
    Shuffle,
)
from buffers import Buffer
from validators import Validator
from collections import Set
from gradbox import Gradbox
from layout.int_tuple import IntArray
from broadcasthelper import ShapeBroadcaster
from ndbuffer import NDBuffer
from utilities import Utils


struct Tensor[dtype: DType = DType.float32](
    Copyable
    & Movable
    & Sized
    & Stringable
    & Representable
    & Writable
    & Absable
    & EqualityComparable
):
    alias Row = List[Scalar[dtype]]
    alias Rows = List[Self.Row]
    alias Block = List[Self.Rows]
    alias Blocks = List[Self.Block]

    alias randint = Tensor[DType.int64].randn
    alias randfloat = Tensor[DType.float64].randn
    alias randint32 = Tensor[DType.int32].randn
    alias randfloat32 = Tensor[DType.float32].randn

    var buffer: NDBuffer[dtype]
    var requires_grad: Bool
    var gradbox: Optional[Gradbox[dtype]]
    var ancestors: Optional[Ancestors[dtype]]
    var backwardFn: Optional[BackwardFn[dtype]]

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        self.buffer = NDBuffer[dtype](shape)
        self.requires_grad = requires_grad
        self.gradbox = None
        self.ancestors = None
        self.backwardFn = None
        self.init_gradbox()

    fn __init__(
        out self,
        shape: Shape,
        ptr: UnsafePointer[Scalar[dtype]],
        requires_grad: Bool = False,
        *,
        copy: Bool = True,
    ):
        self.buffer = NDBuffer[dtype](
            Buffer[dtype](shape.num_elements(), ptr, copy=copy), shape
        )
        self.requires_grad = requires_grad
        self.gradbox = None
        self.ancestors = None
        self.backwardFn = None
        self.init_gradbox()

    fn __init__(
        out self,
        var buffer: NDBuffer[dtype],
        requires_grad: Bool = False,
    ):
        self.buffer = buffer^
        self.requires_grad = requires_grad
        self.gradbox = None
        self.ancestors = None
        self.backwardFn = None
        self.init_gradbox()

    @staticmethod
    fn build_view(
        source: UnsafePointer[Self],
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        ref src_buffer = source[].buffer

        buffer = src_buffer.share(shape, strides, offset)
        return Tensor[dtype](buffer=buffer^, requires_grad=requires_grad)

    fn as_gradbox(self, share: Bool = False) -> Gradbox[dtype]:
        return Gradbox[dtype](self.buffer.contiguous(), share=share)

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox^
        self.ancestors = other.ancestors^
        self.backwardFn = other.backwardFn^

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox.copy()
        self.ancestors = other.ancestors.copy()
        self.backwardFn = other.backwardFn.copy()

    @always_inline
    fn id(self) -> Int:
        return identity(self)

    @always_inline
    fn init_gradbox(mut self):
        if self.requires_grad and self.gradbox == None:
            gradbox = Gradbox[dtype](self.shape())
            gradbox.zero_grad()
            self.gradbox = Optional(gradbox^)

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.buffer.is_contiguous()

    @always_inline
    fn shared(self) -> Bool:
        return self.buffer.shared()

    fn is_leaf(self) -> Bool:
        return self.requires_grad and not self.has_backward_fn()

    @always_inline
    fn __len__(self) -> Int:
        return self.shape()[0] if self.shape() != Shape() else 0

    @always_inline
    fn shape(self) -> Shape:
        return self.buffer.shape

    @always_inline
    fn strides(self) -> Strides:
        return self.buffer.strides

    @always_inline
    fn offset(self) -> Int:
        return self.buffer.offset

    @always_inline
    fn numels(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn rank(self) -> Int:
        return self.buffer.rank()

    @always_inline
    fn max_index(self) -> Int:
        return self.buffer.max_index()

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __getitem__(List[Int]): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __getitem__(IntList): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]


    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        if self.rank() == 0 and indices.size() != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __getitem__(IntArray): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.rank() == 0:  # Tensor with Shape ()
            panic(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(List[Int])"
            )

        return self.buffer[indices]

    fn __getitem__(self, *slices: Slice) -> Tensor[dtype]:
        # Delegate shape/strides/offset computation
        shape, strides, offset = Validator.validate_and_compute_view_metadata(
            self.shape(),
            self.strides(),
            slices,
        )
        return View[dtype].forward[track_grad=True](
            self,
            shape=shape,
            strides=strides,
            offset=offset,
            requires_grad=self.requires_grad,
            validated=True,
        )

    fn __getitem__(self, *indices: Idx) -> Tensor[dtype]:
        # Compute view metadata
        view_shape, view_strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape(), self.strides(), indices
            )
        )

        # Handle scalar (rank-0) case
        is_scalar = len(view_shape) == 0
        shape = Shape() if is_scalar else view_shape
        strides = Strides() if is_scalar else view_strides
        return View[dtype].forward[track_grad=True](
            self,
            shape,
            strides,
            offset,
            self.requires_grad,
            True,
        )

    @always_inline
    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.rank() == 0:  # Tensor with Shape()
            panic(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(List[Int])"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __setitem__(List[Int]): Scalar tensor expects no"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __setitem__(IntList): Scalar tensor expects no"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        if self.rank() == 0 and indices.size() != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __setitem__(IntArray): Scalar tensor expects no"
                " indices"
            )

        self.buffer[indices] = value

    fn item(self) -> Scalar[dtype]:
        return self.buffer.item()

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
        s += self.shape().__str__()
        if self.buffer.shared():
            s += ", strides: " + self.strides().__str__()
            s += ", offset: " + self.offset().__str__()
        s += ", Type: " + dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    # Check if it has a backward fn before calling this API
    @always_inline
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self.backwardFn.value().copy()

    @always_inline
    fn has_backward_fn(self) -> Bool:
        return self.backwardFn is not None

    @always_inline
    fn has_grad(self) -> Bool:
        return self.gradbox != None

    @always_inline
    fn zero_grad(self):
        if self.requires_grad and self.has_grad():
            self.gradbox.value().zero_grad()

    @always_inline
    fn grad(ref self) -> ref [origin_of(self.gradbox.value())] Gradbox[dtype]:
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad or grad not initialized"
            )
        return self.gradbox.value()

    @always_inline
    fn unshared_grad(self) -> Gradbox[dtype]:
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad or grad not initialized"
            )
        return self.gradbox.value().unshared()

    fn rows(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → rows: tensor rank is not 2")
        return self.shape()[0]

    fn cols(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → cols: tensor rank is not 2")
        return self.shape()[1]

    fn is_scalar(self) -> Bool:
        return self.buffer.is_scalar()

    fn __eq__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[Equal](scalar))

    fn __ne__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[NotEqual](scalar))

    fn __lt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[LessThan](scalar))

    fn __le__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[LessThanEqual](scalar)
        )

    fn __gt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThan](scalar)
        )

    fn __ge__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThanEqual](scalar)
        )

    fn __eq__(self, other: Tensor[dtype]) -> Bool:
        return self.eq(other).all_true()

    fn eq(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[Equal](other.buffer))

    fn __ne__(self, other: Tensor[dtype]) -> Bool:
        return self.ne(other).all_true()

    fn ne(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[NotEqual](other.buffer))

    fn __lt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[LessThan](other.buffer))

    fn __le__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[LessThanEqual](other.buffer)
        )

    fn __gt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[GreaterThan](other.buffer)
        )

    fn __ge__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[GreaterThanEqual](other.buffer)
        )

    fn float(self) -> Tensor[DType.float32]:
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Tensor[DType.float64]:
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Tensor[NewType]:
        var new_type_buffer = self.buffer.to_dtype[NewType]()
        return Tensor[NewType](new_type_buffer^)

    @always_inline
    fn add_ancestry(mut self, *parents: Tensor[dtype]):
        # Initialize ancestors if needed
        if not self.ancestors:
            self.ancestors = Optional(Ancestors[dtype].untracked())

        ref ancestors = self.ancestors.value()
        for parent in parents:
            ancestors.append(Ancestor[dtype](parent.copy()))

    fn has_ancestry(self) -> Bool:
        return self.ancestors != None

    @always_inline
    fn ancestry(ref self) -> ref [self.ancestors.value()] Ancestors[dtype]:
        if self.ancestors == None:
            panic("Tensor → ancestry: ancestry not initialized")
        return self.ancestors.value()

    @always_inline
    fn broadcastable(self, to: Tensor[dtype]) -> Bool:
        return ShapeBroadcaster.broadcastable(self.shape(), to.shape())

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        fn all_truthy(elem: Scalar[DType.bool]) -> Bool:
            return elem == Scalar[DType.bool](True)

        return self.for_all(all_truthy)

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        fn any_truthy(elem: Scalar[DType.bool]) -> Bool:
            return elem == Scalar[DType.bool](True)

        return self.any(any_truthy)

    fn for_all[
        simd_width: Int = simd_width_of[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return (
            self.buffer.contiguous_buffer().for_all[simd_width](pred).all_true()
        )

    fn any[
        simd_width: Int = simd_width_of[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.contiguous_buffer().any[simd_width](pred)

    fn log(self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )

        buffer = self.buffer.map[
            Utils[dtype].log_buffer, Utils[dtype].log_scalar
        ]()
        nd_buffer = NDBuffer[dtype](buffer^, self.shape())
        return Tensor[dtype](nd_buffer^, requires_grad=grad_required)

    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self,) -> Bool:
        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    fn address(self) -> UnsafePointer[Self,]:
        return UnsafePointer(to=self)

    fn seed_grad(mut self, with_tensor: Tensor[dtype]):
        if not self.requires_grad:
            return
        if not self.has_grad():
            self.requires_grad_()
        self.gradbox.value().seed_grad(with_tensor)

    fn seed_grad(mut self, value: Scalar[dtype]):
        with_tensor = Tensor[dtype].full(self.shape(), value)
        self.seed_grad(with_tensor)

    @always_inline
    fn fill(self, value: Scalar[dtype]):
        self.buffer.fill(value)

    @staticmethod
    fn full_like(
        like: Tensor[dtype], value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        shape = like.shape()
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
        return Tensor[dtype](
            NDBuffer[dtype].full(shape, value), requires_grad=requires_grad
        )

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

        nd_buffer = NDBuffer[dtype](buffer^, shape)

        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

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

        nd_buffer = NDBuffer[dtype](buffer^, Shape([count]))
        tensor = Tensor[dtype](nd_buffer^, requires_grad=requires_grad)
        return tensor^

    @staticmethod
    fn zeros(
        axes_spans: List[Int], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        return Self.zeros(Shape(axes_spans), requires_grad=requires_grad)

    @staticmethod
    fn zeros(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        shape = Shape(axes_spans)
        return Self.zeros(shape, requires_grad=requires_grad)

    @staticmethod
    fn zeros_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        shape = tensor.shape()
        buffer = Buffer[dtype].zeros(shape.num_elements())
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn ones_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype].full(tensor.shape(), 1, requires_grad=requires_grad)
        return out^

    @staticmethod
    fn zeros(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        buffer = Buffer[dtype].zeros(shape.num_elements())
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    fn onehot(self: Tensor[DType.int64], num_classes: Int) -> Tensor[dtype]:
        """Convert tensor of class indices to one-hot encoding.
        Args:
            self: Tensor of shape (...,) containing class indices.
            num_classes: Number of classes.
        Returns: Tensor of shape (..., num_classes).
        """
        shape = self.shape()
        result = Tensor[dtype](shape + [num_classes])

        result.fill(Scalar[dtype](0))

        # Set appropriate positions to 1.0
        for idx in shape:
            var class_idx = self[idx].__int__()
            if class_idx < 0 or class_idx >= num_classes:
                panic(
                    "Tensor → onehot: invalid class at coordinate: ",
                )
            if class_idx >= 0 and class_idx < num_classes:
                # var one_hot_idx = idx + [class_idx]
                var one_hot_idx = IntArrayHelper.extend(idx, class_idx)
                result[one_hot_idx] = Scalar[dtype](1)

        return result^

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
        memcpy(dest=buffer.data, src=row._data, count=numels)
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntList(len(rows), len(rows[0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                panic("Tensor → d2 → not all rows equal in length")
            flattened.extend(row.copy())
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(dest=buffer.data, src=flattened._data, count=numels)
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad)

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

                flattened.extend(row.copy())
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(dest=buffer.data, src=flattened._data, count=numels)
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

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
                    flattened.extend(row.copy())
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(dest=buffer.data, src=flattened._data, count=numels)
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

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
                        flattened.extend(row.copy())
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        memcpy(dest=buffer.data, src=flattened._data, count=numels)
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn of(*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "of(*elems)")
        shape = Shape(IntList(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor^

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
        return tensor^

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
        return tensor^

    @staticmethod
    fn scalar(val: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        result = Tensor[dtype](Shape(), requires_grad=requires_grad)
        result[IntArray()] = val
        return result^

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
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    fn broadcast_to(
        self, target_shape: Shape, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Tensor → broadcast_to: shape "
                + self.shape().__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        out = Tensor[dtype](broadcasted_buffer^, requires_grad=grad_required)
        return out^

    @always_inline
    fn load[
        simdwidth: Int = 1
    ](self, row: Int, col: Int) -> SIMD[dtype, simdwidth]:
        constrained[
            simdwidth.is_power_of_two(),
            "Tensor → load: SIMD width (simdwidth) must be a power of 2",
        ]()
        rank = self.rank()
        shape = self.shape()
        if rank != 2:
            panic("Tensor → load: supported only for 2D tensors")

        if row < 0 or row >= shape[0] or col < 0 or col + simdwidth > shape[1]:
            panic("Tensor → load: Out-of-bounds access")

        strides = self.strides()
        offset = self.offset()

        if not self.is_contiguous() and strides[1] != 1 and simdwidth > 1:
            panic(
                "Tensor → SIMD load attempted on non-contiguous Tensor - only"
                " single-element loads are permitted for non-contiguous tensor"
            )

        addr = row * strides[0] + col * strides[1] + offset
        return self.buffer.data().load[simdwidth](addr)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, row: Int, col: Int, value: SIMD[dtype, simdwidth]):
        constrained[
            simdwidth.is_power_of_two(),
            "Tensor → store: SIMD width (simdwidth) must be a power of 2",
        ]()
        rank = self.rank()
        shape = self.shape()
        if rank != 2:
            panic("Tensor → store is supported only for 2D tensors")

        if row < 0 or row >= shape[0] or col < 0 or col + simdwidth > shape[1]:
            panic("Tensor → store: out-of-bounds access")

        strides = self.strides()
        offset = self.offset()

        if not self.is_contiguous() and strides[1] != 1 and simdwidth > 1:
            panic(
                "Tensor → SIMD store attempted on non-contiguous Tensor - only"
                " single-element stores are permitted for non-contiguous tensor"
            )

        addr = row * strides[0] + col * strides[1] + offset
        self.buffer.data().store[simdwidth](addr, value)

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

        _ = """fn repeat[
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
        return Tile.forward[track_grad](self, repeat, requires_grad)"""

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
            Shape(), requires_grad=requires_grad, validated=True
        )

    fn reshape[
        track_grad: Bool = True
    ](self, *newdims: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        if len(newdims) == 1 and newdims[0] == 0:
            return self.reshape[track_grad](requires_grad=requires_grad)
        shape = Validator.validate_and_construct_new_shape(
            self.shape(), IntList(newdims)
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
            self.shape(), IntList.new(shape)
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

    fn upstream_grad_share[
        augment: Bool
    ](
        self,
        other: Tensor[dtype],
        upstream_grad: Gradbox[dtype],
    ) -> Gradbox[
        dtype
    ]:
        var grad_contrib: Gradbox[dtype]
        if upstream_grad.shape() == Shape():
            grad_contrib = Gradbox[dtype].full(
                self.shape(), upstream_grad.item(), share=False
            )
        else:

            @parameter
            if augment:
                grad_contrib = upstream_grad * other
            else:
                grad_contrib = upstream_grad.copy()

            if grad_contrib.shape() != self.shape():
                axes = IntList(
                    ShapeBroadcaster.broadcast_mask(
                        self.shape(), grad_contrib.shape()
                    )
                ).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape() != self.shape():
                grad_contrib = grad_contrib.reshape(self.shape())

        return grad_contrib^

        _ = """fn broadcast_op(
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
        if upstream_grad.shape == Shape():
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

        return grad_contrib"""

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

    fn update_grad[opcode: Int](self, incoming: Gradbox[dtype]):
        if opcode == MulTensor:
            self.grad().__imul__(incoming)
        if opcode == AddTensor:
            self.grad().__iadd__(incoming)
        if opcode == SubtractTensor:
            self.grad().__isub__(incoming)
        if opcode == ZeroGrad:
            self.zero_grad()

    fn __iadd__(self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __iadd__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )
        self.buffer.inplace_ops[Add](other.buffer)

    fn __isub__(self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __isub__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.inplace_ops[Subtract](other.buffer)

    fn __imul__(self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __imul__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.inplace_ops[Multiply](other.buffer)

    fn __itruediv__(self, other: Self):
        if self.is_leaf():
            panic(
                "Tensor → __itruediv__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.inplace_ops[Divide](other.buffer)

    fn unique(self) -> Tensor[dtype]:
        return Tensor[dtype](self.buffer.unique(), requires_grad=False)

    fn count(self, key: Scalar[dtype]) -> Int:
        return self.buffer.count(key)

    fn sum_all(self) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → sum_all is for numeric data types only",
        ]()

        return self.buffer.sum_all()

    fn product_all(self) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → product_all is for numeric data types only",
        ]()

        return self.buffer.reduce[
            Utils[dtype].product_buffer,
            Utils[dtype].product_scalars,
            unit = Scalar[dtype](1),
        ]()

    fn exp(self) -> Tensor[dtype]:
        constrained[
            dtype.is_floating_point(),
            "Tensor → exp is for floating point data types only",
        ]()

        var buffer = self.buffer.map[
            Utils[dtype].exp_buffer, Utils[dtype].exp_scalar
        ]()
        var nd_buffer = NDBuffer[dtype](buffer^, self.buffer.shape)
        return Tensor[dtype](nd_buffer^, requires_grad=False)

    fn __neg__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __neg__ is for numeric data types only",
        ]()
        var buffer = self.buffer.map[
            Utils[dtype].negate_buffer, Utils[dtype].negate_scalar
        ]()
        var nd_buffer = NDBuffer[dtype](buffer^, self.buffer.shape)
        return Tensor[dtype](nd_buffer^, requires_grad=False)

    fn __invert__(self: Tensor[DType.bool]) -> Tensor[DType.bool]:
        var buffer = self.buffer.map[
            Utils[dtype].invert_buffer, Utils[dtype].invert_scalar
        ]()
        var nd_buffer = NDBuffer[DType.bool](buffer^, self.buffer.shape)
        # What is the meaning of requires_grad for boolean Tensor?
        return Tensor[DType.bool](nd_buffer^, requires_grad=False)

    fn __abs__(self) -> Tensor[dtype]:
        var buffer = self.buffer.map[
            Utils[dtype].abs_buffer, Utils[dtype].abs_scalar
        ]()
        var nd_buffer = NDBuffer[dtype](buffer^, self.buffer.shape)
        return Tensor[dtype](nd_buffer^, requires_grad=False)

    fn __radd__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return AddScalar[dtype].forward[True](self, scalar)

    fn __add__(self, other: Self) -> Tensor[dtype]:
        return Adder[dtype].forward[True](self, other)

    fn __rsub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractFromScalar[dtype].forward[True](self, scalar)

    fn __sub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractScalar[dtype].forward[True](self, scalar)

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        return Subtractor[dtype].forward[True](self, other)

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, factor: Scalar[dtype]) -> Tensor[dtype]:
        return MultiplyScalar[dtype].forward[True](self, factor)

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Tensor[dtype]:
        return Multiplicator[dtype].forward[True](self, other)

    fn __pow__[
        track_grad: Bool = True
    ](self, exponent: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __pow__ is for numeric data types only",
        ]()

        return Exponentiator[dtype].forward[track_grad](self, exponent)

    fn dot[
        track_grad: Bool = True
    ](self, other: Self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return Dot[dtype].forward[track_grad](self, other, requires_grad)

    fn __iadd__(self, scalar: Scalar[dtype]):
        if self.is_leaf():
            panic(
                "Tensor → __iadd__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Add](scalar)

    fn __isub__(self, scalar: Scalar[dtype]):
        if self.is_leaf():
            panic(
                "Tensor → __isub__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Subtract](scalar)

    fn __imul__(self, scalar: Scalar[dtype]):
        if self.is_leaf():
            panic(
                "Tensor → __imul__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Multiply](scalar)

    fn __itruediv__(self, scalar: Scalar[dtype]):
        if self.is_leaf():
            panic(
                "Tensor → __itruediv__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Divide](scalar)

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(
            "\n",
            self.__str__(),
            end="\n",
        )
        empty = List[Int]()
        print_tensor_recursive[dtype](
            # UnsafePointer(to=self),
            self,
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
            self.grad().print(num_first, num_last)

    fn __del__(deinit self):
        _ = self.buffer^
        _ = self.gradbox^
        _ = self.ancestors^
        _ = self.backwardFn^

        log_debug("Tensor freed: " + self.id().__str__())

    fn mse(self, target: Tensor[dtype]) -> Tensor[dtype]:
        return ((self - target) ** 2).mean()

    fn backward(self, start_grad: Scalar[dtype] = 1.0):
        output = Ancestor(self)
        output.backward(start_grad)

    fn backward(self, seed_tensor: Tensor[dtype]):
        output = Ancestor(self)
        output.backward(seed_tensor)

    fn requires_grad_(mut self, requires_grad: Bool = True):
        self.requires_grad = requires_grad
        self.init_gradbox()

    fn __iter__(ref self) -> ElemIterator[dtype, origin_of(self)]:
        return ElemIterator[dtype, origin_of(self)](Pointer(to=self))

    fn element_at(self, index: Int) -> Scalar[dtype]:
        return self.buffer.element_at(index)

    fn view[
        track_grad: Bool = True
    ](
        self,
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
        self,
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
        self,
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
        self,
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
        self,
        shape: Shape,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return View[dtype].forward[track_grad](
            self, shape, Strides.default(shape), offset, requires_grad, False
        )

    fn transpose[
        track_grad: Bool = True
    ](self, *axes: Int, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return self.transpose[track_grad](IntList(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return self.transpose[track_grad](IntList.new(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](self, axes: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Transpose.forward[track_grad](self, axes, requires_grad)

    fn expand[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Expand[dtype].forward[track_grad](self, target, requires_grad)

    fn expand[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
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
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Squeeze[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    # Squeeze single axis if provided, otherwise squeeze all dims of size 1
    fn squeeze1[
        track_grad: Bool = True
    ](
        self,
        axis: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Squeeze[dtype].forward[track_grad](
            self,
            IntList(axis.value()) if axis else IntList(),
            requires_grad,
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](self, axis: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Unsqueeze[dtype].forward[track_grad](
            self, IntList(axis), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return Unsqueeze[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](self, axes: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Unsqueeze[dtype].forward[track_grad](self, axes, requires_grad)

    fn permute[
        track_grad: Bool = True
    ](
        self, axes: List[Int], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return Permute[dtype].forward[track_grad](
            self, IntList.new(axes), requires_grad
        )

    fn permute[
        track_grad: Bool = True
    ](self, axes: IntList, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Permute[dtype].forward[track_grad](self, axes, requires_grad)

    fn argmax(self, axis: Int = 0, keepdims: Bool = False) -> Tensor[DType.int32]:
        return Argmax[dtype].argmax(tensor=self, axis=axis, keepdims=keepdims)

    fn argmin(self, axis: Int = 0, keepdims: Bool = False) -> Tensor[DType.int32]:
        return Argmin[dtype].argmin(tensor=self, axis=axis, keepdims=keepdims)

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


    fn sum_over_broadcasted_axes(
        batch_tensor: Tensor[dtype], target_shape: Shape
    ) -> Tensor[dtype]:
        var nd_buffer = batch_tensor.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Tensor[dtype](nd_buffer^, requires_grad=False)

    @staticmethod
    fn broadcasted_indices(
        target_indices: IntList, target_shape: Shape, source_shape: Shape
    ) -> IntList:
        # Get coordinates for source tensor given target coordinates
        var source_indices = IntList.with_capacity(len(source_shape))

        for i in range(len(source_shape)):
            target_idx = len(target_shape) - len(source_shape) + i
            if source_shape[i] == 1:
                source_indices.append(0)  # Broadcasted dimension → use 0
            else:
                source_indices.append(
                    target_indices[target_idx]
                )  # Normal dimension

        return source_indices^


@register_passable
struct ElemIterator[dtype: DType, origin: ImmutableOrigin](ImplicitlyCopyable):
    var src: Pointer[Tensor[dtype], origin]
    var index_itr: ShapeIndexIterator[ImmutableAnyOrigin]

    fn __init__(out self, src: Pointer[Tensor[dtype], origin]):
        self.src = src
        self.index_itr = rebind[ShapeIndexIterator[ImmutableAnyOrigin]](
            src[].shape().__iter__()
        )

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Tuple[IntArray, Scalar[dtype]]:
        next = self.index_itr.__next__()
        return next, self.src[][next]

    fn __len__(self) -> Int:
        return self.index_itr.__len__()

    fn __has_next__(self) -> Bool:
        return self.index_itr.__has_next__()

        _ = """@staticmethod
    fn sum_over_broadcasted_axes(
        batch_grad: Tensor[dtype], recipient_shape: Shape
    ) -> Tensor[dtype]:
        result = batch_grad.copy()
        current_shape = batch_grad.shape.copy()

        # Sum over extra leading dimensions
        while len(current_shape) > len(recipient_shape):
            result = result.sum(axes=[0], keepdims=False)
            current_shape = result.shape.copy()

        # Sum over mismatched dimensions
        for i in range(len(recipient_shape)):
            if current_shape[i] != recipient_shape[i] and current_shape[i] > 1:
                result = result.sum(axes=[i], keepdims=True)
                current_shape = result.shape.copy()
        return result^

    fn vector_matrix_mm[
        track_grad: Bool = True
    ](
        A: Tensor[dtype], B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        return VectorMatrixMM[dtype].forward[track_grad](A, B, requires_grad)

    fn matrix_vector_mm[
        track_grad: Bool = True
    ](
        A: Tensor[dtype], B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        return MatrixVectorMM[dtype].forward[track_grad](A, B, requires_grad)




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



    fn matmul[
        track_grad: Bool = True, simd_width: Int = simd_width_of[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        return Matmul[dtype].forward[track_grad](A, B)

    @staticmethod
    fn matmul_2d[
        track_grad: Bool = True, simd_width: Int = simd_width_of[dtype]()
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
        A: Tensor[dtype], B: Tensor[dtype], requires_grad: Bool = True
    ) -> Tensor[dtype]:
        return Matmul_nd[dtype].forward[track_grad](A, B, requires_grad)



    fn into_view[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        if not self.owns_data:
            panic("Tensor → into_view: not allowed on non-owning tensor")
        shape, strides = self.shape, self.strides
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        return View[dtype].forward[track_grad](
            self, shape, strides, 0, grad_required, True
        )

        fn slice[
        track_grad: Bool = True
    ](self, start: Int, end: Int, step: Int = 1, axis: Int = 0) -> Tensor[
        dtype
    ]:
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
        self,
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
                    == Shape() else (
                        tensor.squeeze[track_grad=False](
                            [], requires_grad=False
                        )
                    )[IntList()]
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

    fn set(self, value: Scalar[dtype], *indices: Idx):
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
                sliced[idx] = value"""


from testing import assert_true


fn main() raises:
    #test_view_backward()
    # test_complex_mixed_ops_backward()
    # test_slice_backward()
    pass


fn test_slice_backward() raises:
    print("test_slice_backward")
    alias dtype = DType.float32
    a = Tensor[dtype].d1([1, 2, 3, 4, 5, 6], requires_grad=True)
    r = a.reshape([2, 3])
    s = r[Slice(1, None, None), Slice(0, 3, 1)]
    s.sum().backward(42)
    assert_true(
        a.grad().as_tensor()[Slice(3, None, None)]
        == Tensor[dtype]([42, 42, 42])
    )


fn test_view_backward() raises:
    print("test_view_backward")
    alias dtype = DType.float32
    a = Tensor[dtype].d1([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], requires_grad=True)
    v = a.view(shape=Shape(2, 4), strides=Strides(4, 1), offset=2)
    assert_true(v == Tensor[dtype].d2([[3, 4, 5, 6], [7, 8, 9, 10]]))

    v2 = v.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)
    # v2 = v.view(shape=Shape(2, 2), strides=Strides(1, 1), offset=2)

    assert_true(v2 == Tensor[dtype].d2([[3, 4], [5, 6]]))
    # Test gradients
    loss = v2.mean()
    loss.backward()
    assert_true(
        a.grad() == Tensor[dtype]([0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0])
    )
    assert_true(
        a.grad().as_tensor()[Slice(2, 6, 1)]
        == Tensor[dtype]([0.25, 0.25, 0.25, 0.25])
    )


fn test_complex_mixed_ops_backward() raises:
    print("test_complex_mixed_ops_backward")
    alias dtype = DType.float32
    a = Tensor[dtype].d2(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], requires_grad=True
    )
    v = a.view(shape=Shape(2, 4), strides=Strides(4, 1), offset=2)

    v2 = v.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)

    v3 = v2.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=0)
    c = v3.contiguous()
    s = c.mean()
    s.backward(42)
    assert_true(
        a.grad().as_tensor()[Slice(1, 2, None), Slice(None, None, None)]
        == Tensor[dtype].d2([[10.5, 10.5, 10.5, 10.5]])
    )
