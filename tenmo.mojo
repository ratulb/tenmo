### Mojo Tensor
### Implement tensor library in mojo from first principles
from math import exp, floor, log, cos, sin, sqrt, pi
from random import seed, random_float64
from sys import simd_width_of
from utils.numerics import min_finite
from memory import memcpy, memset, memset_zero
from shapes import Shape, ShapeIndexIterator
from ancestry import Ancestors
from strides import Strides
from common_utils_imports import *
from common_utils import IDGen, log_warning
from operators import *

from backpropagation import BackwardFn
from forwards import *
from buffers import Buffer
from validators import Validator
from collections import Set, Deque
from gradbox import Gradbox
from intarray import IntArray
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
    & ImplicitlyCopyable
):
    alias Row = List[Scalar[dtype]]
    alias Rows = List[Self.Row]
    alias Block = List[Self.Rows]
    alias Blocks = List[Self.Block]
    var _id: Int
    var buffer: NDBuffer[dtype]
    var requires_grad: Bool
    var gradbox: UnsafePointer[Gradbox[dtype], MutAnyOrigin]
    var ancestors: Optional[Ancestors[dtype]]
    var backwardFn: Optional[BackwardFn[dtype]]

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        self._id = IDGen.generate_id()
        self.buffer = NDBuffer[dtype](shape)
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Gradbox[dtype], MutAnyOrigin]()
        self.ancestors = None
        self.backwardFn = None
        self.init_gradbox()

    fn __init__(
        out self,
        shape: Shape,
        ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        requires_grad: Bool = False,
        *,
        copy: Bool = True,
    ):
        self._id = IDGen.generate_id()
        self.buffer = NDBuffer[dtype](
            Buffer[dtype](shape.num_elements(), ptr, copy), shape
        )
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Gradbox[dtype], MutAnyOrigin]()
        self.ancestors = None
        self.backwardFn = None
        self.init_gradbox()

    fn __init__(
        out self,
        var buffer: NDBuffer[dtype],
        requires_grad: Bool = False,
    ):
        self._id = IDGen.generate_id()
        self.buffer = buffer^
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Gradbox[dtype], MutAnyOrigin]()
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

        src_buffer.share(shape, strides, offset)
	var buffer = src_buffer.copy()
        #buffer = src_buffer.share(shape, strides, offset)
        return Tensor[dtype](buffer=buffer^, requires_grad=requires_grad)

    fn as_gradbox(deinit self, share: Bool = False) -> Gradbox[dtype]:
        return Gradbox[dtype](self^.buffer.contiguous(), share=share)

    fn __moveinit__(out self, deinit other: Self):
        self._id = other._id
        self.buffer = other.buffer^
        self.requires_grad = other.requires_grad
        self.gradbox = other.gradbox
        self.ancestors = other.ancestors^
        self.backwardFn = other.backwardFn^

    fn __copyinit__(out self, other: Self):
        self._id = other._id
        self.buffer = other.buffer.copy()
        self.requires_grad = other.requires_grad
        if other.gradbox != UnsafePointer[Gradbox[dtype], MutAnyOrigin]():
            self.gradbox = alloc[Gradbox[dtype]](1)
            self.gradbox.init_pointee_copy(other.gradbox[])
        else:
            self.gradbox = UnsafePointer[Gradbox[dtype], MutAnyOrigin]()
        self.ancestors = other.ancestors.copy()
        self.backwardFn = other.backwardFn.copy()

    @always_inline
    fn id(self) -> Int:
        return self._id

    @always_inline
    fn init_gradbox(mut self):
        if (
            self.requires_grad
            and self.gradbox == UnsafePointer[Gradbox[dtype], MutAnyOrigin]()
        ):
            gradbox = Gradbox[dtype](self.shape())
            gradbox.zero_grad()
            self.gradbox = alloc[Gradbox[dtype]](1)
            self.gradbox.init_pointee_move(gradbox^)

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
    fn shape(ref self) -> ref [self.buffer.shape] Shape:
        return self.buffer.shape

    @always_inline
    fn strides(ref self) -> ref [self.buffer.strides] Strides:
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

    fn __getitem__[
        track_grad: Bool = True
    ](self, *slices: Slice) -> Tensor[dtype]:
        # Delegate shape/strides/offset computation
        shape, strides, offset = Validator.validate_and_compute_view_metadata(
            self.shape(),
            self.strides(),
            slices,
        )
        return View[dtype].forward[track_grad=track_grad](
            self,
            shape=shape,
            strides=strides,
            offset=offset,
            requires_grad=self.requires_grad,
            validated=True,
        )

    fn __getitem__[
        track_grad: Bool = True
    ](self, *indices: Idx) -> Tensor[dtype]:
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
        abs_offset = self.offset() + offset
        return View[dtype].forward[track_grad=track_grad](
            self,
            shape,
            strides,
            abs_offset,
            self.requires_grad,
            validated=True,
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
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        if self.rank() == 0 and indices.size() != 0:  # Tensor with Shape ()
            panic(
                "Tensor → __setitem__(IntArray): Scalar tensor expects no"
                " indices"
            )

        self.buffer[indices] = value

    fn set(self, value: Scalar[dtype], *indices: Idx):
        # Compute view metadata
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape(), self.strides(), indices
            )
        )
        if len(shape) == 0:
            self.buffer.buffer[self.offset() + offset] = value
        else:
            sliced = self.view[track_grad=False](
                shape=shape, strides=strides, offset=self.offset() + offset
            )
            for coord in shape:
                sliced[coord] = value

    fn set(self, tensor: Tensor[dtype], *indices: Idx):
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape(), self.strides(), indices
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
                    tensor.item() if tensor.shape()
                    == Shape() else (tensor.squeeze[track_grad=False]([]))[
                        IntArray()
                    ]
                )
                self.buffer.buffer[self.offset() + offset] = elem
        else:
            tensor_shape = tensor.shape()
            if not ShapeBroadcaster.broadcastable(tensor_shape, shape):
                panic(
                    "Tensor → set: input tensor not broadcastable to shape",
                    shape.__str__(),
                )
            else:
                sliced = self.view[track_grad=False](
                    shape=shape, strides=strides, offset=self.offset() + offset
                )
                if tensor_shape == shape:
                    for coord in shape:
                        sliced[coord] = tensor[coord]
                else:
                    mask = ShapeBroadcaster.broadcast_mask(tensor_shape, shape)
                    for coord in shape:
                        tensor_coord = ShapeBroadcaster.translate_index(
                            tensor_shape, coord, mask, shape
                        )
                        sliced[coord] = tensor[tensor_coord]

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
        s += ", Type: " + Self.dtype.__str__()
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
        return self.gradbox != UnsafePointer[Gradbox[dtype]]()

    @always_inline
    fn zero_grad(self):
        if self.requires_grad and self.has_grad():
            self.gradbox[].zero_grad()

    @always_inline
    fn gradients(self) -> UnsafePointer[Gradbox[dtype], MutAnyOrigin]:
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad or grad not initialized"
            )
        return self.gradbox

    @always_inline
    fn grad(self) -> Gradbox[dtype]:
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad or grad not initialized"
            )
        return self.gradbox[].unshared()

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

    fn gt(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThan](scalar)
        )

    fn __ge__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThanEqual](scalar)
        )

    fn __eq__(self, other: Tensor[dtype]) -> Bool:
        return self.eq(other).buffer.buffer.all_true()

    fn eq(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[Equal](other.buffer))

    fn __ne__(self, other: Tensor[dtype]) -> Bool:
        return self.ne(other).buffer.buffer.all_true()

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
        return Tensor[NewType](
            new_type_buffer^, requires_grad=self.requires_grad
        )

    @always_inline
    fn add_ancestry(mut self, *parents: Tensor[dtype]):
        # Initialize ancestors if needed
        if not self.ancestors:
            self.ancestors = Optional(Ancestors[dtype].untracked())

        ref ancestors = self.ancestors.value()
        for parent in parents:
            ancestors.append(parent)

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

    fn all(self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.buffer.map_to_bool(pred).all_true()

    fn any(self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        return self.buffer.buffer.any(pred)

    fn log[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return Logarithm[dtype].forward[track_grad](self, requires_grad)

    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self,) -> Bool:
        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    fn address(self) -> UnsafePointer[Self, MutAnyOrigin]:
        return UnsafePointer(to=self)

    fn unsafe_address(
        ref self,
    ) -> UnsafePointer[
        Self,
        #mut = Origin(origin_of(self)).mut,
        origin = origin_of(self),
    ]:
        return UnsafePointer(to=self).mut_cast[Origin(origin_of(self)).mut]()

    fn seed_grad(mut self, with_tensor: Tensor[dtype]):
        if not self.requires_grad:
            return
        if not self.has_grad():
            self.requires_grad_()
        self.gradbox[].seed_grad(with_tensor)

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
            "Tensor → randint: is supported only for numeric type",
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
        shape: Shape,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → rand: is supported only for numeric type",
        ]()

        # Set seed if provided
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        # let it use whatever seed state exists

        var numels = shape.num_elements()
        var buffer = Buffer[dtype](numels)

        var min_f64 = min.cast[DType.float64]()
        var max_f64 = max.cast[DType.float64]()

        for i in range(numels):
            buffer[i] = random_float64(min_f64, max_f64).cast[dtype]()

        var nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn randn(
        shape: Shape,
        mean: Float64 = 0.0,
        std: Float64 = 1.0,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → randn: is supported only for numeric type",
        ]()

        # Set seed if provided
        if init_seed:
            seed(init_seed.value())
        else:
            # let it use whatever seed state exists
            seed()
        var numels = shape.num_elements()
        var buffer = Buffer[dtype](numels)

        # Box-Muller transform generates pairs
        var i = 0
        while i < numels:
            var u1 = random_float64(0.0, 1.0)
            var u2 = random_float64(0.0, 1.0)

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            # Box-Muller transform
            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mag * cos(2.0 * pi * u2) + mean
            var z1 = mag * sin(2.0 * pi * u2) + mean

            buffer[i] = z0.cast[dtype]()
            if i + 1 < numels:
                buffer[i + 1] = z1.cast[dtype]()
            i += 2

        var nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn arange(
        *args: Scalar[dtype],
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        nd_buffer = NDBuffer[dtype].arange(args)
        tensor = Tensor[dtype](nd_buffer^, requires_grad=requires_grad)
        return tensor^

    @staticmethod
    fn linspace(
        start: Scalar[dtype],
        end: Scalar[dtype],
        steps: Int,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        nd_buffer = NDBuffer[dtype].linspace(start, end, steps)
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

    fn onehot(self: Tensor[DType.int32], num_classes: Int) -> Tensor[dtype]:
        """Convert tensor of class indices to one-hot encoding.
        Args:
            self: Tensor of shape (...,) containing class indices.
            num_classes: Number of classes.
        Returns: Tensor of shape (..., num_classes).
        """
        shape = self.shape()
        result = Tensor[dtype].zeros(shape + [num_classes])

        # Set appropriate positions to 1.0
        for coord in shape:
            var class_index = self[coord].__int__()
            if class_index < 0 or class_index >= num_classes:
                panic(
                    "Tensor → onehot: invalid class",
                    class_index.__str__(),
                    "at coordinate",
                    coord.__str__(),
                )
            var onehot_coord = coord + class_index
            result[onehot_coord] = Scalar[dtype](1)

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
        shape = Shape(IntArray(numels))
        buffer = Buffer[dtype](numels)
        memcpy(dest=buffer.data, src=row._data, count=numels)
        nd_buffer = NDBuffer[dtype](buffer^, shape)
        return Tensor[dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntArray(len(rows), len(rows[0]))
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
        dims = IntArray(len(blocks), len(blocks[0]), len(blocks[0][0]))
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
        dims = IntArray(
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
        dims = IntArray(
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
        shape = Shape(IntArray(len(elems)))
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
        shape = Shape(IntArray(len(elems)))
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
        grad_required = requires_grad.or_else(self.requires_grad)
        out = Tensor[dtype](broadcasted_buffer^, requires_grad=grad_required)
        return out^

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[dtype, simdwidth]:
        """SIMD load of a row segment from a 2D Tensor.

        Preconditions:
            - Tensor must be 2D.
            - Columns must be contiguous (stride[1] == 1) for SIMD loads.
            - `col + simdwidth` must not exceed the number of columns.
        """
        return self.buffer.load[simdwidth, validated](row, col)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[dtype, simdwidth]):
        """SIMD store of a row segment into a 2D Tensor.

        Preconditions:
            - Tensor must be 2D.
            - Columns must be contiguous for SIMD stores (stride[1] == 1).
            - Caller may set validated=True if these checks are already ensured.
        """
        self.buffer.store[simdwidth, validated](row, col, value)

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
        return Repeat[dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn repeat[
        track_grad: Bool = True
    ](self, *repeat: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Repeat[dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn tile[
        track_grad: Bool = True
    ](self, repeat: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Tile[dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn tile[
        track_grad: Bool = True
    ](self, *repeat: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Tile[dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
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
            self.shape(), IntArray(newdims)
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
            self.shape(), IntArray(shape)
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
                axes = ShapeBroadcaster.broadcast_mask(
                    self.shape(), grad_contrib.shape()
                ).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape() != self.shape():
                grad_contrib = grad_contrib.reshape(self.shape())

        return grad_contrib^

    fn sum[
        track_grad: Bool = True
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.sum[track_grad](IntArray(axes), keepdims, requires_grad)

    fn sum[
        track_grad: Bool = True
    ](
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Summer[dtype].forward[track_grad](
            self, axes, keepdims, requires_grad
        )

    fn sqrt[
        track_grad: Bool = True
    ](
        self,
        epsilon: Scalar[dtype] = Scalar[dtype](1e-12),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Sqrt[dtype].forward[track_grad](self, epsilon, requires_grad)

    fn mean[
        track_grad: Bool = True
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return self.mean[track_grad](IntArray(axes), keepdims, requires_grad)

    fn mean[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Mean[dtype].forward[track_grad](
            self, axes, keepdims, requires_grad
        )

    fn variance[
        track_grad: Bool = True
    ](
        self,
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Variance[dtype].forward[track_grad](
            self, axis, keepdims, unbiased, requires_grad
        )

    fn std[
        track_grad: Bool = True
    ](
        self,
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        epsilon: Scalar[dtype] = Scalar[dtype](1e-12),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return StdDev[dtype].forward[track_grad](
            self, axis, keepdims, unbiased, epsilon, requires_grad
        )

    fn norm(
        self,
        p: Float64 = 2.0,
        axis: Optional[Int] = None,
        keepdims: Bool = False,
    ) -> Tensor[dtype]:
        """Compute Lp norm. Supports L2 (p=2) only."""
        if p == 2.0:
            # L2 norm: sqrt(sum(x²))
            var squared = self.__mul__[track_grad=False](self)
            var dim = [axis.value()] if axis else List[Int]()
            var sum_sq = squared.sum[track_grad=False](dim, keepdims=keepdims)
            return sum_sq.sqrt[track_grad=False]()
        else:
            panic("Only L2 norm (p=2) currently supported")
            return Tensor[dtype].scalar(0)

    fn __rtruediv__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return DivideScalar[dtype].forward[track_grad](self, scalar)

    fn __truediv__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return DivideByScalar[dtype].forward[track_grad](self, scalar)

    # Element wise division of two tensors
    fn __truediv__[track_grad: Bool = True](self, other: Self) -> Tensor[dtype]:
        return Divider[dtype].forward[track_grad](self, other)

    fn update_grad[opcode: Int](self, incoming: Gradbox[dtype]):
        ref gradbox = self.gradbox[]
        if opcode == MulTensor:
            gradbox.__imul__(incoming)
        if opcode == AddTensor:
            gradbox.__iadd__(incoming)
            # self.grad().buffer.fill_equal_shape(incoming.buffer)
        if opcode == SubtractTensor:
            gradbox.__isub__(incoming)
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

    fn __isub__(self, other: Gradbox[dtype]):
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
        # Create a zero tensor with same shape and properties
        var zeros = Tensor[dtype].zeros_like(self)

        # Use subtraction: 0 - self
        return Subtractor[dtype].forward[True](zeros, self)

    fn __invert__(self: Tensor[DType.bool]) -> Tensor[DType.bool]:
        var buffer = self.buffer.map[
            Utils[dtype].invert_buffer, Utils[dtype].invert_scalar
        ]()
        var nd_buffer = NDBuffer[DType.bool](buffer^, self.buffer.shape)
        return Tensor[DType.bool](nd_buffer^, requires_grad=False)

    fn __abs__(self) -> Tensor[dtype]:
        var buffer = self.buffer.map[
            Utils[dtype].abs_buffer, Utils[dtype].abs_scalar
        ]()
        var nd_buffer = NDBuffer[dtype](buffer^, self.buffer.shape)
        return Tensor[dtype](nd_buffer^, requires_grad=False)

    fn __radd__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__add__[track_grad](scalar)

    fn __add__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return AddScalar[dtype].forward[track_grad](self, scalar)

    fn __add__[track_grad: Bool = True](self, other: Self) -> Tensor[dtype]:
        return Adder[dtype].forward[track_grad](self, other)

    fn __rsub__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractFromScalar[dtype].forward[track_grad](self, scalar)

    fn __sub__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return SubtractScalar[dtype].forward[track_grad](self, scalar)

    fn __sub__[track_grad: Bool = True](self, other: Self) -> Tensor[dtype]:
        return Subtractor[dtype].forward[track_grad](self, other)

    fn __rmul__[
        track_grad: Bool = True
    ](self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__[track_grad](scalar)

    fn __mul__[
        track_grad: Bool = True
    ](self, factor: Scalar[dtype]) -> Tensor[dtype]:
        return MultiplyScalar[dtype].forward[track_grad](self, factor)

    # Element wise multiplication of two tensors
    fn __mul__[track_grad: Bool = True](self, other: Self) -> Tensor[dtype]:
        return Multiplicator[dtype].forward[track_grad](self, other)

    fn __mul__(self, other: Gradbox[dtype]) -> Gradbox[dtype]:
        return Multiplicator[dtype].forward(self, other)

    fn __pow__[
        track_grad: Bool = True
    ](self, exponent: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __pow__ is for numeric data types only",
        ]()

        return Exponentiator[dtype].forward[track_grad](self, exponent)

    fn dot[track_grad: Bool = True](self, other: Self) -> Tensor[dtype]:
        return Dot[dtype].forward[track_grad](self, other)

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
            self,
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn __del__(deinit self):
        _ = self.buffer^
        if self.has_grad():
            self.gradbox.destroy_pointee()
            self.gradbox.free()
        _ = self.ancestors^
        _ = self.backwardFn^

        log_debug("Tensor freed: " + self.id().__str__())

    fn mse[
        track_grad: Bool = True
    ](self, target: Tensor[dtype]) -> Tensor[dtype]:
        var diff = Subtractor[dtype].forward[track_grad](self, target)
        var squared = Multiplicator[dtype].forward[track_grad](diff, diff)
        return squared.mean[track_grad]()

    fn backward[
        graph_size: UInt = 50
    ](mut output: Tensor[dtype], start_grad: Scalar[dtype] = 1.0):
        if not output.requires_grad:
            return
        var shape = output.shape()
        var seed_tensor = Tensor[dtype].full(shape, start_grad)
        output.backward[graph_size](seed_tensor)

    fn backward[graph_size: UInt = 50](mut output, seed_tensor: Tensor[dtype]):
        if not output.requires_grad:
            return
        output.seed_grad(seed_tensor)

        try:
            # Phase 1: Discovery
            var topo_ids = List[Int](capacity=graph_size)
            var dfs_stack = List[Int](capacity=graph_size)
            var node_list = List[Tensor[dtype]](capacity=graph_size)
            var visited = Set[Int]()
            var fanin = Dict[Int, Int]()
            var id_to_index = Dict[Int, Int]()  # Map ID to List index

            dfs_stack.append(output.id())

            # Add to list and record index
            node_list.append(output)
            id_to_index[output.id()] = 0

            while len(dfs_stack) > 0:
                var node_id = dfs_stack.pop()

                if node_id in visited:
                    continue

                visited.add(node_id)
                topo_ids.append(node_id)

                var node_idx = id_to_index[node_id]
                ref node = node_list[node_idx]  # Access by index

                if node.has_ancestry():
                    for parent in node.ancestry():
                        var parent_id = parent.id()
                        fanin[parent_id] = fanin.get(parent_id, 0) + 1

                        if parent_id not in id_to_index:
                            var new_idx = len(node_list)
                            node_list.append(parent)
                            id_to_index[parent_id] = new_idx
                            dfs_stack.append(parent_id)

            # Phase 2: Execute backward
            var ready_queue = Deque[Int](capacity=graph_size)
            ready_queue.append(output.id())

            while len(ready_queue) > 0:
                var node_id = ready_queue.popleft()
                var node_idx = id_to_index[node_id]
                ref node = node_list[node_idx]  # Access by index

                if node.has_backward_fn():
                    for result in node.backward_fn()(node):
                        ref target_node = result[0]
                        ref grad = result[1]
                        var op_code = result[2]
                        var target_id = target_node.id()

                        if target_id in id_to_index:
                            var target_idx = id_to_index[target_id]
                            # Access by index each time (safe)
                            if op_code == AddTensor:
                                node_list[target_idx].update_grad[AddTensor](
                                    grad
                                )
                            else:
                                node_list[target_idx].update_grad[
                                    SubtractTensor
                                ](grad)

                        if target_id in fanin:
                            fanin[target_id] -= 1
                            if fanin[target_id] == 0:
                                var target_idx = id_to_index[target_id]
                                if node_list[target_idx].has_backward_fn():
                                    ready_queue.append(target_id)
        except e:
            print(e)

    fn requires_grad_(mut self, requires_grad: Bool = True):
        # Note cleaning up existing gradbox
        self.requires_grad = requires_grad
        if requires_grad and not self.has_grad():
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

    fn into_view[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        if self.shared():
            log_warning("Tensor → into_view: already shared")
            return self.copy()
        shape, strides = self.shape(), self.strides()
        grad_required = requires_grad.or_else(self.requires_grad)
        return View[dtype].forward[track_grad](
            self, shape, strides, 0, grad_required, validated=True
        )

    fn transpose[
        track_grad: Bool = True
    ](self, *axes: Int, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return self.transpose[track_grad](IntArray(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return self.transpose[track_grad](IntArray(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](self, axes: IntArray, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Transpose.forward[track_grad](self, axes, requires_grad)

    fn slice[
        track_grad: Bool = True
    ](self, start: Int, end: Int, step: Int = 1, axis: Int = 0) -> Tensor[
        dtype
    ]:
        # Call Validator to compute everything
        var shape, strides, offset = (
            Validator.validate_and_compute_slice_metadata(
                self.shape(), self.strides(), axis, start, end, step
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
        step_sizes = IntArray(steps) if steps else IntArray.filled(len(axes), 1)
        if len(step_sizes) != len(axes):
            panic("Tensor → slice: length of steps must match axes length")

        # Call Validator
        var shape, strides, offset = (
            Validator.validate_and_compute_slice_metadata_multi(
                self.shape(),
                self.strides(),
                axes,
                starts,
                ends,
                step_sizes,
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
            self, IntArray(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](self, *axes: Int, requires_grad: Optional[Bool] = None) -> Tensor[dtype]:
        return Unsqueeze[dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        return Unsqueeze[dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](self, axes: IntArray, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Unsqueeze[dtype].forward[track_grad](self, axes, requires_grad)

    fn permute[
        track_grad: Bool = True
    ](self, axes: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Permute[dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn permute[
        track_grad: Bool = True
    ](self, axes: IntArray, requires_grad: Optional[Bool] = None) -> Tensor[
        dtype
    ]:
        return Permute[dtype].forward[track_grad](self, axes, requires_grad)

    fn argmax(
        self, axis: Int = 0, keepdims: Bool = False
    ) -> Tensor[DType.int32]:
        return Argmax[dtype].argmax(tensor=self, axis=axis, keepdims=keepdims)

    fn argmin(
        self, axis: Int = 0, keepdims: Bool = False
    ) -> Tensor[DType.int32]:
        return Argmin[dtype].argmin(tensor=self, axis=axis, keepdims=keepdims)

    fn max(
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return MinMax[dtype].forward[True](
            self, IntArray(axes), keepdims, requires_grad
        )

    fn max(
        self,
        axes: IntArray,
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
            self, IntArray(axes), keepdims, requires_grad
        )

    fn min(
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return MinMax[dtype].forward[False](self, axes, keepdims, requires_grad)

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

    fn relu[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None,) -> Tensor[dtype]:
        return ReLU[dtype].forward[track_grad](self, requires_grad)

    fn clip[
        track_grad: Bool = True
    ](
        self,
        min_val: Scalar[dtype],
        max_val: Scalar[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        return Clip[dtype].forward[track_grad](
            self, min_val, max_val, requires_grad
        )

    fn tanh[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None,) -> Tensor[dtype]:
        return Tanh[dtype].forward[track_grad](self, requires_grad)

    fn sigmoid[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None,) -> Tensor[dtype]:
        return Sigmoid[dtype].forward[track_grad](self, requires_grad)

    fn softmax[
        track_grad: Bool = True, log: Bool = False  # Whether to use LogSoftmax
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        @parameter
        if log:
            return LogSoftmax[dtype].forward[track_grad](
                self, IntArray(axes), requires_grad
            )
        else:
            return Softmax[dtype].forward[track_grad](
                self, IntArray(axes), requires_grad
            )

    fn softmax[
        track_grad: Bool = True, log: Bool = False
    ](
        self,
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        @parameter
        if log:
            return LogSoftmax[dtype].forward[track_grad](
                self, axes, requires_grad
            )
        else:
            return Softmax[dtype].forward[track_grad](self, axes, requires_grad)

    fn binary_cross_entropy[
        track_grad: Bool = True
    ](
        pred: Tensor[dtype],
        target: Tensor[dtype],
        epsilon: Scalar[dtype] = Scalar[dtype](1e-9),
    ) -> Tensor[dtype]:
        return BCELoss[dtype].forward[track_grad](pred, target, epsilon)

    fn binary_cross_entropy_with_logits[
        track_grad: Bool = True
    ](
        logits: Tensor[dtype],
        target: Tensor[dtype],
        epsilon: Scalar[dtype] = Scalar[dtype](1e-9),
    ) -> Tensor[dtype]:
        return BCEWithLogitsLoss[dtype].forward[track_grad](
            logits, target, epsilon
        )

    fn sum_over_broadcasted_axes(
        batch_tensor: Tensor[dtype], target_shape: Shape
    ) -> Tensor[dtype]:
        var nd_buffer = batch_tensor.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Tensor[dtype](nd_buffer^, requires_grad=False)

    fn matmul[
        track_grad: Bool = True, mode: Int = mm
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        return Matmul[dtype].forward[track_grad=track_grad, mode=mode](A, B)

    fn matmul(A: Tensor[dtype], B: Gradbox[dtype]) -> Gradbox[dtype]:
        return Matmul[dtype].forward(A, B)


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


fn main():
    pass
