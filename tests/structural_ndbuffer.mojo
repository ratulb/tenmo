from .shapes import Shape
from .strides import Strides
from .buffers import Buffer
from .intarray import IntArray
from .indexhelper import IndexCalculator, IndexIterator
from .matrixshapevalidator import MatrixShapeValidator
from .broadcasthelper import ShapeBroadcaster
from .common_utils import panic, log_debug, print_buffer, Epsilon, One
from .validators import Validator
from std.memory import memcpy, AddressSpace, ArcPointer
from std.gpu.host import DeviceBuffer, DeviceContext
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from .device import Device, CPU, GPU, DeviceState
from std.collections import Set
from std.sys import simd_width_of, has_accelerator
from .scalar_ops_kernel import ScalarOperations
from .scalar_inplace_ops_kernel import InplaceScalarOperations
from .binary_ops_kernel import BinaryOperations
from .binary_inplace_ops_kernel import BinaryInplaceOperations
from .unary_ops_kernel import UnaryOpsKernel
from .matmul_kernel import MatmulNdGpu
from .compare_kernel import AllClose, Compare, CompareScalar
from .reduction_kernel import Reduction
from .minmax_kernel import ReductionMinMax
from .minmax_reducer import MinMaxReducer
from std.math import sqrt, log, exp, tanh
from .mnemonics import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    MAX,
    MIN,
    POW,
    NEGATE,
    SQRT,
    ABS,
    LOG,
    EXP,
    SIGMOID_FORWARD,
    SIGMOID_BACKWARD,
    TANH_FORWARD,
    TANH_BACKWARD,
    RELU_FORWARD,
    Overwrite,
    ReverseDivide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)

# =====================================================================
# Layout - Pure metadata describing data layout in memory.
# =====================================================================


struct Layout(ImplicitlyCopyable & Movable & Equatable):
    """
    Pure metadata describing how data is laid out in memory.
    No data, no device, no allocation.
    Device-agnostic — same for CPU and GPU.
    """

    var shape: Shape
    var strides: Strides
    var offset: Int
    var _contiguous: Bool

    fn __init__(out self):
        """Initialize an empty Layout with zero dimensions."""
        self.shape = Shape()
        self.strides = Strides.Zero()
        self.offset = 0
        self._contiguous = True

    fn __init__(
        out self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
    ):
        """Initialize Layout with given shape, strides, and offset."""
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self._contiguous = strides.is_contiguous(shape)

    fn __copyinit__(out self, copy: Self):
        """Copy Layout with deep copy of shape and strides."""
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self._contiguous = copy._contiguous

    fn __moveinit__(out self, deinit take: Self):
        """Move Layout by transferring ownership of shape and strides."""
        self.shape = take.shape^
        self.strides = take.strides^
        self.offset = take.offset
        self._contiguous = take._contiguous

    fn __eq__(self, other: Self) -> Bool:
        """Check if two Layouts are equal."""
        return (
            self.shape == other.shape
            and self.strides == other.strides
            and self.offset == other.offset
        )

    fn __ne__(self, other: Self) -> Bool:
        """Check if two Layouts are not equal."""
        return not self.__eq__(other)

    @always_inline
    fn is_contiguous(self) -> Bool:
        """Return True if the layout represents contiguous memory."""
        return self._contiguous

    @always_inline
    fn num_elements(self) -> Int:
        """Return the total number of elements in the layout."""
        return self.shape.num_elements()

    @always_inline
    fn rank(self) -> Int:
        """Return the number of dimensions in the layout."""
        return self.shape.rank()

    @always_inline
    fn max_index(self) -> Int:
        """Return the maximum linear index addressable in this layout."""
        var max_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] > 0:
                max_idx += (self.shape[i] - 1) * self.strides[i]
        return max_idx

    @always_inline
    fn min_index(self) -> Int:
        """Return the minimum linear index addressable in this layout."""
        var min_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] < 0:
                min_idx += (self.shape[i] - 1) * self.strides[i]
        return min_idx

    @always_inline
    fn offset_at(self, indices: IntArray) -> Int:
        """Return the absolute linear offset for given multidimensional indices.
        """
        if indices.size() != self.rank():
            panic("Layout.offset_at: Incorrect number of indices")
        return IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )


# =====================================================================
# Storage - Pure data carrier for CPU buffer or GPU device state.
# =====================================================================


struct Storage[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Pure data carrier — CPU buffer or GPU device state.
    No shape knowledge. No layout knowledge.
    copy() is cheap — just a refcount bump.
    """

    var buffer: Buffer[Self.dtype]
    var device_state: Optional[DeviceState[Self.dtype]]

    fn __init__(out self):
        """Initialize empty Storage with no buffer and no device state."""
        self.buffer = Buffer[Self.dtype]()
        self.device_state = None

    fn __init__(out self, var buffer: Buffer[Self.dtype]):
        """Initialize Storage with a CPU buffer."""
        self.buffer = buffer^
        self.device_state = None

    fn __init__(out self, var device_state: DeviceState[Self.dtype]):
        """Initialize Storage with a GPU device state."""
        self.buffer = Buffer[Self.dtype]()
        self.device_state = Optional(device_state^)

    fn __copyinit__(out self, copy: Self):
        """Copy Storage with refcount bump for buffer and device_state."""
        self.buffer = copy.buffer.copy()
        self.device_state = copy.device_state.copy()

    fn __moveinit__(out self, deinit take: Self):
        """Move Storage by transferring ownership of buffer and device_state."""
        self.buffer = take.buffer^
        self.device_state = take.device_state^

    @always_inline
    fn is_on_gpu(self) -> Bool:
        """Return True if data resides on GPU."""
        comptime if has_accelerator():
            return self.device_state is not None
        return False

    @always_inline
    fn is_on_cpu(self) -> Bool:
        """Return True if data resides on CPU."""
        return not self.is_on_gpu()

    fn copy(self) -> Self:
        """Explicit copy — refcount bump only, no data copy."""
        return self


comptime TILE_SIZE = 32


# =====================================================================
# NDBuffer - Multi-dimensional array with Layout + Storage composition.
# =====================================================================


struct NDBuffer[dtype: DType](
    ImplicitlyCopyable & Movable & Equatable & Writable & Sized
):
    """Multi-dimensional array with shape, strides, and dat.storage."""

    # =================================================================
    # Core fields - Layout and Storage composition.
    # =================================================================

    var layout: Layout
    var storage: Storage[Self.dtype]

    # =================================================================
    # Forwarding properties - Maintain zero external API changes.
    # =================================================================

    @always_inline
    fn shape(ref self) -> ref[self.layout.shape] Shape:
        """Forward to layout.shape for backward compatibility."""
        return self.layout.shape

    @always_inline
    fn strides(ref self) -> ref[self.layout.strides] Strides:
        """Forward to layout.strides for backward compatibility."""
        return self.layout.strides

    @always_inline
    fn offset(self) -> Int:
        """Forward to layout.offset for backward compatibility."""
        return self.layout.offset

    @always_inline
    fn buffer(ref self) -> ref[self.storage.buffer] Buffer[Self.dtype]:
        """Forward t.storage.buffer for backward compatibility."""
        return self.storage.buffer

    @always_inline
    fn device_state(
        ref self,
    ) -> ref[self.storage.device_state] Optional[DeviceState[Self.dtype]]:
        """Forward t.storage.device_state for backward compatibility."""
        return self.storage.device_state

    @always_inline
    fn _contiguous(self) -> Bool:
        """Forward to layout._contiguous for backward compatibility."""
        return self.layout._contiguous

    @always_inline
    fn is_on_gpu(self) -> Bool:
        """Return True if data resides on GPU."""
        return self.storage.is_on_gpu()

    @always_inline
    fn is_on_cpu(self) -> Bool:
        """Return True if data resides on CPU."""
        return self.storage.is_on_cpu()

    @always_inline
    fn is_contiguous(self) -> Bool:
        """Return True if the underlying data is contiguous in memory."""
        return self.layout.is_contiguous()

    @always_inline
    fn max_index(self) -> Int:
        """Return the maximum linear index addressable in this buffer."""
        return self.layout.max_index()

    @always_inline
    fn min_index(self) -> Int:
        """Return the minimum linear index addressable in this buffer."""
        return self.layout.min_index()

    @always_inline
    fn numels(self) -> Int:
        """Return the total number of elements in the buffer."""
        return self.layout.num_elements()

    @always_inline
    fn rank(self) -> Int:
        """Return the number of dimensions in the buffer."""
        return self.layout.rank()

    @always_inline
    fn size(self) -> Int:
        """Return the size of the underlying buffer."""
        return self.storage.buffer.size

    @always_inline
    fn data_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Scalar[Self.dtype], origin, address_space=address_space
    ]:
        """Return a pointer to the underlying data."""
        return (
            self.storage.buffer.unsafe_ptr()
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    # =================================================================
    # Constructors.
    # =================================================================

    fn __init__(out self, *values: Scalar[Self.dtype]):
        """Initialize NDBuffer from variadic scalar values."""
        var buffer = Buffer[Self.dtype](len(values))
        for i in range(len(values)):
            buffer[i] = values[i]
        self = NDBuffer[Self.dtype](buffer^)

    fn __init__(
        out self,
        var buffer: Buffer[Self.dtype] = Buffer[Self.dtype](),
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        """Initialize NDBuffer from a Buffer with optional shape, strides, and offset.
        """
        if buffer.size == 0:
            log_debug(
                "NDBuffer → __init__(Buffer, ...): zero sized buffer -"
                " potential danger"
            )
            self.layout = Layout(
                shape.or_else(Shape()),
                strides.or_else(Strides.Zero()),
                offset,
            )
            self.storage = Storage[Self.dtype](buffer^)
        else:
            var _shape = shape.or_else(Shape(buffer.size))
            var _strides = strides.or_else(Strides.default(_shape))
            self.layout = Layout(_shape, _strides, offset)
            self.storage = Storage[Self.dtype](buffer^)

    fn __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        """Initialize NDBuffer with given shape, allocating a new Buffer."""
        var buf = Buffer[Self.dtype](shape.num_elements())
        var _strides = strides.or_else(Strides.default(shape))
        self.layout = Layout(shape, _strides, offset)
        self.storage = Storage[Self.dtype](buf^)

    fn __init__(
        out self,
        device_buffer: DeviceBuffer[Self.dtype],
        shape: Shape,
        *,
        copy: Bool = False,
    ) raises:
        """Initialize NDBuffer from a DeviceBuffer, copying to CPU if requested.
        """
        var buffer: Buffer[Self.dtype]
        with device_buffer.map_to_host() as host_buffer:
            buffer = Buffer[Self.dtype](
                shape.num_elements(), host_buffer.unsafe_ptr(), copy=copy
            )
        self.layout = Layout(shape, Strides.default(shape), 0)
        self.storage = Storage[Self.dtype](buffer^)

    fn __moveinit__(out self, deinit take: Self):
        """Move NDBuffer by transferring layout an.storage ownership."""
        self.layout = take.layout^
        self.storage = take.storage^

    fn __copyinit__(out self, copy: Self):
        """Copy NDBuffer with deep copy of layout and refcount bump fo.storage.
        """
        self.layout = copy.layout.copy()
        self.storage = copy.storage.copy()

    @staticmethod
    fn with_device_state(
        var device_state: DeviceState[Self.dtype], shape: Shape
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer that owns a GPU device state with given shape."""
        var ndb = NDBuffer[Self.dtype]()
        ndb.layout = Layout(shape, Strides.default(shape), 0)
        ndb.storage = Storage[Self.dtype](device_state^)
        return ndb^

    @staticmethod
    @always_inline
    fn Empty() -> NDBuffer[Self.dtype]:
        """Return an empty NDBuffer."""
        return NDBuffer[Self.dtype](Buffer[Self.dtype]())

    # =================================================================
    # Factory methods.
    # =================================================================

    @staticmethod
    @always_inline
    fn zeros(
        shape: Shape, device: Device = CPU().into()
    ) -> NDBuffer[Self.dtype]:
        """Create a zero-initialized NDBuffer on the specified device."""
        var buffer = Buffer[Self.dtype].zeros(shape.num_elements())
        var ndb = NDBuffer[Self.dtype](buffer^, shape)

        if device.is_cpu():
            return ndb^
        else:
            comptime if has_accelerator():
                try:
                    var (_, result) = ndb^.to_device(device)
                    return result^
                except e:
                    print(e)
                    panic("NDBuffer zeros: device transfer failed")
                    return Self.Empty()
            else:
                return ndb^

    @staticmethod
    @always_inline
    fn full(
        shape: Shape, scalar: Scalar[Self.dtype], device: Device = CPU().into()
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer filled with a scalar value on the specified device.
        """
        var buffer = Buffer[Self.dtype].full(scalar, shape.num_elements())
        var ndb = NDBuffer[Self.dtype](buffer^, shape)

        if device.is_cpu():
            return ndb^
        else:
            comptime if has_accelerator():
                try:
                    var (_, result) = ndb^.to_device(device)
                    return result^
                except e:
                    print(e)
                    panic("NDBuffer full: device transfer failed")
                    return Self.Empty()
            else:
                return ndb^

    @staticmethod
    @always_inline
    fn arange(
        args: VariadicList[Scalar[Self.dtype], _],
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer with evenly spaced values."""
        var buffer = Buffer[Self.dtype].arange(args)
        var shape = Shape(buffer.size)
        return NDBuffer[Self.dtype](buffer^, shape^)

    @staticmethod
    @always_inline
    fn arange(
        *args: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer with evenly spaced values."""
        return Self.arange(args)

    @staticmethod
    @always_inline
    fn linspace(
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        steps: Int,
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer with linearly spaced values."""
        var buffer = Buffer[Self.dtype].linspace(start, end, steps)
        var shape = Shape(buffer.size)
        return NDBuffer[Self.dtype](buffer^, shape^)

    @staticmethod
    fn onehot(
        indices: NDBuffer[Self.dtype],
        num_classes: Int,
        device: Optional[Device] = None,
        ignore_index: Optional[Int] = None,
    ) -> NDBuffer[Self.dtype]:
        """Convert NDBuffer of class indices to one-hot encoding."""
        ref shape = indices.layout.shape
        ref target_device = device.or_else(indices.device())
        var result = NDBuffer[Self.dtype].zeros(
            shape + [num_classes], device=target_device
        )

        var ignore_val = ignore_index.or_else(-1000000)

        for coord in shape:
            var class_index = indices[coord].__int__()

            if ignore_index and class_index == ignore_val:
                continue

            if class_index < 0 or class_index >= num_classes:
                panic(
                    "Tensor → onehot: invalid class",
                    String(class_index),
                    "at coordinate",
                    String(coord),
                )
            var onehot_coord = coord + class_index
            result[onehot_coord] = Scalar[Self.dtype](1)

        return result^

    # =================================================================
    # Device management.
    # =================================================================

    fn get_gpu(
        ref self,
    ) raises -> ref[self.storage.device_state.value().gpu] GPU:
        """Return a reference to the GPU device if buffer is on GPU."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.storage.device_state.value().get_gpu()
            else:
                raise ("NDBuffer get_gpu: buffer is not on gpu")
        else:
            raise (
                "NDBuffer get_gpu: buffer not on gpu or system has no"
                " accelerator"
            )

    fn to_cpu(self) raises -> Self:
        """Move the buffer to CPU, returning a new NDBuffer."""
        var _, nd_buffer = self.to_device(CPU().into())
        return nd_buffer^

    fn to_gpu(self, gpu: GPU) raises -> Self:
        """Move the buffer to the specified GPU, returning a new NDBuffer."""
        if self.storage.buffer.size == 0:
            raise "NDBuffer -> to_gpu(): Empty buffer"
        return self.to_device(gpu.into())[1]

    fn device(self) -> Device:
        """Return the device where the buffer resides."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.storage.device_state.value().get_gpu().into()
        return CPU().into()

    fn device_context(self) -> Optional[ArcPointer[DeviceContext]]:
        """Return the device context if buffer is on GPU."""
        if self.is_on_gpu():
            return self.storage.device_state.value().gpu[]
        return None

    fn get_device_state(
        ref self,
    ) raises -> ref[self.storage.device_state.value()] DeviceState[Self.dtype]:
        """Return a reference to the DeviceState if buffer is on GPU."""
        if self.is_on_gpu():
            return self.storage.device_state.value()
        raise "Not on any device"

    fn gpu_id(self) -> Int64:
        """Return the GPU ID if buffer is on GPU, otherwise -1."""
        if self.is_on_gpu():
            return self.storage.device_state.value().get_gpu().id
        return -1

    fn to_device(
        self, device: Device
    ) raises -> Tuple[Int, NDBuffer[Self.dtype]]:
        """
        Materialize this buffer onto another device.

        Returns:
            - Tuple of (status, new NDBuffer).
        """
        if not self.storage.device_state:
            if device.is_cpu():
                print("NDBuffer -> to_device: already on CPU")
                return -1, self

            var gpu = device.kind[GPU]
            var new_device_state = DeviceState[Self.dtype](self.numels(), gpu)
            new_device_state.fill(self)
            var result = NDBuffer[Self.dtype].with_device_state(
                new_device_state, self.layout.shape
            )
            return 0, result^

        var curr_state = self.storage.device_state.value()
        var curr_gpu = curr_state.gpu

        if device.is_gpu():
            var new_gpu = device.kind[GPU]

            if curr_gpu == new_gpu:
                print("NDBuffer -> to_device: current and new device is same")
                return -1, self

            var ndb_buffer = curr_state.into(self.layout.shape)
            return ndb_buffer.to_device(device)

        if self.is_contiguous():
            return 0, curr_state.into(self.layout.shape)
        else:
            var flat_cpu = curr_state.into(Shape(len(curr_state)))
            var viewed = flat_cpu.share(
                self.layout.shape, self.layout.strides, self.layout.offset
            )
            var result = NDBuffer[Self.dtype](self.layout.shape)
            result.copy_from_alike[overwrite=True, validate=False](viewed^)
            return 0, result^

    fn sync(self):
        """Synchronize the device buffer if on GPU."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.storage.device_state.value().sync()
                except e:
                    print(e)
                    print("NDBuffer device state synchronization failed")

    # =================================================================
    # Data access and manipulation.
    # =================================================================

    fn __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        """Return the element at the given multidimensional indices."""
        index = IndexCalculator.flatten_index(
            self.layout.shape, indices, self.layout.strides, self.layout.offset
        )
        return self.get(index)

    fn __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        """Set the element at the given multidimensional indices."""
        index = IndexCalculator.flatten_index(
            self.layout.shape, indices, self.layout.strides, self.layout.offset
        )
        self.set(index, value)

    fn __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        """Return the element at the given multidimensional indices."""
        index = IndexCalculator.flatten_index(
            self.layout.shape, indices, self.layout.strides, self.layout.offset
        )
        return self.get(index)

    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        """Set the element at the given multidimensional indices."""
        index = IndexCalculator.flatten_index(
            self.layout.shape, indices, self.layout.strides, self.layout.offset
        )
        self.set(index, value)

    fn __getitem__(self, indices: VariadicList[Int, _]) -> Scalar[Self.dtype]:
        """Return the element at the given multidimensional indices."""
        index = IndexCalculator.flatten_index(
            self.layout.shape, indices, self.layout.strides, self.layout.offset
        )
        return self.get(index)

    fn __setitem__(
        self, indices: VariadicList[Int, _], value: Scalar[Self.dtype]
    ):
        """Set the element at the given multidimensional indices."""
        index = IndexCalculator.flatten_index(
            self.layout.shape, indices, self.layout.strides, self.layout.offset
        )
        self.set(index, value)

    @always_inline
    fn item(self) -> Scalar[Self.dtype]:
        """Return the scalar value for a zero-dimensional or singleton buffer.
        """
        if self.layout.shape != Shape(1) and self.layout.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim"
                " buffer/singleton, got shape: "
                + String(self.layout.shape)
            )
        return self.get(0)

    fn get(self, index: Int) -> Scalar[Self.dtype]:
        """Return the element at the given linear index."""
        idx = index + self.max_index() if index < 0 else index
        if idx < 0 or idx > self.max_index():
            panic(
                "NDBuffer → element_at: index out of bounds.",
                "NDBuffer max index",
                String(self.max_index()),
                ", provided index",
                String(index),
            )
        if self.is_on_gpu():
            ref device_state = self.storage.device_state.value()
            try:
                return device_state[idx]
            except e:
                print(e)
                panic("Error in NDBuffer → get: ", String(e))
                return Scalar[Self.dtype](0)
        return self.data_ptr()[idx]

    fn set(self, index: Int, value: Scalar[Self.dtype]):
        """Set the element at the given linear index."""
        idx = index + self.max_index() if index < 0 else index
        if idx < 0 or idx > self.max_index():
            panic(
                "NDBuffer → set_element_at: index out of bounds.",
                "NDBuffer max index",
                String(self.max_index()),
                ", provided index",
                String(index),
            )

        if self.is_on_gpu():
            ref device_state = self.storage.device_state.value()
            try:
                device_state[idx] = value
            except e:
                print(e)
                panic("Error in NDBuffer → set: ", String(e))
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            ptr[idx] = value

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from a 2D NDBuffer."""
        comptime assert (
            simdwidth.is_power_of_two()
        ), "NDBuffer → load: SIMD width must be a power of 2"
        if simdwidth > self.numels():
            panic("NDBuffer → load: buffer size is less than simd width")

        comptime if not validated:
            var rank = self.rank()
            ref shape = self.layout.shape

            if rank != 2:
                panic("NDBuffer → load: Only 2D buffers are supported.")

            if (
                row < 0
                or row >= shape[0]
                or col < 0
                or col + simdwidth > shape[1]
            ):
                panic(
                    "NDBuffer → load: Out-of-bounds access. "
                    + "Attempted row "
                    + String(row)
                    + ", col range ["
                    + String(col)
                    + ", "
                    + String((col + simdwidth))
                    + ") "
                    + "for shape "
                    + String(shape)
                    + "."
                )

            if simdwidth > 1 and self.layout.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD load requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + String(self.layout.strides[1])
                    + ". "
                    + "Use .contiguous() or scalar loads."
                )

        var addr = (
            row * self.layout.strides[0]
            + col * self.layout.strides[1]
            + self.layout.offset
        )
        if self.is_on_gpu():
            ref device_state = self.storage.device_state.value()
            try:
                return device_state.load[simdwidth=simdwidth](addr).cast[
                    Self.dtype
                ]()
            except e:
                print(e)
                panic("Error in NDBuffer → get: ", String(e))
                return SIMD[Self.dtype, simdwidth](0)
        return self.data_ptr().load[width=simdwidth](addr)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
        """SIMD store of a row segment into a 2D NDBuffer."""
        comptime assert (
            simdwidth.is_power_of_two()
        ), "NDBuffer → store: SIMD width must be a power of 2"
        if simdwidth > self.numels():
            panic("NDBuffer → store: buffer size is less than simd width")

        comptime if not validated:
            var rank = self.rank()
            ref shape = self.layout.shape

            if rank != 2:
                panic("NDBuffer → store: Only 2D buffers are supported.")

            if (
                row < 0
                or row >= shape[0]
                or col < 0
                or col + simdwidth > shape[1]
            ):
                panic(
                    "NDBuffer → store: Out-of-bounds access. "
                    + "Attempted row "
                    + String(row)
                    + ", col range ["
                    + String(col)
                    + ", "
                    + String((col + simdwidth))
                    + ") "
                    + "for shape "
                    + String(shape)
                    + "."
                )

            if simdwidth > 1 and self.layout.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD store requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + String(self.layout.strides[1])
                    + ". "
                    + "Use .contiguous() or scalar stores."
                )

        var addr = (
            row * self.layout.strides[0]
            + col * self.layout.strides[1]
            + self.layout.offset
        )
        if self.is_on_gpu():
            ref device_state = self.storage.device_state.value()
            try:
                device_state.store[simdwidth=simdwidth](
                    addr, value.cast[DeviceState[Self.dtype].datatype]()
                )
            except e:
                print(e)
                panic("Error in NDBuffer → store: ", String(e))
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            ptr.store[width=simdwidth](addr, value)

    @always_inline
    fn index_iterator(
        self,
    ) -> IndexIterator[
        origin_of(self.layout.shape), origin_of(self.layout.strides)
    ]:
        """Return an iterator over linear indices in the buffer."""
        return IndexIterator(
            shape=Pointer(to=self.layout.shape).get_immutable(),
            strides=Pointer(to=self.layout.strides).get_immutable(),
            start_offset=self.layout.offset,
        )

    @always_inline
    fn data_buffer(ref self) -> ref[self.storage.buffer] Buffer[Self.dtype]:
        """Return a reference to the underlying Buffer."""
        return self.storage.buffer

    @always_inline
    fn is_scalar(self) -> Bool:
        """Return True if the buffer is a scalar (zero-dimensional)."""
        return self.numels() == 1 and self.layout.shape == Shape()

    @always_inline
    fn __len__(self) -> Int:
        """Return the total number of elements."""
        return self.layout.shape.num_elements()

    # =================================================================
    # Buffer operations.
    # =================================================================

    fn fill(self, value: Scalar[Self.dtype]):
        """Fill all elements with the given scalar value."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.storage.device_state.value().fill(value)
                except e:
                    print(e)
                    panic("Error filling NDBuffer value: ", String(value))
            else:
                self.fill_cpu(value)
        else:
            self.fill_cpu(value)

    @always_inline
    fn fill_cpu(self, value: Scalar[Self.dtype]):
        """Fill all elements on CPU with the given scalar value."""
        ref buffer = self.data_buffer()
        if self.is_contiguous():
            buffer.fill(
                value, self.layout.offset, self.layout.offset + self.numels()
            )
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            for index in self.index_iterator():
                (ptr + index)[] = value

    fn fill(self, cpu_buffer: NDBuffer[Self.dtype]):
        """Fill this NDBuffer from a CPU NDBuffer."""
        if cpu_buffer.is_scalar() or cpu_buffer.layout.shape == Shape.Unit():
            self.fill(cpu_buffer.item())
            return

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.storage.device_state.value().fill(cpu_buffer)
                except e:
                    print(e)
                    panic(
                        "NDBuffer → fill: error filling GPU buffer from CPU"
                        " buffer"
                    )
            else:
                self.fill_cpu(cpu_buffer)
        else:
            self.fill_cpu(cpu_buffer)

    fn fill_cpu(self, other: NDBuffer[Self.dtype]):
        """Fill this CPU buffer from another CPU buffer."""
        if self.__is__(other):
            panic("NDBuffer → fill_cpu: cannot fill with self")

        if self.layout.shape == other.layout.shape:
            self.copy_from_alike[overwrite=True, validate=True](other)
        else:
            if not ShapeBroadcaster.broadcastable(
                self.layout.shape, other.layout.shape
            ):
                panic(
                    (
                        "NDBuffer → fill_cpu(other): dimension mismatch:"
                        " self.layout.shape"
                    ),
                    String(self.layout.shape),
                    "≠",
                    "other shape",
                    String(other.layout.shape),
                )
            var broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.layout.shape, other.layout.shape
            )
            if broadcast_shape != self.layout.shape:
                panic(
                    "NDBuffer → fill_cpu: broadcasted shape must match receiver"
                    " shape"
                )

            mask = ShapeBroadcaster.broadcast_mask(
                other.layout.shape, self.layout.shape
            )
            for coord in self.layout.shape:
                src_coord = ShapeBroadcaster.translate_index(
                    other.layout.shape, coord, mask, self.layout.shape
                )
                self[coord] = other[src_coord]

    fn zero(self):
        """Set all elements to zero."""
        self.fill(Scalar[Self.dtype](0))

    def contiguous_buffer(self) -> Buffer[Self.dtype]:
        """Return a contiguous copy of the buffer with the same data - CPU only.
        """
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            return self.storage.buffer[start:end]
        else:
            var buffer = Buffer[Self.dtype](self.numels())
            var index = 0
            for idx in self.index_iterator():
                buffer[index] = self.storage.buffer[idx]
                index += 1
            return buffer^

    fn contiguous_device_state(self) raises -> DeviceState[Self.dtype]:
        """Return a fresh independent contiguous DeviceState. Caller must ensure self is on GPU.
        """
        ref curr_state = self.storage.device_state.value()
        ref gpu = curr_state.get_gpu()
        var new_state = DeviceState[Self.dtype](self.numels(), gpu)

        if self.is_contiguous():
            curr_state.buffer.enqueue_copy_to(new_state.buffer)
            new_state.sync()
        else:
            new_state.fill(self)

        return new_state^

    fn contiguous(
        self, new_shape: Optional[Shape] = None
    ) -> NDBuffer[Self.dtype]:
        """Return a contiguous copy of the buffer, optionally with new shape."""
        var target_shape = new_shape.or_else(self.layout.shape)

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var new_state = self.contiguous_device_state()
                    return NDBuffer[Self.dtype].with_device_state(
                        new_state^, target_shape
                    )
                except e:
                    panic(
                        "NDBuffer → contiguous: GPU materialisation failed: "
                        + String(e)
                    )
                    return self

        if (
            self.is_contiguous()
            and not self.shared()
            and target_shape == self.layout.shape
        ):
            return self
        return NDBuffer[Self.dtype](self.contiguous_buffer(), target_shape)

    # =================================================================
    # View and sharing operations.
    # =================================================================

    fn shared(self) -> Bool:
        """Check if underlying buffer is shared."""
        return self.storage.buffer.is_shared()

    fn share(
        mut self,
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[Self.dtype]:
        """Create a shared view of this buffer."""
        var size = len(self.storage.buffer) if self.is_on_cpu() else len(
            self.storage.device_state.value()
        )

        if self.is_on_cpu() and size > 0 and not self.shared():
            self.storage.buffer.shared()

        var new_shape = shape.or_else(self.layout.shape)
        var new_strides = strides.or_else(Strides.default(new_shape))
        var max_index = IndexCalculator.max_index(
            new_shape, new_strides, offset
        )

        if max_index > size:
            panic(
                "NDBuffer → share: invalid view [max_index="
                + String(max_index)
                + " > buffer_size="
                + String(size)
                + "] shape="
                + String(new_shape)
                + " strides="
                + String(new_strides)
                + " offset="
                + String(offset)
            )

        var ndb = NDBuffer[Self.dtype]()
        ndb.layout = Layout(new_shape, new_strides, offset)
        ndb.storage = Storage[Self.dtype](self.storage.buffer.copy())
        ndb.storage.device_state = self.storage.device_state.copy()
        return ndb^

    # =================================================================
    # Shape manipulation.
    # =================================================================

    fn reshape(
        self, new_shape: Shape, validated: Bool = False
    ) -> NDBuffer[Self.dtype]:
        """Return a reshaped view of the buffer."""
        var shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            self.layout.shape, new_shape.intarray()
        )

        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.reshape_gpu(shape)
        return self.contiguous(shape)

    fn reshape_gpu(
        self,
        shape: Shape,
    ) -> NDBuffer[Self.dtype]:
        """Return a reshaped view of the buffer on GPU."""
        var out: NDBuffer[Self.dtype]

        try:
            ref device_state = self.storage.device_state.value()
            var new_state = device_state.new(self.numels(), 0, sync=False)
            new_state.fill(self, sync=True)
            out = NDBuffer[Self.dtype].with_device_state(new_state, shape)
        except e:
            print(e)
            panic("Error reshaping device buffer")
            out = NDBuffer[Self.dtype].Empty()
        return out^

    fn transpose(
        mut self,
        axes: IntArray = IntArray(),
        *,
        shared: Bool = True,
    ) -> NDBuffer[Self.dtype]:
        """Return a transposed view or copy of the buffer."""
        ref shape = self.layout.shape
        var normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntArray.range(0, shape.rank()).reversed()
        )
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.layout.strides.permute(normalized_axes)

        if shared:
            return self.share(new_shape, new_strides, self.layout.offset)
        else:
            var view = self.share(new_shape, new_strides, self.layout.offset)
            return view.contiguous()

    fn flatten(
        self,
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
    ) -> NDBuffer[Self.dtype]:
        """Flatten the buffer between start_dim and end_dim."""
        rank = self.rank()
        if rank == 0:
            return self.contiguous()
        var endd = end_dim.or_else(rank - 1)

        if endd < start_dim:
            panic("NDBuffer → flatten: end_dim must be >= start_dim")

        var original_shape = self.layout.shape
        collapsed = original_shape[start_dim : endd + 1].product()
        new_shape = (
            original_shape[:start_dim]
            + [collapsed]
            + original_shape[endd + 1 :]
        )
        return self.contiguous(new_shape)

    fn squeeze(
        mut self, axes: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        """Remove dimensions of size 1."""
        var shape = self.layout.shape
        var rank = shape.rank()

        var axes_to_squeeze: IntArray
        if axes == IntArray():
            axes_to_squeeze = shape.indices_of_axes_with_size(1)
        else:
            axes_to_squeeze = IntArray.with_capacity(len(axes))
            var seen = IntArray.with_capacity(len(axes))
            for axis in axes:
                var normalized = axis if axis >= 0 else axis + rank
                if normalized < 0 or normalized >= rank:
                    panic(
                        "NDBuffer → squeeze: axis ",
                        String(axis),
                        " out of range",
                    )
                if shape[normalized] != 1:
                    panic(
                        "NDBuffer → squeeze: cannot squeeze axis ",
                        String(normalized),
                        " with size ",
                        String(shape[normalized]),
                    )
                if normalized in seen:
                    panic("NDBuffer → squeeze: duplicate axis ", String(axis))
                seen.append(normalized)
                axes_to_squeeze.append(normalized)
            axes_to_squeeze.sort()

        if len(axes_to_squeeze) == 0:
            return self

        var new_size = rank - len(axes_to_squeeze)
        var new_shape_dims = IntArray.with_capacity(new_size)
        var new_strides_arr = IntArray.with_capacity(new_size)

        for i in range(rank):
            if i not in axes_to_squeeze:
                new_shape_dims.append(shape[i])
                new_strides_arr.append(self.layout.strides[i])

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        if shared:
            return self.share(new_shape, new_strides, self.layout.offset)
        else:
            var view = self.share(new_shape, new_strides, self.layout.offset)
            return view.contiguous()

    fn unsqueeze(
        mut self, axes: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        """Insert dimensions of size 1 at specified axes."""
        var rank = self.layout.shape.rank()
        var new_axes_count = len(axes)

        if new_axes_count == 0:
            return self

        var new_rank = rank + new_axes_count

        var normalized_axes = IntArray.with_capacity(new_axes_count)
        var seen = IntArray.with_capacity(new_axes_count)

        for axis in axes:
            var normalized = axis if axis >= 0 else new_rank + axis
            if normalized < 0 or normalized >= new_rank:
                panic(
                    "NDBuffer → unsqueeze: axis ",
                    String(axis),
                    " out of range",
                )
            if normalized in seen:
                panic("NDBuffer → unsqueeze: duplicate axis ", String(axis))
            seen.append(normalized)
            normalized_axes.append(normalized)

        normalized_axes.sort()

        var new_shape_dims = IntArray.with_capacity(new_rank)
        var new_strides_arr = IntArray.with_capacity(new_rank)
        var orig_i = 0
        var ins_i = 0

        for i in range(new_rank):
            if ins_i < new_axes_count and i == normalized_axes[ins_i]:
                new_shape_dims.append(1)
                var insert_stride = (
                    self.layout.strides[orig_i] if orig_i < rank else 1
                )
                new_strides_arr.append(insert_stride)
                ins_i += 1
            else:
                new_shape_dims.append(self.layout.shape[orig_i])
                new_strides_arr.append(self.layout.strides[orig_i])
                orig_i += 1

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        if shared:
            return self.share(new_shape, new_strides, self.layout.offset)
        else:
            var view = self.share(new_shape, new_strides, self.layout.offset)
            return view.contiguous()

    fn permute(
        mut self, perm: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        """Permute the axes of the buffer."""
        var shape = self.layout.shape
        var rank = shape.rank()

        if len(perm) != rank:
            panic(
                "NDBuffer → permute: number of axes (",
                String(len(perm)),
                ") must match rank (",
                String(rank),
                ")",
            )

        var visited = IntArray.filled(rank, 0)
        for i in range(len(perm)):
            var normalized = perm[i] if perm[i] >= 0 else perm[i] + rank
            if normalized < 0 or normalized >= rank:
                panic(
                    "NDBuffer → permute: invalid axis ",
                    String(perm[i]),
                    " for rank ",
                    String(rank),
                )
            if visited[normalized] == 1:
                panic("NDBuffer → permute: duplicate axis ", String(perm[i]))
            visited[normalized] = 1

        var new_shape_dims = IntArray.with_capacity(rank)
        var new_strides_arr = IntArray.with_capacity(rank)
        for i in range(len(perm)):
            new_shape_dims.append(shape[perm[i]])
            new_strides_arr.append(self.layout.strides[perm[i]])

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        if shared:
            return self.share(new_shape, new_strides, self.layout.offset)
        else:
            var view = self.share(new_shape, new_strides, self.layout.offset)
            return view.contiguous()

    fn broadcast_to(self, target_shape: Shape) -> NDBuffer[Self.dtype]:
        """
        Broadcast this NDBuffer to target_shape.
        Uses stride=0 trick for broadcast dims — pure metadata, no data copy.
        """
        if not ShapeBroadcaster.expandable_to(self.layout.shape, target_shape):
            panic(
                "NDBuffer.broadcast_to: cannot expand "
                + String(self.layout.shape)
                + " to "
                + String(target_shape)
            )

        var own_shape = self.layout.shape
        var own_rank = own_shape.rank()
        var target_rank = target_shape.rank()

        var extra_dims = target_rank - own_rank

        var new_strides = IntArray.with_capacity(target_rank)

        for _ in range(extra_dims):
            new_strides.append(0)

        for i in range(own_rank):
            var target_i = i + extra_dims
            if own_shape[i] == 1 and target_shape[target_i] > 1:
                new_strides.append(0)
            else:
                new_strides.append(self.layout.strides[i])

        var self_copy = self.copy()
        var view = self_copy.share(
            target_shape, Strides(new_strides), self.layout.offset
        )

        return view.contiguous()

    # =================================================================
    # Type conversion.
    # =================================================================

    fn to_dtype[NewType: DType](self) -> NDBuffer[NewType]:
        """Convert the buffer to a different data type."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var cpu_ndb = self.contiguous_device_state().into(
                        self.layout.shape
                    )
                    var cast_ndb = cpu_ndb.to_dtype[NewType]()
                    var new_state = DeviceState[NewType](
                        self.numels(), self.storage.device_state.value().gpu
                    )
                    new_state.fill(cast_ndb)
                    return NDBuffer[NewType].with_device_state(
                        new_state^, self.layout.shape
                    )
                except e:
                    panic("NDBuffer to_dtype GPU failed: " + String(e))
                    return NDBuffer[NewType].Empty()

        var new_buffer = self.contiguous_buffer().to_dtype[NewType]()
        return NDBuffer[NewType](new_buffer^, self.layout.shape)

    # =================================================================
    # Comparison operations.
    # =================================================================

    fn __eq__(self, other: Self) -> Bool:
        """Return True if all elements are equal."""
        var ndb = self.compare[Equal](other)
        if ndb.is_on_gpu():
            return ndb.storage.device_state.value().all_true()
        return ndb.storage.buffer.all_true()

    fn __ne__(self, other: Self) -> Bool:
        """Return True if any elements are not equal."""
        var ndb = self.compare[NotEqual](other)
        if ndb.is_on_gpu():
            return ndb.storage.device_state.value().all_true()
        return ndb.storage.buffer.all_true()

    @always_inline
    fn compare[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        DType.bool
    ]:
        """Compare two buffers element-wise, returning a boolean buffer."""
        if not self.layout.shape == other.layout.shape:
            panic(
                "NDBuffer → compare(self, other): dimension mismatch: "
                + String(self.layout.shape)
                + "≠"
                + String(other.layout.shape)
            )
        var result: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    result = Compare[Self.dtype].launch[op_code](self, other)
                except e:
                    print(e)
                    panic("NDBuffer compare → GPU operation failed")
                    result = NDBuffer[DType.bool].Empty()
            elif (self.is_on_gpu() and other.is_on_cpu()) or (
                self.is_on_cpu() and other.is_on_gpu()
            ):
                panic(
                    "NDBuffer compare → not both buffers are on the same device"
                )
                result = NDBuffer[DType.bool].Empty()
            else:
                result = self.compare_cpu[op_code](other)
        else:
            result = self.compare_cpu[op_code](other)

        return result^

    @always_inline
    fn compare_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        DType.bool
    ]:
        """Compare two buffers element-wise on CPU."""
        if self.is_contiguous() and other.is_contiguous():
            var self_contiguous = self.contiguous_buffer()
            var other_contiguous = other.contiguous_buffer()
            var result_buffer = self_contiguous.compare_buffer_full[op_code](
                other_contiguous
            )
            return NDBuffer[DType.bool](result_buffer^, self.layout.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())
            var iterator = other.index_iterator()
            for idx in self.index_iterator():
                var self_val = self.storage.buffer[idx]
                var next_index = -1
                try:
                    next_index = iterator.__next__()
                except e:
                    print(e)
                    panic("Raised StopIteration in NDBuffer → compare")

                var other_val = other.storage.buffer[next_index]

                comptime if op_code == Equal:
                    buffer[index] = self_val == other_val
                elif op_code == NotEqual:
                    buffer[index] = self_val != other_val
                elif op_code == LessThan:
                    buffer[index] = self_val < other_val
                elif op_code == LessThanEqual:
                    buffer[index] = self_val <= other_val
                elif op_code == GreaterThan:
                    buffer[index] = self_val > other_val
                else:
                    buffer[index] = self_val >= other_val

                index += 1

            return NDBuffer[DType.bool](buffer^, self.layout.shape)

    @always_inline
    fn compare_scalar[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        DType.bool
    ]:
        """Compare buffer with scalar element-wise, returning a boolean buffer.
        """
        var result: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    result = CompareScalar[Self.dtype].launch[op_code](
                        self, scalar
                    )
                except e:
                    print(e)
                    panic("NDBuffer compare_scalar → GPU operation failed.")
                    result = NDBuffer[DType.bool].Empty()
            else:
                result = self.compare_scalar_cpu[op_code](scalar)
        else:
            result = self.compare_scalar_cpu[op_code](scalar)

        return result^

    @always_inline
    fn compare_scalar_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        DType.bool
    ]:
        """Compare buffer with scalar element-wise on CPU."""
        if self.is_contiguous():
            var contiguous_data = self.contiguous_buffer()
            var result_buffer = contiguous_data.compare_scalar_full[op_code](
                scalar
            )
            return NDBuffer[DType.bool](result_buffer^, self.layout.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())

            for idx in self.index_iterator():
                var value = self.storage.buffer[idx]

                comptime if op_code == Equal:
                    buffer[index] = value == scalar
                elif op_code == NotEqual:
                    buffer[index] = value != scalar
                elif op_code == LessThan:
                    buffer[index] = value < scalar
                elif op_code == LessThanEqual:
                    buffer[index] = value <= scalar
                elif op_code == GreaterThan:
                    buffer[index] = value > scalar
                else:
                    buffer[index] = value >= scalar

                index += 1

            return NDBuffer[DType.bool](buffer^, self.layout.shape)

    @always_inline
    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        """Return True if all elements are close within tolerance."""
        comptime assert (
            Self.dtype.is_floating_point()
        ), "NDBuffer → all_close is for floating point data types only"

        if self.layout.shape != other.layout.shape:
            panic(
                "NDBuffer → all_close(other) expects same shaped buffers: "
                + String(self.layout.shape)
                + "≠"
                + String(other.layout.shape)
            )
        var result: Bool

        comptime if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    result = AllClose[Self.dtype].launch[rtol=rtol, atol=atol](
                        self, other
                    )
                except e:
                    print(e)
                    panic("NDBuffer all_close → GPU operation failed")
                    result = False
            elif (self.is_on_gpu() and other.is_on_cpu()) or (
                self.is_on_cpu() and other.is_on_gpu()
            ):
                panic(
                    "NDBuffer all_close → both buffers must be on the same"
                    " device"
                )
                result = False
            else:
                result = self.contiguous_buffer().all_close[
                    rtol=rtol, atol=atol
                ](other.contiguous_buffer())
        else:
            result = self.contiguous_buffer().all_close[rtol=rtol, atol=atol](
                other.contiguous_buffer()
            )

        return result

    fn all_true(self: NDBuffer[DType.bool]) -> Bool:
        """Return True if all elements are True."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.storage.device_state.value().all_true()

        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            for i in range(start, end):
                if not self.storage.buffer[i]:
                    return False
            return True

        for idx in self.index_iterator():
            if not self.storage.buffer[idx]:
                return False
        return True

    fn any_true(self: NDBuffer[DType.bool]) -> Bool:
        """Return True if any element is True."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.storage.device_state.value().any_true()

        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            for i in range(start, end):
                if self.storage.buffer[i]:
                    return True
            return False

        for idx in self.index_iterator():
            if self.storage.buffer[idx]:
                return True
        return False

    # =================================================================
    # Arithmetic operations.
    # =================================================================

    @always_inline
    fn __add__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Element-wise addition of two buffers."""
        return self.arithmetic_ops[Add](other)

    @always_inline
    fn __sub__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Element-wise subtraction of two buffers."""
        return self.arithmetic_ops[Subtract](other)

    @always_inline
    fn __mul__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Element-wise multiplication of two buffers."""
        return self.arithmetic_ops[Multiply](other)

    @always_inline
    fn __truediv__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Element-wise division of two buffers."""
        return self.arithmetic_ops[Divide](other)

    @always_inline
    fn __add__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Add a scalar to each element."""
        return self.scalar_ops[Add](scalar)

    @always_inline
    fn __sub__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Subtract a scalar from each element."""
        return self.scalar_ops[Subtract](scalar)

    @always_inline
    fn __mul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Multiply each element by a scalar."""
        return self.scalar_ops[Multiply](scalar)

    @always_inline
    fn __truediv__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Divide each element by a scalar."""
        return self.scalar_ops[Divide](scalar)

    @always_inline
    fn __rmul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Multiply each element by a scalar (commutative)."""
        return self.__mul__(scalar)

    @always_inline
    fn __rtruediv__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Divide a scalar by each element."""
        return self.scalar_ops[ReverseDivide](scalar)

    @always_inline
    fn __neg__(self) -> NDBuffer[Self.dtype]:
        """Negate each element."""
        return self.unary_ops[NEGATE]()

    @always_inline
    fn __pow__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Raise each element to a scalar power."""
        return self.scalar_ops[POW](scalar)

    @always_inline
    fn __imul__(self, factor: Scalar[Self.dtype]):
        """In-place multiplication by a scalar."""
        self.inplace_scalar_ops[Multiply](factor)

    @always_inline
    fn __iadd__(self, scalar: Scalar[Self.dtype]):
        """In-place addition of a scalar."""
        self.inplace_scalar_ops[Add](scalar)

    @always_inline
    fn __isub__(self, scalar: Scalar[Self.dtype]):
        """In-place subtraction of a scalar."""
        self.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[Self.dtype]):
        """In-place division by a scalar."""
        self.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, other: NDBuffer[Self.dtype]):
        """In-place element-wise multiplication."""
        self.inplace_ops[Multiply](other)

    @always_inline
    fn __iadd__(self, other: NDBuffer[Self.dtype]):
        """In-place element-wise addition."""
        self.inplace_ops[Add](other)

    @always_inline
    fn __isub__(self, other: NDBuffer[Self.dtype]):
        """In-place element-wise subtraction."""
        self.inplace_ops[Subtract](other)

    fn __itruediv__(self, other: NDBuffer[Self.dtype]):
        """In-place element-wise division."""
        self.inplace_ops[Divide](other)

    @always_inline
    fn max(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Element-wise maximum with a scalar."""
        return self.scalar_ops[MAX](scalar)

    @always_inline
    fn min(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Element-wise minimum with a scalar."""
        return self.scalar_ops[MIN](scalar)

    @always_inline
    fn log[
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Natural logarithm of each element."""
        return self.float_unary_ops[LOG, epsilon]()

    @always_inline
    fn exp(self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Exponential of each element."""
        return self.float_unary_ops[EXP]()

    @always_inline
    fn sigmoid(
        self,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Sigmoid activation function applied to each element."""
        return self.float_unary_ops[SIGMOID_FORWARD]()

    @always_inline
    fn tanh(self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Hyperbolic tangent applied to each element."""
        return self.float_unary_ops[TANH_FORWARD]()

    @always_inline
    fn clamp(
        self: NDBuffer[Self.dtype],
        lower_bound: Scalar[Self.dtype],
        upper_bound: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Clamp each element between lower_bound and upper_bound."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            var result_buffer = self.storage.buffer.clamp(
                lower_bound, upper_bound
            ) if start == 0 else self.storage.buffer[start:end].clamp(
                lower_bound, upper_bound
            )
            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = self.storage.buffer[idx].clamp(
                    lower_bound, upper_bound
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

    fn clamp_in_place(
        self: NDBuffer[Self.dtype],
        lower_bound: Scalar[Self.dtype],
        upper_bound: Scalar[Self.dtype],
    ):
        """Clamp each element in-place between lower_bound and upper_bound."""
        if self.is_contiguous():
            self.storage.buffer.clamp_in_place(lower_bound, upper_bound)
        else:
            for idx in self.index_iterator():
                self.storage.buffer[idx] = self.storage.buffer[idx].clamp(
                    lower_bound, upper_bound
                )

    # =================================================================
    # Arithmetic operation implementations.
    # =================================================================

    @always_inline
    fn arithmetic_ops[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        """Element-wise arithmetic operation between two buffers."""
        if not ShapeBroadcaster.broadcastable(
            self.layout.shape, other.layout.shape
        ):
            panic(
                "NDBuffer → arithmetic_ops: dimension mismatch: "
                + String(self.layout.shape)
                + ", "
                + String(other.layout.shape)
            )

        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    out = BinaryOperations[Self.dtype].launch[op_code](
                        self, other, epsilon
                    )
                except e:
                    print(e)
                    print(
                        (
                            "NDBuffer arithmetic_ops → GPU operation failed for"
                            " opcode: "
                        ),
                        String(op_code),
                    )
                    out = NDBuffer[Self.dtype].Empty()
            else:
                out = self.arithmetic_ops_cpu[op_code](other, epsilon)
        else:
            out = self.arithmetic_ops_cpu[op_code](other, epsilon)

        return out^

    @always_inline
    fn arithmetic_ops_cpu[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        """Element-wise arithmetic operation between two buffers on CPU."""
        if self.layout.shape != other.layout.shape:
            return self.broadcast_buffer[op_code](other)

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.layout.offset
            self_end = self_start + self.numels()
            other_start = other.layout.offset
            other_end = other_start + other.numels()
            var result_buffer = self.storage.buffer.arithmetic_ops[
                op_code=op_code
            ](
                other.storage.buffer,
                self_start,
                self_end,
                other_start,
                other_end,
                epsilon=epsilon,
            )
            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

        else:
            var result_buffer = Buffer[Self.dtype](self.numels())
            var index = 0

            if self.is_contiguous() and not other.is_contiguous():
                var offset = self.layout.offset
                for idx in other.index_iterator():
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.storage.buffer[offset + index],
                        other.storage.buffer[idx],
                        epsilon,
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var offset = other.layout.offset
                for idx in self.index_iterator():
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.storage.buffer[idx],
                        other.storage.buffer[offset + index],
                        epsilon,
                    )
                    index += 1

            else:
                var iterator = other.index_iterator()
                for idx in self.index_iterator():
                    var next_index = -1
                    try:
                        next_index = iterator.__next__()
                    except e:
                        print(e)
                        panic(
                            "Raised StopIteration in NDBuffer → arithmetic_ops"
                        )

                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.storage.buffer[idx],
                        other.storage.buffer[next_index],
                        epsilon,
                    )
                    index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

    @always_inline
    fn broadcast_buffer[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        """Broadcast and apply arithmetic operation between two buffers."""
        if self.layout.shape.rank() == 0 or other.layout.shape.rank() == 0:
            return self.broadcast_scalar_buffer[op_code](other)
        else:
            return self.broadcast_nd_buffer[op_code](other)

    @always_inline
    fn broadcast_scalar_buffer[
        op_code: Int
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        """Broadcast where one buffer is scalar."""
        result_shape = (
            other.layout.shape if self.layout.shape.rank()
            == 0 else self.layout.shape
        )
        var buffer = Buffer[Self.dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            self_val = (
                self.item() if self.layout.shape.rank() == 0 else self[coord]
            )
            other_val = (
                other.item() if other.layout.shape.rank() == 0 else other[coord]
            )
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[op_code](self_val, other_val)

        return NDBuffer[Self.dtype](buffer^, result_shape)

    @always_inline
    fn broadcast_nd_buffer[
        op_code: Int
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        """Broadcast two multi-dimensional buffers and apply arithmetic operation.
        """
        result_shape = ShapeBroadcaster.broadcast_shape(
            self.layout.shape, other.layout.shape
        )

        mask1 = ShapeBroadcaster.broadcast_mask(self.layout.shape, result_shape)
        mask2 = ShapeBroadcaster.broadcast_mask(
            other.layout.shape, result_shape
        )

        var buffer = Buffer[Self.dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            self_coord = ShapeBroadcaster.translate_index(
                self.layout.shape, coord, mask1, result_shape
            )
            other_coord = ShapeBroadcaster.translate_index(
                other.layout.shape, coord, mask2, result_shape
            )
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[op_code](
                self[self_coord], other[other_coord]
            )
        return NDBuffer[Self.dtype](buffer^, result_shape)

    @always_inline
    fn scalar_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        """Apply scalar operation to each element."""
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → scalar_ops: cannot divide by zero")

        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    comptime if op_code == POW:
                        out = ScalarOperations[Self.dtype].launch_pow(
                            self, scalar
                        )
                    else:
                        out = ScalarOperations[Self.dtype].launch[op_code](
                            self, scalar
                        )
                except e:
                    print(e)
                    panic(
                        (
                            "NDBuffer scalar_ops → GPU operation failed for"
                            " opcode: "
                        ),
                        String(op_code),
                    )
                    out = Self.Empty()
            else:
                out = self.scalar_ops_cpu[op_code](scalar)
        else:
            out = self.scalar_ops_cpu[op_code](scalar)

        return out^

    @always_inline
    fn scalar_ops_cpu[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        """Apply scalar operation to each element on CPU."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            var result_buffer: Buffer[Self.dtype]

            comptime if op_code == POW:
                result_buffer = self.storage.buffer[start:end] ** scalar
            else:
                result_buffer = self.storage.buffer.arithmetic_ops_scalar[
                    op_code
                ](scalar, start, end)
            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = Self.scalar_fn[op_code](
                    self.storage.buffer[idx], scalar, epsilon
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

    @always_inline
    fn inplace_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]):
        """In-place element-wise operation with another buffer."""
        if not ShapeBroadcaster.broadcastable(
            self.layout.shape, other.layout.shape
        ):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + String(self.layout.shape)
                + ", "
                + String(other.layout.shape)
            )

        comptime if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    BinaryInplaceOperations[Self.dtype].launch[op_code](
                        self, other
                    )
                except e:
                    print(e)
                    print(
                        (
                            "NDBuffer inplace_ops → GPU operation failed for"
                            " opcode: "
                        ),
                        String(op_code),
                    )
            else:
                self.inplace_ops_cpu[op_code](other)
        else:
            self.inplace_ops_cpu[op_code](other)

    @always_inline
    fn inplace_ops_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]):
        """In-place element-wise operation with another buffer on CPU."""
        if not ShapeBroadcaster.broadcastable(
            self.layout.shape, other.layout.shape
        ):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + String(self.layout.shape)
                + ", "
                + String(other.layout.shape)
            )

        if self.layout.shape != other.layout.shape:
            broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.layout.shape, other.layout.shape
            )

            if broadcast_shape != self.layout.shape:
                panic(
                    "NDBuffer → inplace_ops: broadcasted shape "
                    + String(broadcast_shape)
                    + " must match receiver shape "
                    + String(self.layout.shape)
                )

            var broadcast_result = self.broadcast_buffer[op_code](other)
            self.copy_from_alike[overwrite=True, validate=False](
                broadcast_result^
            )

        else:
            if self.is_contiguous() and other.is_contiguous():
                self_start = self.layout.offset
                self_end = self_start + self.numels()
                other_start = other.layout.offset
                other_end = other_start + other.numels()
                self.storage.buffer.inplace_ops[op_code](
                    other.storage.buffer,
                    self_start,
                    self_end,
                    other_start,
                    other_end,
                )

            elif self.is_contiguous() and not other.is_contiguous():
                var index = self.layout.offset
                for idx in other.index_iterator():
                    self.storage.buffer[index] = Self.scalar_fn[op_code](
                        self.storage.buffer[index], other.storage.buffer[idx]
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var index = other.layout.offset
                for idx in self.index_iterator():
                    self.storage.buffer[idx] = Self.scalar_fn[op_code](
                        self.storage.buffer[idx], other.storage.buffer[index]
                    )
                    index += 1
            else:
                var iterator = other.index_iterator()
                for index in self.index_iterator():
                    var next_index = -1
                    try:
                        next_index = iterator.__next__()
                    except e:
                        print(e)
                        panic("Raised StopIteration in NDBuffer → inplace_ops")

                    self.storage.buffer[index] = Self.scalar_fn[op_code](
                        self.storage.buffer[index],
                        other.storage.buffer[next_index],
                    )

    @always_inline
    fn inplace_scalar_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]):
        """In-place scalar operation on all elements."""
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    InplaceScalarOperations[Self.dtype].launch[op_code](
                        self, scalar
                    )
                except e:
                    print(e)
                    panic(
                        (
                            "NDBuffer inplace_scalar_ops → GPU operation failed"
                            " for opcode: "
                        ),
                        String(op_code),
                    )
            else:
                self.inplace_scalar_ops_cpu[op_code](scalar)
        else:
            self.inplace_scalar_ops_cpu[op_code](scalar)

    @always_inline
    fn inplace_scalar_ops_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]):
        """In-place scalar operation on all elements on CPU."""
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        if self.is_contiguous():
            start = self.layout.offset
            end = start + self.numels()
            self.storage.buffer.inplace_ops_scalar[op_code](scalar, start, end)

        else:
            for index in self.index_iterator():
                self.storage.buffer[index] = Self.scalar_fn[op_code](
                    self.storage.buffer[index], scalar
                )

    # =================================================================
    # Unary operations.
    # =================================================================

    @always_inline
    fn unary_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Apply unary operation to each element."""
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    out = UnaryOpsKernel[Self.dtype].launch[op_code](self)
                except e:
                    print(e)
                    panic(
                        (
                            "NDBuffer unary_ops → GPU operation failed for"
                            " opcode: "
                        ),
                        String(op_code),
                    )
                    out = Self.Empty()
            else:
                out = self.unary_ops_cpu[op_code]()
        else:
            out = self.unary_ops_cpu[op_code]()

        return out^

    @always_inline
    fn unary_ops_cpu[
        op_code: Int
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        """Apply unary operation to each element on CPU."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            var result_buffer: Buffer[Self.dtype]

            comptime if op_code == NEGATE:
                result_buffer = self.storage.buffer[start:end].__neg__()
            elif op_code == SQRT:
                result_buffer = self.storage.buffer.unary_ops[SQRT](start, end)
            else:
                result_buffer = self.storage.buffer[start:end].__abs__()

            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = Self.unary_fn_helper[op_code](
                    self.storage.buffer[idx]
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

    @always_inline
    fn float_unary_ops[
        op_code: Int,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Apply floating-point unary operation to each element."""
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    out = UnaryOpsKernel[Self.dtype].launch[op_code, epsilon](
                        self
                    )
                except e:
                    print(e)
                    panic(
                        (
                            "NDBuffer float_unary_ops → GPU operation failed"
                            " for opcode: "
                        ),
                        String(op_code),
                    )
                    out = Self.Empty()
            else:
                out = self.float_unary_ops_cpu[op_code, epsilon]()
        else:
            out = self.float_unary_ops_cpu[op_code, epsilon]()

        return out^

    @always_inline
    fn float_unary_ops_cpu[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Apply floating-point unary operation to each element on CPU."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            var result_buffer = self.storage.buffer.float_unary_ops[
                op_code, epsilon
            ](start, end)
            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)
        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = Self.float_unary_fn_helper[
                    op_code, epsilon
                ](self.storage.buffer[idx])
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.layout.shape)

    @always_inline
    fn unary_ops_with_mask[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """Apply unary operation returning both output and mask."""
        var out: NDBuffer[Self.dtype]
        var mask: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var result = UnaryOpsKernel[Self.dtype].launch_with_mask[
                        op_code
                    ](self)
                    out = result[0]
                    mask = result[1]
                except e:
                    panic(
                        "NDBuffer unary_ops_with_mask → GPU launch failed: ",
                        String(e),
                    )
                    out = Self.Empty()
                    mask = Self.Empty()
            else:
                (out, mask) = self.unary_ops_with_mask_cpu[op_code]()
        else:
            (out, mask) = self.unary_ops_with_mask_cpu[op_code]()

        return (out^, mask^)

    @always_inline
    fn unary_ops_with_mask_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """Apply unary operation returning both output and mask on CPU."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            var result = self.storage.buffer.unary_ops_with_mask[op_code](
                start, end
            )
            var out_ndb = NDBuffer[Self.dtype](result[0], self.layout.shape)
            var mask_ndb = NDBuffer[Self.dtype](result[1], self.layout.shape)
            return (out_ndb^, mask_ndb^)
        else:
            var numels = self.numels()
            var out_buf = Buffer[Self.dtype](numels)
            var mask_buf = Buffer[Self.dtype](numels)
            var zero = Scalar[Self.dtype](0)
            var one = Scalar[Self.dtype](1)
            var index = 0
            for idx in self.index_iterator():
                var val = self.storage.buffer[idx]
                comptime if op_code == RELU_FORWARD:
                    if val > zero:
                        out_buf[index] = val
                        mask_buf[index] = one
                    else:
                        out_buf[index] = zero
                        mask_buf[index] = zero
                else:
                    out_buf[index] = val
                    mask_buf[index] = one
                index += 1
            return (
                NDBuffer[Self.dtype](out_buf^, self.layout.shape),
                NDBuffer[Self.dtype](mask_buf^, self.layout.shape),
            )

    # =================================================================
    # Reduction operations.
    # =================================================================

    fn sum_all(self) -> Scalar[Self.dtype]:
        """Return the sum of all elements."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            return self.storage.buffer.sum(start, end)
        else:
            var accum_sum: Scalar[Self.dtype] = Scalar[Self.dtype](0)
            for index in self.index_iterator():
                accum_sum += self.storage.buffer[index]
            return accum_sum

    fn sum(
        self, normalized_axes: IntArray, keepdims: Bool = False
    ) -> NDBuffer[Self.dtype]:
        """Return the sum over specified axes."""
        return self.reduce[mean=False](normalized_axes, keepdims)

    fn reduce[
        mean: Bool = False
    ](self, normalized_axes: IntArray, keepdims: Bool = False) -> NDBuffer[
        Self.dtype
    ]:
        """Reduce over specified axes with optional mean calculation."""
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    out = Reduction[Self.dtype].launch[mean=mean](
                        self, normalized_axes, keepdims
                    )
                except e:
                    print(e)
                    panic(
                        "NDBuffer sum - GPU operation failed for: ",
                        "mean: ",
                        String(mean),
                    )
                    out = NDBuffer[Self.dtype].Empty()
            else:
                out = self.reduce_cpu[mean=mean](normalized_axes, keepdims)
        else:
            out = self.reduce_cpu[mean=mean](normalized_axes, keepdims)

        return out^

    fn reduce_cpu[
        mean: Bool = False
    ](self, normalized_axes: IntArray, keepdims: Bool,) -> NDBuffer[Self.dtype]:
        """Reduce over specified axes on CPU with optional mean calculation."""
        var reduced_volume = Scalar[Self.dtype](1)

        comptime if mean:
            var volume = self.layout.shape.reduced_shape(
                normalized_axes
            ).product()
            reduced_volume = reduced_volume if volume == 0 else Scalar[
                Self.dtype
            ](volume)

        var out_shape = self.layout.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            comptime if mean:
                out[IntArray()] = self.sum_all() / reduced_volume
            else:
                out[IntArray()] = self.sum_all()
        else:
            reduction_axes_shape = self.layout.shape.reduced_shape(
                normalized_axes
            )

            for out_coord in out_shape:
                var accum_sum = Scalar[Self.dtype](0)
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum_sum += self[self_coord]

                comptime if mean:
                    out[out_coord] = accum_sum / reduced_volume
                else:
                    out[out_coord] = accum_sum

        return out^

    fn log_sum(
        self, normalized_axes: IntArray, keepdims: Bool = False
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Return log(sum(exp(x))) over specified axes."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return Reduction[Self.dtype].launch_log_sum(
                        self, normalized_axes, keepdims
                    )
                except e:
                    print(e)
                    panic("NDBuffer log_sum - GPU failed, falling back to CPU")
                    return NDBuffer[Self.dtype].Empty()
            else:
                return self.reduce_log_sum_cpu(normalized_axes, keepdims)
        else:
            return self.reduce_log_sum_cpu(normalized_axes, keepdims)

    fn reduce_log_sum_cpu(
        self, normalized_axes: IntArray, keepdims: Bool
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Return log(sum(exp(x))) over specified axes on CPU."""
        var out_shape = self.layout.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            var accum = Scalar[Self.dtype](0)
            for index in self.index_iterator():
                accum += exp(self.storage.buffer[index])
            out[IntArray()] = log(max(accum, Epsilon[Self.dtype].value()))
        else:
            var reduction_axes_shape = self.layout.shape.reduced_shape(
                normalized_axes
            )
            for out_coord in out_shape:
                var accum = Scalar[Self.dtype](0)
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum += exp(self[self_coord])
                out[out_coord] = log(max(accum, Epsilon[Self.dtype].value()))

        return out^

    fn minmax[
        is_max: Bool
    ](
        self, axes: IntArray, keepdims: Bool = False, paired: Bool = False
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Return min or max over specified axes with optional mask."""
        ref shape = self.layout.shape
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var (result_ndb, mask_ndb) = ReductionMinMax[
                        Self.dtype
                    ].launch[is_max=is_max](self, normalized_axes, keepdims)
                    return result_ndb, mask_ndb
                except e:
                    print(e)
                    panic("NDBuffer minmax: gpu path failed")
                    return (
                        NDBuffer[Self.dtype].Empty(),
                        NDBuffer[Self.dtype].Empty(),
                    )
            else:
                return self.minmax_cpu[is_max](
                    normalized_axes, keepdims, paired
                )
        else:
            return self.minmax_cpu[is_max](normalized_axes, keepdims, paired)

    fn minmax_cpu[
        is_max: Bool
    ](
        self,
        normalized_axes: IntArray,
        keepdims: Bool = False,
        paired: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Return min or max over specified axes on CPU."""
        var result_ndb = MinMaxReducer[Self.dtype].reduce_minmax[is_max](
            self, normalized_axes, keepdims
        )
        if paired:
            var mask_ndb = MinMaxReducer[Self.dtype].build_minmax_mask[is_max](
                self, result_ndb, normalized_axes, keepdims
            )
            return result_ndb, mask_ndb
        else:
            return result_ndb, NDBuffer[Self.dtype].Empty()

    # =================================================================
    # Softmax operations.
    # =================================================================

    fn softmax(
        self, axes: IntArray, validated: Bool = False
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Apply softmax activation over specified axes."""
        var normalized_axes = (
            axes if validated else Validator.validate_and_normalize_axes(
                self.layout.shape, axes
            )
        )
        var (_, stable_exp) = self._softmax_components(normalized_axes)
        var exp_sum = stable_exp.sum(normalized_axes, keepdims=True)
        return stable_exp / exp_sum

    fn log_softmax(
        self, axes: IntArray, validated: Bool = False
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Apply log softmax activation over specified axes."""
        var normalized_axes = (
            axes if validated else Validator.validate_and_normalize_axes(
                self.layout.shape, axes
            )
        )
        var (stable, stable_exp) = self._softmax_components(normalized_axes)
        var log_sum_exp = stable.log_sum(normalized_axes, keepdims=True)
        var exp_sum = stable_exp.sum(normalized_axes, keepdims=True)
        return stable - log_sum_exp, stable_exp / exp_sum

    fn _softmax_components(
        self, normalized_axes: IntArray
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Return (stable, stable_exp) components for softmax."""
        var (max_values, _) = self.minmax[is_max=True](
            normalized_axes, keepdims=True
        )
        var stable = self - max_values
        var stable_exp = stable.exp()
        return stable, stable_exp

    # =================================================================
    # Matrix multiplication.
    # =================================================================

    fn matmul_2d(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """2D matrix multiplication."""
        ref A_shape = A.layout.shape
        ref B_shape = B.layout.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var C: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if A.is_on_gpu() and B.is_on_gpu():
                try:
                    C = MatmulNdGpu[Self.dtype].launch[tile_size=TILE_SIZE](
                        A, B
                    )
                except e:
                    print(e)
                    panic("NDBuffer matmul_2d → GPU operation failed")
                    C = NDBuffer[Self.dtype](Shape())
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                panic(
                    (
                        "NDBuffer matmul_2d → both buffers must be on gpu. A is"
                        " on gpu?"
                    ),
                    String(A.is_on_gpu()),
                    ", B is on gpu?",
                    String(B.is_on_gpu()),
                )
                C = NDBuffer[Self.dtype](Shape())
            else:
                C = A.matmul_2d_cpu(B)
        else:
            C = A.matmul_2d_cpu(B)

        return C^

    fn matmul_2d_cpu(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """2D matrix multiplication on CPU."""
        ref A_shape = A.layout.shape
        ref B_shape = B.layout.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        comptime tile_size = TILE_SIZE
        comptime simdwidth = simd_width_of[Self.dtype]()

        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]

        var C = NDBuffer[Self.dtype].zeros(Shape([m, p]))

        ref A_strides = A.layout.strides
        var A_stride0 = A_strides[0]
        var A_stride1 = A_strides[1]
        var A_offset = A.layout.offset
        var A_data = A.data_ptr()

        ref B_strides = B.layout.strides
        var B_stride0 = B_strides[0]
        var B_stride1 = B_strides[1]
        var B_offset = B.layout.offset
        var B_data = B.data_ptr()
        var C_data = C.data_ptr()

        if B.is_contiguous():
            var num_tiles_i = (m + tile_size - 1) // tile_size

            @parameter
            fn process_row_tile(tile_idx: Int):
                var i_tile = tile_idx * tile_size
                var i_end = min(i_tile + tile_size, m)

                for j_tile in range(0, p, tile_size):
                    for k_tile in range(0, n, tile_size):
                        var j_end = min(j_tile + tile_size, p)
                        var k_end = min(k_tile + tile_size, n)

                        for i in range(i_tile, i_end):
                            var a_row_base = i * A_stride0 + A_offset
                            var c_row_base = i * p

                            var j = j_tile

                            while j + simdwidth <= j_end:
                                var c_addr = c_row_base + j
                                var accumulator = C_data.load[width=simdwidth](
                                    c_addr
                                )

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    var a_ik = A_data[a_addr]

                                    var b_addr = (
                                        k * B_stride0 + B_offset + j * B_stride1
                                    )
                                    var b_vec = B_data.load[width=simdwidth](
                                        b_addr
                                    )

                                    accumulator += a_ik * b_vec

                                C_data.store[width=simdwidth](
                                    c_addr, accumulator
                                )
                                j += simdwidth

                            while j < j_end:
                                var c_addr = c_row_base + j
                                var accumulator = C_data[c_addr]

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    var b_addr = (
                                        k * B_stride0 + B_offset + j * B_stride1
                                    )
                                    accumulator += (
                                        A_data[a_addr] * B_data[b_addr]
                                    )

                                C_data[c_addr] = accumulator
                                j += 1

            parallelize[process_row_tile](num_tiles_i, num_physical_cores())

        else:
            for i in range(m):
                var a_row_base = i * A_stride0 + A_offset
                var c_row_base = i * p

                for j in range(p):
                    var accumulator: Scalar[Self.dtype] = 0

                    for k in range(n):
                        var a_addr = a_row_base + k * A_stride1
                        var b_addr = k * B_stride0 + B_offset + j * B_stride1
                        accumulator += A_data[a_addr] * B_data[b_addr]

                    var c_addr = c_row_base + j
                    C_data[c_addr] = accumulator

        return C^

    fn matmul_nd(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """Batch matrix multiplication."""
        var A_shape = A.layout.shape
        var B_shape = B.layout.shape

        var A_rank = A_shape.rank()
        var B_rank = B_shape.rank()

        if A_rank < 2 or B_rank < 2:
            panic("NDBuffer → matmul_nd: inputs must be at least 2D")

        var k_A = A_shape[A_rank - 1]
        var k_B = B_shape[B_rank - 2]

        if k_A != k_B:
            panic(
                "NDBuffer → matmul_nd: inner dims must match, got "
                + String(k_A)
                + " and "
                + String(k_B)
            )

        var C: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if A.is_on_gpu() and B.is_on_gpu():
                try:
                    C = MatmulNdGpu[Self.dtype].launch[tile_size=TILE_SIZE](
                        A, B
                    )
                except e:
                    print(e)
                    panic("NDBuffer matmul_nd → GPU operation failed")
                    C = Self.Empty()
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                panic(
                    (
                        "NDBuffer matmul_nd → both buffers must be on gpu. A is"
                        " on gpu?"
                    ),
                    String(A.is_on_gpu()),
                    ", B is on gpu?",
                    String(B.is_on_gpu()),
                )
                C = NDBuffer[Self.dtype](Shape())
            else:
                C = A.matmul_nd_cpu(B)
        else:
            C = A.matmul_nd_cpu(B)

        return C^

    fn matmul_nd_cpu(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """Batch matrix multiplication on CPU."""
        var A_shape = A.layout.shape
        var B_shape = B.layout.shape

        var A_rank = A_shape.rank()
        var B_rank = B_shape.rank()

        var m = A_shape[A_rank - 2]
        var k_A = A_shape[A_rank - 1]
        var k_B = B_shape[B_rank - 2]
        var p = B_shape[B_rank - 1]

        if k_A != k_B:
            panic(
                "NDBuffer → matmul_nd: inner dims must match, got "
                + String(k_A)
                + " and "
                + String(k_B)
            )

        comptime tile_size = TILE_SIZE
        comptime simdwidth = simd_width_of[Self.dtype]()

        var k = k_A

        var A_batch_shape = A_shape[:-2]
        var B_batch_shape = B_shape[:-2]

        if not ShapeBroadcaster.broadcastable(A_batch_shape, B_batch_shape):
            panic(
                "NDBuffer → matmul_nd: batch shapes not broadcastable: "
                + String(A_batch_shape)
                + " vs "
                + String(B_batch_shape)
            )

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            A_batch_shape, B_batch_shape
        )
        var total_batch = batch_shape.product()
        if total_batch == 0:
            total_batch = 1

        var out_shape = batch_shape + Shape(m, p)
        var C = NDBuffer[Self.dtype].zeros(out_shape)

        var A_batch_rank = A_batch_shape.rank()
        var B_batch_rank = B_batch_shape.rank()
        var batch_rank = batch_shape.rank()

        ref A_strides = A.layout.strides
        ref B_strides = B.layout.strides

        var A_batch_strides = A_strides[:-2]
        var B_batch_strides = B_strides[:-2]

        var A_row_stride = A_strides[A_rank - 2]
        var A_col_stride = A_strides[A_rank - 1]
        var B_row_stride = B_strides[B_rank - 2]
        var B_col_stride = B_strides[B_rank - 1]

        var A_offset = A.layout.offset
        var B_offset = B.layout.offset

        var A_data = A.data_ptr()
        var B_data = B.data_ptr()
        var C_data = C.data_ptr()

        var B_contiguous = B.is_contiguous()

        var num_tiles_i = (m + tile_size - 1) // tile_size
        var total_tiles = total_batch * num_tiles_i

        @parameter
        fn process_tile(flat_idx: Int):
            var batch = flat_idx // num_tiles_i
            var tile_idx = flat_idx % num_tiles_i

            var A_base_off = A_offset
            var B_base_off = B_offset

            if batch_rank > 0:
                var coords = List[Int](capacity=batch_rank)
                for _ in range(batch_rank):
                    coords.append(0)
                var remaining = batch
                for dim in reversed(range(batch_rank)):
                    coords[dim] = remaining % batch_shape[dim]
                    remaining //= batch_shape[dim]

                var A_rank_off = batch_rank - A_batch_rank
                for i in range(A_batch_rank):
                    var coord = (
                        coords[A_rank_off + i] if A_batch_shape[i] > 1 else 0
                    )
                    A_base_off += coord * A_batch_strides[i]

                var B_rank_off = batch_rank - B_batch_rank
                for i in range(B_batch_rank):
                    var coord = (
                        coords[B_rank_off + i] if B_batch_shape[i] > 1 else 0
                    )
                    B_base_off += coord * B_batch_strides[i]

            var C_base_off = batch * m * p

            var i_tile = tile_idx * tile_size
            var i_end = min(i_tile + tile_size, m)

            if B_contiguous:
                for j_tile in range(0, p, tile_size):
                    for k_tile in range(0, k, tile_size):
                        var j_end = min(j_tile + tile_size, p)
                        var k_end = min(k_tile + tile_size, k)

                        for i in range(i_tile, i_end):
                            var a_row_base = A_base_off + i * A_row_stride
                            var c_row_base = C_base_off + i * p

                            var j = j_tile

                            while j + simdwidth <= j_end:
                                var c_addr = c_row_base + j
                                var accumulator = C_data.load[width=simdwidth](
                                    c_addr
                                )

                                for kk in range(k_tile, k_end):
                                    var a_addr = a_row_base + kk * A_col_stride
                                    var a_ik = A_data[a_addr]
                                    var b_addr = (
                                        B_base_off
                                        + kk * B_row_stride
                                        + j * B_col_stride
                                    )
                                    var b_vec = B_data.load[width=simdwidth](
                                        b_addr
                                    )
                                    accumulator += a_ik * b_vec

                                C_data.store[width=simdwidth](
                                    c_addr, accumulator
                                )
                                j += simdwidth

                            while j < j_end:
                                var c_addr = c_row_base + j
                                var accumulator = C_data[c_addr]

                                for kk in range(k_tile, k_end):
                                    var a_addr = a_row_base + kk * A_col_stride
                                    var b_addr = (
                                        B_base_off
                                        + kk * B_row_stride
                                        + j * B_col_stride
                                    )
                                    accumulator += (
                                        A_data[a_addr] * B_data[b_addr]
                                    )

                                C_data[c_addr] = accumulator
                                j += 1

            else:
                for i in range(i_tile, i_end):
                    var a_row_base = A_base_off + i * A_row_stride
                    var c_row_base = C_base_off + i * p

                    for j in range(p):
                        var accumulator = Scalar[Self.dtype](0)

                        for kk in range(k):
                            var a_addr = a_row_base + kk * A_col_stride
                            var b_addr = (
                                B_base_off
                                + kk * B_row_stride
                                + j * B_col_stride
                            )
                            accumulator += A_data[a_addr] * B_data[b_addr]

                        C_data[c_row_base + j] = accumulator

        parallelize[process_tile](total_tiles, num_physical_cores())

        return C^

    # =================================================================
    # Copy and fill operations.
    # =================================================================

    fn copy_from_alike[
        overwrite: Bool = True, validate: Bool = True
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]):
        """Copy data from another buffer with matching shape."""
        comptime if validate:
            if not self.layout.shape == other.layout.shape:
                panic(
                    (
                        "NDBuffer → copy_from_alike(other): dimension mismatch:"
                        " self.layout.shape"
                    ),
                    String(self.layout.shape),
                    "≠",
                    "other shape",
                    String(other.layout.shape),
                )

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.layout.offset
            other_start = other.layout.offset
            self_end = self_start + self.numels()
            other_end = other_start + other.numels()

            comptime if overwrite:
                self.storage.buffer.inplace_ops[Overwrite, validate=validate](
                    other.storage.buffer,
                    self_start,
                    self_end,
                    other_start,
                    other_end,
                )
            else:
                self.storage.buffer.inplace_ops[Add, validate=validate](
                    other.storage.buffer,
                    self_start,
                    self_end,
                    other_start,
                    other_end,
                )

        elif self.is_contiguous() and not other.is_contiguous():
            var index = self.layout.offset
            for idx in other.index_iterator():
                comptime if overwrite:
                    self.storage.buffer[index] = other.storage.buffer[idx]
                else:
                    self.storage.buffer[index] += other.storage.buffer[idx]
                index += 1

        elif not self.is_contiguous() and other.is_contiguous():
            var index = other.layout.offset
            for idx in self.index_iterator():
                comptime if overwrite:
                    self.storage.buffer[idx] = other.storage.buffer[index]
                else:
                    self.storage.buffer[idx] += other.storage.buffer[index]
                index += 1

        else:
            var iterator = other.index_iterator()
            for index in self.index_iterator():
                comptime if overwrite:
                    try:
                        self.storage.buffer[index] = other.storage.buffer[
                            iterator.__next__()
                        ]
                    except e:
                        print(e)
                        panic(
                            "Raised StopIteration in NDBuffer → copy_from_alike"
                        )
                else:
                    try:
                        self.storage.buffer[index] += other.storage.buffer[
                            iterator.__next__()
                        ]
                    except e:
                        print(e)
                        panic(
                            "Raised StopIteration in NDBuffer → copy_from_alike"
                        )

    # =================================================================
    # Counting and unique operations.
    # =================================================================

    fn count(self, key: Scalar[Self.dtype]) -> Int:
        """Count occurrences of key in the buffer."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var contig = self.contiguous_device_state()
                    var numels = self.numels()
                    var total = 0

                    with contig.buffer.map_to_host() as host_buffer:
                        var ptr = host_buffer.unsafe_ptr()

                        comptime if Self.dtype == DType.bool:
                            var key_u8 = UInt8(1) if key.cast[
                                DType.bool
                            ]() else UInt8(0)
                            var key_storage = key_u8.cast[
                                DeviceState[Self.dtype].datatype
                            ]()
                            for i in range(numels):
                                if ptr[i] == key_storage:
                                    total += 1
                        else:
                            comptime simd_width = simd_width_of[
                                DeviceState[Self.dtype].datatype
                            ]()
                            var key_storage = key.cast[
                                DeviceState[Self.dtype].datatype
                            ]()
                            var idx = 0
                            while idx + simd_width <= numels:
                                var block = ptr.load[width=simd_width](idx)
                                var result = block.eq(key_storage)
                                if result.reduce_and():
                                    total += simd_width
                                elif result.reduce_or():
                                    for i in range(simd_width):
                                        if result[i]:
                                            total += 1
                                idx += simd_width
                            for i in range(idx, numels):
                                if ptr[i] == key_storage:
                                    total += 1

                    return total
                except e:
                    panic("NDBuffer count GPU failed: " + String(e))
                    return 0

        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            return self.storage.buffer.count(key, start, end)

        var _count = 0
        for index in self.index_iterator():
            if self.storage.buffer[index] == key:
                _count += 1
        return _count

    fn unique(self) -> NDBuffer[Self.dtype]:
        """Return a buffer containing unique values."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var contig = self.contiguous_device_state()
                    var numels = self.numels()
                    var uniques = Set[Scalar[Self.dtype]]()

                    with contig.buffer.map_to_host() as host_buffer:
                        var ptr = host_buffer.unsafe_ptr()

                        comptime if Self.dtype == DType.bool:
                            var key_storage = UInt8(1).cast[
                                DeviceState[Self.dtype].datatype
                            ]()
                            for i in range(numels):
                                uniques.add(
                                    Scalar[Self.dtype](ptr[i] == key_storage)
                                )
                        else:
                            for i in range(numels):
                                uniques.add(ptr[i].cast[Self.dtype]())

                    var distincts = List[Scalar[Self.dtype]](
                        capacity=len(uniques)
                    )
                    for elem in uniques:
                        distincts.append(elem)
                    var unique_shape = Shape(len(distincts))
                    return NDBuffer[Self.dtype](
                        Buffer[Self.dtype](distincts^), unique_shape
                    )
                except e:
                    panic("NDBuffer unique GPU failed: " + String(e))
                    return NDBuffer[Self.dtype].Empty()

        var uniques = Set[Scalar[Self.dtype]]()
        if self.is_contiguous():
            if not self.shared():
                for i in range(self.numels()):
                    uniques.add(self.storage.buffer[i])
            else:
                var start = self.layout.offset
                var end = start + self.numels()
                for i in range(start, end):
                    uniques.add(self.storage.buffer[i])
        else:
            for index in self.index_iterator():
                uniques.add(self.storage.buffer[index])

        var distincts = List[Scalar[Self.dtype]](capacity=Int(len(uniques)))
        for elem in uniques:
            distincts.append(elem)
        var unique_shape = Shape(len(distincts))
        return NDBuffer[Self.dtype](
            Buffer[Self.dtype](distincts^), unique_shape
        )

    # =================================================================
    # Map operations.
    # =================================================================

    fn map_where(
        self, pred: fn(Scalar[Self.dtype]) -> Bool, value: Scalar[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """Replace elements where predicate is true with given value."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var mapped_ds = DeviceState[Self.dtype].map_where(
                        self, pred, value
                    )
                    return NDBuffer[Self.dtype].with_device_state(
                        mapped_ds^, self.layout.shape
                    )
                except e:
                    print(e)
                    panic("NDBuffer map_where error")
                    return NDBuffer[Self.dtype].Empty()
            else:
                return self.map_where_cpu(pred, value)
        return self.map_where_cpu(pred, value)

    @always_inline
    fn map_where_cpu(
        self, pred: fn(Scalar[Self.dtype]) -> Bool, value: Scalar[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """Replace elements where predicate is true with given value on CPU."""
        var buffer = Buffer[Self.dtype](len(self))
        var index = 0
        for next in self.index_iterator():
            buffer[index] = value if pred(self.get(next)) else self.get(next)
            index += 1
        return NDBuffer[Self.dtype](buffer^, self.layout.shape)

    fn map[
        map_buffer: fn(Buffer[Self.dtype]) -> Buffer[Self.dtype],
        map_element: fn(Scalar[Self.dtype]) -> Scalar[Self.dtype],
    ](self) -> Buffer[Self.dtype]:
        """Apply a mapping function to each element."""
        if self.is_contiguous():
            var start = self.layout.offset
            var end = start + self.numels()
            return map_buffer(self.storage.buffer[start:end])
        else:
            var buffer = Buffer[Self.dtype](self.numels())
            var index = 0
            for idx in self.index_iterator():
                buffer[index] = map_element(self.storage.buffer[idx])
                index += 1
            return buffer^

    fn map_to_bool(
        self,
        pred: fn(Scalar[Self.dtype]) -> Bool,
    ) -> NDBuffer[DType.bool]:
        """Apply predicate to each element, returning a boolean NDBuffer."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var cpu_ndb = self.to_cpu()
                    var bool_buffer = cpu_ndb.contiguous_buffer().map_to_bool(
                        pred
                    )
                    return NDBuffer[DType.bool](bool_buffer^, self.layout.shape)
                except e:
                    panic("NDBuffer map_to_bool: GPU→CPU failed: " + String(e))
                    return NDBuffer[DType.bool].Empty()

        var bool_buffer = self.contiguous_buffer().map_to_bool(pred)
        return NDBuffer[DType.bool](bool_buffer^, self.layout.shape)

    # =================================================================
    # Shuffle operation.
    # =================================================================

    fn shuffle(
        self,
        permutation: List[Int],
        axis: Int,
    ) -> NDBuffer[Self.dtype]:
        """Shuffle the buffer along specified axis using permutation."""
        var shape = self.layout.shape
        var result = NDBuffer[Self.dtype].zeros(shape)
        for coord in shape:
            var src_coord = coord
            src_coord[axis] = permutation[coord[axis]]
            result[coord] = self[src_coord]
        return result^

    # =================================================================
    # Reduction helper.
    # =================================================================

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_buffer: NDBuffer[Self.dtype], target_shape: Shape
    ) -> NDBuffer[Self.dtype]:
        """Reduce broadcasted dimensions to match target shape."""
        if extended_buffer.layout.shape == target_shape:
            return extended_buffer
        var result: NDBuffer[Self.dtype]
        if extended_buffer.is_on_cpu():
            result = extended_buffer.contiguous()
        else:
            result = extended_buffer
        var current_shape = result.layout.shape
        while len(current_shape) > len(target_shape):
            result = result.reduce(normalized_axes=IntArray(0), keepdims=False)
            current_shape = result.layout.shape
        for i in range(len(target_shape)):
            if current_shape[i] != target_shape[i] and current_shape[i] > 1:
                result = result.reduce(
                    normalized_axes=IntArray(i), keepdims=True
                )
                current_shape = result.layout.shape
        return result^

    # =================================================================
    # Identity and comparison.
    # =================================================================

    @always_inline
    fn __is__(self, other: NDBuffer[Self.dtype]) -> Bool:
        """Check if two buffers share the same underlying data."""
        if self.is_on_cpu() and other.is_on_cpu():
            return self.data_ptr() == other.data_ptr()
        elif self.is_on_gpu() and other.is_on_gpu():
            return (
                self.storage.device_state.value()
                == other.storage.device_state.value()
            )
        return False

    # =================================================================
    # Helper functions.
    # =================================================================

    @staticmethod
    @always_inline
    fn scalar_fn[
        op_code: Int,
    ](
        lhs: Scalar[Self.dtype],
        rhs: Scalar[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Scalar[Self.dtype]:
        """Apply scalar operation to two scalars."""
        comptime if op_code == Add:
            return lhs + rhs
        elif op_code == Subtract:
            return lhs - rhs
        elif op_code == ReverseSubtract:
            return rhs - lhs
        elif op_code == Multiply:
            return lhs * rhs
        elif op_code == Divide:
            return lhs / rhs
        elif op_code == MAX:
            return max(lhs, rhs)
        elif op_code == MIN:
            return min(lhs, rhs)
        elif op_code == SIGMOID_BACKWARD:
            return rhs * lhs * (One[Self.dtype].value() - lhs)
        elif op_code == TANH_BACKWARD:
            return rhs * (One[Self.dtype].value() - lhs * lhs)
        elif op_code == POW:
            return lhs**rhs
        else:
            return rhs / lhs

    @staticmethod
    @always_inline
    fn unary_fn_helper[
        op_code: Int
    ](scalar: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Apply unary operation to a scalar."""
        comptime if op_code == NEGATE:
            return -scalar
        elif op_code == SQRT:
            return sqrt(scalar)
        else:
            return scalar.__abs__()

    @staticmethod
    @always_inline
    fn float_unary_fn_helper[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](scalar: Scalar[Self.dtype]) -> Scalar[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Apply floating-point unary operation to a scalar."""
        comptime if op_code == LOG:
            return log(max(scalar, epsilon))
        elif op_code == SIGMOID_FORWARD:
            return One[Self.dtype].value() / (
                One[Self.dtype].value() + exp(scalar)
            )
        elif op_code == TANH_FORWARD:
            return tanh(scalar)
        else:
            return exp(scalar)

    # =================================================================
    # String representation.
    # =================================================================

    fn __str__(self) -> String:
        """Return a string representation of the buffer."""
        s = String("NDBuffer [")
        s += "Shape: " + String(self.layout.shape)
        s += ", Type: " + String(Self.dtype)
        s += ", Shared : " + String(self.shared())
        s += ", Strides : " + String(self.layout.strides)
        s += ", Offset : " + String(self.layout.offset)
        s += ", Contiguous : " + String(self.layout.is_contiguous())
        s += ", Buffer size : " + String(self.size())
        s += (
            ", Device : "
            + "gpu: "
            + String(self.gpu_id()) if self.is_on_gpu() else ", Device : "
            + "cpu"
        )
        s += "]"
        return s

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        """Print the buffer contents."""
        print(
            "\n",
            String(self),
            end="\n",
        )
        empty = List[Int]()
        print_buffer[Self.dtype](
            self,
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn __repr__(self) -> String:
        """Return a string representation of the buffer."""
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        """Write the string representation to a writer."""
        writer.write(self.__str__())
