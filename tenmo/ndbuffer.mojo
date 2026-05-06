from .shapes import Shape
from .strides import Strides
from .buffers import Buffer
from .intarray import IntArray
from .indexhelper import IndexCalculator, IndexIterator
from .matrixshapevalidator import MatrixShapeValidator
from .broadcasthelper import ShapeBroadcaster
from .common_utils import panic, log_debug, print_buffer, Epsilon, One
from .validators import Validator
from std.memory import memcpy, AddressSpace
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
from .reduction_kernel import Reduction, ProductArg
from .layernorm_kernel import LayerNormKernel
from .minmax_kernel import ReductionMinMax
from .minmax_reducer import MinMaxReducer
from .std_variance_backward_kernel import StdVarianceBackwardKernel
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
    INVERT,
    LOG,
    EXP,
    SIGMOID_FORWARD,
    SIGMOID_BACKWARD,
    TANH_FORWARD,
    TANH_BACKWARD,
    RELU_FORWARD,
    SQRT_BACKWARD,
    Overwrite,
    ReverseDivide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
    SUM,
    MEAN,
)


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
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self._contiguous = strides.is_contiguous(shape)

    fn __copyinit__(out self, copy: Self):
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self._contiguous = copy._contiguous

    fn __moveinit__(out self, deinit take: Self):
        self.shape = take.shape^
        self.strides = take.strides^
        self.offset = take.offset
        self._contiguous = take._contiguous

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.shape == other.shape
            and self.strides == other.strides
            and self.offset == other.offset
        )

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self._contiguous

    @always_inline
    fn num_elements(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn rank(self) -> Int:
        return self.shape.rank()


    @always_inline
    fn max_index(self) -> Int:
        """Calculate the highest accessible memory offset.

        For dimensions with positive strides, the maximum is reached at the
        last index of that dimension. For negative strides, the highest
        address is already at index 0 (the base offset), so those dimensions
        do not contribute.

        Returns:
            The highest valid memory offset.

        Example:
            ```mojo
            var buf = NDBuffer[DType.float32](Shape(3, 2), strides=Strides(4, -1), offset=10)
            print(buf.max_index())  # 10 + 1*4 = 14
            ```
        """
        var max_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] > 0:
                max_idx += (self.shape[i] - 1) * self.strides[i]
        return max_idx


struct Storage[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Pure data carrier — CPU buffer or GPU device state.
    No shape knowledge. No layout knowledge.
    copy() is cheap — just a refcount bump.
    """

    var buffer: Buffer[Self.dtype]
    var device_state: Optional[DeviceState[Self.dtype]]

    fn __init__(out self):
        self.buffer = Buffer[Self.dtype]()
        self.device_state = None

    fn __init__(out self, var buffer: Buffer[Self.dtype]):
        self.buffer = buffer^
        self.device_state = None

    fn __init__(out self, var device_state: DeviceState[Self.dtype]):
        self.buffer = Buffer[Self.dtype]()
        self.device_state = Optional(device_state^)

    fn __copyinit__(out self, copy: Self):
        self.buffer = copy.buffer.copy()  # Buffer refcount bump if shared
        self.device_state = copy.device_state.copy()  # Ref count bump for GPU

    fn __moveinit__(out self, deinit take: Self):
        self.buffer = take.buffer^
        self.device_state = take.device_state^

    @always_inline
    fn is_on_gpu(self) -> Bool:
        comptime if has_accelerator():
            return self.device_state is not None
        return False

    @always_inline
    fn is_on_cpu(self) -> Bool:
        return not self.is_on_gpu()

    fn copy(self) -> Self:
        """Explicit copy — refcount bump only, no data copy."""
        return self


comptime TILE_SIZE = 32


struct NDBuffer[dtype: DType](
    ImplicitlyCopyable & Movable & Equatable & Writable & Sized
):
    var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: Buffer[Self.dtype]
    var _contiguous: Bool
    var device_state: Optional[DeviceState[Self.dtype]]

    fn __init__(out self, *values: Scalar[Self.dtype]):
        buffer = Buffer[Self.dtype](len(values))
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
        self.device_state = None

        if buffer.size == 0:
            log_debug(
                "NDBuffer →__init__(Buffer, ...): zero sized buffer - potential"
                " danger"
            )
            self.buffer = buffer^
            self.shape = shape.or_else(Shape())
            self.strides = strides.or_else(Strides.Zero())
            self.offset = offset
            self._contiguous = True

        else:
            _shape = shape.or_else(Shape(buffer.size))
            self.shape = _shape.copy()
            self.buffer = buffer^
            self.strides = strides.or_else(Strides.default(_shape))
            self.offset = offset
            self._contiguous = False
            self._contiguous = self.is_contiguous()

    fn __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.buffer = Buffer[Self.dtype](shape.num_elements())
        self.shape = shape
        self.strides = strides.or_else(Strides.default(shape))
        self.offset = offset
        self._contiguous = False
        self.device_state = None
        self._contiguous = self.is_contiguous()

    fn __init__(
        out self,
        device_buffer: DeviceBuffer[Self.dtype],
        shape: Shape,
        *,
        copy: Bool = False,
    ) raises:
        var buffer: Buffer[Self.dtype]
        with device_buffer.map_to_host() as host_buffer:
            buffer = Buffer[Self.dtype](
                shape.num_elements(), host_buffer.unsafe_ptr(), copy=copy
            )
        self.buffer = buffer^
        self.shape = shape
        self.strides = Strides.default(shape)
        self.offset = 0
        self._contiguous = False
        self.device_state = None
        self._contiguous = self.is_contiguous()

    fn __moveinit__(out self, deinit take: Self):
        self.buffer = take.buffer^
        self.shape = take.shape^
        self.strides = take.strides^
        self.offset = take.offset
        self._contiguous = take._contiguous
        self.device_state = take.device_state^

    fn __copyinit__(out self, copy: Self):
        """Copy NDBuffer - Buffer handles ref counting automatically."""
        self.buffer = copy.buffer.copy()  # Buffer copy handles shared/unshared!
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self._contiguous = copy._contiguous
        self.device_state = copy.device_state.copy()

    @staticmethod
    fn with_device_state(
        var device_state: DeviceState[Self.dtype], shape: Shape
    ) -> NDBuffer[Self.dtype]:
        var empty_cpu_buffer = Buffer[Self.dtype]()
        var ndb = NDBuffer[Self.dtype](
            empty_cpu_buffer^,
            shape=shape,
            strides=Strides.default(shape),
            offset=0,
        )
        ndb.device_state = device_state^
        return ndb^

    fn buffer_layout(self) -> Layout:
        return Layout(self.shape, self.strides, self.offset)

    fn buffer_storage(self) -> Storage[Self.dtype]:
        var storage = Storage[Self.dtype]()
        storage.buffer = self.buffer.copy()
        storage.device_state = self.device_state.copy()
        return storage^

    fn sync(self):
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.device_state.value().sync()
                except e:
                    print(e)
                    print("NDBuffer device state synchronization failed")

    @staticmethod
    @always_inline
    fn zeros(
        shape: Shape, device: Device = CPU().into()
    ) -> NDBuffer[Self.dtype]:
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
                    # Unreachable
                    return Self.Empty()
            else:
                return ndb^

    fn get_gpu(
        ref self,
    ) raises -> ref[self.device_state.value().gpu] GPU:
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.device_state.value().get_gpu()
            else:
                raise ("NDBuffer get_gpu: buffer is not on gpu")
        else:
            raise (
                "NDBuffer get_gpu: buffer not on gpu or system has no"
                " accelerator"
            )

    fn to_cpu(self) raises -> Self:
        var _, nd_buffer = self.to_device(CPU().into())
        return nd_buffer^

    fn to_gpu(self, gpu: GPU) raises -> Self:
        if self.buffer.size == 0:
            raise "NDBuffer -> to_gpu(): Empty buffer"
        return self.to_device(gpu.into())[1]

    fn device(self) -> Device:
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.device_state.value().get_gpu().into()
        return CPU().into()

    fn device_context(self) -> Optional[DeviceContext]:
        if self.is_on_gpu():
            return self.device_state.value().gpu[]
        return None

    fn get_device_state(
        ref self,
    ) raises -> ref[self.device_state.value()] DeviceState[Self.dtype]:
        if self.is_on_gpu():
            return self.device_state.value()
        raise "Not on any device"

    fn to_device(
        self, device: Device
    ) raises -> Tuple[Int, NDBuffer[Self.dtype]]:
        """
        Materialize this buffer onto another device.

        Returns:
            - None if already on target device.
            - New NDBuffer if transfer occurs.
        """

        # 1) Currently on CPU

        if not self.device_state:
            if device.is_cpu():
                print("NDBuffer -> to_device: already on CPU")
                return -1, self

            # CPU -> GPU
            var gpu = device.kind[GPU]
            # Allocate device storage
            var new_device_state = DeviceState[Self.dtype](self.numels(), gpu)
            # Fill from logical view (handles offset/strides)
            new_device_state.fill(self)
            # Create new NDBuffer:
            #   - contiguous
            #   - offset = 0
            #   - no CPU buffer
            var result = NDBuffer[Self.dtype].with_device_state(
                new_device_state, self.shape
            )
            return 0, result^

        # 2) Currently on GPU
        var curr_state = self.device_state.value()
        var curr_gpu = curr_state.gpu

        if device.is_gpu():
            var new_gpu = device.kind[GPU]

            if curr_gpu == new_gpu:
                print("NDBuffer -> to_device: current and new device is same")
                # Already on this GPU
                return -1, self

            # GPU -> different GPU
            # We materialize through CPU

            # First bring to CPU
            var ndb_buffer = curr_state.into(self.shape)

            # Then move CPU -> new GPU
            # This would return 0, NDBuffer
            return ndb_buffer.to_device(device)

        # ---------------------------------------
        # 3) GPU -> CPU
        # ---------------------------------------
        # Materialize contiguous CPU buffer
        # New NDBuffer alltogether!
        if self.is_contiguous():
            return 0, curr_state.into(self.shape)
        else:
            # Materialise respecting strides
            # Step 1: bring raw flat device buffer to CPU
            var flat_cpu = curr_state.into(Shape(len(curr_state)))
            # Step 2: create view with correct shape/strides/offset over flat data
            var viewed = flat_cpu.share(self.shape, self.strides, self.offset)
            # Step 3: materialise into contiguous CPU buffer
            var result = NDBuffer[Self.dtype](self.shape)
            result.copy_from_alike[overwrite=True, validate=False](viewed^)
            return 0, result^

    fn is_on_gpu(self) -> Bool:
        comptime if has_accelerator():
            return not self.device_state == None
        return False

    fn gpu_id(self) -> Int64:
        if self.is_on_gpu():
            return self.device_state.value().get_gpu().id
        return -1

    fn is_on_cpu(self) -> Bool:
        return self.is_on_gpu() == False

    @staticmethod
    @always_inline
    fn full(
        shape: Shape, scalar: Scalar[Self.dtype], device: Device = CPU().into()
    ) -> NDBuffer[Self.dtype]:
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
                    # Unreachable
                    return NDBuffer[Self.dtype].Empty()
            else:
                return ndb^

    @staticmethod
    fn onehot(
        indices: NDBuffer[Self.dtype],
        num_classes: Int,
        device: Optional[Device] = None,
        ignore_index: Optional[Int] = None,
    ) -> NDBuffer[Self.dtype]:
        """Convert NDBuffer of class indices to one-hot encoding.
        Args:
            indices: Tensor of shape (...,) containing class indices.
            num_classes: Number of classes.
            device: Target device.
            ignore_index: If provided, rows where index == ignore_index become all zeros.
        Returns: Tensor of shape (..., num_classes).
        """
        ref shape = indices.shape
        ref target_device = device.or_else(indices.device())
        var result = NDBuffer[Self.dtype].zeros(
            shape + [num_classes], device=target_device
        )

        var ignore_val = ignore_index.or_else(-1000000)  # sentinel

        for coord in shape:
            var class_index = indices[coord].__int__()

            # Skip ignored indices entirely — leave row as zeros
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

    fn shuffle(
        self,
        permutation: List[Int],
        axis: Int,
    ) -> NDBuffer[Self.dtype]:
        var shape = self.shape
        var result = NDBuffer[Self.dtype].zeros(shape)
        for coord in shape:
            var src_coord = coord
            src_coord[axis] = permutation[coord[axis]]
            result[coord] = self[src_coord]
        return result^

    @always_inline
    fn index_iterator(
        self,
    ) -> IndexIterator[origin_of(self.shape), origin_of(self.strides)]:
        return IndexIterator(
            shape=Pointer(to=self.shape).get_immutable(),
            strides=Pointer(to=self.strides).get_immutable(),
            start_offset=self.offset,
        )

    @staticmethod
    @always_inline
    fn Empty() -> NDBuffer[Self.dtype]:
        return NDBuffer[Self.dtype](Buffer[Self.dtype]())

    @staticmethod
    @always_inline
    fn arange(
        args: VariadicList[Scalar[Self.dtype], _],
    ) -> NDBuffer[Self.dtype]:
        var buffer = Buffer[Self.dtype].arange(args)
        var shape = Shape(buffer.size)
        return NDBuffer[Self.dtype](buffer^, shape^)

    @staticmethod
    @always_inline
    fn arange(
        *args: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        return Self.arange(args)

    @staticmethod
    @always_inline
    fn linspace(
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        steps: Int,
    ) -> NDBuffer[Self.dtype]:
        var buffer = Buffer[Self.dtype].linspace(start, end, steps)
        var shape = Shape(buffer.size)
        return NDBuffer[Self.dtype](buffer^, shape^)

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.strides.is_contiguous(self.shape)

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    fn __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.get(index)

    fn __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.set(index, value)

    fn __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, IntArray(indices), self.strides, self.offset
        )
        return self.get(index)

    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, IntArray(indices), self.strides, self.offset
        )
        self.set(index, value)

    fn __getitem__(self, indices: VariadicList[Int, _]) -> Scalar[Self.dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, IntArray(indices), self.strides, self.offset
        )
        return self.get(index)

    fn __setitem__(
        self, indices: VariadicList[Int, _], value: Scalar[Self.dtype]
    ):
        index = IndexCalculator.flatten_index(
            self.shape, IntArray(indices), self.strides, self.offset
        )
        self.set(index, value)

    @always_inline
    fn item(self) -> Scalar[Self.dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim"
                " buffer/singleton, got shape: "
                + String(self.shape)
            )
        return self.get(0)

    fn get(self, index: Int) -> Scalar[Self.dtype]:
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
            ref device_state = self.device_state.value()
            try:
                return device_state[idx]
            except e:
                print(e)
                panic("Error in NDBuffer → get: ", String(e))
                # Unreachable
                return Scalar[Self.dtype](0)
        return self.data_ptr()[idx]

    fn set(self, index: Int, value: Scalar[Self.dtype]):
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
            ref device_state = self.device_state.value()
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
            ref shape = self.shape

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

            if simdwidth > 1 and self.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD load requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + String(self.strides[1])
                    + ". "
                    + "Use .contiguous() or scalar loads."
                )

        var addr = row * self.strides[0] + col * self.strides[1] + self.offset
        if self.is_on_gpu():
            ref device_state = self.device_state.value()
            try:
                return device_state.load[simdwidth=simdwidth](addr).cast[
                    Self.dtype
                ]()
            except e:
                print(e)
                panic("Error in NDBuffer → get: ", String(e))
                # Unreachable
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
            ref shape = self.shape

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

            if simdwidth > 1 and self.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD store requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + String(self.strides[1])
                    + ". "
                    + "Use .contiguous() or scalar stores."
                )

        var addr = row * self.strides[0] + col * self.strides[1] + self.offset
        if self.is_on_gpu():
            ref device_state = self.device_state.value()
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

    fn __str__(self) -> String:
        s = String("NDBuffer [")
        s += "Shape: " + String(self.shape)
        s += ", Type: " + String(Self.dtype)
        s += ", Shared : " + String(self.shared())
        s += ", Strides : " + String(self.strides)
        s += ", Offset : " + String(self.offset)
        s += ", Contiguous : " + String(self.is_contiguous())
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
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    fn data_buffer(ref self) -> ref[self.buffer] Buffer[Self.dtype]:
        return self.buffer

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape()

    @always_inline
    fn numels(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn __len__(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn rank(self) -> Int:
        return self.shape.rank()

    @always_inline
    fn max_index(self) -> Int:
        var max_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] > 0:
                max_idx += (self.shape[i] - 1) * self.strides[i]
            # negative stride: highest address is already at offset,
            # no addition needed for this dimension
        return max_idx

    @always_inline
    fn min_index(self) -> Int:
        """Calculate the lowest accessible memory offset.

        For dimensions with negative strides, the minimum is reached at the
        last index of that dimension. For positive strides, the lowest
        address is already at index 0 (the base offset), so those dimensions
        do not contribute.

        Returns:
            The lowest valid memory offset.

        Example:
            ```mojo
            var buf = NDBuffer[DType.float32](Shape(3, 2), strides=Strides(4, -1), offset=10)
            print(buf.min_index())  # 10 + 1*(-1) = 9
            ```
        """
        var min_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] < 0:
                min_idx += (self.shape[i] - 1) * self.strides[i]
        return min_idx

    @always_inline
    fn offset_at(self, indices: IntArray) -> Int:
        """Return the absolute linear offset in the underlying buffer
        for the given multidimensional indices."""
        if indices.size() != self.rank():
            panic("NDBuffer.offset_at: Incorrect number of indices")

        return IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )

    fn to_dtype[NewType: DType](self) -> NDBuffer[NewType]:
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    # GPU → CPU, cast, CPU → GPU
                    var cpu_ndb = self.contiguous_device_state().into(
                        self.shape
                    )
                    var cast_ndb = cpu_ndb.to_dtype[
                        NewType
                    ]()  # CPU handles all cases correctly
                    var new_state = DeviceState[NewType](
                        self.numels(), self.device_state.value().gpu
                    )
                    new_state.fill(cast_ndb)
                    return NDBuffer[NewType].with_device_state(
                        new_state^, self.shape
                    )
                except e:
                    panic("NDBuffer to_dtype GPU failed: " + String(e))
                    return NDBuffer[NewType].Empty()  # unreachable

        # CPU path — unchanged
        var new_buffer = self.contiguous_buffer().to_dtype[NewType]()
        return NDBuffer[NewType](new_buffer^, self.shape)

    @always_inline
    fn __imul__(self, factor: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Multiply](factor)

    @always_inline
    fn __iadd__(self, scalar: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Add](scalar)

    @always_inline
    fn __isub__(self, scalar: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Multiply](other)

    @always_inline
    fn __iadd__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Add](other)

    @always_inline
    fn __isub__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Subtract](other)

    fn __itruediv__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Divide](other)

    @always_inline
    fn shared(self) -> Bool:
        """Check if underlying buffer is shared."""
        return self.buffer.is_shared()

    fn share(
        mut self,
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[Self.dtype]:
        """
        Create shared view of this buffer.
        First call enables ref counting. Subsequent calls just create views.
        """
        var size = len(self.buffer) if self.is_on_cpu() else len(
            self.device_state.value()
        )
        # Enable ref counting if not already shared
        if self.is_on_cpu() and size > 0 and not self.shared():
            self.buffer.shared()
        var new_shape = shape.or_else(self.shape)
        var new_strides = strides.or_else(Strides.default(new_shape))
        var max_index = IndexCalculator.max_index(
            new_shape, new_strides, offset
        )
        if max_index >= size:
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

        var ndb = NDBuffer[Self.dtype](
            buffer=self.buffer.copy(),
            shape=new_shape,
            strides=new_strides,
            offset=offset,
        )
        ndb.device_state = self.device_state.copy()
        return ndb^

    fn transpose(
        mut self,
        axes: IntArray = IntArray(),
        *,
        shared: Bool = True,
    ) -> NDBuffer[Self.dtype]:
        ref shape = self.shape
        var normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntArray.range(0, shape.rank()).reversed()
        )
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)

        if shared:
            # View: shared buffer — for Tensor view ops
            return self.share(new_shape, new_strides, self.offset)
        else:
            # Owned contiguous copy — for Gradbox
            var view = self.share(new_shape, new_strides, self.offset)
            return view.contiguous()

    @always_inline
    fn __is__(self, other: NDBuffer[Self.dtype]) -> Bool:
        if self.is_on_cpu() and other.is_on_cpu():
            return self.data_ptr() == other.data_ptr()
        elif self.is_on_gpu() and other.is_on_gpu():
            return self.device_state.value() == other.device_state.value()
        return False

    @always_inline
    fn data_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Scalar[Self.dtype], origin, address_space=address_space
    ]:
        return (
            self.buffer.unsafe_ptr()
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    @always_inline
    fn zero(self):
        self.fill(Scalar[Self.dtype](0))

    fn softmax(
        self, axes: IntArray, validated: Bool = False
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        var normalized_axes = (
            axes if validated else Validator.validate_and_normalize_axes(
                self.shape, axes
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
        var normalized_axes = (
            axes if validated else Validator.validate_and_normalize_axes(
                self.shape, axes
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
        var (max_values, _) = self.minmax[is_max=True](
            normalized_axes, keepdims=True
        )
        var stable = self - max_values
        var stable_exp = stable.exp()
        return stable, stable_exp

    @always_inline
    fn fill(self, value: Scalar[Self.dtype]):
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.device_state.value().fill(value)
                except e:
                    print(e)
                    panic("Error filling NDBuffer value: ", String(value))
            else:
                self.fill_cpu(value)
        else:
            self.fill_cpu(value)

    @always_inline
    fn fill_cpu(self, value: Scalar[Self.dtype]):
        ref buffer = self.data_buffer()
        if self.is_contiguous():
            buffer.fill(value, self.offset, self.offset + self.numels())
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            for index in self.index_iterator():
                (ptr + index)[] = value

    fn reshape(
        self, new_shape: Shape, validated: Bool = False
    ) -> NDBuffer[Self.dtype]:
        var shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            self.shape, new_shape.intarray()
        )

        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.reshape_gpu(shape)
        return self.contiguous(shape)

    fn reshape_gpu(
        self,
        shape: Shape,
    ) -> NDBuffer[Self.dtype]:
        var out: NDBuffer[Self.dtype]

        try:
            ref device_state = self.device_state.value()
            var new_state = device_state.new(self.numels(), 0, sync=False)
            new_state.fill(self, sync=True)
            out = NDBuffer[Self.dtype].with_device_state(new_state, shape)
        except e:
            print(e)
            panic("Error reshaping device buffer")
            # Unreachable
            out = NDBuffer[Self.dtype].Empty()
        return out^

    fn map_where(
        self, pred: fn(Scalar[Self.dtype]) -> Bool, value: Scalar[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var mapped_ds = DeviceState[Self.dtype].map_where(
                        self, pred, value
                    )
                    return NDBuffer[Self.dtype].with_device_state(
                        mapped_ds^, self.shape
                    )
                except e:
                    print(e)
                    panic("NDBuffer mapp_where error")
                    return NDBuffer[Self.dtype].Empty()
            else:
                return self.map_where_cpu(pred, value)
        return self.map_where_cpu(pred, value)

    @always_inline
    fn map_where_cpu(
        self, pred: fn(Scalar[Self.dtype]) -> Bool, value: Scalar[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        var buffer = Buffer[Self.dtype](len(self))
        var index = 0
        for next in self.index_iterator():
            buffer[index] = value if pred(self.get(next)) else self.get(next)
            index += 1
        return NDBuffer[Self.dtype](buffer^, self.shape)

    fn product_all(self) -> Scalar[Self.dtype]:
        """CPU only operation."""
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer.product(start, end)
        else:
            var product: Scalar[Self.dtype] = Scalar[Self.dtype](1)
            for index in self.index_iterator():
                product *= self.buffer[index]
            return product

    fn sum_all(self) -> Scalar[Self.dtype]:
        """CPU only operation."""
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer.sum(start, end)
        else:
            var accum_sum: Scalar[Self.dtype] = Scalar[Self.dtype](0)
            for index in self.index_iterator():
                accum_sum += self.buffer[index]
            return accum_sum

    fn sum(
        self, normalized_axes: IntArray, keepdims: Bool = False
    ) -> NDBuffer[Self.dtype]:
        return self.reduce[op_code=SUM](normalized_axes, keepdims)

    fn reduce[
        op_code: Int = SUM
    ](self, normalized_axes: IntArray, keepdims: Bool = False) -> NDBuffer[
        Self.dtype
    ]:
        """Sum / mean reduction. Axes must be already normalized.
        op_code: SUM or MEAN. For PRODUCT use product()."""

        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    out = Reduction[Self.dtype].launch[op_code](
                       self, normalized_axes, keepdims
                    )
                except e:
                    print(e)
                    panic(
                        "NDBuffer reduce — GPU operation failed for op_code: ",
                        String(op_code),
                    )
                    out = NDBuffer[Self.dtype].Empty()
            else:
                out = self.reduce_cpu[op_code](normalized_axes, keepdims)
        else:
            out = self.reduce_cpu[op_code](normalized_axes, keepdims)

        return out^

    fn reduce_cpu[
        op_code: Int = SUM
    ](self, normalized_axes: IntArray, keepdims: Bool) -> NDBuffer[Self.dtype]:
        """CPU sum / mean. op_code: SUM or MEAN."""
        var reduced_volume = Scalar[Self.dtype](1)

        comptime if op_code == MEAN:
            var volume = self.shape.reduced_shape(normalized_axes).product()
            reduced_volume = reduced_volume if volume == 0 else Scalar[
                Self.dtype
            ](volume)

        var out_shape = self.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            comptime if op_code == MEAN:
                out[IntArray()] = self.sum_all() / reduced_volume
            else:
                out[IntArray()] = self.sum_all()
        else:
            var reduction_axes_shape = self.shape.reduced_shape(normalized_axes)
            for out_coord in out_shape:
                var accum = Scalar[Self.dtype](0)
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum += self[self_coord]
                comptime if op_code == MEAN:
                    out[out_coord] = accum / reduced_volume
                else:
                    out[out_coord] = accum

        return out^

    # =============================================================================
    # Returns (mean_ndb, var_ndb) — variance already divided by n or n-1.
    # Callers:
    #   Variance.forward  → uses mean_ndb and var_ndb directly
    #   StdDev.forward    → uses mean_ndb, var_ndb, and sqrt(var_ndb) for std
    #
    # Both mean and var are returned so forward can save them into BwdArg
    # without any extra computation — they were computed for free by Welford.
    # =============================================================================

    fn welford(
        self: NDBuffer[Self.dtype],
        axes: IntArray,
        unbiased: Bool,
        keepdims: Bool,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Compute mean and variance in a single Welford pass.

        Returns (mean_ndb, var_ndb). Variance is already divided by n or n-1.
        Both are shaped according to keepdims and axis.

        No extra cost over computing variance alone — mean is free from Welford.
        Caller saves both into BwdArg for zero-recomputation backward.

        Args:
            axes:     Reduction axes.
            unbiased: If True divide by n-1, else divide by n.
            keepdims: If True keep reduced dim with size 1.

        Returns:
            Tuple of (mean NDBuffer, variance NDBuffer).
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return self._welford_gpu(axes, unbiased, keepdims)
                except e:
                    print(e)
                    panic("NDBuffer welford → GPU operation failed")
                    return (Self.Empty(), Self.Empty())
        return self._welford_cpu(axes, unbiased, keepdims)

    fn _welford_gpu(
        self: NDBuffer[Self.dtype],
        axes: IntArray,
        unbiased: Bool,
        keepdims: Bool,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var (mean_ndb, M2_ndb) = Reduction[Self.dtype].launch_welford(
            self, axes, keepdims
        )
        var n = self.shape.reduced_shape(axes).product()
        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)
        var var_ndb = M2_ndb.scalar_ops[Divide](divisor)
        return (mean_ndb^, var_ndb^)

    fn _welford_cpu(
        self: NDBuffer[Self.dtype],
        axes: IntArray,
        unbiased: Bool,
        keepdims: Bool,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var out_shape = self.shape.compute_output_shape(
            axes, keepdims, validated=True
        )
        var mean_out = NDBuffer[Self.dtype].zeros(out_shape)
        var var_out  = NDBuffer[Self.dtype].zeros(out_shape)

        var n = self.shape.reduced_shape(axes).product()
        var divisor = Scalar[Self.dtype](
            n - 1 if unbiased and n > 1 else n
        )

        if out_shape == Shape():
            # Global scalar reduction
            var local_mean = Scalar[Self.dtype](0)
            var local_M2   = Scalar[Self.dtype](0)
            var count = 0
            for idx in self.index_iterator():
                var x = self.buffer[idx]
                count += 1
                var delta  = x - local_mean
                local_mean += delta / Scalar[Self.dtype](count)
                var delta2 = x - local_mean
                local_M2   += delta * delta2
            mean_out[IntArray()] = local_mean
            var_out[IntArray()]  = local_M2 / divisor
        else:
            # Multi-axis reduction — mirrors reduce_cpu coord pattern exactly
            var reduction_axes_shape = self.shape.reduced_shape(axes)
            for out_coord in out_shape:
                var local_mean = Scalar[Self.dtype](0)
                var local_M2   = Scalar[Self.dtype](0)
                var count = 0
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        axes, red_coord
                    ) if keepdims else out_coord.insert(axes, red_coord)
                    var x = self[self_coord]
                    count += 1
                    var delta  = x - local_mean
                    local_mean += delta / Scalar[Self.dtype](count)
                    var delta2 = x - local_mean
                    local_M2   += delta * delta2
                mean_out[out_coord] = local_mean
                var_out[out_coord]  = local_M2 / divisor

        return (mean_out^, var_out^)

    # Returns Tuple[NDBuffer, ProductArg] — result + backward arg.
    # ProductArg is passed up to Product.forward which stores it in BackwardFnArg.

    fn product[
        store_excl_product: Bool = True,
    ](
        self,
        normalized_axes: IntArray,
        keepdims: Bool = False,
    ) -> Tuple[
        NDBuffer[Self.dtype], ProductArg[Self.dtype]
    ]:
        """Product reduction. Axes must be already normalized.

        Returns (result NDBuffer, ProductArg) — caller stores ProductArg
        in BackwardFnArg for backward pass.

        All dtypes. Float64 log-space accumulation — overflow safe.
        """
        var out: NDBuffer[Self.dtype]
        var arg: ProductArg[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var result = Reduction[Self.dtype].launch_product[
                        store_excl_product
                    ](self, normalized_axes, keepdims)
                    out = result[0]
                    arg = result[1]
                except e:
                    print(e)
                    panic(
                        "NDBuffer product — GPU operation failed: ", String(e)
                    )
                    # Unreachable
                    out = NDBuffer[Self.dtype].Empty()
                    arg = ProductArg[Self.dtype].Empty()
            else:
                (out, arg) = self.product_cpu[store_excl_product](
                    normalized_axes, keepdims
                )
        else:
            (out, arg) = self.product_cpu[store_excl_product](
                normalized_axes, keepdims
            )

        return (out^, arg^)

    fn product_cpu[
        store_excl_product: Bool = True,
    ](
        self,
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> Tuple[
        NDBuffer[Self.dtype], ProductArg[Self.dtype]
    ]:
        """CPU product reduction. Float64 log-space — overflow safe.

        Mirrors GPU product_reduce kernel logic exactly so CPU and GPU
        produce identical results (within float64 precision).
        """
        var out_shape = self.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)
        var zero_counts = NDBuffer[DType.int32].zeros(out_shape)

        var f64_zero = Scalar[DType.float64](0)

        if out_shape == Shape():
            # Full reduction to scalar
            var log_abs_sum = f64_zero
            var neg_count = 0
            var zero_count = 0
            for idx in self.index_iterator():
                var val = self.buffer[idx].cast[DType.float64]()
                if val == f64_zero:
                    zero_count += 1
                else:
                    if val < f64_zero:
                        neg_count += 1
                    log_abs_sum += log(abs(val))
            zero_counts[IntArray()] = Scalar[DType.int32](zero_count)
            if zero_count > 0:
                out[IntArray()] = Scalar[Self.dtype](0)
            else:
                var sign = Scalar[DType.float64](
                    -1 if neg_count % 2 == 1 else 1
                )
                # out[IntArray()] = (sign * exp(log_abs_sum)).cast[Self.dtype]()
                out[IntArray()] = Self._cast_result[Self.dtype](
                    sign * exp(log_abs_sum)
                )
        else:
            var reduction_axes_shape = self.shape.reduced_shape(normalized_axes)
            for out_coord in out_shape:
                var log_abs_sum = f64_zero
                var neg_count = 0
                var zero_count = 0
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    var val = self[self_coord].cast[DType.float64]()
                    if val == f64_zero:
                        zero_count += 1
                    else:
                        if val < f64_zero:
                            neg_count += 1
                        log_abs_sum += log(abs(val))
                zero_counts[out_coord] = Scalar[DType.int32](zero_count)
                if zero_count > 0:
                    out[out_coord] = Scalar[Self.dtype](0)
                else:
                    var sign = Scalar[DType.float64](
                        -1 if neg_count % 2 == 1 else 1
                    )
                    # out[out_coord] = (sign * exp(log_abs_sum)).cast[Self.dtype]()
                    out[out_coord] = Self._cast_result[Self.dtype](
                        sign * exp(log_abs_sum)
                    )

        # excl_product: compute now or defer to backward
        var excl_optional: Optional[NDBuffer[Self.dtype]] = None
        comptime if store_excl_product:
            excl_optional = Optional(
                self.excl_product_cpu(normalized_axes, keepdims)
            )

        var reduced_volume = self.shape.reduced_shape(normalized_axes).product()

        var arg = ProductArg[Self.dtype](
            input=self,
            excl_product=excl_optional^,
            zero_counts=zero_counts^,
            axes=normalized_axes,
            keepdims=keepdims,
            reduced_volume=reduced_volume,
        )

        return (out^, arg^)

    fn compute_excl_product(
        self,
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        """Compute exclusive product buffer for backward recompute path.

        Called by ProductBackward when store_excl_product=False.
        Dispatches to GPU kernel or CPU reference.
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var (_, productArg) = Reduction[Self.dtype].launch_product[
                        store_excl_product=True
                    ](self, normalized_axes, keepdims)
                    return productArg.excl_product.value()
                except e:
                    panic(
                        "NDBuffer compute_excl_product — GPU failed: ",
                        String(e),
                    )
                    return NDBuffer[Self.dtype].Empty()  # Unreachable
        return self.excl_product_cpu(normalized_axes, keepdims)

    fn excl_product_cpu(
        self,
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        """CPU exclusive product: excl[i] = product of all others in slice.

        Mirrors excl_product_kernel logic. Float64 log-space — overflow safe.
        Output is input-shaped.
        """
        var excl = NDBuffer[Self.dtype].zeros(self.shape)
        var f64_zero = Scalar[DType.float64](0)
        var out_shape = self.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var reduction_axes_shape = self.shape.reduced_shape(normalized_axes)

        # For each output (slice), compute total log_abs, neg_count, zero_count
        # then for each element in the slice compute excl via subtraction
        if out_shape == Shape():
            # Full reduction — one slice, all elements
            var total_log = f64_zero
            var total_neg = 0
            var total_zero = 0
            for idx in self.index_iterator():
                var val = self.buffer[idx].cast[DType.float64]()
                if val == f64_zero:
                    total_zero += 1
                else:
                    if val < f64_zero:
                        total_neg += 1
                    total_log += log(abs(val))

            for idx in self.index_iterator():
                var val = self.buffer[idx].cast[DType.float64]()
                excl.set(
                    idx,
                    Self._excl_one_cpu(
                        val, total_log, total_neg, total_zero, f64_zero
                    ),
                )
        else:
            for out_coord in out_shape:
                # Pass 1: totals for this slice
                var total_log = f64_zero
                var total_neg = 0
                var total_zero = 0
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    var val = self[self_coord].cast[DType.float64]()
                    if val == f64_zero:
                        total_zero += 1
                    else:
                        if val < f64_zero:
                            total_neg += 1
                        total_log += log(abs(val))

                # Pass 2: excl for each element in slice
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    var val = self[self_coord].cast[DType.float64]()
                    excl[self_coord] = Self._excl_one_cpu(
                        val, total_log, total_neg, total_zero, f64_zero
                    )

        return excl^

    @always_inline
    @staticmethod
    fn _cast_result[
        datatype: DType
    ](val: Scalar[DType.float64]) -> Scalar[datatype]:
        """Cast float64 log-space result back to dtype.
        Rounds to nearest integer for integral types before casting —
        prevents log/exp precision loss from producing 23 instead of 24.
        For floating point types, direct cast (no rounding needed).
        """
        comptime if datatype.is_integral():
            return round(val).cast[datatype]()
        else:
            return val.cast[datatype]()

    @staticmethod
    fn _excl_one_cpu(
        val: Scalar[DType.float64],
        total_log: Scalar[DType.float64],
        total_neg: Int,
        total_zero: Int,
        f64_zero: Scalar[DType.float64],
    ) -> Scalar[Self.dtype]:
        """Compute exclusive product for one element. CPU helper."""
        if total_zero > 1:
            return Scalar[Self.dtype](0)
        elif total_zero == 1:
            if val == f64_zero:
                # This is the zero — excl = product of all non-zero others
                var sign = Scalar[DType.float64](
                    -1 if total_neg % 2 == 1 else 1
                )
                # return (sign * exp(total_log)).cast[Self.dtype]()
                return Self._cast_result[Self.dtype](sign * exp(total_log))
            else:
                # Non-zero element in a one-zero slice → excl contains zero
                return Scalar[Self.dtype](0)
        else:
            # No zeros
            if val == f64_zero:
                return Scalar[Self.dtype](0)  # Shouldn't reach here
            var val_neg = 1 if val < f64_zero else 0
            var excl_log = total_log - log(abs(val))
            var excl_neg = total_neg - val_neg
            var sign = Scalar[DType.float64](-1 if excl_neg % 2 == 1 else 1)
            # return (sign * exp(excl_log)).cast[Self.dtype]()
            return Self._cast_result[Self.dtype](sign * exp(excl_log))

    fn log_sum(
        self, normalized_axes: IntArray, keepdims: Bool = False
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return Reduction[Self.dtype].launch_log_sum(
                        self, normalized_axes, keepdims
                    )
                except e:
                    print(e)
                    panic("NDBuffer log_sum - GPU failed, falling back to CPU")
                    # Not reachable
                    return NDBuffer[Self.dtype].Empty()
            else:
                return self.reduce_log_sum_cpu(normalized_axes, keepdims)
        else:
            return self.reduce_log_sum_cpu(normalized_axes, keepdims)

    fn reduce_log_sum_cpu(
        self, normalized_axes: IntArray, keepdims: Bool
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        var out_shape = self.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            var accum = Scalar[Self.dtype](0)
            for index in self.index_iterator():
                accum += exp(self.buffer[index])
            out[IntArray()] = log(max(accum, Epsilon[Self.dtype].value()))
        else:
            var reduction_axes_shape = self.shape.reduced_shape(normalized_axes)
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

    fn flatten(
        self,
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
    ) -> NDBuffer[Self.dtype]:
        rank = self.rank()
        if rank == 0:
            return self.contiguous()
        var endd = end_dim.or_else(rank - 1)

        if endd < start_dim:
            panic("NDBuffer → flatten: end_dim must be >= start_dim")

        var original_shape = self.shape
        # compute new shape
        collapsed = original_shape[start_dim : endd + 1].product()
        new_shape = (
            original_shape[:start_dim]
            + [collapsed]
            + original_shape[endd + 1 :]
        )
        return self.contiguous(new_shape)

    fn contiguous_buffer(self) -> Buffer[Self.dtype]:
        """Returns a contiguous copy of the buffer with the same data - CPU only.
        """
        # - same shape
        # - contiguous strides
        # - offset = 0
        # - copies data from original
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer[start:end]
        else:
            var buffer = Buffer[Self.dtype](self.numels())
            var index = 0
            for idx in self.index_iterator():
                buffer[index] = self.buffer[idx]
                index += 1
            return buffer^

    fn contiguous_device_state(self) raises -> DeviceState[Self.dtype]:
        """
        Returns a fresh independent contiguous DeviceState.
        Caller must ensure self is on GPU.
        Fast path: enqueue_copy_to for contiguous source.
        Slow path: DeviceState.fill for non-contiguous (handles strided iteration).
        """
        ref curr_state = self.device_state.value()
        ref gpu = curr_state.get_gpu()
        var new_state = DeviceState[Self.dtype](self.numels(), gpu)

        if self.is_contiguous():
            # Fast path: direct DeviceBuffer → DeviceBuffer copy, no host round-trip
            curr_state.buffer.enqueue_copy_to(new_state.buffer)
            new_state.sync()
        else:
            # Slow path: fill handles non-contiguous strided GPU source correctly
            # It materialises through CPU: GPU→CPU (strided) then CPU→GPU (contiguous)
            new_state.fill(
                self
            )  # fill(ref source: NDBuffer) — already handles this

        return new_state^

    fn contiguous(
        self, new_shape: Optional[Shape] = None
    ) -> NDBuffer[Self.dtype]:
        var target_shape = new_shape.or_else(self.shape)

        comptime if has_accelerator():
            if self.is_on_gpu():
                # Already contiguous on GPU with no shape change — but still need
                # a fresh independent DeviceState (unshared), so always materialise
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
                    # unreachable — satisfies compiler
                    return self

        # CPU path — unchanged
        if (
            self.is_contiguous()
            and not self.shared()
            and target_shape == self.shape
        ):
            return self
        return NDBuffer[Self.dtype](self.contiguous_buffer(), target_shape)

    fn squeeze(
        mut self, axes: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        var shape = self.shape
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
                new_strides_arr.append(self.strides[i])

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        if shared:
            # View: shared buffer, ref-counted — for Tensor view ops
            return self.share(new_shape, new_strides, self.offset)
        else:
            # Owned contiguous copy — for Gradbox
            # First build the view to get correct shape/strides, then materialise
            var view = self.share(new_shape, new_strides, self.offset)
            return view.contiguous()

    fn unsqueeze(
        mut self, axes: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        var rank = self.shape.rank()
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
                var insert_stride = self.strides[orig_i] if orig_i < rank else 1
                new_strides_arr.append(insert_stride)
                ins_i += 1
            else:
                new_shape_dims.append(self.shape[orig_i])
                new_strides_arr.append(self.strides[orig_i])
                orig_i += 1

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        if shared:
            return self.share(new_shape, new_strides, self.offset)
        else:
            var view = self.share(new_shape, new_strides, self.offset)
            return view.contiguous()

    fn permute(
        mut self, perm: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        """
        Permute axes of this NDBuffer.
        perm[i] = j means: new axis i takes old axis j.
        Example: perm=[2,0,1] on shape [A,B,C] → shape [C,A,B].

        shared=True  → view with reordered shape/strides, same buffer.
                       used by Tensor.permute — no data movement.
        shared=False → owned contiguous copy with permuted layout.
                       used by Gradbox.permute — GPU safe via contiguous().
        GPU safety: shared=True is always safe — just metadata.
                    shared=False delegates to contiguous() which handles.
                    GPU via contiguous_device_state().
        """
        var shape = self.shape
        var rank = shape.rank()

        if len(perm) != rank:
            panic(
                "NDBuffer → permute: number of axes (",
                String(len(perm)),
                ") must match rank (",
                String(rank),
                ")",
            )

        # Validate permutation — one pass, no O(n) `in` scan per element
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

        # Build permuted shape and strides
        var new_shape_dims = IntArray.with_capacity(rank)
        var new_strides_arr = IntArray.with_capacity(rank)
        for i in range(len(perm)):
            new_shape_dims.append(shape[perm[i]])
            new_strides_arr.append(self.strides[perm[i]])

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        if shared:
            # View — shared buffer, ref-counted — for Tensor view ops
            # No data movement, GPU safe
            return self.share(new_shape, new_strides, self.offset)
        else:
            # Owned contiguous copy — for Gradbox
            # Build view first, then materialise via contiguous()
            # contiguous() handles GPU via contiguous_device_state()
            var view = self.share(new_shape, new_strides, self.offset)
            return view.contiguous()

    fn count(self, key: Scalar[Self.dtype]) -> Int:
        """
        Count occurrences of key in the buffer.

        GPU path:
            1. contiguous_device_state() — materialises logical view correctly,
               handles offset (contiguous fast path) and non-contiguous strides
               (slow path via fill/index_iterator). Result is flat, offset=0.
            2. map_to_host once — SIMD vectorized count on flat buffer.
        CPU path:
            Contiguous: delegates to Buffer.count (SIMD vectorized).
            Non-contiguous: index_iterator.
        Result is always a CPU scalar Int.
        """

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Materialise logical view — handles offset + strides
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
                            # SIMD vectorized on flat contiguous buffer
                            comptime simd_width = simd_width_of[
                                DeviceState[Self.dtype].datatype
                            ]()
                            var key_storage = key.cast[
                                DeviceState[Self.dtype].datatype
                            ]()
                            var idx = 0
                            # Full SIMD chunks
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
                            # Scalar tail
                            for i in range(idx, numels):
                                if ptr[i] == key_storage:
                                    total += 1

                    return total
                except e:
                    panic("NDBuffer count GPU failed: " + String(e))
                    return 0  # unreachable
        # CPU contiguous fast path
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer.count(key, start, end)

        # CPU non-contiguous fallback
        var _count = 0
        for index in self.index_iterator():
            if self.buffer[index] == key:
                _count += 1
        return _count

    fn unique(self) -> NDBuffer[Self.dtype]:
        """
        Get unique values in the buffer.

        GPU path:
            1. contiguous_device_state() — materialises logical view correctly,
               handles offset (contiguous fast path) and non-contiguous strides
               (slow path via fill/index_iterator). Result is flat, offset=0.
            2. map_to_host once — collect uniques via Set on CPU.
        CPU path:
            Contiguous: iterates buffer directly respecting offset.
            Non-contiguous: index_iterator.
        Result is always a CPU NDBuffer — Set is CPU-side and unique
        results are typically small, no benefit keeping on GPU.
        """

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Materialise logical view — handles offset + strides
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
                    return NDBuffer[Self.dtype].Empty()  # unreachable

        # CPU contiguous fast path
        var uniques = Set[Scalar[Self.dtype]]()
        if self.is_contiguous():
            if not self.shared():
                for i in range(self.numels()):
                    uniques.add(self.buffer[i])
            else:
                var start = self.offset
                var end = start + self.numels()
                for i in range(start, end):
                    uniques.add(self.buffer[i])
        else:
            # CPU non-contiguous fallback
            for index in self.index_iterator():
                uniques.add(self.buffer[index])

        var distincts = List[Scalar[Self.dtype]](capacity=Int(len(uniques)))
        for elem in uniques:
            distincts.append(elem)
        var unique_shape = Shape(len(distincts))
        return NDBuffer[Self.dtype](
            Buffer[Self.dtype](distincts^), unique_shape
        )

    fn copy_from_alike[
        overwrite: Bool = True, validate: Bool = True
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]):
        comptime if validate:
            if not self.shape == other.shape:
                panic(
                    (
                        "NDBuffer → copy_from_alike(other):"
                        " dimension mismatch: self shape"
                    ),
                    String(self.shape),
                    "≠",
                    "other shape",
                    String(other.shape),
                )

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            other_start = other.offset
            self_end = self_start + self.numels()
            other_end = other_start + other.numels()

            comptime if overwrite:
                self.buffer.inplace_ops[Overwrite, validate=validate](
                    other.buffer, self_start, self_end, other_start, other_end
                )
            else:
                self.buffer.inplace_ops[Add, validate=validate](
                    other.buffer, self_start, self_end, other_start, other_end
                )

        elif self.is_contiguous() and not other.is_contiguous():
            var index = self.offset
            for idx in other.index_iterator():
                comptime if overwrite:
                    self.buffer[index] = other.buffer[idx]
                else:
                    self.buffer[index] += other.buffer[idx]
                index += 1

        elif not self.is_contiguous() and other.is_contiguous():
            var index = other.offset
            for idx in self.index_iterator():
                comptime if overwrite:
                    self.buffer[idx] = other.buffer[index]
                else:
                    self.buffer[idx] += other.buffer[index]
                index += 1

        else:
            var iterator = other.index_iterator()
            for index in self.index_iterator():
                comptime if overwrite:
                    try:
                        self.buffer[index] = other.buffer[iterator.__next__()]
                    except e:
                        print(e)
                        panic(
                            "Raised StopIteration in NDBuffer → "
                            " copy_from_alike"
                        )
                else:
                    try:
                        self.buffer[index] += other.buffer[iterator.__next__()]
                    except e:
                        print(e)
                        panic(
                            "Raised StopIteration in NDBuffer → "
                            " copy_from_alike"
                        )

    fn fill(self, cpu_buffer: NDBuffer[Self.dtype]):
        """Fill this NDBuffer from a CPU NDBuffer."""
        if cpu_buffer.is_scalar() or cpu_buffer.shape == Shape.Unit():
            self.fill(
                cpu_buffer.item()
            )  # Scalar/Singleton NDBuffer - shared or otherwise
            return

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.device_state.value().fill(cpu_buffer)
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
        if self.__is__(other):
            panic("NDBuffer → fill_cpu: cannot fill with self")

        if self.shape == other.shape:
            self.copy_from_alike[overwrite=True, validate=True](other)
        else:
            # Handle broadcast
            if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
                panic(
                    (
                        "NDBuffer → fill_cpu(other): dimension mismatch: self"
                        " shape"
                    ),
                    String(self.shape),
                    "≠",
                    "other shape",
                    String(other.shape),
                )
            var broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.shape, other.shape
            )
            if broadcast_shape != self.shape:
                panic(
                    "NDBuffer → fill_cpu: broadcasted shape must match receiver"
                    " shape"
                )

            # self.shape -> Target shape
            # other.shape -> Source shape

            mask = ShapeBroadcaster.broadcast_mask(other.shape, self.shape)
            for coord in self.shape:
                src_coord = ShapeBroadcaster.translate_index(
                    other.shape, coord, mask, self.shape
                )
                self[coord] = other[src_coord]

    @always_inline
    fn inplace_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]):
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + String(self.shape)
                + ", "
                + String(other.shape)
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
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + String(self.shape)
                + ", "
                + String(other.shape)
            )

        # Handle broadcasting case
        if self.shape != other.shape:
            broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.shape, other.shape
            )

            # PyTorch's rule: broadcasted shape must match receiver shape
            if broadcast_shape != self.shape:
                panic(
                    "NDBuffer → inplace_ops: broadcasted shape "
                    + String(broadcast_shape)
                    + " must match receiver shape "
                    + String(self.shape)
                )

            # Get the broadcasted result which is now of self's shape
            var broadcast_result = self.broadcast_buffer[op_code](other)
            self.copy_from_alike[overwrite=True, validate=False](
                broadcast_result^
            )

        else:
            # Same shape case
            if self.is_contiguous() and other.is_contiguous():
                self_start = self.offset
                self_end = self_start + self.numels()
                other_start = other.offset
                other_end = other_start + other.numels()
                self.buffer.inplace_ops[op_code](
                    other.buffer, self_start, self_end, other_start, other_end
                )

            elif self.is_contiguous() and not other.is_contiguous():
                var index = self.offset
                for idx in other.index_iterator():
                    self.buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[index], other.buffer[idx]
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var index = other.offset
                for idx in self.index_iterator():
                    self.buffer[idx] = Self.scalar_fn[op_code](
                        self.buffer[idx], other.buffer[index]
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

                    self.buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[index], other.buffer[next_index]
                    )

    @always_inline
    fn inplace_scalar_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]):
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
                    # Unreacahble
            else:
                self.inplace_scalar_ops_cpu[op_code](scalar)
        else:
            self.inplace_scalar_ops_cpu[op_code](scalar)

    @always_inline
    fn inplace_scalar_ops_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]):
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        if self.is_contiguous():
            start = self.offset
            end = start + self.numels()
            self.buffer.inplace_ops_scalar[op_code](scalar, start, end)

        else:
            for index in self.index_iterator():
                self.buffer[index] = Self.scalar_fn[op_code](
                    self.buffer[index], scalar
                )

    @always_inline
    fn __add__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Add](other)

    @always_inline
    fn __neg__(self) -> NDBuffer[Self.dtype]:
        return self.unary_ops[NEGATE]()

    @always_inline
    fn __abs__(self) -> NDBuffer[Self.dtype]:
        return self.unary_ops[ABS]()

    @always_inline
    fn log[
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[LOG, epsilon]()

    @always_inline
    fn exp(self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[EXP]()

    @always_inline
    fn sigmoid(
        self,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[SIGMOID_FORWARD]()

    @always_inline
    fn tanh(self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[TANH_FORWARD]()

    @always_inline
    fn __mul__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Multiply](other)

    @always_inline
    fn __mul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Multiply](scalar)

    @always_inline
    fn __add__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Add](scalar)

    @always_inline
    fn __sub__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Subtract](scalar)

    @always_inline
    fn max(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[MAX](scalar)

    @always_inline
    fn min(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[MIN](scalar)

    @always_inline
    fn __pow__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[POW](scalar)

    @always_inline
    fn __rmul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.__mul__(scalar)

    @always_inline
    fn __sub__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Subtract](other)

    @always_inline
    fn __truediv__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Divide](other)

    @always_inline
    fn __truediv__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Divide](scalar)

    @always_inline
    fn __rtruediv__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[ReverseDivide](scalar)

    @always_inline
    fn arithmetic_ops[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → arithmetic_ops: dimension mismatch: "
                + String(self.shape)
                + ", "
                + String(other.shape)
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
                    # Unreachable
                    out = NDBuffer[Self.dtype].Empty()
            elif (self.is_on_gpu() and other.is_on_cpu()) or (
                self.is_on_cpu() and other.is_on_gpu()
            ):
                panic(
                    "NDBuffer arithmetic_ops - both tensors must be on the same"
                    " device - they are not"
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
        # Handle broadcasting case
        if self.shape != other.shape:
            return self.broadcast_buffer[op_code](other)

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            self_end = self_start + self.numels()
            other_start = other.offset
            other_end = other_start + other.numels()
            var result_buffer = self.buffer.arithmetic_ops[op_code=op_code](
                other.buffer,
                self_start,
                self_end,
                other_start,
                other_end,
                epsilon=epsilon,
            )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var result_buffer = Buffer[Self.dtype](self.numels())
            var index = 0

            if self.is_contiguous() and not other.is_contiguous():
                var offset = self.offset
                for idx in other.index_iterator():
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[offset + index], other.buffer[idx], epsilon
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var offset = other.offset
                for idx in self.index_iterator():
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[idx], other.buffer[offset + index], epsilon
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
                        self.buffer[idx], other.buffer[next_index], epsilon
                    )
                    index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @always_inline
    fn broadcast_buffer[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        if self.shape.rank() == 0 or other.shape.rank() == 0:
            return self.broadcast_scalar_buffer[op_code](other)
        else:
            return self.broadcast_nd_buffer[op_code](other)

    @always_inline
    fn broadcast_scalar_buffer[
        op_code: Int
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        result_shape = other.shape if self.shape.rank() == 0 else self.shape
        var buffer = Buffer[Self.dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[coord]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[coord]
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
        result_shape = ShapeBroadcaster.broadcast_shape(self.shape, other.shape)

        mask1 = ShapeBroadcaster.broadcast_mask(self.shape, result_shape)
        mask2 = ShapeBroadcaster.broadcast_mask(other.shape, result_shape)

        var buffer = Buffer[Self.dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            self_coord = ShapeBroadcaster.translate_index(
                self.shape, coord, mask1, result_shape
            )
            other_coord = ShapeBroadcaster.translate_index(
                other.shape, coord, mask2, result_shape
            )
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[op_code](
                self[self_coord], other[other_coord]
            )
        return NDBuffer[Self.dtype](buffer^, result_shape)

    fn broadcast_to(self, target_shape: Shape) -> NDBuffer[Self.dtype]:
        """
        Broadcast this NDBuffer to target_shape.
        Uses stride=0 trick for broadcast dims — pure metadata, no data copy.
        Then contiguous() materialises the view correctly on CPU or GPU.

        GPU safe: share() is metadata only, contiguous() uses
                  contiguous_device_state() on GPU.
        CPU safe: contiguous() uses contiguous_buffer().
        """
        if not ShapeBroadcaster.expandable_to(self.shape, target_shape):
            panic(
                "NDBuffer.broadcast_to: cannot expand "
                + String(self.shape)
                + " to "
                + String(target_shape)
            )

        var own_shape = self.shape
        var own_rank = own_shape.rank()
        var target_rank = target_shape.rank()

        var extra_dims = target_rank - own_rank

        # Build expanded strides — prepend zeros for extra leading dims
        var new_strides = IntArray.with_capacity(target_rank)

        # Extra leading dims — stride 0 (broadcast)
        for _ in range(extra_dims):
            new_strides.append(0)

        # Align existing dims — stride 0 where dim==1 and target>1
        for i in range(own_rank):
            var target_i = i + extra_dims
            if own_shape[i] == 1 and target_shape[target_i] > 1:
                new_strides.append(0)  # broadcast dim
            else:
                new_strides.append(self.strides[i])  # keep original stride

        # Create non-contiguous view with broadcast strides
        var self_copy = self.copy()
        var view = self_copy.share(
            target_shape, Strides(new_strides), self.offset
        )

        # Materialise — handles GPU via contiguous_device_state()
        return view.contiguous()

    @staticmethod
    @always_inline
    fn scalar_fn[
        op_code: Int,
    ](
        lhs: Scalar[Self.dtype],
        rhs: Scalar[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Scalar[Self.dtype]:
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
            # lhs = sigmoid output, rhs = grad
            return rhs * lhs * (One[Self.dtype].value() - lhs)

        elif op_code == SQRT_BACKWARD:
            return rhs * (
                One[Self.dtype].value()
                / (epsilon + Scalar[Self.dtype](2) * sqrt(lhs))
            )
        elif op_code == TANH_BACKWARD:
            # lhs = tanh output, rhs = grad
            return rhs * (One[Self.dtype].value() - lhs * lhs)

        elif op_code == POW:
            return lhs**rhs

        else:  # op_code == ReverseDivide
            return rhs / lhs

    @staticmethod
    @always_inline
    fn float_unary_fn_helper[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](scalar: Scalar[Self.dtype]) -> Scalar[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        comptime if op_code == LOG:
            return log(max(scalar, epsilon))
        elif op_code == SIGMOID_FORWARD:
            return One[Self.dtype].value() / (
                One[Self.dtype].value() + exp(scalar)
            )
        elif op_code == TANH_FORWARD:
            return tanh(scalar)
        else:  # op_code == EXP:
            return exp(scalar)

    @always_inline
    fn scalar_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
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
                    # Unreacahble
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
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer: Buffer[Self.dtype]

            comptime if op_code == POW:
                result_buffer = self.buffer[start:end] ** scalar
            else:
                result_buffer = self.buffer.arithmetic_ops_scalar[op_code](
                    scalar, start, end
                )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = Self.scalar_fn[op_code](
                    self.buffer[idx], scalar, epsilon
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @always_inline
    fn unary_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
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
                    # Unreacahble
                    out = Self.Empty()
            else:
                out = self.unary_ops_cpu[op_code]()
        else:
            out = self.unary_ops_cpu[op_code]()

        return out^

    @always_inline
    fn unary_ops_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer = self.buffer.unary_ops[op_code](start, end)

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = Self.unary_fn_helper[op_code](
                    self.buffer[idx]
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @staticmethod
    @always_inline
    fn unary_fn_helper[
        op_code: Int
    ](scalar: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        comptime if op_code == NEGATE:
            return -scalar
        elif op_code == SQRT:
            return sqrt(scalar)
        else:  # op_code == ABS:
            return scalar.__abs__()

    @always_inline
    fn unary_ops_with_mask[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """Device-aware unary op + mask. Returns (output, mask).

        CPU path: delegates to Buffer.unary_ops_with_mask (vectorized).
        GPU path: single kernel pass via UnaryOpsKernel.launch_with_mask.

        Non-contiguous input is handled efficiently on both paths:
        - CPU: index_iterator() loop (acceptable, matches existing slow path).
        - GPU: contiguous_device_state() does ONE map_to_host copy, then
               the kernel runs on the flat buffer — no per-element host calls.

        Returns:
            Tuple of (output NDBuffer, mask NDBuffer).
            Both share the same device as self.
        """
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
                    # Unreachable
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
        """CPU path for unary_ops_with_mask. Delegates to Buffer."""
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result = self.buffer.unary_ops_with_mask[op_code](start, end)
            var out_ndb = NDBuffer[Self.dtype](result[0], self.shape)
            var mask_ndb = NDBuffer[Self.dtype](result[1], self.shape)
            return (out_ndb^, mask_ndb^)
        else:
            # Non-contiguous CPU slow path
            var numels = self.numels()
            var out_buf = Buffer[Self.dtype](numels)
            var mask_buf = Buffer[Self.dtype](numels)
            var zero = Scalar[Self.dtype](0)
            var one = Scalar[Self.dtype](1)
            var index = 0
            for idx in self.index_iterator():
                var val = self.buffer[idx]
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
                NDBuffer[Self.dtype](out_buf^, self.shape),
                NDBuffer[Self.dtype](mask_buf^, self.shape),
            )

    @always_inline
    fn float_unary_ops[
        op_code: Int,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """For LOG/EXP/SIGMOID/TANH."""
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
                    # Unreacahble
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
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer = self.buffer.float_unary_ops[op_code, epsilon](
                start, end
            )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)
        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = Self.float_unary_fn_helper[
                    op_code, epsilon
                ](self.buffer[idx])
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    fn layernorm_normalize(
        self: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        var_: NDBuffer[Self.dtype],
        gamma: NDBuffer[Self.dtype],
        beta: NDBuffer[Self.dtype],
        eps: Scalar[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """Fused normalize: rstd + x_hat + out in single pass.

        Pass 2 of LayerNorm forward — Welford (Pass 1) already ran.
        Returns (out_ndb, x_hat_ndb, rstd_ndb).
        out and x_hat are shape (*, D). rstd is shape (*, 1).

        Args:
         self:  Input (*, D). Must be contiguous.
         mean:  Per-row mean (*, 1) from Welford.
         var_:  Per-row variance (*, 1) from Welford.
         gamma: Scale (D,).
         beta:  Shift (D,).
         eps:   Numerical stability constant.
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return LayerNormKernel[Self.dtype].launch(
                        self, mean, var_, gamma, beta, eps
                    )
                except e:
                    print(e)
                    panic("NDBuffer layernorm_normalize → GPU operation failed")
                    return (
                        Self.Empty(),
                        Self.Empty(),
                        Self.Empty(),
                    )  # unreachable
        return self.layernorm_normalize_cpu(mean, var_, gamma, beta, eps)

    fn layernorm_normalize_cpu(
        self: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        var_: NDBuffer[Self.dtype],
        gamma: NDBuffer[Self.dtype],
        beta: NDBuffer[Self.dtype],
        eps: Scalar[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """CPU fused normalize — serial over rows, element-wise per row."""
        var D = self.shape[-1]
        var outer_size = self.numels() // D
        var out_shape = self.shape
        var rstd_shape = out_shape[0:-1] + [1]

        var out_buf = Buffer[Self.dtype](self.numels())
        var x_hat_buf = Buffer[Self.dtype](self.numels())
        var rstd_buf = Buffer[Self.dtype](outer_size)

        for row in range(outer_size):
            var row_mean = mean.buffer[row]
            var row_var = var_.buffer[row]
            var safe_var = row_var + eps
            # rsqrt(var + eps) = 1/sqrt(var+eps) — rstd directly
            var rstd = rsqrt(
                safe_var if safe_var
                > Scalar[Self.dtype](0) else Scalar[Self.dtype](eps)
            )
            rstd_buf[row] = rstd
            var row_base = row * D
            for i in range(D):
                var x_i = self.buffer[self.offset + row_base + i]
                var x_hat_i = (x_i - row_mean) * rstd
                var out_i = gamma.buffer[i] * x_hat_i + beta.buffer[i]
                x_hat_buf[row_base + i] = x_hat_i
                out_buf[row_base + i] = out_i

        var out_ndb = NDBuffer[Self.dtype](out_buf^, out_shape)
        var x_hat_ndb = NDBuffer[Self.dtype](x_hat_buf^, out_shape)
        var rstd_ndb = NDBuffer[Self.dtype](rstd_buf^, rstd_shape)
        return (out_ndb^, x_hat_ndb^, rstd_ndb^)

    fn variance_backward_normalize(
        self: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        scale: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Fused variance backward: (x - mean) * scale.

        Single pass over (*, D).
        x (self) may be non-contiguous — stride-aware on CPU and GPU.
        out is always a fresh contiguous allocation.

        Mean is (*, 1) keepdims=True — one scalar per row.
        scale = 2 / divisor — uniform across all rows.

        Caller multiplies result by upstream grad (pass 2).

        Replaces in VarianceBackward:
            diff       = x - mean         pass 1
            local_grad = diff * scale     pass 2
            (upstream multiply is still pass 2, done by caller)

        Args:
            self:  Input x (*, D). May be strided view.
            mean:  Per-row mean (*, 1) from VarianceBwdArg.
            scale: 2 / divisor scalar.

        Returns:
            Contiguous NDBuffer (*, D) — (x - mean) * scale.
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return StdVarianceBackwardKernel[
                        Self.dtype
                    ].launch_variance_backward(self, mean, scale)
                except e:
                    print(e)
                    panic("NDBuffer variance_backward_normalize → GPU failed")
                    return Self.Empty()
        return self._variance_backward_normalize_cpu(mean, scale)

    fn _variance_backward_normalize_cpu(
        self: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        scale: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """CPU variance backward normalize — stride-aware via self[coord]."""
        var D = self.shape[-1]
        # var outer_size = self.numels() // D
        var out_buf = Buffer[Self.dtype](self.numels())
        var out_shape = self.shape

        # Outer loop — iterate over all row coordinates
        # Use index_iterator on the non-last dims to handle any stride layout
        var row = 0
        for outer_coord in self.shape[0:-1]:
            var row_mean = mean.buffer[row]  # mean is (*, 1) contiguous
            var out_base = row * D
            for i in range(D):
                # Build full coord: outer_coord + [i]
                var full_coord = outer_coord.insert(len(outer_coord), i)
                out_buf[out_base + i] = (self[full_coord] - row_mean) * scale
                # Equivalent: (self[coord] - row_mean) * scale
                # Written as two muls to avoid a temporary sub tensor
            row += 1

        return NDBuffer[Self.dtype](out_buf^, out_shape)

    fn std_backward_normalize(
        self: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        denom: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Fused std backward: (x - mean) / denom.

        Single pass over (*, D).
        x (self) may be non-contiguous — stride-aware on CPU and GPU.
        out is always a fresh contiguous allocation.

        Mean is (*, 1) keepdims=True — one scalar per row.
        Denom is (*, 1) — (std + eps) * divisor, one scalar per row.

        Caller multiplies result by upstream grad (pass 2).

        Replaces in StdBackward:
            diff       = x - mean              pass 1
            local_grad = diff / denom          pass 2
            (upstream multiply is still pass 2, done by caller)

        Args:
            self:  Input x (*, D). May be strided view.
            mean:  Per-row mean (*, 1) from StdBwdArg.
            denom: Per-row denominator (*, 1) — (std+eps)*divisor.

        Returns:
            Contiguous NDBuffer (*, D) — (x - mean) / denom.
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return StdVarianceBackwardKernel[
                        Self.dtype
                    ].launch_std_backward(self, mean, denom)
                except e:
                    print(e)
                    panic("NDBuffer std_backward_normalize → GPU failed")
                    return Self.Empty()
        return self._std_backward_normalize_cpu(mean, denom)

    fn _std_backward_normalize_cpu(
        self: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        denom: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """CPU std backward normalize — stride-aware via self[coord]."""
        var D = self.shape[-1]
        # var outer_size = self.numels() // D
        var out_buf = Buffer[Self.dtype](self.numels())
        var out_shape = self.shape

        var row = 0
        for outer_coord in self.shape[0:-1]:
            var row_mean = mean.buffer[row]  # (*, 1) contiguous
            var row_denom = denom.buffer[row]  # (*, 1) contiguous
            var out_base = row * D
            for i in range(D):
                var full_coord = outer_coord.insert(len(outer_coord), i)
                out_buf[out_base + i] = (
                    (self[full_coord] - row_mean)
                ) / row_denom
            row += 1

        return NDBuffer[Self.dtype](out_buf^, out_shape)

    fn minmax[
        is_max: Bool
    ](
        self, axes: IntArray, keepdims: Bool = False, paired: Bool = False
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        ref shape = self.shape
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
                    # Unreachable
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

    fn clamp(
        self: NDBuffer[Self.dtype],
        lower_bound: Scalar[Self.dtype],
        upper_bound: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer = self.buffer.clamp(
                lower_bound, upper_bound
            ) if start == 0 else self.buffer[start:end].clamp(
                lower_bound, upper_bound
            )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = self.buffer[idx].clamp(
                    lower_bound, upper_bound
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    fn clamp_in_place(
        self: NDBuffer[Self.dtype],
        lower_bound: Scalar[Self.dtype],
        upper_bound: Scalar[Self.dtype],
    ):
        if (
            self.is_contiguous()
        ):  # Use only when whole underlying buffer(Buffer) is contiguous - for example Gradbox buffer
            self.buffer.clamp_in_place(lower_bound, upper_bound)
        else:
            for idx in self.index_iterator():
                self.buffer[idx] = self.buffer[idx].clamp(
                    lower_bound, upper_bound
                )

    fn __eq__(self, other: Self) -> Bool:
        var ndb = self.compare[Equal](other)
        if ndb.is_on_gpu():
            return ndb.device_state.value().all_true()
        return ndb.buffer.all_true()

    fn __ne__(self, other: Self) -> Bool:
        var ndb = self.compare[NotEqual](other)
        if ndb.is_on_gpu():
            return ndb.device_state.value().all_true()
        return ndb.buffer.all_true()

    @always_inline
    fn compare[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        DType.bool
    ]:
        if not self.shape == other.shape:
            panic(
                "NDBuffer → compare(self, other): dimension mismatch: "
                + String(self.shape)
                + "≠"
                + String(other.shape)
            )
        var result: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    result = Compare[Self.dtype].launch[op_code](self, other)
                except e:
                    print(e)
                    panic("NDBuffer compare → GPU operation failed")
                    # Not reachable
                    result = NDBuffer[DType.bool].Empty()
            elif (self.is_on_gpu() and other.is_on_cpu()) or (
                self.is_on_cpu() and other.is_on_gpu()
            ):
                panic(
                    "NDBuffer compare → not both buffers are no the same device"
                )
                # Not reachable
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
        if self.is_contiguous() and other.is_contiguous():
            var self_contiguous = self.contiguous_buffer()
            var other_contiguous = other.contiguous_buffer()
            var result_buffer = self_contiguous.compare_buffer_full[op_code](
                other_contiguous
            )
            return NDBuffer[DType.bool](result_buffer^, self.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())
            var iterator = other.index_iterator()
            for idx in self.index_iterator():
                var self_val = self.buffer[idx]
                var next_index = -1
                try:
                    next_index = iterator.__next__()
                except e:
                    print(e)
                    panic("Raised StopIteration in NDBuffer → compare")

                var other_val = other.buffer[next_index]

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
                else:  # GreaterThanEqual
                    buffer[index] = self_val >= other_val

                index += 1

            return NDBuffer[DType.bool](buffer^, self.shape)

    @always_inline
    fn compare_scalar[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        DType.bool
    ]:
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
                    # Unreachable
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
        if self.is_contiguous():
            var contiguous_data = self.contiguous_buffer()
            var result_buffer = contiguous_data.compare_scalar_full[op_code](
                scalar
            )
            return NDBuffer[DType.bool](result_buffer^, self.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())

            for idx in self.index_iterator():
                var value = self.buffer[idx]

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
                else:  # GreaterThanEqual
                    buffer[index] = value >= scalar

                index += 1

            return NDBuffer[DType.bool](buffer^, self.shape)

    @always_inline
    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        comptime assert (
            Self.dtype.is_floating_point()
        ), "NDBuffer → all_close is for floating point data types only"

        if self.shape != other.shape:
            panic(
                "NDBuffer → all_close(other) expects same shaped buffers: "
                + String(self.shape)
                + "≠"
                + String(other.shape)
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

    fn map_to_bool(
        self,
        pred: fn(Scalar[Self.dtype]) -> Bool,
    ) -> NDBuffer[DType.bool]:
        """Apply predicate to each element, returning a boolean NDBuffer.
        GPU path: transfers to CPU, applies pred, transfers back if needed.
        CPU path: delegates to Buffer.map_to_bool via contiguous buffer.
        Note: pred is a CPU function — GPU path materialises through CPU.
        """

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Materialise to CPU — pred is a CPU fn
                    var cpu_ndb = self.to_cpu()
                    var bool_buffer = cpu_ndb.contiguous_buffer().map_to_bool(
                        pred
                    )
                    return NDBuffer[DType.bool](bool_buffer^, self.shape)
                except e:
                    panic("NDBuffer map_to_bool: GPU→CPU failed: " + String(e))
                    return NDBuffer[DType.bool].Empty()  # unreachable

        # CPU path
        var bool_buffer = self.contiguous_buffer().map_to_bool(pred)
        return NDBuffer[DType.bool](bool_buffer^, self.shape)

    fn all_true(self: NDBuffer[DType.bool]) -> Bool:
        """
        Returns True if all elements are True.
        GPU path: delegates to DeviceState[DType.bool].all_true().
                  which checks all uint8 values == 1 internally.
        CPU path: delegates to Buffer[DType.bool].all_true().
        """

        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.device_state.value().all_true()

        # CPU path — contiguous fast path
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            for i in range(start, end):
                if not self.buffer[i]:
                    return False
            return True

        # CPU non-contiguous fallback
        for idx in self.index_iterator():
            if not self.buffer[idx]:
                return False
        return True

    fn any_true(self: NDBuffer[DType.bool]) -> Bool:
        """
        Returns True if any element is True.
        GPU path: delegates to DeviceState[DType.bool].any_true()
                  which checks any uint8 value == 1 internally.
        CPU path: delegates to Buffer[DType.bool] iteration.
        """

        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.device_state.value().any_true()

        # CPU path — contiguous fast path
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            for i in range(start, end):
                if self.buffer[i]:
                    return True
            return False

        # CPU non-contiguous fallback
        for idx in self.index_iterator():
            if self.buffer[idx]:
                return True
        return False

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_buffer: NDBuffer[Self.dtype], target_shape: Shape
    ) -> NDBuffer[Self.dtype]:
        if extended_buffer.shape == target_shape:
            return extended_buffer
        var result: NDBuffer[Self.dtype]
        if extended_buffer.is_on_cpu():
            result = extended_buffer.contiguous()
        else:
            result = extended_buffer
        var current_shape = result.shape
        # Sum over extra leading dimensions
        while len(current_shape) > len(target_shape):
            result = result.reduce(normalized_axes=IntArray(0), keepdims=False)
            current_shape = result.shape
        # Sum over mismatched dimensions
        for i in range(len(target_shape)):
            if current_shape[i] != target_shape[i] and current_shape[i] > 1:
                result = result.reduce(
                    normalized_axes=IntArray(i), keepdims=True
                )
                current_shape = result.shape
        return result^

    fn matmul_2d(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        ref A_shape = A.shape
        ref B_shape = B.shape
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
                    panic("NDBuffer matmaul_2d → GPU operation failed")
                    # Unreachable - make the compiler happy
                    C = NDBuffer[Self.dtype](Shape())
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                panic(
                    (
                        " NDBuffer matmaul_2d → both buffers must be on gpu. A"
                        " is on gpu?"
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
        ref A_shape = A.shape
        ref B_shape = B.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        comptime tile_size = TILE_SIZE
        comptime simdwidth = simd_width_of[Self.dtype]()

        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]

        var C = NDBuffer[Self.dtype].zeros(Shape([m, p]))

        # Hoist metadata
        ref A_strides = A.strides
        var A_stride0 = A_strides[0]
        var A_stride1 = A_strides[1]
        var A_offset = A.offset
        var A_data = A.data_ptr()

        ref B_strides = B.strides
        var B_stride0 = B_strides[0]
        var B_stride1 = B_strides[1]
        var B_offset = B.offset
        var B_data = B.data_ptr()
        var C_data = C.data_ptr()

        if B.is_contiguous():
            # ========================================
            # PARALLELIZED TILED MATMUL
            # ========================================
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
                            # C.offset = 0, C_stride0 = p
                            # var c_row_base = i * C_stride0 + C_offset
                            var c_row_base = i * p

                            var j = j_tile

                            # Main vectorized loop
                            while j + simdwidth <= j_end:
                                # var c_addr = c_row_base + j * C_stride1, C_stride1 = 1
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

                            # Tail handling
                            while j < j_end:
                                # var c_addr = c_row_base + j * C_stride1, C_stride1 =1
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
            # Non-contiguous path (scalar)
            for i in range(m):
                var a_row_base = i * A_stride0 + A_offset
                # C_stride0 = p, C_offset = 0
                # var c_row_base = i * C_stride0 + C_offset
                var c_row_base = i * p

                for j in range(p):
                    var accumulator: Scalar[Self.dtype] = 0

                    for k in range(n):
                        var a_addr = a_row_base + k * A_stride1
                        var b_addr = k * B_stride0 + B_offset + j * B_stride1
                        accumulator += A_data[a_addr] * B_data[b_addr]

                    # var c_addr = c_row_base + j * C_stride1, C_stride1
                    var c_addr = c_row_base + j
                    C_data[c_addr] = accumulator

        return C^

    fn matmul_nd(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        var A_shape = A.shape
        var B_shape = B.shape

        # ── Validate inner dims ───────────────────────────────────────────────────
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
                    panic("NDBuffer matmaul_nd → GPU operation failed")
                    # Unreachable - make the compiler happy
                    C = Self.Empty()
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                panic(
                    (
                        " NDBuffer matmaul_nd → both buffers must be on gpu. A"
                        " is on gpu?"
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
        var A_shape = A.shape
        var B_shape = B.shape

        # ── Validate inner dims ───────────────────────────────────────────────────
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

        # ── Batch shapes and broadcasting ─────────────────────────────────────────
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

        # ── Output ────────────────────────────────────────────────────────────────
        var out_shape = batch_shape + Shape(m, p)
        var C = NDBuffer[Self.dtype].zeros(out_shape)

        # ── Hoist metadata ────────────────────────────────────────────────────────
        var A_batch_rank = A_batch_shape.rank()
        var B_batch_rank = B_batch_shape.rank()
        var batch_rank = batch_shape.rank()

        var A_batch_strides = A.strides[:-2]
        var B_batch_strides = B.strides[:-2]

        var A_row_stride = A.strides[A_rank - 2]
        var A_col_stride = A.strides[A_rank - 1]
        var B_row_stride = B.strides[B_rank - 2]
        var B_col_stride = B.strides[B_rank - 1]

        var A_offset = A.offset
        var B_offset = B.offset

        var A_data = A.data_ptr()
        var B_data = B.data_ptr()
        var C_data = C.data_ptr()
        # C is always contiguous, offset 0
        # C strides: batch dims row-major, then (p, 1) for inner dims

        var B_contiguous = B.is_contiguous()

        # ── Parallelise over batch × m tiles ──────────────────────────────────────
        var num_tiles_i = (m + tile_size - 1) // tile_size
        var total_tiles = total_batch * num_tiles_i

        @parameter
        fn process_tile(flat_idx: Int):
            var batch = flat_idx // num_tiles_i
            var tile_idx = flat_idx % num_tiles_i

            # ── Recover batch coords and compute A/B base offsets ─────────────────
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

                # Right-aligned broadcast clamping for A
                var A_rank_off = batch_rank - A_batch_rank
                for i in range(A_batch_rank):
                    var coord = (
                        coords[A_rank_off + i] if A_batch_shape[i] > 1 else 0
                    )
                    A_base_off += coord * A_batch_strides[i]

                # Right-aligned broadcast clamping for B
                var B_rank_off = batch_rank - B_batch_rank
                for i in range(B_batch_rank):
                    var coord = (
                        coords[B_rank_off + i] if B_batch_shape[i] > 1 else 0
                    )
                    B_base_off += coord * B_batch_strides[i]

            # C base offset for this batch — C is contiguous row-major
            var C_base_off = batch * m * p

            # ── Tiled matmul for this batch slice ─────────────────────────────────
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

                            # Vectorised loop
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

                            # Tail
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
                # Non-contiguous B — scalar path
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
