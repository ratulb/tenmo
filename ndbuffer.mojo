from shapes import Shape
from strides import Strides
from buffers import Buffer
from intarray import IntArray
from indexhelper import IndexCalculator, IndexIterator
from matrixshapevalidator import MatrixShapeValidator
from broadcasthelper import ShapeBroadcaster
from common_utils import panic, log_warning, print_buffer
from validators import Validator
from memory import memcpy, AddressSpace, ArcPointer
from gpu.host import DeviceBuffer, DeviceContext
from algorithm import parallelize
from sys.info import num_physical_cores
from device import Device, CPU, GPU, DeviceState
from collections import Set
from sys import simd_width_of, has_accelerator
from scalar_ops_kernel import ScalarOperations
from scalar_inplace_ops_kernel import InplaceScalarOperations
from binary_ops_kernel import BinaryOperations
from binary_inplace_ops_kernel import BinaryInplaceOperations
from matmul_kernel import MatmulNdGpu
from compare_kernel import AllClose, Compare, CompareScalar
from reduction_kernel import Reduction
from mnemonics import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    MAX,
    MIN,
    POW,
    Overwrite,
    ReverseDivide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)

comptime TILE_SIZE = 32


struct NDBuffer[dtype: DType](
    ImplicitlyCopyable
    & Movable
    & Equatable
    & Stringable
    & Representable
    & Writable
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
            log_warning(
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

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^
        self.shape = other.shape^
        self.strides = other.strides^
        self.offset = other.offset
        self._contiguous = other._contiguous
        self.device_state = other.device_state^

    fn __copyinit__(out self, other: Self):
        """Copy NDBuffer - Buffer handles ref counting automatically."""
        self.buffer = (
            other.buffer.copy()
        )  # Buffer copy handles shared/unshared!
        self.shape = other.shape.copy()
        self.strides = other.strides.copy()
        self.offset = other.offset
        self._contiguous = other._contiguous
        self.device_state = other.device_state.copy()

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

    fn sync(self):
        @parameter
        if has_accelerator():
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

            @parameter
            if has_accelerator():
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
    ) raises -> ref [self.device_state.value().gpu] GPU:
        @parameter
        if has_accelerator():
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

    fn device(self) raises -> Device:
        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                return self.device_state.value().get_gpu().into()
        return CPU().into()

    fn device_context(self) -> Optional[ArcPointer[DeviceContext]]:
        if self.is_on_gpu():
            return self.device_state.value().gpu[]
        return None

    fn get_device_state(
        ref self,
    ) raises -> ref [self.device_state.value()] DeviceState[Self.dtype]:
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
        @parameter
        if has_accelerator():
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

            @parameter
            if has_accelerator():
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
        args: VariadicList[Scalar[Self.dtype]],
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
            self.shape, indices, self.strides, self.offset
        )
        return self.get(index)

    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.set(index, value)

    fn __getitem__(self, indices: VariadicList[Int]) -> Scalar[Self.dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.get(index)

    fn __setitem__(self, indices: VariadicList[Int], value: Scalar[Self.dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.set(index, value)

    @always_inline
    fn item(self) -> Scalar[Self.dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim"
                " buffer/singleton, got shape: "
                + self.shape.__str__()
            )
        return self.get(0)

    fn get(self, index: Int) -> Scalar[Self.dtype]:
        idx = index + self.max_index() if index < 0 else index
        if idx < 0 or idx > self.max_index():
            panic(
                "NDBuffer → element_at: index out of bounds.",
                "NDBuffer max index",
                self.max_index().__str__(),
                ", provided index",
                index.__str__(),
            )
        if self.is_on_gpu():
            ref device_state = self.device_state.value()
            try:
                return device_state[idx]
            except e:
                print(e)
                panic("Error in NDBuffer → get: ", e.__str__())
                # Unreachable
                return Scalar[Self.dtype](0)
        return self.data_ptr()[idx]

    fn set(self, index: Int, value: Scalar[Self.dtype]):
        idx = index + self.max_index() if index < 0 else index
        if idx < 0 or idx > self.max_index():
            panic(
                "NDBuffer → set_element_at: index out of bounds.",
                "NDBuffer max index",
                self.max_index().__str__(),
                ", provided index",
                index.__str__(),
            )

        if self.is_on_gpu():
            ref device_state = self.device_state.value()
            try:
                device_state[idx] = value
            except e:
                print(e)
                panic("Error in NDBuffer → set: ", e.__str__())
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            ptr[idx] = value

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from a 2D NDBuffer."""

        constrained[
            simdwidth.is_power_of_two(),
            "NDBuffer → load: SIMD width must be a power of 2",
        ]()
        if simdwidth > self.numels():
            panic("NDBuffer → load: buffer size is less than simd width")

        @parameter
        if not validated:
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
                    + row.__str__()
                    + ", col range ["
                    + col.__str__()
                    + ", "
                    + (col + simdwidth).__str__()
                    + ") "
                    + "for shape "
                    + shape.__str__()
                    + "."
                )

            if simdwidth > 1 and self.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD load requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + self.strides[1].__str__()
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
                panic("Error in NDBuffer → get: ", e.__str__())
                # Unreachable
                return SIMD[Self.dtype, simdwidth](0)
        return self.data_ptr().load[width=simdwidth](addr)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
        """SIMD store of a row segment into a 2D NDBuffer."""

        constrained[
            simdwidth.is_power_of_two(),
            "NDBuffer → store: SIMD width must be a power of 2",
        ]()
        if simdwidth > self.numels():
            panic("NDBuffer → store: buffer size is less than simd width")

        @parameter
        if not validated:
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
                    + row.__str__()
                    + ", col range ["
                    + col.__str__()
                    + ", "
                    + (col + simdwidth).__str__()
                    + ") "
                    + "for shape "
                    + shape.__str__()
                    + "."
                )

            if simdwidth > 1 and self.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD store requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + self.strides[1].__str__()
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
                panic("Error in NDBuffer → store: ", e.__str__())
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            ptr.store[width=simdwidth](addr, value)

    fn __str__(self) -> String:
        s = String("NDBuffer [")
        s += "Shape: " + self.shape.__str__()
        s += ", Type: " + Self.dtype.__str__()
        s += ", Shared : " + self.shared().__str__()
        s += ", Strides : " + self.strides.__str__()
        s += ", Offset : " + self.offset.__str__()
        s += ", Contiguous : " + self.is_contiguous().__str__()
        s += ", Buffer size : " + self.size().__str__()
        s += (
            ", Device : "
            + "gpu: "
            + self.gpu_id().__str__() if self.is_on_gpu() else ", Device : "
            + "cpu"
        )
        s += "]"
        return s

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(
            "\n",
            self.__str__(),
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
    fn data_buffer(ref self) -> ref [self.buffer] Buffer[Self.dtype]:
        return self.buffer

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape()

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
    fn offset_at(self, indices: IntArray) -> Int:
        """Return the absolute linear offset in the underlying buffer
        for the given multidimensional indices."""
        if indices.size() != self.rank():
            panic("NDBuffer.offset_at: Incorrect number of indices")

        return IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )

    fn to_dtype[NewType: DType](self) -> NDBuffer[NewType]:
        @parameter
        if has_accelerator():
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
                    panic("NDBuffer to_dtype GPU failed: " + e.__str__())
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
        if max_index > size:
            panic(
                "NDBuffer → share: invalid view [max_index="
                + max_index.__str__()
                + " > buffer_size="
                + size.__str__()
                + "] shape="
                + new_shape.__str__()
                + " strides="
                + new_strides.__str__()
                + " offset="
                + offset.__str__()
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
    ](ref [origin, address_space]self) -> UnsafePointer[
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

    @always_inline
    fn fill(self, value: Scalar[Self.dtype]):
        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.device_state.value().fill(value)
                except e:
                    print(e)
                    panic("Error filling NDBuffer value: ", value.__str__())
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

        @parameter
        if has_accelerator():
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
            out = NDBuffer[Self.dtype](Shape())
        return out^

    fn map[
        map_buffer: fn (Buffer[Self.dtype]) -> Buffer[Self.dtype],
        map_element: fn (Scalar[Self.dtype]) -> Scalar[Self.dtype],
    ](self) -> Buffer[Self.dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return map_buffer(self.buffer[start:end])
        else:
            var buffer = Buffer[Self.dtype](self.numels())
            var index = 0
            for idx in self.index_iterator():
                buffer[index] = map_element(self.buffer[idx])
                index += 1
            return buffer^

    fn reduce[
        reduce_buffer: fn (Buffer[Self.dtype], Int, Optional[Int]) -> Scalar[
            Self.dtype
        ],
        reduce_elements: fn (Scalar[Self.dtype], Scalar[Self.dtype]) -> Scalar[
            Self.dtype
        ],
        unit: Scalar[Self.dtype] = Scalar[Self.dtype](0),
    ](self) -> Scalar[Self.dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return reduce_buffer(self.buffer, start, end)
        else:
            var accum: Scalar[Self.dtype] = unit
            for index in self.index_iterator():
                accum = reduce_elements(self.buffer[index], accum)
            return accum

    fn sum_all(self) -> Scalar[Self.dtype]:
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
        return self.reduce[mean=False](normalized_axes, keepdims)

    fn reduce[
        mean: Bool = False
    ](self, normalized_axes: IntArray, keepdims: Bool = False) -> NDBuffer[
        Self.dtype
    ]:
        """Axes must be already normalized."""

        var out: NDBuffer[Self.dtype]

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    out = Reduction[Self.dtype].launch[mean=mean](
                        self, normalized_axes, keepdims
                    )
                except e:
                    print(e)
                    print(
                        (
                            "NDBuffer sum - GPU operation failed for"
                            ". Failling back on CPU"
                        ),
                    )
                    out = self.reduce_cpu[mean=mean](normalized_axes, keepdims)
            else:
                out = self.reduce_cpu[mean=mean](normalized_axes, keepdims)
        else:
            out = self.reduce_cpu[mean=mean](normalized_axes, keepdims)

        return out^

    fn reduce_cpu[
        mean: Bool = False
    ](self, normalized_axes: IntArray, keepdims: Bool,) -> NDBuffer[Self.dtype]:
        var reduced_volume = Scalar[Self.dtype](1)

        @parameter
        if mean:
            var volume = self.shape.reduced_shape(normalized_axes).product()
            reduced_volume = reduced_volume if volume == 0 else Scalar[
                Self.dtype
            ](volume)

        var out_shape = self.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var out = NDBuffer[Self.dtype].zeros(out_shape)

        # Handle scalar output cases
        if out_shape == Shape():
            # This covers both scalar input AND full reduction cases
            @parameter
            if mean:
                out[IntArray()] = self.sum_all() / reduced_volume
            else:
                out[IntArray()] = self.sum_all()
        else:
            # Handle partial reduction with proper coordinate mapping
            reduction_axes_shape = self.shape.reduced_shape(normalized_axes)

            for out_coord in out_shape:
                var accum_sum = Scalar[Self.dtype](0)
                for red_coord in reduction_axes_shape:
                    # Use normalized_axes (sorted) for coordinate reconstruction
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum_sum += self[self_coord]

                @parameter
                if mean:
                    out[out_coord] = accum_sum / reduced_volume
                else:
                    out[out_coord] = accum_sum

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

        @parameter
        if has_accelerator():
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
                        + e.__str__()
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
                        axis.__str__(),
                        " out of range",
                    )
                if shape[normalized] != 1:
                    panic(
                        "NDBuffer → squeeze: cannot squeeze axis ",
                        normalized.__str__(),
                        " with size ",
                        shape[normalized].__str__(),
                    )
                if normalized in seen:
                    panic("NDBuffer → squeeze: duplicate axis ", axis.__str__())
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
                    axis.__str__(),
                    " out of range",
                )
            if normalized in seen:
                panic("NDBuffer → unsqueeze: duplicate axis ", axis.__str__())
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
        mut self, axes: IntArray, *, shared: Bool = True
    ) -> NDBuffer[Self.dtype]:
        """
        Permute axes of this NDBuffer.

        shared=True  → view with reordered shape/strides, same buffer
                       used by Tensor.permute — no data movement
        shared=False → owned contiguous copy with permuted layout
                       used by Gradbox.permute — GPU safe via contiguous()

        GPU safety: shared=True is always safe — just metadata.
                    shared=False delegates to contiguous() which handles
                    GPU via contiguous_device_state().
        """
        var shape = self.shape
        var rank = shape.rank()

        if len(axes) != rank:
            panic(
                "NDBuffer → permute: number of axes (",
                len(axes).__str__(),
                ") must match rank (",
                rank.__str__(),
                ")",
            )

        # Validate permutation
        var seen = IntArray.with_capacity(rank)
        for axis in axes:
            var normalized = axis if axis >= 0 else axis + rank
            if normalized < 0 or normalized >= rank:
                panic(
                    "NDBuffer → permute: invalid axis index ",
                    axis.__str__(),
                    " for rank ",
                    rank.__str__(),
                )
            if normalized in seen:
                panic("NDBuffer → permute: duplicate axis ", axis.__str__())
            seen.append(normalized)

        # Build permuted shape and strides
        var new_shape_dims = IntArray.with_capacity(rank)
        var new_strides_arr = IntArray.with_capacity(rank)
        for axis in axes:
            new_shape_dims.append(shape[axis])
            new_strides_arr.append(self.strides[axis])

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

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Materialise logical view — handles offset + strides
                    var contig = self.contiguous_device_state()
                    var numels = self.numels()
                    var total = 0

                    with contig.buffer.map_to_host() as host_buffer:
                        var ptr = host_buffer.unsafe_ptr()

                        @parameter
                        if Self.dtype == DType.bool:
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
                    panic("NDBuffer count GPU failed: " + e.__str__())
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

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Materialise logical view — handles offset + strides
                    var contig = self.contiguous_device_state()
                    var numels = self.numels()
                    var uniques = Set[Scalar[Self.dtype]]()

                    with contig.buffer.map_to_host() as host_buffer:
                        var ptr = host_buffer.unsafe_ptr()

                        @parameter
                        if Self.dtype == DType.bool:
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
                    panic("NDBuffer unique GPU failed: " + e.__str__())
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
        @parameter
        if validate:
            if not self.shape == other.shape:
                panic(
                    (
                        "NDBuffer → copy_from_alike(other):"
                        " dimension mismatch: self shape"
                    ),
                    self.shape.__str__(),
                    "≠",
                    "other shape",
                    other.shape.__str__(),
                )

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            other_start = other.offset
            self_end = self_start + self.numels()
            other_end = other_start + other.numels()

            @parameter
            if overwrite:
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

                @parameter
                if overwrite:
                    self.buffer[index] = other.buffer[idx]
                else:
                    self.buffer[index] += other.buffer[idx]
                index += 1

        elif not self.is_contiguous() and other.is_contiguous():
            var index = other.offset
            for idx in self.index_iterator():

                @parameter
                if overwrite:
                    self.buffer[idx] = other.buffer[index]
                else:
                    self.buffer[idx] += other.buffer[index]
                index += 1

        else:
            var iterator = other.index_iterator()
            for index in self.index_iterator():

                @parameter
                if overwrite:
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

        @parameter
        if has_accelerator():
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
                    self.shape.__str__(),
                    "≠",
                    "other shape",
                    other.shape.__str__(),
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
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        @parameter
        if has_accelerator():
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
                        op_code.__str__(),
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
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
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
                    + broadcast_shape.__str__()
                    + " must match receiver shape "
                    + self.shape.__str__()
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
        @parameter
        if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        @parameter
        if has_accelerator():
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
                        op_code.__str__(),
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
        @parameter
        if op_code == Divide:
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
    fn __mul__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Multiply](other)

    @always_inline
    fn __mul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Multiply](scalar)

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

    fn __truediv__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Divide](other)

    @always_inline
    fn arithmetic_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → arithmetic_ops: dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        var out: NDBuffer[Self.dtype]

        @parameter
        if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    out = BinaryOperations[Self.dtype].launch[op_code](
                        self, other
                    )
                except e:
                    print(e)
                    print(
                        (
                            "NDBuffer arithmetic_ops → GPU operation failed for"
                            " opcode: "
                        ),
                        op_code.__str__(),
                    )
                    # Unreachable
                    out = NDBuffer[Self.dtype].Empty()
            else:
                out = self.arithmetic_ops_cpu[op_code](other)
        else:
            out = self.arithmetic_ops_cpu[op_code](other)

        return out^

    @always_inline
    fn arithmetic_ops_cpu[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        # Handle broadcasting case
        if self.shape != other.shape:
            return self.broadcast_buffer[op_code](other)

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            self_end = self_start + self.numels()
            other_start = other.offset
            other_end = other_start + other.numels()

            var result_buffer = self.buffer.arithmetic_ops[op_code](
                other.buffer, self_start, self_end, other_start, other_end
            )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var result_buffer = Buffer[Self.dtype](self.numels())
            var index = 0

            if self.is_contiguous() and not other.is_contiguous():
                var offset = self.offset
                for idx in other.index_iterator():
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[offset + index], other.buffer[idx]
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var offset = other.offset
                for idx in self.index_iterator():
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[idx], other.buffer[offset + index]
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
                        self.buffer[idx], other.buffer[next_index]
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
        var own_shape = self.shape
        if not ShapeBroadcaster.broadcastable(own_shape, target_shape):
            panic(
                "NDBuffer → broadcast_to(target_shape): "
                + own_shape.__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        var target_rank = target_shape.rank()
        var own_rank = own_shape.rank()
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
        op_code: Int
    ](lhs: Scalar[Self.dtype], rhs: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        @parameter
        if op_code == Add:
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
        elif op_code == POW:
            return lhs**rhs

        else:  # op_code == ReverseDivide
            return rhs / lhs

    @always_inline
    fn scalar_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        @parameter
        if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → scalar_ops: cannot divide by zero")

        var out: NDBuffer[Self.dtype]

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:

                    @parameter
                    if op_code == POW:
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
                        op_code.__str__(),
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
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer: Buffer[Self.dtype]

            @parameter
            if op_code == POW:
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
                    self.buffer[idx], scalar
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

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
                + self.shape.__str__()
                + "≠"
                + other.shape.__str__()
            )
        var result: NDBuffer[DType.bool]

        @parameter
        if has_accelerator():
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

                @parameter
                if op_code == Equal:
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

        @parameter
        if has_accelerator():
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

                @parameter
                if op_code == Equal:
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
        constrained[
            Self.dtype.is_floating_point(),
            "NDBuffer → all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            panic(
                "NDBuffer → all_close(other) expects same shaped buffers: "
                + self.shape.__str__()
                + "≠"
                + other.shape.__str__()
            )
        var result: Bool

        @parameter
        if has_accelerator():
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
        pred: fn (Scalar[Self.dtype]) -> Bool,
    ) -> NDBuffer[DType.bool]:
        """Apply predicate to each element, returning a boolean NDBuffer.
        GPU path: transfers to CPU, applies pred, transfers back if needed.
        CPU path: delegates to Buffer.map_to_bool via contiguous buffer.
        Note: pred is a CPU function — GPU path materialises through CPU.
        """

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Materialise to CPU — pred is a CPU fn
                    var cpu_ndb = self.to_cpu()
                    var bool_buffer = cpu_ndb.contiguous_buffer().map_to_bool(
                        pred
                    )
                    return NDBuffer[DType.bool](bool_buffer^, self.shape)
                except e:
                    panic(
                        "NDBuffer map_to_bool: GPU→CPU failed: " + e.__str__()
                    )
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

        @parameter
        if has_accelerator():
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

        @parameter
        if has_accelerator():
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

        @parameter
        if has_accelerator():
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
                    A.is_on_gpu().__str__(),
                    ", B is on gpu?",
                    B.is_on_gpu().__str__(),
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
                + k_A.__str__()
                + " and "
                + k_B.__str__()
            )

        var C: NDBuffer[Self.dtype]

        @parameter
        if has_accelerator():
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
                    A.is_on_gpu().__str__(),
                    ", B is on gpu?",
                    B.is_on_gpu().__str__(),
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
                + k_A.__str__()
                + " and "
                + k_B.__str__()
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
                + A_batch_shape.__str__()
                + " vs "
                + B_batch_shape.__str__()
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


fn main() raises:
    comptime dtype = DType.float32
    pass
