from .shared.shapes import Shape
from .shared.strides import Strides
from .shared.buffers import Buffer
from .shared.intarray import IntArray
from .shared.indexhelper import IndexCalculator, IndexIterator
from .matrixshapevalidator import MatrixShapeValidator
from .shared.broadcasthelper import ShapeBroadcaster
from .shared.constants import Epsilon, CloseTol, UNKNOWN_VALUE
from .shared.indexhelper import Idx
from .shared.logging import log_debug
from .shared.panic import panic

from .validators import Validator
from std.memory import unsafe_memcpy, Pointer
from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic, Ordering, fence
from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext
from .gpu.device import (
    Device,
    CPU,
    GPU,
    DeviceState,
)
from .shared.layout import Layout
from std.collections import Set
from std.sys import simd_width_of, has_accelerator
from std.sys.intrinsics import strided_load
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .kernels.scalar_ops_kernel import ScalarKernel
from .kernels.scalar_inplace_ops_kernel import ScalarInplaceKernel
from .kernels.binary_ops_kernel import BinaryKernel
from .kernels.binary_inplace_ops_kernel import BinaryInplaceKernel
from .kernels.unary_ops_kernel import UnaryKernel
from .kernels.onehot import OnehotKernel
from .kernels.matmul_kernel import MatmulKernel
from .matmul_cpu import MmCpu2d, MmCpuNd
from .cpu_arithmetics import CpuArithmeticOps
from .shared.scalar_ops import compare_pair
from .kernels.compare_kernel import AllClose, Compare, CompareScalar

from std.math import sqrt, log, exp, tanh
from std.random import seed, random_float64, random_ui64
from .kernels.random_kernel import RandomKernel
from .shared.mnemonics import (
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

comptime TILE_SIZE = 32


struct NDBuffer[dtype: DType](
    ImplicitlyCopyable & Equatable & Writable & Sized
):
    var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: Buffer[Self.dtype]
    var device_state: Optional[DeviceState[Self.dtype]]

    def __init__(out self, *values: Scalar[Self.dtype]):
        var buffer = Buffer[Self.dtype](len(values))
        for i in range(len(values)):
            buffer[i] = values[i]
        self = NDBuffer[Self.dtype](buffer^)

    def __init__(
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

        else:
            var _shape = shape.or_else(Shape(buffer.size))
            self.shape = _shape.copy()
            self.buffer = buffer^
            self.strides = strides.or_else(Strides.default(_shape))
            self.offset = offset

    def __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.buffer = Buffer[Self.dtype](shape.num_elements())
        self.shape = shape
        self.strides = strides.or_else(Strides.default(shape))
        self.offset = offset
        self.device_state = None

    def __init__(
        out self,
        device_buffer: DeviceBuffer[Self.dtype],
        shape: Shape,
        *,
        copy: Bool = False,
    ) raises:
        var buffer: Buffer[Self.dtype]
        with device_buffer.map_to_host() as host_buffer:
            buffer = Buffer[Self.dtype](
                shape.num_elements(),
                host_buffer.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                copy=copy,
            )
        self.buffer = buffer^
        self.shape = shape
        self.strides = Strides.default(shape)
        self.offset = 0
        self.device_state = None

    def __init__(out self, *, deinit move: Self):
        self.buffer = move.buffer^
        self.shape = move.shape^
        self.strides = move.strides^
        self.offset = move.offset
        self.device_state = move.device_state^

    def __init__(out self, *, copy: Self):
        """Copy NDBuffer - Buffer handles ref counting automatically."""
        self.buffer = copy.buffer.copy()  # Buffer copy handles shared/unshared!
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self.device_state = copy.device_state.copy()

    def clone(self, sync: Bool = True) -> Self:
        """Create an independent deep-copy of this buffer (opt-out of sharing).

        Unlike copy-init (which aliases shared-from-birth buffers), `clone`
        materialises fresh storage — the result shares no memory with the
        source. Layout (shape, strides, offset) is preserved on CPU. On GPU the
        clone is materialised as an independent device buffer.
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var cpu_copy = self.to_cpu(sync=sync)
                    var gpu = self.get_gpu()
                    return cpu_copy.to_gpu(gpu)
                except e:
                    print(e)
                    panic("NDBuffer.clone: GPU clone failed: " + String(e))
                    return self  # unreachable — satisfies compiler

        var out = NDBuffer[Self.dtype](
            self.contiguous_buffer(), self.shape, None, 0
        )
        return out^

    def layout(self) -> Layout:
        """Canonical `Layout` (shape/strides/offset/contiguous) of this buffer.

        Rebuilds from the legacy fields on every call. Layout is a small
        RegisterPassable value object — the rebuild cost is negligible.
        """
        return Layout(self.shape.copy(), self.strides.copy(), self.offset)

    @staticmethod
    def with_device_state(
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

    @staticmethod
    def from_device_state(
        state: DeviceState[Self.dtype],
        shape: Shape,
        sync: Bool = True,
    ) raises -> NDBuffer[Self.dtype]:
        """Copy DeviceState content to a contiguous CPU NDBuffer with 0 offset.

        bool: converts uint8 0/1 back to bool. (NDBuffer-coupled bridge; the
        NDBuffer-free core is `gpu.transfer.device_to_host`.)
        """
        var numels = len(state)
        comptime if Self.dtype == DType.bool:
            var cpu_buf = Buffer[DType.bool](numels)
            with state.buffer.map_to_host() as host_buffer:
                for i in range(numels):
                    cpu_buf[i] = host_buffer[i].cast[DType.uint8]() == UInt8(1)
            var casted_to_dtype = cpu_buf.to_dtype[Self.dtype]()
            return NDBuffer[Self.dtype](casted_to_dtype^, shape)
        else:
            var cpu_buf = Buffer[Self.dtype](numels)
            with state.buffer.map_to_host() as host_buffer:
                var src_ptr = host_buffer.unsafe_ptr()
                var dst_ptr = cpu_buf.unsafe_ptr()
                unsafe_memcpy(
                    dest=dst_ptr,
                    src=src_ptr.unsafe_bitcast[Scalar[Self.dtype]](),
                    count=len(state),
                )
            return NDBuffer[Self.dtype](cpu_buf^, shape)

    def fill_device_state(
        self,
        device_state: DeviceState[Self.dtype],
        sync: Bool = True,
    ) raises:
        """Fill the DeviceState's device buffer from the source NDBuffer."""

        comptime storage = DType.uint8 if Self.dtype == DType.bool else Self.dtype

        if self.is_on_gpu():
            if self.is_contiguous():
                self.device_state.value().buffer.enqueue_copy_to(
                    device_state.buffer
                )
            else:
                with device_state.buffer.map_to_host() as host_buffer:
                    var next_index = 0
                    for index in self.index_iterator():
                        comptime if Self.dtype == DType.bool:
                            var v = self.storage_get(index).cast[DType.bool]()
                            host_buffer[next_index] = rebind[Scalar[storage]](
                                UInt8(1) if v else UInt8(0)
                            )
                        else:
                            host_buffer[next_index] = rebind[Scalar[storage]](
                                self.storage_get(index)
                            )
                        next_index += 1
            if sync:
                device_state.sync()
            return

        with device_state.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            var src_ptr = self.data_ptr()

            if self.is_contiguous():
                var src_offset = self.offset
                src_ptr = src_ptr.unsafe_offset(src_offset)
                var numels = self.numels()

                comptime if Self.dtype == DType.bool:
                    for i in range(numels):
                        device_ptr[unsafe_offset=i] = rebind[Scalar[storage]](
                            UInt8(1) if (src_ptr.unsafe_offset(i))[].cast[
                                DType.bool
                            ]() else UInt8(0)
                        )
                else:
                    unsafe_memcpy(
                        dest=device_ptr,
                        src=src_ptr.unsafe_bitcast[Scalar[storage]](),
                        count=numels,
                    )
            else:
                var next_index = 0
                for index in self.index_iterator():
                    comptime if Self.dtype == DType.bool:
                        device_ptr[unsafe_offset=next_index] = rebind[
                            Scalar[storage]
                        ](
                            UInt8(1) if (src_ptr.unsafe_offset(index))[].cast[
                                DType.bool
                            ]() else UInt8(0)
                        )
                    else:
                        device_ptr[unsafe_offset=next_index] = rebind[
                            Scalar[storage]
                        ]((src_ptr.unsafe_offset(index))[])
                    next_index += 1
        if sync:
            device_state.sync()

    @staticmethod
    def with_layout_device_state(
        layout: Layout, var device_state: DeviceState[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """Reconstruct NDBuffer from GPU kernel result (Layout + DeviceState).
        """
        var ndb = NDBuffer[Self.dtype](
            Buffer[Self.dtype](),
            shape=layout.shape,
            strides=layout.strides,
            offset=layout.offset,
        )
        ndb.device_state = device_state^
        return ndb^

    @staticmethod
    def with_layout_buffer(
        layout: Layout, var buffer: Buffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        """Reconstruct NDBuffer from CPU engine result (Layout + Buffer)."""
        return NDBuffer[Self.dtype](
            buffer^,
            shape=layout.shape,
            strides=layout.strides,
            offset=layout.offset,
        )

    def sync(self):
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    self.device_state.value().sync()
                except e:
                    print(e)
                    print("NDBuffer device state synchronization failed")

    @staticmethod
    def rand(
        shape: Shape,
        min: Scalar[Self.dtype] = 0,
        max: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        device: Device = CPU().into(),
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer with uniform random values in [min, max)."""
        comptime if has_accelerator():
            if device.is_gpu():
                try:
                    var rng_seed = UInt64(
                        init_seed.value()
                    ) if init_seed else random_ui64(0, UInt64.MAX)
                    var gpu = device.gpu()
                    var (layout, dev_state) = RandomKernel[
                        Self.dtype
                    ].launch_uniform(shape, min, max, rng_seed, gpu)
                    return NDBuffer[Self.dtype].with_layout_device_state(
                        layout, dev_state
                    )
                except e:
                    panic("NDBuffer.rand GPU launch failed: " + String(e))

        # CPU path
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)
        var min_f64 = min.cast[DType.float64]()
        var max_f64 = max.cast[DType.float64]()
        for i in range(numels):
            buffer[i] = random_float64(min_f64, max_f64).cast[Self.dtype]()
        return NDBuffer[Self.dtype](buffer^, shape)

    @staticmethod
    def randn(
        shape: Shape,
        mean: Float64 = 0.0,
        std: Float64 = 1.0,
        init_seed: Optional[Int] = None,
        device: Device = CPU().into(),
    ) -> NDBuffer[Self.dtype]:
        """Create an NDBuffer with values from a normal distribution."""
        comptime if has_accelerator():
            if device.is_gpu():
                try:
                    var rng_seed = UInt64(
                        init_seed.value()
                    ) if init_seed else random_ui64(0, UInt64.MAX)
                    var gpu = device.gpu()
                    var (layout, dev_state) = RandomKernel[
                        Self.dtype
                    ].launch_normal(
                        shape,
                        mean.cast[DType.float32](),
                        std.cast[DType.float32](),
                        rng_seed,
                        gpu,
                    )
                    return NDBuffer[Self.dtype].with_layout_device_state(
                        layout, dev_state
                    )
                except e:
                    panic("NDBuffer.randn GPU launch failed: " + String(e))

        # CPU path — polar method
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)
        var i = 0
        while i < numels:
            var u: Float64 = 0.0
            var v: Float64 = 0.0
            var s: Float64 = 2.0
            while s >= 1.0 or s == 0.0:
                u = random_float64(-1.0, 1.0)
                v = random_float64(-1.0, 1.0)
                s = u * u + v * v
            var multiplier = sqrt(-2.0 * log(s) / s) * std
            var z0 = u * multiplier + mean
            var z1 = v * multiplier + mean
            buffer[i] = z0.cast[Self.dtype]()
            if i + 1 < numels:
                buffer[i + 1] = z1.cast[Self.dtype]()
            i += 2
        return NDBuffer[Self.dtype](buffer^, shape)

    @staticmethod
    @always_inline
    def zeros(
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

    def tolist(self) raises -> List[Scalar[Self.dtype]]:
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    return self.to_cpu().buffer.tolist()
                except e:
                    print(e)
                    raise e^
        if self.is_contiguous():
            var start = self.offset
            var end = self.numels() + start
            return self.buffer[start:end].tolist()
        else:
            return self.contiguous_buffer().tolist()

    def get_gpu(
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

    def to_cpu(self, sync: Bool = True) raises -> Self:
        var _, nd_buffer = self.to_device(CPU().into(), sync=sync)
        return nd_buffer^

    def to_gpu(self, gpu: GPU) raises -> Self:
        if self.buffer.size == 0:
            raise "NDBuffer -> to_gpu(): Empty buffer"
        return self.to_device(gpu.into())[1]

    def device(self) -> Device:
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.device_state.value().get_gpu().into()
        return CPU().into()

    def device_context(self) -> Optional[DeviceContext]:
        if self.is_on_gpu():
            return self.device_state.value().gpu[]
        return None

    def get_device_state(
        ref self,
    ) raises -> ref[self.device_state.value()] DeviceState[Self.dtype]:
        if self.is_on_gpu():
            return self.device_state.value()
        raise "Not on any device"

    def to_device(
        self, device: Device, sync: Bool = False
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
            self.fill_device_state(new_device_state, sync=sync)
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
            var ndb_buffer = Self.from_device_state(
                curr_state, self.shape, sync=sync
            )

            # Then move CPU -> new GPU
            # This would return 0, NDBuffer
            return ndb_buffer.to_device(device, sync=sync)

        # 3) GPU -> CPU
        # Materialize contiguous CPU buffer
        # New NDBuffer alltogether!
        if self.is_contiguous() and self.offset == 0:
            return 0, Self.from_device_state(curr_state, self.shape, sync=sync)
        else:
            # Materialise respecting strides
            # Step 1: bring raw flat device buffer to CPU
            var flat_cpu = Self.from_device_state(
                curr_state, Shape(len(curr_state)), sync=sync
            )
            # Step 2: create view with correct shape/strides/offset over flat data
            var viewed = flat_cpu.share(self.shape, self.strides, self.offset)
            # Step 3: materialise into contiguous CPU buffer
            var result = NDBuffer[Self.dtype](self.shape)
            result.copy_from_alike[overwrite=True, validate=False](viewed^)
            return 0, result^

    def is_on_gpu(ref self) -> Bool:
        comptime if has_accelerator():
            return not self.device_state == None
        return False

    def ref_count(self) -> Int:
        if self.is_on_gpu():
            return UNKNOWN_VALUE
        return self.buffer.ref_count()

    def gpu_id(self) -> Int64:
        if self.is_on_gpu():
            return self.device_state.value().get_gpu().id
        return -1

    def is_on_cpu(ref self) -> Bool:
        return self.is_on_gpu() == False

    @staticmethod
    @always_inline
    def full(
        shape: Shape,
        scalar: Scalar[Self.dtype],
        device: Device = CPU().into(),
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype]:
        var buffer = Buffer[Self.dtype].full(scalar, shape.num_elements())
        var ndb = NDBuffer[Self.dtype](buffer^, shape)

        if device.is_cpu():
            return ndb^
        else:
            comptime if has_accelerator():
                try:
                    var (_, result) = ndb^.to_device(device, sync=sync)
                    return result^
                except e:
                    print(e)
                    panic("NDBuffer full: device transfer failed")
                    # Unreachable
                    return NDBuffer[Self.dtype].Empty()
            else:
                return ndb^

    @staticmethod
    def onehot(
        indices: NDBuffer[Self.dtype],
        num_classes: Int,
        device: Optional[Device] = None,
        ignore_index: Optional[Int] = None,
        sync: Bool = False,
    ) -> NDBuffer[Self.dtype]:
        """Convert NDBuffer of class indices to one-hot encoding.

        GPU path: delegates to OnehotKernel[dtype, dtype].launch().
        CPU path: direct iteration on CPU buffer.

        The `device` parameter is honored exactly: when present, indices
        are first transferred to that device (no-op if already there) and
        the result is produced on it. When absent, behavior follows the
        indices' current device.
        """
        var src = indices
        comptime if has_accelerator():
            if device:
                var target = device.value()
                if not (src.device() == target):
                    try:
                        src = src.to_device(target, sync=sync)[1]
                    except e:
                        panic(
                            "NDBuffer.onehot: index transfer failed: "
                            + String(e)
                        )
            if src.is_on_gpu():
                try:
                    var ign = ignore_index.or_else(-1000000)
                    var (l, s) = OnehotKernel[Self.dtype, Self.dtype].launch(
                        src.layout(),
                        src.device_state.value(),
                        num_classes,
                        ign,
                        sync=sync,
                    )
                    return NDBuffer[Self.dtype].with_layout_device_state(l, s^)
                except e:
                    panic("OnehotKernel GPU launch failed: " + String(e))
        # CPU path — direct iteration on CPU buffer
        var result_shape = src.shape + [num_classes]
        var result_buf = Buffer[Self.dtype].zeros(result_shape.num_elements())
        var result_strides = Strides.default(result_shape)
        var ign = ignore_index.or_else(-1000000)
        var ndb_buf = src.buffer
        var ndb_offset = src.offset
        var ndb_strides = src.strides
        for coord in src.shape:
            var coord_list = List[Int]()
            for i in range(len(coord)):
                coord_list.append(coord[i])
            var flat = IndexCalculator.flatten_index(
                src.shape,
                coord_list,
                ndb_strides,
                ndb_offset,
            )
            var ci = ndb_buf[flat].__int__()
            if ignore_index and ci == ign:
                continue
            if ci < 0 or ci >= num_classes:
                panic(
                    "OnehotKernel: invalid class",
                    String(ci),
                    "at coordinate",
                    String(coord),
                )
            var out_coord = coord + ci
            var out_coord_list = List[Int]()
            for i in range(len(out_coord)):
                out_coord_list.append(out_coord[i])
            var out_flat = IndexCalculator.flatten_index(
                result_shape,
                out_coord_list,
                result_strides,
                0,
            )
            result_buf[out_flat] = Scalar[Self.dtype](1)
        return NDBuffer[Self.dtype](result_buf^, result_shape)

    def shuffle(
        self,
        permutation: List[Int],
        axis: Int,
    ) -> NDBuffer[Self.dtype]:
        var shape = self.shape
        # Defensive: Shuffle.forward normalizes, but the range(axis) /
        # range(axis+1, rank) arithmetic below silently miscomputes for
        # axis<0 — never trust the raw value here.
        var rank = shape.rank()
        var ax = axis if axis >= 0 else rank + axis
        var result = NDBuffer[Self.dtype].zeros(shape)

        # Fast path: contiguous source — for a fixed outer coordinate each row
        # along the axis maps to a contiguous `inner`-element run in both source
        # and output, so use per-row memcpy blocks (parallelized) instead of
        # per-coordinate IndexCalculator get/set.
        if self.is_contiguous():
            var axis_size = shape[ax]
            var inner = 1
            for d in range(ax + 1, rank):
                inner *= shape[d]
            var outer = 1
            for d in range(ax):
                outer *= shape[d]
            var seg_stride = axis_size * inner
            var n_blocks = outer * axis_size
            var self_off = self.offset
            var src_ptr = self.data_ptr()
            var dst_ptr = result.data_ptr()
            var n_threads = num_physical_cores()

            def shuffle_row(b: Int) {imm}:
                var o = b // axis_size
                var k = b % axis_size
                var src = self_off + (o * axis_size + permutation[k]) * inner
                var dst = (o * axis_size + k) * inner
                unsafe_memcpy(
                    dest=dst_ptr.unsafe_offset(dst),
                    src=src_ptr.unsafe_offset(src),
                    count=inner,
                )

            if n_blocks >= n_threads and shape.numels() >= n_threads * 32768:
                parallelize(shuffle_row, n_blocks, n_threads)
            else:
                for b in range(n_blocks):
                    shuffle_row(b)
        else:
            # General path: coordinate-by-coordinate copy (any strides)
            for coord in shape:
                var src_coord = coord
                src_coord[ax] = permutation[coord[ax]]
                result[coord] = self[src_coord]
        return result^

    @always_inline
    def index_iterator(
        self,
    ) -> IndexIterator[origin_of(self.shape), origin_of(self.strides)]:
        return IndexIterator(
            shape=Pointer(to=self.shape).as_imm(),
            strides=Pointer(to=self.strides).as_imm(),
            start_offset=self.offset,
        )

    @staticmethod
    @always_inline
    def Empty() -> NDBuffer[Self.dtype]:
        return NDBuffer[Self.dtype](Buffer[Self.dtype]())

    @staticmethod
    @always_inline
    def arange(
        args: VariadicList[Scalar[Self.dtype], _],
    ) -> NDBuffer[Self.dtype]:
        var buffer = Buffer[Self.dtype].arange(args)
        var shape = Shape(buffer.size)
        return NDBuffer[Self.dtype](buffer^, shape^)

    @staticmethod
    @always_inline
    def arange(
        *args: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        return Self.arange(args)

    @staticmethod
    @always_inline
    def linspace(
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        steps: Int,
    ) -> NDBuffer[Self.dtype]:
        var buffer = Buffer[Self.dtype].linspace(start, end, steps)
        var shape = Shape(buffer.size)
        return NDBuffer[Self.dtype](buffer^, shape^)

    @always_inline
    def is_contiguous(self) -> Bool:
        return self.strides.is_contiguous(self.shape)

    @always_inline
    def size(self) -> Int:
        return self.buffer.size

    def __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        var idx_list = List[Int]()
        for i in range(len(indices)):
            idx_list.append(indices[i])
        var index = IndexCalculator.flatten_index(
            self.shape, idx_list, self.strides, self.offset
        )
        return self.storage_get(index)

    def __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        var idx_list = List[Int]()
        for i in range(len(indices)):
            idx_list.append(indices[i])
        var index = IndexCalculator.flatten_index(
            self.shape, idx_list, self.strides, self.offset
        )
        self.storage_set(index, value)

    def __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        var index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.storage_get(index)

    def __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        var index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.storage_set(index, value)

    def __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        var index = IndexCalculator.flatten_variadic(
            self.shape, indices, self.strides, self.offset
        )
        return self.storage_get(index)

    def __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
        var index = IndexCalculator.flatten_variadic(
            self.shape, indices, self.strides, self.offset
        )
        self.storage_set(index, value)

    def __getitem__(self, indices: VariadicList[Int, _]) -> Scalar[Self.dtype]:
        var index = IndexCalculator.flatten_variadic(
            self.shape, indices, self.strides, self.offset
        )
        return self.storage_get(index)

    def __setitem__(
        self, indices: VariadicList[Int, _], value: Scalar[Self.dtype]
    ):
        var index = IndexCalculator.flatten_variadic(
            self.shape, indices, self.strides, self.offset
        )
        self.storage_set(index, value)

    @always_inline
    def item(self) -> Scalar[Self.dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim"
                " buffer/singleton, got shape: "
                + String(self.shape)
            )
        return self.get(0)

    def get(self, index: Int) -> Scalar[Self.dtype]:
        """Get element at a logical flat index (C-order over this view).

        Respects view offset/strides: on a slice/transpose view, `get(i)`
        is the i-th logical element. Bounds-checked against the logical
        size (`numels()`); negative indices wrap from the end.
        """
        var n = self.numels()
        var idx = index + n if index < 0 else index
        if idx < 0 or idx >= n:
            panic(
                "NDBuffer → get: index out of bounds.",
                "Logical size",
                String(n),
                ", provided index",
                String(index),
            )
        return self.storage_get(self._logical_to_storage(idx))

    def set(self, index: Int, value: Scalar[Self.dtype]):
        """Set element at a logical flat index (mirror of `get`)."""
        var n = self.numels()
        var idx = index + n if index < 0 else index
        if idx < 0 or idx >= n:
            panic(
                "NDBuffer → set: index out of bounds.",
                "Logical size",
                String(n),
                ", provided index",
                String(index),
            )
        self.storage_set(self._logical_to_storage(idx), value)

    @always_inline
    def _logical_to_storage(self, idx: Int) -> Int:
        """Map a normalized logical flat index to a storage address."""
        if self.is_contiguous():
            return self.offset + idx
        var storage = self.offset
        var rem = idx
        for d in range(self.shape.rank() - 1, -1, -1):
            var c = rem % self.shape[d]
            rem //= self.shape[d]
            storage += c * self.strides[d]
        return storage

    def storage_get[checked: Bool = True](
        self, index: Int
    ) -> Scalar[Self.dtype]:
        """Raw storage read: `index` is a buffer address, no view mapping.

        Negative indices wrap against the last touched address
        (`max_storage_index`), mirroring logical `get`. Bounds-checked
        against the touched storage range
        (`min_storage_index()..max_storage_index()`).

        Internal use only — callers holding addresses from
        `index_iterator()` or `flatten_index(..., offset)`. Prefer `get`
        for logical indexing.

        Args:
            checked: When True (default), wrap negatives and panic on
                out-of-range addresses. Pass `checked=False` only when the
                caller guarantees a non-negative in-range storage address
                (e.g. offsets from `index_iterator()`, pre-validated
                indices) — the min/max recomputation and branches are
                skipped entirely.
        """
        var idx = index
        comptime if checked:
            if idx < 0:
                idx += self.max_storage_index() + 1
            if idx < self.min_storage_index() or idx > self.max_storage_index():
                panic(
                    "NDBuffer → storage_get: index out of bounds.",
                    "NDBuffer storage range",
                    String(self.min_storage_index())
                    + ".."
                    + String(self.max_storage_index()),
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
        return self.data_ptr()[unsafe_offset=idx]

    def storage_set[checked: Bool = True](
        self, index: Int, value: Scalar[Self.dtype]
    ):
        """Raw storage write (mirror of `storage_get`).

        Args:
            checked: When True (default), wrap negatives and panic on
                out-of-range addresses. Pass `checked=False` only when the
                caller guarantees a non-negative in-range storage address.
        """
        var idx = index
        comptime if checked:
            if idx < 0:
                idx += self.max_storage_index() + 1
            if idx < self.min_storage_index() or idx > self.max_storage_index():
                panic(
                    "NDBuffer → storage_set: index out of bounds.",
                    "NDBuffer storage range",
                    String(self.min_storage_index())
                    + ".."
                    + String(self.max_storage_index()),
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
            ptr[unsafe_offset=idx] = value

    @always_inline
    def load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from tenmo. 2D NDBuffer."""
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
        return self.data_ptr().unsafe_load[width=simdwidth](addr)

    @always_inline
    def store[
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
            ptr.unsafe_store[width=simdwidth](addr, value)

    def __str__(self) -> String:
        var s = String("NDBuffer [")
        s += "Shape: " + String(self.shape)
        s += ", Type: " + String(Self.dtype)
        s += ", Shared : " + String(self.is_shared())
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

    def print(self, num_first: Int = 10, num_last: Int = 10) raises:
        print(
            "\n",
            String(self),
            end="\n",
        )
        var empty = List[Int]()
        print_buffer(
            self,
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    def __repr__(self) -> String:
        return self.__str__()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    def data_buffer(ref self) -> ref[self.buffer] Buffer[Self.dtype]:
        return self.buffer

    @always_inline
    def is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape()

    @always_inline
    def numels(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    def __len__(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    def rank(self) -> Int:
        return self.shape.rank()

    @always_inline
    def max_index(self) -> Int:
        """Highest valid LOGICAL flat index (C-order over this view).

        Always `numels() - 1`, regardless of offset/strides. Use with
        `get`/`set`; for the highest touched buffer address use
        `max_storage_index` with `storage_get`/`storage_set`.
        """
        return self.numels() - 1

    @always_inline
    def max_storage_index(self) -> Int:
        """Highest touched buffer address (storage space).

        `offset + Σ(shape[i]-1)*strides[i]` over positive-stride dims.
        Use with `storage_get`/`storage_set`; for logical indexing use
        `max_index` with `get`/`set`.
        """
        var max_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] > 0:
                max_idx += (self.shape[i] - 1) * self.strides[i]
            # negative stride: highest address is already at offset,
            # no addition needed for this dimension
        return max_idx

    @always_inline
    def min_storage_index(self) -> Int:
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
            print(buf.min_storage_index())  # 10 + 1*(-1) = 9
            ```
        """
        var min_idx = self.offset
        for i in range(self.shape.rank()):
            if self.strides[i] < 0:
                min_idx += (self.shape[i] - 1) * self.strides[i]
        return min_idx

    @always_inline
    def offset_at(self, indices: IntArray) -> Int:
        """Return the absolute linear offset in the underlying buffer
        for the given multidimensional indices."""
        if indices.size() != self.rank():
            panic("NDBuffer.offset_at: Incorrect number of indices")

        return IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )

    def to_dtype[NewType: DType](self) -> NDBuffer[NewType]:
        comptime if has_accelerator():
            if self.is_on_gpu():
                comptime if Self.dtype == NewType:
                    # Same dtype on GPU: zero-copy view via create_sub_buffer.
                    # No allocation, no copy — just GPU pointer offset.
                    # size_of[Self.dtype]() == size_of[NewType]() by comptime
                    # guarantee, so element counts match exactly.
                    try:
                        var src_state = self.contiguous_device_state()
                        var sub_buf = src_state.buffer.create_sub_buffer[
                            NewType
                        ](0, len(src_state.buffer))
                        var new_state = DeviceState[NewType](
                            sub_buf, src_state.gpu
                        )
                        return NDBuffer[NewType].with_device_state(
                            new_state^, self.shape
                        )
                    except e:
                        panic(
                            "NDBuffer to_dtype same-dtype GPU failed: "
                            + String(e)
                        )
                        return NDBuffer[NewType].Empty()
                else:
                    # Cross-dtype on GPU: CPU round-trip.
                    try:
                        var cpu_ndb = Self.from_device_state(
                            self.contiguous_device_state(), self.shape
                        )
                        var cast_ndb = cpu_ndb.to_dtype[NewType]()
                        var new_state = DeviceState[NewType](
                            self.numels(), self.device_state.value().gpu
                        )
                        cast_ndb.fill_device_state(new_state)
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
    def __imul__(self, factor: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Multiply](factor)

    @always_inline
    def __iadd__(self, scalar: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Add](scalar)

    @always_inline
    def __isub__(self, scalar: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Subtract](scalar)

    def __itruediv__(self, scalar: Scalar[Self.dtype]):
        self.inplace_scalar_ops[Divide](scalar)

    @always_inline
    def __imul__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Multiply](other)

    @always_inline
    def __iadd__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Add](other)

    @always_inline
    def __isub__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Subtract](other)

    def __itruediv__(self, other: NDBuffer[Self.dtype]):
        self.inplace_ops[Divide](other)

    @always_inline
    def is_shared(self) -> Bool:
        """Check if underlying buffer is shared."""
        comptime if has_accelerator():
            if self.is_on_gpu():
                return True  # DeviceBuffer is always ref-counted
        return self.buffer.is_shared()

    def share(
        self,
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[Self.dtype]:
        # Correct size depending on device
        var size: Int
        comptime if has_accelerator():
            if self.is_on_gpu():
                size = len(self.device_state.value())
            else:
                size = len(self.buffer)
        else:
            size = len(self.buffer)

        var new_shape = shape.or_else(self.shape)
        var new_strides = strides.or_else(Strides.default(new_shape))
        var max_storage_index = IndexCalculator.max_storage_index(
            new_shape, new_strides, offset
        )

        if max_storage_index >= size:
            print(
                "NDBuffer → share FAILURE: self.shape=",
                self.shape,
                " size=",
                size,
                " cpu_buffer_len=",
                len(self.buffer),
                " view_shape=",
                new_shape,
                " strides=",
                new_strides,
                " offset=",
                offset,
            )
            panic(
                "NDBuffer → share: invalid view [max_storage_index="
                + String(max_storage_index)
                + " > buffer_size="
                + String(size)
                + "] shape="
                + String(new_shape)
                + " strides="
                + String(new_strides)
                + " offset="
                + String(offset)
            )
        # All owned buffers are shared-from-birth; no conversion needed.
        var ndb = NDBuffer[Self.dtype](
            buffer=self.buffer.copy(),
            shape=new_shape,
            strides=new_strides,
            offset=offset,
        )
        ndb.device_state = self.device_state.copy()
        return ndb^

    def __getitem__(self, *slices: Slice) -> NDBuffer[Self.dtype]:
        var (
            shape,
            strides,
            offset,
        ) = Validator.validate_and_compute_view_metadata(
            self.shape, self.strides, slices
        )
        return self.share(shape, strides, self.offset + offset)

    def __getitem__(self, *indices: Idx) -> NDBuffer[Self.dtype]:
        var (
            view_shape,
            view_strides,
            offset,
        ) = Validator.validate_and_compute_advanced_indexing_metadata(
            self.shape, self.strides, indices
        )
        var is_scalar = len(view_shape) == 0
        return self.share(
            Shape() if is_scalar else view_shape,
            Strides() if is_scalar else view_strides,
            self.offset + offset,
        )

    def view(self, indices: List[Idx]) -> NDBuffer[Self.dtype]:
        """List[Idx]-based view (non-variadic) — used by the Python bindings.

        Convenience entry for the *indices: Idx overload; identical metadata
        semantics.
        """
        var (
            view_shape,
            view_strides,
            offset,
        ) = Validator.validate_and_compute_advanced_indexing_metadata(
            self.shape, self.strides, indices
        )
        var is_scalar = len(view_shape) == 0
        return self.share(
            Shape() if is_scalar else view_shape,
            Strides() if is_scalar else view_strides,
            self.offset + offset,
        )

    def chunk(self, *indices: Idx) -> NDBuffer[Self.dtype]:
        var (
            shape,
            strides,
            offset,
        ) = Validator.validate_and_compute_advanced_indexing_metadata(
            self.shape, self.strides, indices
        )
        var result = NDBuffer[Self.dtype](shape)
        var absolute_offset = self.offset + offset
        if strides.is_contiguous(shape):
            unsafe_memcpy(
                dest=result.data_ptr(),
                src=self.data_ptr().unsafe_offset(absolute_offset),
                count=shape.num_elements(),
            )
        else:
            var index = 0
            var index_iterator = IndexIterator(
                shape=Pointer(to=shape),
                strides=Pointer(to=strides),
                start_offset=absolute_offset,
            )
            ref result_buffer = result.data_buffer()
            ref src_buffer = self.data_buffer()
            for idx in index_iterator:
                result_buffer[index] = src_buffer[idx]
                index += 1
        return result^

    def transpose(
        self,
        axes: IntArray = IntArray(),
    ) -> NDBuffer[Self.dtype]:
        ref shape = self.shape
        var normalized_axes = Validator.validate_and_normalize_axes(
            shape, axes, ordered=False, fill_missing=True
        ) if len(axes) > 0 else IntArray.range(
            start=shape.rank() - 1, end=-1, step=-1
        )
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)

        # Always a view: shared buffer, new metadata. Tensor view ops alias;
        # Gradbox callers materialise explicitly via contiguous(owned=True).
        return self.share(new_shape, new_strides, self.offset)

    @always_inline
    def __is__(self, other: NDBuffer[Self.dtype]) -> Bool:
        if self.is_on_cpu() and other.is_on_cpu():
            return self.data_ptr() == other.data_ptr()
        elif self.is_on_gpu() and other.is_on_gpu():
            return self.device_state.value() == other.device_state.value()
        return False

    @always_inline
    def data_ptr(ref self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.buffer.unsafe_ptr()

    @always_inline
    def zero(self):
        self.fill(Scalar[Self.dtype](0))

    @always_inline
    def fill(self, value: Scalar[Self.dtype]):
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
    def fill_cpu(self, value: Scalar[Self.dtype]):
        ref buffer = self.data_buffer()
        if self.is_contiguous():
            buffer.fill(value, self.offset, self.offset + self.numels())
        else:
            var ptr = self.data_ptr().unsafe_mut_cast[True]()
            for index in self.index_iterator():
                (ptr.unsafe_offset(index))[] = value

    def reshape(
        self, new_shape: Shape, validated: Bool = False
    ) -> NDBuffer[Self.dtype]:
        var shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            self.shape, new_shape.intarray()
        )

        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.reshape_gpu(shape)
        # Lazy reshape: a contiguous shared source needs only metadata (refcount
        # bump) — the row-major elements of shape == new_shape exactly. Read-only
        # NDBuffer callers (e.g. CE forward views) skip the alloc+unsafe_memcpy. NOTE:
        # Gradbox.reshape must NOT use this — it materializes its own independent
        # buffer (see gradbox.mojo) so a reshaped gradbox is always contiguous,
        # zero-offset and allocated. Unshared and/or non-contiguous sources still
        # materialize via contiguous().
        if self.is_contiguous() and self.is_shared():
            return self.share(shape, Strides.default(shape), self.offset)
        return self.contiguous(shape)

    def reshape_gpu(
        self, shape: Shape, sync: Bool = True
    ) -> NDBuffer[Self.dtype]:
        var out: NDBuffer[Self.dtype]

        try:
            ref device_state = self.device_state.value()
            var new_state = device_state.new(self.numels(), 0, sync=False)
            self.fill_device_state(new_state, sync=sync)
            out = NDBuffer[Self.dtype].with_device_state(new_state, shape)
        except e:
            print(e)
            panic("Error reshaping device buffer")
            # Unreachable
            out = NDBuffer[Self.dtype].Empty()
        return out^

    def flatten(
        self,
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
    ) -> NDBuffer[Self.dtype]:
        var rank = self.rank()
        if rank == 0:
            return self.contiguous()
        # Normalize negative dims against rank (PyTorch parity:
        # flatten(x, -2, -1) flattens the last two dims).
        var start = start_dim if start_dim >= 0 else rank + start_dim
        var end_raw = end_dim.or_else(rank - 1)
        var endd = end_raw if end_raw >= 0 else rank + end_raw

        if start < 0 or start >= rank:
            panic(
                "NDBuffer → flatten: start_dim ",
                String(start_dim),
                " out of range for rank ",
                String(rank),
            )
        if endd < 0 or endd >= rank:
            panic(
                "NDBuffer → flatten: end_dim ",
                String(end_raw),
                " out of range for rank ",
                String(rank),
            )
        if endd < start:
            panic("NDBuffer → flatten: end_dim must be >= start_dim")

        var original_shape = self.shape
        # compute new shape
        var collapsed = original_shape[start : endd + 1].product()
        var new_shape = (
            original_shape[:start] + [collapsed] + original_shape[endd + 1 :]
        )
        return self.contiguous(new_shape)

    @always_inline
    def contiguous_buffer(self) -> Buffer[Self.dtype]:
        """Returns a contiguous copy of the buffer with the same data - CPU only.
        """
        # - same shape
        # - contiguous strides
        # - offset = 0
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer[start:end]
        else:
            var total = self.numels()
            var buffer = Buffer[Self.dtype](total)
            var rank = self.shape.rank()
            # Row-wise SIMD gather when the last dim is a plain strided run:
            # per-row strided_load + contiguous store (parallelized). Scalar
            # fallback for rank-0 / empty / bool / stride-0 rows.
            if (
                rank > 0
                and total > 0
                and Self.dtype != DType.bool
                and self.shape[rank - 1] > 0
            ):
                var last_dim = self.shape[rank - 1]
                var rows = total // last_dim
                var sptr = self.buffer.unsafe_ptr()
                var dptr = buffer.unsafe_ptr()
                var s_offset = self.offset
                var s_shape = self.shape
                var s_strides = self.strides
                var s_inner = self.strides[rank - 1]
                var n_threads = num_physical_cores()
                var n_segments = (
                    n_threads if rows >= n_threads
                    and total >= n_threads * 32768 else 1
                )
                comptime sw = simd_width_of[Self.dtype]()

                def copy_row(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=s_offset,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        if s_inner == 0:
                            var v = sptr[unsafe_offset=row_off]
                            var vec = SIMD[Self.dtype, sw](v)
                            var c = 0
                            while c + sw <= last_dim:
                                dptr.unsafe_store[width=sw](flat + c, vec)
                                c += sw
                            while c < last_dim:
                                dptr[unsafe_offset=flat + c] = v
                                c += 1
                        else:
                            var j = 0
                            while j + sw <= last_dim:
                                dptr.unsafe_store[width=sw](
                                    flat + j,
                                    strided_load[sw](
                                        sptr.unsafe_offset(
                                            row_off + j * s_inner
                                        ),
                                        s_inner,
                                    ),
                                )
                                j += sw
                            for k in range(j, last_dim):
                                dptr[unsafe_offset=flat + k] = sptr[
                                    unsafe_offset=row_off + k * s_inner
                                ]
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(copy_row, n_segments, n_threads)
                else:
                    copy_row(0)
                return buffer^
            var index = 0
            for idx in self.index_iterator():
                buffer[index] = self.buffer[idx]
                index += 1
            return buffer^

    def contiguous_device_state(
        self, sync: Bool = True
    ) raises -> DeviceState[Self.dtype]:
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
            if sync:
                new_state.sync()
        else:
            # Slow path: fill handles non-contiguous strided GPU source correctly
            # It materialises through CPU: GPU→CPU (strided) then CPU→GPU (contiguous)
            self.fill_device_state(new_state, sync=sync)  # handles this already

        return new_state^

    def contiguous(
        self,
        new_shape: Optional[Shape] = None,
        *,
        owned: Bool = False,
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype]:
        """Contiguous copy of this buffer, optionally under a new shape.

        owned=False (default): fast path — returns self (an alias) when
            already contiguous+shared with unchanged shape.
        owned=True: ALWAYS materialises fresh independent storage, even for
            layout no-ops. This is the default Gradbox shape-op path
            (transpose/permute/squeeze/unsqueeze/broadcast_to): gradient
            mutation isolation requires a different buffer, never a view.
            (Gradbox.reshape is the exception: it forwards NDBuffer.reshape
            as-is — a view when contiguous. See its docstring.)
            External (borrowed) sources are read and cloned into owned
            storage. The GPU leg always yields a fresh DeviceState either way.
        """
        var target_shape = new_shape.or_else(self.shape)

        comptime if has_accelerator():
            if self.is_on_gpu():
                # Already contiguous on GPU with no shape change — but still need
                # a fresh independent DeviceState (unshared), so always materialise
                try:
                    var new_state = self.contiguous_device_state(sync=sync)
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
            not owned
            and self.is_contiguous()
            and self.is_shared()
            and target_shape == self.shape
        ):
            return self
        return NDBuffer[Self.dtype](self.contiguous_buffer(), target_shape^)

    def squeeze(self, axes: IntArray) -> NDBuffer[Self.dtype]:
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

        # Always a view: shared buffer. Gradbox callers materialise
        # explicitly via contiguous(owned=True).
        return self.share(new_shape, new_strides, self.offset)

    def unsqueeze(self, axes: IntArray) -> NDBuffer[Self.dtype]:
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

        # Always a view: shared buffer. Gradbox callers materialise
        # explicitly via contiguous(owned=True).
        return self.share(new_shape, new_strides, self.offset)

    def permute(self, perm: IntArray) -> NDBuffer[Self.dtype]:
        """
        Permute axes of this NDBuffer — always a view.
        perm[i] = j means: new axis i takes old axis j.
        Example: perm=[2,0,1] on shape [A,B,C] → shape [C,A,B].

        Returns a view with reordered shape/strides over the same buffer
        (no data movement; GPU safe — just metadata). Used by Tensor.permute.
        Gradbox.permute materialises explicitly via contiguous(owned=True)
        (GPU safe via contiguous_device_state()).
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

        # Validate permutation — one pass, no O(n) `in` scan per element.
        # Collect normalized indices for the geometry build below so it
        # never depends on raw (possibly negative) input.
        var visited = IntArray.filled(rank, 0)
        var normed = IntArray.with_capacity(rank)
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
            normed.append(normalized)

        # Build permuted shape and strides
        var new_shape_dims = IntArray.with_capacity(rank)
        var new_strides_arr = IntArray.with_capacity(rank)
        for i in range(len(perm)):
            new_shape_dims.append(shape[normed[i]])
            new_strides_arr.append(self.strides[normed[i]])

        var new_shape = Shape(new_shape_dims)
        var new_strides = Strides(new_strides_arr)

        # Always a view: shared buffer, no data movement, GPU safe.
        return self.share(new_shape, new_strides, self.offset)

    def count(self, key: Scalar[Self.dtype]) -> Int:
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
                                if ptr[unsafe_offset=i] == key_storage:
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
                                var block = ptr.unsafe_load[width=simd_width](
                                    idx
                                )
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
                                if ptr[unsafe_offset=i] == key_storage:
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

    def unique(self) -> NDBuffer[Self.dtype]:
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
                            for i in range(numels):
                                uniques.add(
                                    ptr[unsafe_offset=i].cast[Self.dtype]()
                                )

                        else:
                            for i in range(numels):
                                uniques.add(
                                    ptr[unsafe_offset=i].cast[Self.dtype]()
                                )

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
            if not self.is_shared():
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

    def copy_from_alike[
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
            var self_start = self.offset
            var other_start = other.offset
            var self_end = self_start + self.numels()
            var other_end = other_start + other.numels()

            comptime if overwrite:
                self.buffer.inplace_ops[Overwrite, validate=validate](
                    other.buffer, self_start, self_end, other_start, other_end
                )
            else:
                self.buffer.inplace_ops[Add, validate=validate](
                    other.buffer, self_start, self_end, other_start, other_end
                )

        elif self.is_contiguous() and not other.is_contiguous():
            var total = self.numels()
            var rank = self.shape.rank()
            # Row-wise SIMD gather from the strided source (parallelized);
            # scalar fallback for degenerate/bool/stride-0 rows.
            if (
                rank > 0
                and total > 0
                and Self.dtype != DType.bool
                and self.shape[rank - 1] > 0
                and other.strides[rank - 1] != 0
            ):
                var last_dim = self.shape[rank - 1]
                var rows = total // last_dim
                var sptr = other.buffer.unsafe_ptr()
                var dptr = self.buffer.unsafe_ptr()
                var s_offset = other.offset
                var s_shape = other.shape
                var s_strides = other.strides
                var s_inner = other.strides[rank - 1]
                var d_base = self.offset
                var n_threads = num_physical_cores()
                var n_segments = (
                    n_threads if rows >= n_threads
                    and total >= n_threads * 32768 else 1
                )
                comptime sw = simd_width_of[Self.dtype]()

                def copy_row_c2(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=s_offset,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        var j = 0
                        while j + sw <= last_dim:
                            var v = strided_load[sw](
                                sptr.unsafe_offset(row_off + j * s_inner),
                                s_inner,
                            )
                            comptime if overwrite:
                                dptr.unsafe_store[width=sw](
                                    d_base + flat + j, v
                                )
                            else:
                                var cur = dptr.unsafe_load[width=sw](
                                    d_base + flat + j
                                )
                                dptr.unsafe_store[width=sw](
                                    d_base + flat + j, cur + v
                                )
                            j += sw
                        for k in range(j, last_dim):
                            comptime if overwrite:
                                dptr[unsafe_offset=d_base + flat + k] = sptr[
                                    unsafe_offset=row_off + k * s_inner
                                ]
                            else:
                                dptr[unsafe_offset=d_base + flat + k] += sptr[
                                    unsafe_offset=row_off + k * s_inner
                                ]
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(copy_row_c2, n_segments, n_threads)
                else:
                    copy_row_c2(0)
            else:
                var index = self.offset
                var it = other.index_iterator()
                while it.__has_next__():
                    var idx = it.peek()
                    comptime if overwrite:
                        self.buffer[index] = other.buffer[idx]
                    else:
                        self.buffer[index] += other.buffer[idx]
                    index += 1
                    it.skip(1)

        elif not self.is_contiguous() and other.is_contiguous():
            var total = self.numels()
            var rank = self.shape.rank()
            # Row workers (parallelized); strided destination has no SIMD
            # store intrinsic, so the inner run stays scalar with hoisted
            # row bases — the win is parallelism + one iterator seek per row.
            if rank > 0 and total > 0 and self.shape[rank - 1] > 0:
                var last_dim = self.shape[rank - 1]
                var rows = total // last_dim
                var sptr = other.buffer.unsafe_ptr()
                var dptr = self.buffer.unsafe_ptr()
                var s_base = other.offset
                var d_shape = self.shape
                var d_strides = self.strides
                var d_inner = self.strides[rank - 1]
                var d_off0 = self.offset
                var n_threads = num_physical_cores()
                var n_segments = (
                    n_threads if rows >= n_threads
                    and total >= n_threads * 32768 else 1
                )

                def copy_row_c3(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=d_shape).as_imm(),
                        strides=Pointer(to=d_strides).as_imm(),
                        start_offset=d_off0,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        for k in range(last_dim):
                            var sv = sptr[unsafe_offset=s_base + flat + k]
                            comptime if overwrite:
                                dptr[unsafe_offset=row_off + k * d_inner] = sv
                            else:
                                dptr[unsafe_offset=row_off + k * d_inner] += sv
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(copy_row_c3, n_segments, n_threads)
                else:
                    copy_row_c3(0)
            else:
                var index = other.offset
                var it = self.index_iterator()
                while it.__has_next__():
                    var idx = it.peek()
                    comptime if overwrite:
                        self.buffer[idx] = other.buffer[index]
                    else:
                        self.buffer[idx] += other.buffer[index]
                    index += 1
                    it.skip(1)

        else:
            var total = self.numels()
            var rank = self.shape.rank()
            # Both strided: dual row iterators, scalar inner runs,
            # parallelized across rows.
            if rank > 0 and total > 0 and self.shape[rank - 1] > 0:
                var last_dim = self.shape[rank - 1]
                var rows = total // last_dim
                var sptr = other.buffer.unsafe_ptr()
                var dptr = self.buffer.unsafe_ptr()
                var s_shape = other.shape
                var s_strides = other.strides
                var s_inner = other.strides[rank - 1]
                var d_shape = self.shape
                var d_strides = self.strides
                var d_inner = self.strides[rank - 1]
                var s_off0 = other.offset
                var d_off0 = self.offset
                var n_threads = num_physical_cores()
                var n_segments = (
                    n_threads if rows >= n_threads
                    and total >= n_threads * 32768 else 1
                )

                def copy_row_c4(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it_s = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=s_off0,
                    )
                    var it_d = IndexIterator(
                        shape=Pointer(to=d_shape).as_imm(),
                        strides=Pointer(to=d_strides).as_imm(),
                        start_offset=d_off0,
                    )
                    it_s.skip(r0 * last_dim)
                    it_d.skip(r0 * last_dim)
                    for _ in range(r0, r1):
                        var s_row = it_s.peek()
                        var d_row = it_d.peek()
                        for k in range(last_dim):
                            var sv = sptr[unsafe_offset=s_row + k * s_inner]
                            comptime if overwrite:
                                dptr[unsafe_offset=d_row + k * d_inner] = sv
                            else:
                                dptr[unsafe_offset=d_row + k * d_inner] += sv
                        it_s.skip(last_dim)
                        it_d.skip(last_dim)

                if n_segments > 1:
                    parallelize(copy_row_c4, n_segments, n_threads)
                else:
                    copy_row_c4(0)
            else:
                var it_self = self.index_iterator()
                var it_other = other.index_iterator()
                while it_self.__has_next__():
                    var index = it_self.peek()
                    var next_index = it_other.peek()
                    comptime if overwrite:
                        self.buffer[index] = other.buffer[next_index]
                    else:
                        self.buffer[index] += other.buffer[next_index]
                    it_self.skip(1)
                    it_other.skip(1)

    def fill(self, cpu_buffer: NDBuffer[Self.dtype]):
        """Fill this NDBuffer from tenmo. CPU NDBuffer."""
        if cpu_buffer.is_scalar() or cpu_buffer.shape == Shape.Unit():
            self.fill(
                cpu_buffer.item()
            )  # Scalar/Singleton NDBuffer - shared or otherwise
            return

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    cpu_buffer.fill_device_state(self.device_state.value())
                except e:
                    print(e)
                    panic(
                        "NDBuffer → fill: error filling GPU buffer from"
                        " tenmo.PU buffer"
                    )
            else:
                self.fill_cpu(cpu_buffer)
        else:
            self.fill_cpu(cpu_buffer)

    def fill_cpu(self, other: NDBuffer[Self.dtype]):
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

            # Fast path: both contiguous and the source last dim maps 1:1
            # onto the target last dim — every output row is a contiguous
            # run, so bulk-memcpy per row (parallelized) instead of
            # per-element coord translate + get/set. The (B,1)-style case
            # (source last dim broadcasts) becomes a per-row SIMD splat.
            var tgt_shape = self.shape
            var src_shape = other.shape
            var rank = tgt_shape.rank()
            var src_rank = src_shape.rank()
            var last = tgt_shape[rank - 1]
            var src_last_size = src_shape[src_rank - 1] if src_rank > 0 else 1
            if self.is_contiguous() and other.is_contiguous() and rank > 0:
                if last == 0:
                    return
                var outer = self.numels() // last
                # Effective source stride per target dim (0 = broadcast),
                # right-aligned; contiguous source ⇒ row-major strides.
                var eff = List[Int]()
                var run = 1
                for d in range(rank - 1, -1, -1):
                    var sd = d - (rank - src_rank)
                    if sd < 0 or src_shape[sd] == 1:
                        eff.append(0)
                    else:
                        eff.append(run)
                    if sd >= 0:
                        run *= src_shape[sd]
                # eff built back-to-front: eff[rank-1-d] == stride of dim d.
                var src_off_base = other.offset
                var dst_off_base = self.offset
                var sptr = other.buffer.unsafe_ptr()
                var dptr = self.buffer.unsafe_ptr()
                var n_threads = num_physical_cores()
                comptime sw = simd_width_of[Self.dtype]()

                def fill_row(r: Int) {imm}:
                    var rem = r
                    var src_off = src_off_base
                    for d in range(rank - 2, -1, -1):
                        var sz = tgt_shape[d]
                        var c = rem % sz
                        rem //= sz
                        src_off += c * eff[rank - 1 - d]
                    var dst = dst_off_base + r * last
                    if src_last_size == last:
                        unsafe_memcpy(
                            dest=dptr.unsafe_offset(dst),
                            src=sptr.unsafe_offset(src_off),
                            count=last,
                        )
                    else:
                        # Broadcast last dim: single source value splatted.
                        var v = sptr[unsafe_offset=src_off]
                        var vec = SIMD[Self.dtype, sw](v)
                        var c = 0
                        while c + sw <= last:
                            dptr.unsafe_store[width=sw](dst + c, vec)
                            c += sw
                        while c < last:
                            dptr[unsafe_offset=dst + c] = v
                            c += 1

                if outer >= n_threads and self.numels() >= n_threads * 32768:
                    parallelize(fill_row, outer, n_threads)
                else:
                    for r in range(outer):
                        fill_row(r)
                return

            # self.shape -> Target shape
            # other.shape -> Source shape

            var mask = ShapeBroadcaster.broadcast_mask(other.shape, self.shape)
            for coord in self.shape:
                var src_coord = ShapeBroadcaster.translate_index(
                    other.shape, coord, mask, self.shape
                )
                self[coord] = other[src_coord]

    @always_inline
    def inplace_ops[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        sync: Bool = False,
    ):
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
                    BinaryInplaceKernel[Self.dtype].launch[op_code](
                        self.layout(),
                        self.device_state.value(),
                        other.layout(),
                        other.device_state.value(),
                        sync=sync,
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
                CpuArithmeticOps[Self.dtype].inplace_ops[op_code](
                    self.layout(),
                    self.buffer,
                    other.layout(),
                    other.buffer,
                )
        else:
            CpuArithmeticOps[Self.dtype].inplace_ops[op_code](
                self.layout(),
                self.buffer,
                other.layout(),
                other.buffer,
            )

    @always_inline
    def inplace_scalar_ops[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ):
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    comptime if op_code == POW:
                        ScalarInplaceKernel[Self.dtype].launch_inplace_pow(
                            self.layout(),
                            self.device_state.value(),
                            scalar,
                            sync=sync,
                        )
                    else:
                        ScalarInplaceKernel[Self.dtype].launch[op_code](
                            self.layout(),
                            self.device_state.value(),
                            scalar,
                            sync=sync,
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
                CpuArithmeticOps[Self.dtype].inplace_scalar_ops[op_code](
                    self.layout(), self.buffer, scalar
                )
        else:
            CpuArithmeticOps[Self.dtype].inplace_scalar_ops[op_code](
                self.layout(), self.buffer, scalar
            )

    @always_inline
    def __add__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Add](other)

    @always_inline
    def __neg__(self) -> NDBuffer[Self.dtype]:
        return self.unary_ops[NEGATE]()

    @always_inline
    def __abs__(self) -> NDBuffer[Self.dtype]:
        return self.unary_ops[ABS]()

    @always_inline
    def log[
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[LOG, epsilon]()

    @always_inline
    def exp(self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[EXP]()

    @always_inline
    def sigmoid(
        self,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[SIGMOID_FORWARD]()

    @always_inline
    def tanh(self) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        return self.float_unary_ops[TANH_FORWARD]()

    @always_inline
    def __mul__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Multiply](other)

    @always_inline
    def __mul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Multiply](scalar)

    @always_inline
    def __add__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Add](scalar)

    @always_inline
    def __sub__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Subtract](scalar)

    @always_inline
    def max(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[MAX](scalar)

    @always_inline
    def min(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[MIN](scalar)

    @always_inline
    def __pow__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[POW](scalar)

    @always_inline
    def __rmul__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Multiply](scalar)

    @always_inline
    def __sub__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Subtract](other)

    @always_inline
    def __truediv__(self, other: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.arithmetic_ops[Divide](other)

    @always_inline
    def __truediv__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[Divide](scalar)

    @always_inline
    def __rtruediv__(self, scalar: Scalar[Self.dtype]) -> NDBuffer[Self.dtype]:
        return self.scalar_ops[ReverseDivide](scalar)

    @always_inline
    def arithmetic_ops[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        sync: Bool = False,
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
                    var (l, s) = BinaryKernel[Self.dtype].launch[op_code](
                        self.layout(),
                        self.device_state.value(),
                        other.layout(),
                        other.device_state.value(),
                        epsilon,
                        sync=sync,
                    )
                    out = NDBuffer[Self.dtype].with_layout_device_state(l, s^)
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
                var (l, s) = CpuArithmeticOps[Self.dtype].compute[op_code](
                    self.layout(),
                    self.buffer,
                    other.layout(),
                    other.buffer,
                    epsilon,
                )
                out = NDBuffer[Self.dtype].with_layout_buffer(l, s^)
        else:
            var (l2, s2) = CpuArithmeticOps[Self.dtype].compute[op_code](
                self.layout(),
                self.buffer,
                other.layout(),
                other.buffer,
                epsilon,
            )
            out = NDBuffer[Self.dtype].with_layout_buffer(l2, s2^)

        return out^

    def broadcast_to(
        self, target_shape: Shape, sync: Bool = True
    ) -> NDBuffer[Self.dtype]:
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

        # GPU: materialise — keeps today's behaviour (grad buffers on device stay
        # contiguous; BroadcastToBackward's sum/reduce expects a full shape).
        comptime if has_accelerator():
            if self.is_on_gpu():
                return view.contiguous(sync=sync)

        # CPU: return the stride-0 view (pure metadata for shared sources, mirrors
        # expand.mojo). Read paths are already stride-aware; the materialization
        # (alloc + full-size unsafe_memcpy) is skipped. NOTE: Gradbox.broadcast_to
        # materializes the view (see gradbox.mojo) so a broadcast gradbox is
        # always contiguous and independent.
        return view^

    @always_inline
    def scalar_ops[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](
        self: NDBuffer[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) -> NDBuffer[Self.dtype]:
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → scalar_ops: cannot divide by zero")

        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    comptime if op_code == POW:
                        var (l, s) = ScalarKernel[Self.dtype].launch_pow(
                            self.layout(),
                            self.device_state.value(),
                            scalar,
                            sync=sync,
                        )
                        out = NDBuffer[Self.dtype].with_layout_device_state(
                            l, s^
                        )
                    else:
                        var (l2, s2) = ScalarKernel[Self.dtype].launch[op_code](
                            self.layout(),
                            self.device_state.value(),
                            scalar,
                            sync=sync,
                        )
                        out = NDBuffer[Self.dtype].with_layout_device_state(
                            l2, s2^
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
                var (l, s) = CpuArithmeticOps[Self.dtype].compute[
                    op_code, epsilon
                ](self.layout(), self.buffer, scalar)
                out = NDBuffer[Self.dtype].with_layout_buffer(l, s^)
        else:
            var (l2, s2) = CpuArithmeticOps[Self.dtype].compute[
                op_code, epsilon
            ](self.layout(), self.buffer, scalar)
            out = NDBuffer[Self.dtype].with_layout_buffer(l2, s2^)

        return out^

    @always_inline
    def unary_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], sync: Bool = False) -> NDBuffer[Self.dtype]:
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var (l, s) = UnaryKernel[Self.dtype].launch[op_code](
                        self.layout(), self.device_state.value(), sync=sync
                    )
                    out = NDBuffer[Self.dtype].with_layout_device_state(l, s^)
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
                var (l, s) = CpuArithmeticOps[Self.dtype].unary_ops[op_code](
                    self.layout(), self.buffer
                )
                out = NDBuffer[Self.dtype].with_layout_buffer(l, s^)
        else:
            var (l2, s2) = CpuArithmeticOps[Self.dtype].unary_ops[op_code](
                self.layout(), self.buffer
            )
            out = NDBuffer[Self.dtype].with_layout_buffer(l2, s2^)

        return out^

    @always_inline
    def float_unary_ops[
        op_code: Int,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](self: NDBuffer[Self.dtype], sync: Bool = False) -> NDBuffer[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """For LOG/EXP/SIGMOID/TANH."""
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var (l, s) = UnaryKernel[Self.dtype].launch[
                        op_code, epsilon
                    ](self.layout(), self.device_state.value(), sync=sync)
                    out = NDBuffer[Self.dtype].with_layout_device_state(l, s^)
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
                var (l, s) = CpuArithmeticOps[Self.dtype].unary_ops_constrained[
                    op_code, epsilon
                ](self.layout(), self.buffer)
                out = NDBuffer[Self.dtype].with_layout_buffer(l, s^)
        else:
            var (l2, s2) = CpuArithmeticOps[Self.dtype].unary_ops_constrained[
                op_code, epsilon
            ](self.layout(), self.buffer)
            out = NDBuffer[Self.dtype].with_layout_buffer(l2, s2^)

        return out^

    def clamp(
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

    def clamp_in_place(
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

    def __eq__(self, other: Self) -> Bool:
        var ndb = self.compare[Equal](other)
        if ndb.is_on_gpu():
            return ndb.device_state.value().all_true()
        return ndb.buffer.all_true()

    def __ne__(self, other: Self) -> Bool:
        var ndb = self.compare[NotEqual](other)
        if ndb.is_on_gpu():
            return ndb.device_state.value().all_true()
        return ndb.buffer.all_true()

    @always_inline
    def compare[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        sync: Bool = False,
    ) -> NDBuffer[DType.bool]:
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
                    var (l, s) = Compare[Self.dtype].launch[op_code](
                        self.layout(),
                        self.device_state.value(),
                        other.layout(),
                        other.device_state.value(),
                        sync=sync,
                    )
                    result = NDBuffer[DType.bool].with_layout_device_state(
                        l, s^
                    )
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
    def compare_cpu[
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
            var it_self = self.index_iterator()
            var it_other = other.index_iterator()
            while it_self.__has_next__():
                var idx = it_self.peek()
                var next_index = it_other.peek()
                var self_val = self.buffer[idx]
                var other_val = other.buffer[next_index]

                buffer[index] = compare_pair[op_code, Self.dtype](
                    self_val, other_val
                )

                index += 1
                it_self.skip(1)
                it_other.skip(1)

            return NDBuffer[DType.bool](buffer^, self.shape)

    @always_inline
    def compare_scalar[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) -> NDBuffer[DType.bool]:
        var result: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var (l, s) = CompareScalar[Self.dtype].launch[op_code](
                        self.layout(),
                        self.device_state.value(),
                        scalar,
                        sync=sync,
                    )
                    result = NDBuffer[DType.bool].with_layout_device_state(
                        l, s^
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
    def compare_scalar_cpu[
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

                buffer[index] = compare_pair[op_code, Self.dtype](value, scalar)

                index += 1

            return NDBuffer[DType.bool](buffer^, self.shape)

    @always_inline
    def all_close[
        rtol: Scalar[Self.dtype] = CloseTol[Self.dtype].rtol(),
        atol: Scalar[Self.dtype] = CloseTol[Self.dtype].atol(),
    ](self, other: Self, sync: Bool = False) -> Bool:
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
                        self.layout(),
                        self.device_state.value(),
                        other.layout(),
                        other.device_state.value(),
                        sync=sync,
                    )
                except e:
                    print(e)
                    panic("NDBuffer all_close → GPU operation failed")
                    result = False
            elif self.is_on_gpu() and other.is_on_cpu():
                try:
                    var cpu_self = self.to_cpu()
                    result = cpu_self.contiguous_buffer().all_close[
                        rtol=rtol, atol=atol
                    ](other.contiguous_buffer())
                except e:
                    print(e)
                    panic("NDBuffer all_close → to_cpu failed: " + String(e))
                    result = False
            elif self.is_on_cpu() and other.is_on_gpu():
                try:
                    var cpu_other = other.to_cpu()
                    result = self.contiguous_buffer().all_close[
                        rtol=rtol, atol=atol
                    ](cpu_other.contiguous_buffer())
                except e:
                    print(e)
                    panic("NDBuffer all_close → to_cpu failed: " + String(e))
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

    def map_to_bool(
        self, pred: def(Scalar[Self.dtype]) thin -> Bool
    ) -> NDBuffer[DType.bool]:
        """Apply predicate to each element, returning a boolean NDBuffer.
        GPU path: transfers to CPU, applies pred, transfers back if needed.
        CPU path: iterates through elements, applying pred.
        """

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var cpu_ndb = self.to_cpu()
                    var buf = Buffer[DType.bool](len(self))
                    var idx = 0
                    for next in cpu_ndb.index_iterator():
                        buf[idx] = pred(
                            cpu_ndb.storage_get[checked=False](next)
                        )
                        idx += 1
                    return NDBuffer[DType.bool](buf^, self.shape)
                except e:
                    panic("NDBuffer map_to_bool: GPU→CPU failed: " + String(e))
                    return NDBuffer[DType.bool].Empty()

        var buf = Buffer[DType.bool](len(self))
        var idx = 0
        for next in self.index_iterator():
            buf[idx] = pred(self.storage_get[checked=False](next))
            idx += 1
        return NDBuffer[DType.bool](buf^, self.shape)

    def all_true(self) -> Bool where Self.dtype == DType.bool:
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

    def any_true(self) -> Bool where Self.dtype == DType.bool:
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

    def matmul_2d(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype], sync: Bool = False
    ) -> NDBuffer[Self.dtype]:
        ref A_shape = A.shape
        ref B_shape = B.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var C: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if A.is_on_gpu() and B.is_on_gpu():
                try:
                    var (l, s) = MatmulKernel[Self.dtype].launch[
                        tile_size=TILE_SIZE
                    ](
                        A.layout(),
                        A.device_state.value(),
                        B.layout(),
                        B.device_state.value(),
                        sync=sync,
                    )
                    C = NDBuffer[Self.dtype].with_layout_device_state(l, s^)
                except e:
                    print(e)
                    panic("NDBuffer matmul_2d → GPU operation failed")
                    C = NDBuffer[Self.dtype](Shape())  # unreachable
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                panic(
                    (
                        "NDBuffer matmul_2d → both buffers must be on same"
                        " device. A on gpu? "
                    ),
                    String(A.is_on_gpu()),
                    ", B on gpu? ",
                    String(B.is_on_gpu()),
                )
                C = NDBuffer[Self.dtype](Shape())  # unreachable
            else:
                # CPU path — sync parameter irrelevant, CPU is always synchronous
                var (l, s) = MmCpu2d[Self.dtype].tiled_matmul(
                    A.layout(), A.buffer, B.layout(), B.buffer
                )
                C = NDBuffer[Self.dtype].with_layout_buffer(l, s^)
        else:
            # No accelerator — CPU path always. sync parameter ignored.
            var (l2, s2) = MmCpu2d[Self.dtype].tiled_matmul(
                A.layout(), A.buffer, B.layout(), B.buffer
            )
            C = NDBuffer[Self.dtype].with_layout_buffer(l2, s2^)

        return C^

    def matmul_nd(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype], sync: Bool = False
    ) -> NDBuffer[Self.dtype]:
        var A_shape = A.shape
        var B_shape = B.shape

        # Validate inner dims
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
                    var (l, s) = MatmulKernel[Self.dtype].launch[
                        tile_size=TILE_SIZE
                    ](
                        A.layout(),
                        A.device_state.value(),
                        B.layout(),
                        B.device_state.value(),
                        sync=sync,
                    )
                    C = NDBuffer[Self.dtype].with_layout_device_state(l, s^)
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
                var (l, s) = MmCpuNd[Self.dtype].tiled_matmul(
                    A.layout(), A.buffer, B.layout(), B.buffer
                )
                C = NDBuffer[Self.dtype].with_layout_buffer(l, s^)
        else:
            var (l2, s2) = MmCpuNd[Self.dtype].tiled_matmul(
                A.layout(), A.buffer, B.layout(), B.buffer
            )
            C = NDBuffer[Self.dtype].with_layout_buffer(l2, s2^)

        return C^


def print_buffer[
    dtype: DType,
    //,
](
    imm buffer: NDBuffer[dtype],
    mut indices: List[Int],
    level: Int,
    num_first: Int = 10,
    num_last: Int = 10,
) raises:
    """Pretty-print an NDBuffer with elision.

    Recursively prints each dimension, showing the first `num_first` and
    last `num_last` elements along each axis. GPU buffers are transferred
    to host first. Moved here from tenmo/common_utils.mojo so
    common_utils stops importing NDBuffer — breaks the `common_utils ⇄
    ndbuffer` two-cycle.
    """
    comptime if has_accelerator():
        if buffer.is_on_gpu():
            var cpu_buffer = buffer.to_cpu()
            print_buffer(cpu_buffer, indices, level, num_first, num_last)
            return
    if buffer.buffer.size == 0 and buffer.device_state == None:
        print("  Empty")
        return
    if buffer.rank() == 0:  # Tensor with Shape ()
        print(buffer[[]])
        return
    var current_dim = len(indices)
    var indent = " " * (level * 2)

    if current_dim >= buffer.rank():
        print(
            "ERROR: current_dim (",
            current_dim,
            ") >= ndim (",
            buffer.rank(),
            ")",
        )
        return

    var size = buffer.shape[current_dim]

    if size < 0 or size > 100_000_000:
        print(
            "ERROR: suspicious size: ",
            size,
            "at dim ",
            current_dim,
            String(buffer.shape),
        )
        return

    # Base case: last dimension (print actual elements)
    if current_dim == buffer.rank() - 1:
        print(indent + "[", end="")

        for i in range(size):
            if i < num_first:
                indices.append(i)
                print(buffer[indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")
            elif i == num_first and size > num_first + num_last:
                print("..., ", end="")
            elif i >= size - num_last:
                indices.append(i)
                print(buffer[indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")

        print("]", end="")

    else:
        print(indent + "[")
        for i in range(size):
            if i < num_first:
                indices.append(i)
                print_buffer(buffer, indices, level + 1, num_first, num_last)
                _ = indices.pop()
            elif i == num_first and size > num_first + num_last:
                print(indent + "  ...,")
            elif i >= size - num_last:
                indices.append(i)
                print_buffer(buffer, indices, level + 1, num_first, num_last)
                _ = indices.pop()

            # Print comma and newline for all but last element
            if i != size - 1 and (i < num_first or i >= size - num_last):
                print(",")
            # Special case: last element needs newline before closing bracket
            elif i == size - 1:
                print()  # Newline before closing bracket

        print(indent + "]", end="")


struct NDBufferLite[dtype: DType](ImplicitlyCopyable):
    """NDBufferLite — refcounted heap handle to an NDBuffer
    A light, dtype-generic refcounted wrapper over a single NDBuffer. It lets a
    small owning struct (16 B) share one descriptor blob ([Atomic rc | NDBuffer])
    across many copies in O(1) instead of deep-copying the NDBuffer's
    Shape/Strides/offset descriptor on every copy. Lives here in the storage
    layer (alongside NDBuffer) so both Ancestor.ndb (forward-value snapshots)
    and Gradbox (gradient storage) can use it without reaching into the autograd
    graph files (same handle pattern as ParentNode).
    Empty state (both pointers None) is 16 B and signals "no backing NDBuffer" —
    used by Ancestor nodes that need no forward parent data.
    """
    var _src: Optional[Pointer[NDBuffer[Self.dtype], MutUntrackedOrigin]]
    var _refcount: Optional[Pointer[Atomic[UInt64], MutUntrackedOrigin]]

    def __init__(out self):
        self._src = None
        self._refcount = None

    @always_inline
    def is_empty(self) -> Bool:
        return self._src is None

    def __init__(out self, var ndb: NDBuffer[Self.dtype]):
        var ndb_sz = size_of[NDBuffer[Self.dtype]]()
        var rc_sz = size_of[Atomic[UInt64]]()
        var base = unsafe_alloc[UInt8](rc_sz + ndb_sz)
        var rc = base.unsafe_bitcast[Atomic[UInt64]]()
        rc[] = Atomic[UInt64](1)
        var np = base.unsafe_offset(rc_sz).unsafe_bitcast[
            NDBuffer[Self.dtype]
        ]()
        np.unsafe_write(ndb^)
        self._src = np
        self._refcount = rc

    def __init__(out self, *, deinit move: Self):
        self._src = move._src
        self._refcount = move._refcount
        _ = move._src = {}
        _ = move._refcount = {}

    def __init__(out self, *, copy: Self):
        self._src = copy._src
        self._refcount = copy._refcount
        if copy._refcount:
            _ = copy._refcount.unsafe_value()[].fetch_add[
                ordering=Ordering.RELAXED
            ](1)

    def __deinit__(deinit self):
        if self._refcount == None or self._src == None:
            return
        if (
            self._refcount.unsafe_value()[].fetch_sub[
                ordering=Ordering.RELEASE
            ](1)
            != 1
        ):
            return
        fence[ordering=Ordering.ACQUIRE]()
        self._src.unsafe_value().unsafe_deinit_pointee()
        var alloc_start = self._refcount.unsafe_value().unsafe_bitcast[UInt8]()
        alloc_start.unsafe_free()
        _ = self._refcount = {}
        _ = self._src = {}

    @always_inline
    def value(ref self) -> ref[self] NDBuffer[Self.dtype]:
        return self._src.unsafe_value()[]
