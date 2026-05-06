from std.math import exp, floor, log, cos, sin, sqrt, pi
from std.random import seed, random_float64
from std.sys import simd_width_of
from std.utils.numerics import min_finite
from std.memory import memcpy, memset, memset_zero, AddressSpace
from .shapes import Shape, ShapeIndexIterator
from .ancestry import Ancestors, Ancestor
from .strides import Strides
from std.os.atomic import Atomic, Consistency, fence
from .common_utils import (
    IDGen,
    log_warning,
    now,
    Idx,
    print_buffer,
    panic,
    Epsilon,
    One,
)
from .mnemonics import *
from .indexhelper import IndexIterator
from .backpropagation import Backward, BackwardFnArg
from .forwards import *
from .buffers import Buffer
from .validators import Validator
from std.collections import Set, Deque
from .gradbox import Gradbox
from .intarray import IntArray
from .broadcasthelper import ShapeBroadcaster
from .ndbuffer import NDBuffer
from std.gpu.host import DeviceBuffer, DeviceContext
from .device import Device, CPU, GPU
from std.sys.info import has_accelerator


struct Tensor[dtype: DType](
    Copyable
    & Movable
    & Sized
    & Writable
    & Absable
    & Equatable
    & ImplicitlyCopyable
    & Iterable
):

    """A multi-dimensional array with automatic differentiation support.

    Tensor is the central type for all tensor operations in tenmo. It wraps an
    NDBuffer for memory layout and tracks gradients through the autograd graph.

    A Tensor can live on CPU or GPU, supports broadcasting, views, and gradient
    computation via backpropagation.

    Example:
    ```mojo
    var a = Tensor[DType.float32].zeros(3, 4)
    var b = Tensor[DType.float32].randn(3, 4)
    var c = a + b
    c.backward()
    ```
    """

    comptime Row = List[Scalar[Self.dtype]]
    comptime Rows = List[Self.Row]
    comptime Block = List[Self.Rows]
    comptime Blocks = List[Self.Block]
    var _id: UInt
    var buffer: NDBuffer[Self.dtype]
    var requires_grad: Bool
    var gradbox: UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]
    var ancestors: Optional[Ancestors[Self.dtype]]

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        self._id = IDGen.generate_id()
        self.buffer = NDBuffer[Self.dtype](shape)
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.ancestors = None
        self.init_gradbox()

    fn __init__(out self):
        self._id = 0
        self.buffer = NDBuffer[Self.dtype].Empty()
        self.requires_grad = False
        self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.ancestors = None

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
        requires_grad: Bool = False,
        *,
        copy: Bool = True,
    ):
        self._id = IDGen.generate_id()
        var num_elements = shape.num_elements()
        var buffer = Buffer[Self.dtype](
            num_elements, ptr + offset, copy=True
        ) if copy else Buffer[Self.dtype](num_elements, ptr, copy=False)
        var offset_adjusted = 0 if copy else offset
        self.buffer = NDBuffer[Self.dtype](
            buffer^, shape, strides, offset_adjusted
        )
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.ancestors = None
        self.init_gradbox()

    fn __init__(
        out self,
        var buffer: NDBuffer[Self.dtype],
        requires_grad: Bool = False,
    ):
        self._id = IDGen.generate_id()
        self.buffer = buffer^
        self.requires_grad = requires_grad
        self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.ancestors = None
        self.init_gradbox()

    @staticmethod
    fn from_device_buffer(
        buffer: DeviceBuffer[Self.dtype],
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
        requires_grad: Bool = False,
    ) raises -> Tensor[Self.dtype]:
        """Create a tensor from a GPU device buffer.

        Args:
            buffer: GPU device buffer containing the data.
            shape: Tensor shape. If None, inferred from buffer length.
            strides: Memory strides. If None, inferred from shape.
            offset: Base memory offset (default: 0).
            requires_grad: Whether this tensor requires gradient tracking.

        Returns:
            A new tensor wrapping the device buffer.

        Raises:
            Any error from mapping device buffer to host.
        """
        var out: Tensor[Self.dtype]
        with buffer.map_to_host() as host_buffer:
            var shape_realized = shape.or_else(Shape(len(host_buffer)))
            out = Tensor[Self.dtype](
                host_buffer.unsafe_ptr(),
                shape_realized,
                strides,
                offset,
                requires_grad,
                copy=True,
            )
        return out

    fn as_gradbox(
        deinit self, share: Bool = False, *, contiguous: Bool = True
    ) -> Gradbox[Self.dtype]:
        """Convert tensor to a Gradbox for gradient operations.

        Args:
            share: If True, shared gradbox. If False, gradbox is not shared.
            contiguous: If True, materializes a contiguous copy first.

        Returns:
            A Gradbox containing the tensor's data.
        """
        if contiguous:
            return Gradbox[Self.dtype](self^.buffer.contiguous(), share=share)
        else:
            return Gradbox[Self.dtype](self^.buffer^, share=share)

    fn as_list(self) -> List[Scalar[Self.dtype]]:
        """Copy tensor data into a Mojo List.

        Returns:
            List containing all tensor elements in row-major order.
        """
        var count = self.numels()
        var tensor_data = List[Scalar[Self.dtype]](capacity=count)
        memcpy(dest=tensor_data.unsafe_ptr(), src=self.data_ptr(), count=count)
        return tensor_data^

    fn __moveinit__(out self, deinit take: Self):
        self._id = take._id
        self.buffer = take.buffer^
        self.requires_grad = take.requires_grad
        self.gradbox = take.gradbox
        self.ancestors = take.ancestors^

    fn __copyinit__(out self, copy: Self):
        self._id = copy._id
        self.buffer = copy.buffer.copy()
        self.requires_grad = copy.requires_grad
        if copy.gradbox != UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]():
            self.gradbox = alloc[Gradbox[Self.dtype]](1)
            self.gradbox.init_pointee_copy(copy.gradbox[])
        else:
            self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.ancestors = copy.ancestors.copy()

    fn shallow_copy(self) -> Tensor[Self.dtype]:
        """Create a shallow copy with the underlying buffer.

        Returns:
            A new tensor with its which just copies the buffer.
        """
        var out = Tensor[Self.dtype]()
        out._id = IDGen.generate_id()
        out.buffer = self.buffer.copy()
        return out^

    @always_inline
    fn id(self) -> UInt:
        """Get the unique identifier for this tensor.

        Returns:
            Unique UInt ID assigned at tensor creation.
        """
        return self._id

    fn init_gradbox(mut self):
        """Initialize gradient storage if requires_grad is True.

        Allocates GPU memory for gradients if the tensor is on GPU.
        """
        if (
            self.requires_grad
            and self.gradbox
            == UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        ):
            var gradbox: Gradbox[Self.dtype]

            comptime if has_accelerator():
                if self.is_on_gpu():
                    try:
                        var device_state = self.buffer.device_state.value().new(
                            self.numels(), Scalar[Self.dtype](0)
                        )
                        var ndb = NDBuffer[Self.dtype].with_device_state(
                            device_state^, self.shape()
                        )
                        gradbox = Gradbox[Self.dtype](ndb^, share=False)
                    except e:
                        print(e)
                        panic(
                            "init_gradbox: failed to allocate GPU gradbox: "
                            + String(e)
                        )
                        gradbox = Gradbox[
                            Self.dtype
                        ].Empty  # unreachable, satisfies compiler
                else:
                    gradbox = Gradbox[Self.dtype](self.shape())
                    gradbox.zero_grad()
            else:
                gradbox = Gradbox[Self.dtype](self.shape())
                gradbox.zero_grad()
            self.gradbox = alloc[Gradbox[Self.dtype]](1)
            self.gradbox.init_pointee_move(gradbox^)

    @always_inline
    fn is_contiguous(self) -> Bool:
        """Check if tensor memory layout is contiguous.

        Returns:
            True if the tensor is stored contiguously in row-major order.
        """
        return self.buffer.is_contiguous()

    @always_inline
    fn shared(self) -> Bool:
        """Check if the underlying buffer is shared by multiple views.

        Returns:
            True if the buffer is reference-counted and shared.
        """
        return self.buffer.shared()

    fn is_leaf(self) -> Bool:
        """Check if this tensor is a leaf in the autograd graph.

        A leaf tensor is one that requires gradients but has no ancestors
        (operations that produced it). Leaf tensors are the starting points
        of gradient computation.

        Returns:
            True if this is a leaf tensor, False otherwise.
        """
        return self.requires_grad and not self.ancestors is None

    @always_inline
    fn __len__(self) -> Int:
        """Get the total number of elements in the tensor.

        Returns:
            The total number of elements (product of shape dimensions).
            Returns 0 for scalar tensors with Shape().
        """
        return self.shape()[0] if self.shape() != Shape() else 0

    @always_inline
    fn shape(ref self) -> ref[self.buffer.shape] Shape:
        """Get the shape of this tensor.

        Returns:
            Reference to the tensor's shape.
        """
        return self.buffer.shape

    @always_inline
    fn strides(ref self) -> ref[self.buffer.strides] Strides:
        """Get the strides of this tensor.

        Returns:
            Reference to the tensor's strides.
        """
        return self.buffer.strides

    @always_inline
    fn offset(self) -> Int:
        """Get the base memory offset of this tensor's view.

        Returns:
            The offset into the underlying buffer.
        """
        return self.buffer.offset

    @always_inline
    fn numels(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer.numels()

    @always_inline
    fn num_elements(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer.numels()

    @always_inline
    fn rank(self) -> Int:
        """Get the number of dimensions.

        Returns:
            The number of axes in the tensor's shape.
        """
        return self.buffer.rank()

    @always_inline
    fn max_index(self) -> Int:
        """Get the highest valid memory offset.

        Returns:
            The maximum flat index accessible in this tensor.
        """
        return self.buffer.max_index()

    @always_inline
    fn index_iterator(
        self,
    ) -> IndexIterator[
        origin_of(self.buffer.shape), origin_of(self.buffer.strides)
    ]:
        """Get an iterator over memory offsets.

        Returns:
            IndexIterator for iterating over physical memory offsets.
        """
        return self.buffer.index_iterator()

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        """Index tensor with a List of integers.

        Args:
            indices: List of axis indices.

        Returns:
            Scalar value at the specified coordinates.

        Raises:
            Panic if tensor is scalar but indices are provided.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Tensor → __getitem__(List[Int]): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]

    @always_inline
    fn __getitem__(ref self, indices: IntArray) -> Scalar[Self.dtype]:
        """Index tensor with an IntArray of indices.

        Args:
            indices: IntArray of axis indices.

        Returns:
            Scalar value at the specified coordinates.

        Raises:
            Panic if tensor is scalar but indices are provided.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Tensor → __getitem__(IntArray): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Index tensor with variadic integer indices.

        Args:
            *indices: One index per axis.

        Returns:
            Scalar value at the specified coordinates.

        Raises:
            Panic if tensor is scalar or if index count is unsupported.
        """
        if self.rank() == 0:
            panic(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(List[Int])"
            )

        return self.buffer[indices]

    fn __getitem__[
        track_grad: Bool = True
    ](mut self, *slices: Slice) -> Tensor[Self.dtype]:
        """Slice tensor using Slice objects along each axis.

        Args:
            *slices: One Slice per axis, specifying start, stop, step.

        Returns:
            A view tensor with computed shape, strides, and offset.

        Example:
            ```mojo
            var t = Tensor[DType.float32].zeros(10, 10)
            var view = t[1:5, 2:8]
            ```
        """
        var (
            shape,
            strides,
            offset,
        ) = Validator.validate_and_compute_view_metadata(
            self.shape(),
            self.strides(),
            slices,
        )
        return View[Self.dtype].forward[track_grad=track_grad](
            self,
            shape=shape,
            strides=strides,
            offset=offset,
            requires_grad=self.requires_grad,
            validated=True,
        )

    fn chunk(
        self, *indices: Idx, requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Extract independent chunks along the first axis by indices.

        Allocates a new buffer — gradients do not flow back to the source.

        Args:
            *indices: Indices along the first axis to extract.
            requires_grad: Whether the result tracks gradients.

        Returns:
            A new tensor containing the selected chunks.
        """
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape(),
                self.strides(),
                indices,
            )
        )
        var result = Tensor[Self.dtype](shape, requires_grad=requires_grad)
        var absolute_offset = self.offset() + offset
        if strides.is_contiguous(shape):
            memcpy(
                dest=result.data_ptr(),
                src=self.data_ptr() + absolute_offset,
                count=shape.num_elements(),
            )
        else:
            var index = 0
            var index_iterator = IndexIterator(
                shape=Pointer(to=shape),
                strides=Pointer(to=strides),
                start_offset=absolute_offset,
            )
            ref result_buffer = result.buffer.data_buffer()
            ref src_buffer = self.buffer.data_buffer()

            for idx in index_iterator:
                result_buffer[index] = src_buffer[idx]
                index += 1

        return result^

    fn __getitem__[
        track_grad: Bool = True
    ](mut self, *indices: Idx) -> Tensor[Self.dtype]:
        """Advanced indexing with Idx objects (integers or slices).

        Args:
            *indices: Idx per axis — either an integer or a Slice.
                Missing axes default to full slices.

        Returns:
            A view tensor over the indexed region.
        """
        var (
            view_shape,
            view_strides,
            offset,
        ) = Validator.validate_and_compute_advanced_indexing_metadata(
            self.shape(), self.strides(), indices
        )

        is_scalar = len(view_shape) == 0
        shape = Shape() if is_scalar else view_shape
        strides = Strides() if is_scalar else view_strides
        abs_offset = self.offset() + offset
        return View[Self.dtype].forward[track_grad=track_grad](
            self,
            shape,
            strides,
            abs_offset,
            self.requires_grad,
            validated=True,
        )

    fn gather[track_grad: Bool = True](
        mut self, indices: IntArray, axis: Int = 0
    ) -> Tensor[Self.dtype]:
        """Gather slices along `axis` at the given indices.

        Returns a zero-copy view if indices form a regular stride pattern,
        otherwise copies data.
        """
        var rank = self.shape().rank()

        # Validate and normalize axis
        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic("gather: axis ", String(axis), " out of bounds for rank ", String(rank))

        if len(indices) == 0:
            panic("gather: indices cannot be empty")

        # Validate and normalize indices
        var ax_dim = self.shape()[ax]
        var normalized = IntArray.with_capacity(len(indices))
        for k in range(len(indices)):
            var idx = indices[k]
            if idx < 0:
                idx += ax_dim
            if idx < 0 or idx >= ax_dim:
                panic(
                    "gather: index ", String(indices[k]),
                    " out of bounds for axis ", String(ax),
                    " with size ", String(ax_dim),
                )
            normalized.append(idx)

        # ── Fast path: single index ───────────────────────────────────────────────
        if len(normalized) == 1:
            return self.slice[track_grad](
                start = normalized[0], end = normalized[0] + 1, step = 1, axis = ax
            )

        # ── Fast path: regular stride pattern → zero-copy view ───────────────────
        var step = normalized[1] - normalized[0]
        if step != 0:
            var is_regular = True
            for k in range(2, len(normalized)):
                if normalized[k] - normalized[k - 1] != step:
                    is_regular = False
                    break

            if is_regular:
                var end = normalized[len(normalized) - 1] + (1 if step > 0 else -1)
                return self.slice[track_grad](
                    start = normalized[0], end = end, step = step, axis = ax
                )

        # ── Slow path: irregular or repeated indices → copy ───────────────────────
        return self._gather_copy(ax, normalized)

    fn _gather_copy(mut self, ax: Int, normalized: IntArray) -> Tensor[Self.dtype]:
        """Copy-based gather for non-contiguous index patterns."""
        var rank = self.shape().rank()

        var out_shape_arr = IntArray.with_capacity(rank)
        for d in range(rank):
            out_shape_arr.append(len(normalized) if d == ax else self.shape()[d])

        var result = Tensor[Self.dtype].zeros(Shape(out_shape_arr))
        var total  = result.shape().num_elements()

        for flat in range(total):
            var coords = IntArray.with_capacity(rank)
            var rem = flat
            for d in range(rank - 1, -1, -1):
                #coords.insert(0, rem % result.shape()[d])
                coords.prepend(rem % result.shape()[d])
                rem //= result.shape()[d]

            var src_idx = normalized[coords[ax]]   # already normalized

            var src_offset = self.offset()
            for d in range(rank):
                src_offset += (src_idx if d == ax else coords[d]) * self.strides()[d]

            result.set(flat, self.get(src_offset))

        return result

    @always_inline
    fn __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            *indices: One index per axis.
            value: Scalar value to write.

        Raises:
            Panic if tensor is scalar.
        """
        if self.rank() == 0:
            panic(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(List[Int])"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: List of axis indices.
            value: Scalar value to write.

        Raises:
            Panic if tensor is scalar but indices are provided.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Tensor → __setitem__(List[Int]): Scalar tensor expects no"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, coord: IntArray, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            coord: IntArray of axis indices.
            value: Scalar value to write.

        Raises:
            Panic if tensor is scalar but indices are provided.
        """
        if self.rank() == 0 and len(coord) != 0:
            panic(
                "Tensor → __setitem__(IntArray): Scalar tensor expects no"
                " indices"
            )

        self.buffer[coord] = value

    fn fill(self, value: Scalar[Self.dtype], *indices: Idx):
        """Fill a region with a scalar value using advanced indexing.

        Args:
            value: Scalar value to write.
            *indices: Idx objects (integers or slices) defining the region.
        """
        Filler[Self.dtype].fill(self.buffer, value, indices)

    fn fill(self, tensor: Tensor[Self.dtype], *indices: Idx):
        """Copy data from another tensor into a region using advanced indexing.

        Args:
            tensor: Source tensor to copy from.
            *indices: Idx objects (integers or slices) defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer, tensor.buffer, indices)

    fn fill(self, gradbox: Gradbox[Self.dtype], *indices: Idx):
        """Copy data from a Gradbox into a region using advanced indexing.

        Args:
            gradbox: Source Gradbox to copy from.
            *indices: Idx objects (integers or slices) defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer, gradbox.buffer, indices)

    fn item(self) -> Scalar[Self.dtype]:
        return self.buffer.item()

    @no_inline
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
        s += String(self.shape())
        if self.buffer.shared():
            s += ", strides: " + String(self.strides())
            s += ", offset: " + String(self.offset())
        s += ", Type: " + String(Self.dtype)
        s += ", requires_grad: " + String(self.requires_grad)
        s += (
            ", Device : "
            + "gpu: "
            + String(
                self.buffer.gpu_id()
            ) if self.is_on_gpu() else ", Device : "
            + "cpu"
        )
        s += "]"
        return s

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn write_to(self, buffer: DeviceBuffer[Self.dtype]) raises:
        with buffer.map_to_host() as host_buffer:
            if self.is_contiguous():
                memcpy(
                    dest=host_buffer.unsafe_ptr(),
                    src=self.data_ptr() + self.offset(),
                    count=self.numels(),
                )
            else:
                var ptr = host_buffer.unsafe_ptr()
                ref data_buffer = self.buffer.data_buffer()
                var offset = 0
                for index in self.index_iterator():
                    (ptr + offset)[] = data_buffer[index]
                    offset += 1

    fn to_gpu(
        self,
        gpu: Optional[GPU] = None,
        requires_grad: Optional[Bool] = None,
        stop_grad: Bool = False,
    ) raises -> Tensor[Self.dtype]:
        """Transfer tensor to GPU.

        Args:
            gpu: Target GPU. Uses default GPU if None.
            stop_grad: If True, gradient stops at this GPU tensor —
                       no DeviceTransferBackward registered. Use for
                       permanent GPU residents (model weights).
                       Default False preserves existing grad flow behaviour.
        """
        comptime if has_accelerator():
            var target_gpu = gpu.or_else(GPU())
            return DeviceTransfer[Self.dtype].forward(
                self,
                target_gpu.into(),
                stop_grad=stop_grad,
            )
        else:
            raise Error(
                "Can not move to GPU. System does not have any accelerator"
                " device"
            )

    fn to_cpu(
        self,
        requires_grad: Optional[Bool] = None,
        stop_grad: Bool = False,
    ) raises -> Tensor[Self.dtype]:
        """Transfer tensor to CPU.

        Args:
            requires_grad: If provided, overrides the requires_grad flag
                on the returned tensor.
            stop_grad: If True, gradient stops at this CPU tensor —
                       no DeviceTransferBackward registered.
                       Default False preserves existing grad flow behaviour.
        Returns:
            A new tensor on CPU, with data copied over.

        Raises:
            Error if system has no accelerator.
        """
        comptime if has_accelerator():
            return DeviceTransfer[Self.dtype].forward[True](
                self,
                CPU().into(),
                requires_grad=requires_grad,
                stop_grad=stop_grad,
            )

        raise Error("System does not have any accelerator")

    fn device(self) -> Device:
        """Get the device this tensor is on.

        Returns:
            The CPU or GPU device the tensor resides on.
        """
        return self.buffer.device()

    fn device_context(self) -> Optional[DeviceContext]:
        """Get GPU device context for this tensor.

        Returns:
            GPU device context if tensor is on GPU, None otherwise.
        """
        comptime if has_accelerator():
            if self.is_on_gpu():
                return self.buffer.device_context()
            else:
                return None
        return None

    @always_inline
    fn has_grad(self) -> Bool:
        """Check if gradient storage has been initialized.

        Returns:
            True if a gradient buffer exists for this tensor, False otherwise.
        """
        return (
            self.gradbox != UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        )

    @always_inline
    fn zero_grad(self):
        """Zero out accumulated gradients.

        Call this before backward() to reset gradients to zero.
        Only affects tensors that require and have gradients.
        """
        if self.requires_grad and self.has_grad():
            ref gradbox = self.gradbox[]
            gradbox.zero_grad()

    @always_inline
    fn gradients(self) -> UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]:
        """Get raw pointer to the gradient buffer.

        Returns:
            Pointer to the Gradbox storing accumulated gradients.

        Raises:
            Panic if called on a tensor that does not require grad or has no gradient.
        """
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → gradients(self): called on a tensor that does not"
                " require grad or grad not initialized"
            )
        return self.gradbox

    @always_inline
    fn grad(self) -> Gradbox[Self.dtype]:
        """Get accumulated gradients as a Gradbox.

        Returns:
            The Gradbox containing accumulated gradients.

        Raises:
            Panic if called on a tensor that does not require grad or has no gradient.
        """
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad or grad not initialized"
            )
        return self.gradbox[].detach()

    fn rows(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → rows: tensor rank is not 2")
        return self.shape()[0]

    fn cols(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → cols: tensor rank is not 2")
        return self.shape()[1]

    fn is_scalar(self) -> Bool:
        """Check if this tensor is a scalar (zero-dimensional).

        Returns:
            True if the tensor has a single element, False otherwise.
        """
        return self.buffer.is_scalar()

    fn is_on_gpu(self) -> Bool:
        """Check if this tensor is stored on a GPU.

        Returns:
            True if the tensor is on a GPU, False otherwise.
        """
        return self.buffer.is_on_gpu()

    fn is_on_cpu(self) -> Bool:
        """Check if this tensor is stored on the CPU.

        Returns:
            True if the tensor is on the CPU, False otherwise.
        """
        return self.is_on_gpu() == False

    fn __eq__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[Equal](scalar))

    fn __ne__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[NotEqual](scalar))

    fn __lt__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[LessThan](scalar))

    fn __le__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[LessThanEqual](scalar)
        )

    fn gt(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThan](scalar)
        )

    fn lt(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[LessThan](scalar))

    fn __gt__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThan](scalar)
        )

    fn __ge__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThanEqual](scalar)
        )

    fn __eq__(self, other: Tensor[Self.dtype]) -> Bool:
        return self.buffer == other.buffer

    fn eq(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[Equal](other.buffer))

    fn __ne__(self, other: Tensor[Self.dtype]) -> Bool:
        return self.buffer != other.buffer

    fn ne(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[NotEqual](other.buffer))

    fn __lt__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[LessThan](other.buffer))

    fn __le__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[LessThanEqual](other.buffer)
        )

    fn __gt__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[GreaterThan](other.buffer)
        )

    fn __ge__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[GreaterThanEqual](other.buffer)
        )

    fn float(self) -> Tensor[DType.float32]:
        """Convert tensor to float32 dtype.

        Returns:
            A new tensor with dtype float32.
        """
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Tensor[DType.float64]:
        """Convert tensor to float64 dtype.

        Returns:
            A new tensor with dtype float64.
        """
        return self.to_dtype[DType.float64]()

    fn to_dtype[
        NewType: DType
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[NewType]:
        """Convert tensor to a different data type.

        Args:
            requires_grad: If provided, overrides the requires_grad flag
                on the returned tensor.

        Returns:
            A new tensor with the specified dtype.
        """
        var new_type_buffer = self.buffer.to_dtype[NewType]()
        var grad_required = requires_grad.or_else(self.requires_grad)
        return Tensor[NewType](new_type_buffer^, requires_grad=grad_required)

    fn add_ancestry(
        mut self,
        var backwardFnArg: BackwardFnArg[Self.dtype],
        *parents: Tensor[Self.dtype],
    ):
        """Register parent tensors and backward function for autograd.

        Args:
            backwardFnArg: Backward pass function and metadata.
            *parents: Parent tensors whose gradients this tensor depends on.
        """
        if not self.ancestors:
            self.ancestors = Optional(Ancestors[Self.dtype](backwardFnArg^))
        else:
            self.ancestors.value().set_backward_fn_arg(backwardFnArg^)

        ref ancestors = self.ancestors.value()
        for parent in parents:
            if not parent.buffer.shared():
                var parent_copy = parent.copy()
                var parent_buffer = parent_copy.buffer.copy()
                parent_buffer = parent_buffer.share(
                    parent_buffer.shape,
                    parent_buffer.strides,
                    parent_buffer.offset,
                )
                parent_copy.buffer = parent_buffer^
                ancestors.append(parent_copy^)

            else:
                ancestors.append(parent)

    fn has_ancestry(self) -> Bool:
        """Check if this tensor has registered parent dependencies.

        Returns:
            True if parent tensors have been registered, False otherwise.
        """
        return self.ancestors != None

    @always_inline
    fn ancestry(ref self) -> ref[self.ancestors.value()] Ancestors[Self.dtype]:
        """Get the ancestry graph for backward pass traversal.

        Returns:
            Reference to the Ancestors containing parent dependencies.

        Raises:
            Panic if ancestry has not been initialized.
        """
        if self.ancestors == None:
            panic("Tensor → ancestry: ancestry not initialized")
        return self.ancestors.value()

    @always_inline
    fn broadcastable(self, to: Tensor[Self.dtype]) -> Bool:
        """Check if this tensor can broadcast to a target shape.

        Args:
            to: Target tensor to check broadcasting compatibility with.

        Returns:
            True if this tensor's shape can broadcast to to.shape().
        """
        return ShapeBroadcaster.broadcastable(self.shape(), to.shape())

    fn log[
        track_grad: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        self,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        return Logarithm[Self.dtype].forward[track_grad, epsilon](
            self, requires_grad
        )

    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Self,) -> Bool:
        """Check if two tensors are element-wise close within tolerance.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            other: Tensor to compare against.

        Returns:
            True if all elements are within tolerance, False otherwise.
        """
        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    fn all(self, pred: fn(Scalar[Self.dtype]) -> Bool) -> Bool:
        """Returns True if pred holds for all elements.
        Uses NDBuffer.map_to_bool — handles GPU via CPU materialisation.
        """
        return self.buffer.map_to_bool(pred).all_true()

    fn any(self, pred: fn(Scalar[Self.dtype]) -> Bool) -> Bool:
        """Returns True if pred holds for any element.
        Uses NDBuffer.map_to_bool — handles GPU via CPU materialisation.
        """
        return self.buffer.map_to_bool(pred).any_true()

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        """Returns True if all elements are True.
        GPU path: NDBuffer.all_true → DeviceState.all_true (maps to host).
        CPU path: NDBuffer.all_true → Buffer.all_true.
        """

        return self.buffer.all_true()

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        """Returns True if any element is True.
        GPU path: NDBuffer.any_true → DeviceState.any_true (maps to host).
        CPU path: NDBuffer.any_true → Buffer.any_true.
        """
        return self.buffer.any_true()

    fn unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Tensor[Self.dtype], origin, address_space=address_space
    ]:
        return (
            UnsafePointer(to=self)
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    fn data_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Scalar[Self.dtype], origin, address_space=address_space
    ]:
        return (
            self.buffer.data_ptr()
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    fn seed_grad(mut self, with_tensor: Tensor[Self.dtype]):
        """Seed gradient accumulation with a specific tensor.

        Args:
            with_tensor: Tensor containing gradient values to seed with.
        """
        if not self.requires_grad:
            return
        if not self.has_grad():
            self.requires_grad_()
        self.gradbox[].seed_grad(with_tensor)

    fn seed_grad(mut self, value: Scalar[Self.dtype]):
        """Seed gradient accumulation with a scalar value.

        Args:
            value: Scalar gradient value to seed with.
        """
        with_tensor = Tensor[Self.dtype].full(self.shape(), value)
        self.seed_grad(with_tensor)

    @always_inline
    fn fill(self, value: Scalar[Self.dtype]):
        """Fill the entire tensor with a scalar value.

        Args:
            value: Scalar value to write to all elements.
        """
        self.buffer.fill(value)

    fn map_where(
        self,
        pred: fn(Scalar[Self.dtype]) -> Bool,
        value: Scalar[Self.dtype],
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Replace elements matching a predicate with a scalar value.

        Args:
            pred: Function that returns True for elements to replace.
            value: Scalar value to write where pred returns True.
            requires_grad: Whether the result tracks gradients.

        Returns:
            A new tensor with replaced values.
        """
        var ndb = self.buffer.map_where(pred, value)
        return Tensor[Self.dtype](ndb^, requires_grad=requires_grad)

    @staticmethod
    fn full_like(
        like: Tensor[Self.dtype],
        value: Scalar[Self.dtype],
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor filled with a value, matching another tensor's shape.

        Args:
            like: Reference tensor whose shape to copy.
            value: Scalar value to fill with.
            requires_grad: Whether the result tracks gradients.
            device: Target device. Defaults to like's device.

        Returns:
            A new tensor with the same shape as like, filled with value.
        """
        var shape = like.shape()
        return Tensor[Self.dtype].full(
            shape^,
            value,
            requires_grad=requires_grad,
            device=device.or_else(like.device()),
        )

    @staticmethod
    fn full(
        shape: List[Int],
        value: Scalar[Self.dtype],
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor filled with a scalar value.

        Args:
            shape: Dimensions as a list of ints.
            value: Scalar value to fill with.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with value.
        """
        return Self.full(Shape(shape), value, requires_grad, device=device)

    @staticmethod
    fn full(
        shape: Shape,
        scalar: Scalar[Self.dtype],
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor filled with a scalar value.

        Args:
            shape: Tensor shape.
            scalar: Scalar value to fill with.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with scalar.
        """
        var target_device = device.or_else(CPU().into())
        return Tensor[Self.dtype](
            NDBuffer[Self.dtype].full(shape, scalar, target_device),
            requires_grad=requires_grad,
        )

    @staticmethod
    fn rand(
        *dims: Int,
        low: Scalar[Self.dtype] = 0,
        high: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a tensor with uniform random values in [low, high).

        Args:
            *dims: Shape dimensions as variadic ints.
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            init_seed: Random seed. If None, randomizes each call.
            requires_grad: Whether to track gradients.

        Returns:
            A tensor of given shape with uniform random values.
        """
        return Self.rand(Shape(dims), low, high, init_seed, requires_grad)

    @staticmethod
    fn rand(
        shape: List[Int],
        low: Scalar[Self.dtype] = 0,
        high: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a tensor with uniform random values in [low, high).

        Args:
            shape: Shape as a list of ints.
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            init_seed: Random seed. If None, randomizes each call.
            requires_grad: Whether to track gradients.

        Returns:
            A tensor of given shape with uniform random values.
        """
        return Self.rand(Shape(shape), low, high, init_seed, requires_grad)

    @staticmethod
    fn rand(
        shape: Shape,
        min: Scalar[Self.dtype] = 0,
        max: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a tensor with uniform random values in [min, max).

        Args:
            shape: Tensor shape.
            min: Lower bound (inclusive).
            max: Upper bound (exclusive).
            init_seed: Random seed. If None, randomizes each call.
            requires_grad: Whether to track gradients.

        Returns:
            A tensor of given shape with uniform random values.
        """
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

        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn randn(
        *dims: Int,
        mean: Float64 = 0.0,
        std: Float64 = 1.0,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a tensor with values from a normal distribution.

        Args:
            *dims: Shape dimensions as variadic ints.
            mean: Distribution mean.
            std: Distribution standard deviation.
            init_seed: Random seed. If None, randomizes each call.
            requires_grad: Whether to track gradients.

        Returns:
            A tensor of given shape with normally distributed values.
        """
        return Self.randn(Shape(dims), mean, std, init_seed, requires_grad)

    @staticmethod
    fn randn(
        shape: Shape,
        mean: Float64 = 0.0,
        std: Float64 = 1.0,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a tensor with values from a normal distribution.

        Args:
            shape: Tensor shape.
            mean: Distribution mean.
            std: Distribution standard deviation.
            init_seed: Random seed. If None, randomizes each call.
            requires_grad: Whether to track gradients.

        Returns:
            A tensor of given shape with normally distributed values.
        """
        if init_seed:
            seed(init_seed.value())
        else:
            seed()

        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)

        var i = 0
        while i < numels:
            var u1 = random_float64(0.0, 1.0)
            var u2 = random_float64(0.0, 1.0)

            if u1 < 1e-10:
                u1 = 1e-10

            var mag = std * sqrt(-2.0 * log(u1))
            var z0 = mag * cos(2.0 * pi * u2) + mean
            var z1 = mag * sin(2.0 * pi * u2) + mean

            buffer[i] = z0.cast[Self.dtype]()
            if i + 1 < numels:
                buffer[i + 1] = z1.cast[Self.dtype]()
            i += 2

        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn arange(
        *args: Scalar[Self.dtype],
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a 1D tensor with evenly spaced values.

        Args:
            *args: Start, stop, and optionally step values.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with values from start to stop.

        Example:
            ```mojo
            Tensor[DType.float32].arange(0, 5)      # [0, 1, 2, 3, 4]
            Tensor[DType.float32].arange(0, 10, 2)  # [0, 2, 4, 6, 8]
            ```
        """
        nd_buffer = NDBuffer[Self.dtype].arange(args)
        tensor = Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)
        return tensor^

    @staticmethod
    fn linspace(
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        steps: Int,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a 1D tensor with linearly spaced values.

        Args:
            start: Starting value.
            end: Ending value.
            steps: Number of samples.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with steps values from start to end (inclusive).
        """
        nd_buffer = NDBuffer[Self.dtype].linspace(start, end, steps)
        tensor = Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)
        return tensor^

    @staticmethod
    fn zeros(
        axes_spans: List[Int],
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of zeros.

        Args:
            axes_spans: Shape as a list of ints.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with zeros.
        """
        return Self.zeros(
            Shape(axes_spans), requires_grad=requires_grad, device=device
        )

    @staticmethod
    fn eye(
        n: Int, requires_grad: Bool = False, device: Optional[Device] = None
    ) -> Tensor[Self.dtype]:
        """Create a 2D identity matrix of size n x n.

        Args:
            n: Number of rows and columns.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            An n x n tensor with ones on the diagonal and zeros elsewhere.
        """
        var out = Self.zeros(
            Shape(n, n), requires_grad=requires_grad, device=device
        )
        for i in range(n):
            out[i, i] = Scalar[Self.dtype](1)
        return out^

    @staticmethod
    fn zeros(
        *axes_spans: Int,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of zeros.

        Args:
            *axes_spans: Shape dimensions as variadic ints.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with zeros.
        """
        shape = Shape(axes_spans)
        return Self.zeros(shape, requires_grad=requires_grad, device=device)

    @staticmethod
    fn zeros_like(
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a zeros tensor matching another tensor's shape.

        Args:
            requires_grad: If provided, overrides requires_grad.
            device: Target device. Defaults to self's device.

        Returns:
            A tensor of zeros with the same shape as self.
        """
        var target_device: Optional[Device]

        comptime if has_accelerator():
            if self.is_on_gpu():
                target_device = device.or_else(
                    self.buffer.device_state.value().get_gpu().into()
                )
            else:
                target_device = device.or_else(CPU().into())
        else:
            target_device = CPU().into()

        return Tensor[Self.dtype].zeros(
            self.shape(),
            requires_grad=requires_grad.or_else(self.requires_grad),
            device=target_device,
        )

    @staticmethod
    fn zeros(
        shape: Shape,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of zeros.

        Args:
            shape: Tensor shape.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with zeros.
        """
        var target_device = device.or_else(CPU().into())
        return Tensor[Self.dtype](
            NDBuffer[Self.dtype].zeros(shape, target_device),
            requires_grad=requires_grad,
        )

    @staticmethod
    fn ones_like(
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a ones tensor matching another tensor's shape.

        Args:
            requires_grad: If provided, overrides requires_grad.
            device: Target device. Defaults to self's device.

        Returns:
            A tensor of ones with the same shape as self.
        """
        var target_device: Optional[Device]

        comptime if has_accelerator():
            if self.is_on_gpu():
                target_device = device.or_else(self.device())
            else:
                target_device = device.or_else(CPU().into())
        else:
            target_device = CPU().into()

        return Tensor[Self.dtype].ones(
            self.shape(),
            requires_grad=requires_grad.or_else(self.requires_grad),
            device=target_device,
        )

    @staticmethod
    fn onehot(
        indices: Tensor[Self.dtype],
        num_classes: Int,
        device: Optional[Device] = None,
        ignore_index: Optional[Int] = None,
    ) -> Tensor[Self.dtype]:
        """Convert tensor of class indices to one-hot encoding.
        Args:
            indices: Tensor of shape (...,) containing class indices.
            num_classes: Number of classes.
            device: Target device.
            ignore_index: If provided, rows where index == ignore_index become all zeros.
        Returns: Tensor of shape (..., num_classes).
        """
        var onehot_ndb = NDBuffer[Self.dtype].onehot(
            indices.buffer, num_classes, device, ignore_index
        )
        return Tensor[Self.dtype](onehot_ndb^, requires_grad=False)

    @staticmethod
    fn d1(row: Self.Row, requires_grad: Bool = False) -> Tensor[Self.dtype]:
        """Create a 1D tensor from a list of scalar values.

        Args:
            row: List of scalar values forming the single dimension.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with len(row) elements.
        """
        Validator.validate_dtype_consistency(Self.dtype, requires_grad, "d1")
        if len(row) == 0:
            return Tensor[Self.dtype].scalar(
                min_finite[Self.dtype](), requires_grad=requires_grad
            )
        numels = len(row)
        shape = Shape(IntArray(numels))
        buffer = Buffer[Self.dtype](numels)
        memcpy(dest=buffer.data, src=row.unsafe_ptr(), count=numels)
        nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn d2(
        rows: List[Self.Row], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 2D tensor from a list of rows.

        Args:
            rows: List of rows, each row must have equal length.
            requires_grad: Whether to track gradients.

        Returns:
            A 2D tensor of shape (len(rows), len(rows[0])).

        Raises:
            Panic if rows have inconsistent lengths.
        """
        Validator.validate_dtype_consistency(Self.dtype, requires_grad, "d2")
        dims = IntArray(len(rows), len(rows[0]))
        flattened = List[Scalar[Self.dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                panic("Tensor → d2 → not all rows equal in length")
            flattened.extend(row.copy())
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[Self.dtype](numels)
        memcpy(dest=buffer.data, src=flattened.unsafe_ptr(), count=numels)
        nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad)

    @staticmethod
    fn d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 3D tensor from a list of 2D blocks.

        Args:
            blocks: List of 2D matrices, each matrix must have equal dimensions.
            requires_grad: Whether to track gradients.

        Returns:
            A 3D tensor.

        Raises:
            Panic if blocks have inconsistent dimensions.
        """
        Validator.validate_dtype_consistency(Self.dtype, requires_grad, "d3")
        dims = IntArray(len(blocks), len(blocks[0]), len(blocks[0][0]))
        flattened = List[Scalar[Self.dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                panic("Tensor → d3 → not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    panic("Tensor → d3 → not all rows equal in length")

                flattened.extend(row.copy())
        shape = Shape(dims)
        numels = shape.num_elements()
        buffer = Buffer[Self.dtype](numels)
        memcpy(dest=buffer.data, src=flattened.unsafe_ptr(), count=numels)
        nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn d4(
        blockgrid: List[Self.Block], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 4D tensor from a nested list structure.

        Args:
            blockgrid: List of 3D blocks, each block must have equal dimensions.
            requires_grad: Whether to track gradients.

        Returns:
            A 4D tensor.

        Raises:
            Panic if blockgrid has inconsistent dimensions.
        """
        Validator.validate_dtype_consistency(Self.dtype, requires_grad, "d4")
        dims = IntArray(
            len(blockgrid),
            len(blockgrid[0]),
            len(blockgrid[0][0]),
            len(blockgrid[0][0][0]),
        )
        flattened = List[Scalar[Self.dtype]](capacity=dims.product())
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
        buffer = Buffer[Self.dtype](numels)
        memcpy(dest=buffer.data, src=flattened.unsafe_ptr(), count=numels)
        nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn d5(
        blockhive: List[Self.Blocks], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 5D tensor from a deeply nested list structure.

        Args:
            blockhive: List of 4D blocks, each block must have equal dimensions.
            requires_grad: Whether to track gradients.

        Returns:
            A 5D tensor.

        Raises:
            Panic if blockhive has inconsistent dimensions.
        """
        Validator.validate_dtype_consistency(Self.dtype, requires_grad, "d5")
        dims = IntArray(
            len(blockhive),
            len(blockhive[0]),
            len(blockhive[0][0]),
            len(blockhive[0][0][0]),
            len(blockhive[0][0][0][0]),
        )
        flattened = List[Scalar[Self.dtype]](capacity=dims.product())
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
        buffer = Buffer[Self.dtype](numels)
        memcpy(dest=buffer.data, src=flattened.unsafe_ptr(), count=numels)
        nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    fn of(
        *elems: Scalar[Self.dtype], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 1D tensor from variadic scalar elements.

        Args:
            *elems: Scalar values as variadic arguments.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with len(elems) elements.
        """
        Validator.validate_dtype_consistency(
            Self.dtype, requires_grad, "of(*elems)"
        )
        shape = Shape(IntArray(len(elems)))
        tensor = Tensor[Self.dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor^

    @staticmethod
    fn of(
        elems: Self.Row,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a 1D tensor from a list of scalar elements.

        Args:
            elems: List of scalar values.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with len(elems) elements.
        """
        Validator.validate_dtype_consistency(
            Self.dtype, requires_grad, "of(elems)"
        )
        shape = Shape(IntArray(len(elems)))
        tensor = Tensor[Self.dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]

        return tensor^

    @staticmethod
    fn scalar(
        val: Scalar[Self.dtype], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a scalar (0D) tensor from a single value.

        Args:
            val: The scalar value.
            requires_grad: Whether to track gradients.

        Returns:
            A scalar tensor containing val.
        """
        result = Tensor[Self.dtype](Shape(), requires_grad=requires_grad)
        result[IntArray()] = val
        return result^

    @staticmethod
    fn ones(
        *axes_spans: Int,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of ones.

        Args:
            *axes_spans: Shape dimensions as variadic ints.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with ones.
        """
        return Self.ones(Shape(axes_spans), requires_grad, device)

    @staticmethod
    fn ones(
        shape: Shape,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of ones.

        Args:
            shape: Tensor shape.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with ones.
        """
        var target_device = device.or_else(CPU().into())
        var value = One[Self.dtype].value()
        return Tensor[Self.dtype](
            NDBuffer[Self.dtype].full(shape, value, target_device),
            requires_grad=requires_grad,
        )

    fn broadcast_to(
        self, target_shape: Shape, requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Broadcast tensor to a target shape.

        Args:
            target_shape: Shape to broadcast to.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with the target shape.

        Raises:
            Panic if current shape cannot broadcast to target_shape.
        """
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Tensor → broadcast_to: shape "
                + String(self.shape())
                + " not broadcastable to "
                + String(target_shape)
            )

        broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        grad_required = requires_grad.or_else(self.requires_grad)
        out = Tensor[Self.dtype](
            broadcasted_buffer^, requires_grad=grad_required
        )
        return out^

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from a 2D Tensor.

        Preconditions:
            - Tensor must be 2D.
            - Columns must be contiguous (stride[1] == 1) for SIMD loads.
            - `col + simdwidth` must not exceed the number of columns.
        """
        return self.buffer.load[simdwidth, validated](row, col)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
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
    ) -> Tensor[Self.dtype]:
        """Flatten the tensor by collapsing a range of dimensions.

        Args:
            start_dim: First dimension to flatten (default: 0).
            end_dim: Last dimension to flatten (default: last).
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with flattened dimensions.
        """
        return FlattenForward[Self.dtype].forward[track_grad](
            self, start_dim, end_dim, requires_grad
        )

    fn repeat[
        track_grad: Bool = True
    ](self, repeat: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Repeat tensor along each axis.

        Args:
            repeat: Number of repeats per dimension.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with repeated data.
        """
        return Repeat[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn repeat[
        track_grad: Bool = True
    ](self, *repeat: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Repeat tensor along each axis.

        Args:
            *repeat: Number of repeats per dimension as variadic ints.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with repeated data.
        """
        return Repeat[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn tile[
        track_grad: Bool = True
    ](self, repeat: List[Int], requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Tile the tensor by repeating it along each axis.

        Args:
            repeat: Number of tiles per dimension.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tiled tensor.
        """
        return Tile[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn tile[
        track_grad: Bool = True
    ](self, *repeat: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Tile the tensor by repeating it along each axis.

        Args:
            *repeat: Number of tiles per dimension as variadic ints.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tiled tensor.
        """
        return Tile[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad
        )

    fn contiguous[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[Self.dtype]:
        """Return a contiguous copy of the tensor.

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A contiguous tensor with the same data.
        """
        return Contiguous[Self.dtype].forward[track_grad](self, requires_grad)

    fn reshape[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[Self.dtype]:
        """Reshape a single-element tensor to a scalar.

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A scalar tensor.

        Raises:
            Panic if tensor has more than one element.
        """
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
        Self.dtype
    ]:
        """Reshape tensor to new dimensions.

        Args:
            *newdims: New shape as variadic ints.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with the new shape.
        """
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
    ](
        self,
        shape: List[Int],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        Self.dtype
    ]:
        """Reshape tensor to new dimensions.

        Args:
            shape: New shape as a list of ints.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with the new shape.
        """
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
    ) -> Tensor[Self.dtype]:
        """Reshape tensor to new dimensions.

        Args:
            new_shape: Target shape.
            requires_grad: If provided, overrides requires_grad.
            validated: If True, skips shape validation.

        Returns:
            A tensor with the new shape.
        """
        return Reshape[Self.dtype].forward[track_grad](
            self, new_shape, requires_grad, validated
        )

    fn upstream_grad_share[
        augment: Bool
    ](
        self,
        other: Tensor[Self.dtype],
        upstream_grad: Gradbox[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        """Handle gradient broadcasting and augmentation during backprop.

        Args:
            augment: If True, multiplies upstream_grad by other before accumulating.
            other: The other tensor involved in the operation.
            upstream_grad: Incoming gradient to be processed.

        Returns:
            Processed Gradbox compatible with self's shape for accumulation.
        """
        var grad_contrib: Gradbox[Self.dtype]
        if upstream_grad.shape() == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                self.shape(),
                upstream_grad.item(),
                share=False,
                device=upstream_grad.device(),
            )
        else:
            comptime if augment:
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
    ) -> Tensor[Self.dtype]:
        """Sum tensor elements along given axes.

        Args:
            axes: Axes along which to sum.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with summed values along the specified axes.
        """
        return self.sum[track_grad](IntArray(axes), keepdims, requires_grad)

    fn sum[
        track_grad: Bool = True
    ](
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Sum tensor elements along given axes.

        Args:
            axes: Axes along which to sum.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with summed values along the specified axes.
        """

        return Summer[Self.dtype].forward[track_grad](
            self, axes, keepdims, requires_grad
        )

    fn product[
        track_grad: Bool = True,
        store_excl_product: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute product along given axes.

        Args:
            axes: Axes along which to compute product.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with product values along the specified axes.
        """
        return self.product[track_grad, store_excl_product](
            IntArray(axes), keepdims, requires_grad
        )

    fn product[
        track_grad: Bool = True,
        store_excl_product: Bool = True,
    ](
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Product tensor elements along given axes.

        Args:
            axes: Axes along which to compute product.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with product values along the specified axes.
        """
        return Product[Self.dtype].forward[track_grad, store_excl_product](
            self, axes, keepdims, requires_grad
        )

    fn sqrt[
        track_grad: Bool = True
    ](
        self,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Element-wise square root.

        Args:
            epsilon: Small value added for numerical stability.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with square root of each element.
        """
        return Sqrt[Self.dtype].forward[track_grad](
            self, epsilon, requires_grad
        )

    fn mean[
        track_grad: Bool = True
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute mean along given axes.

        Args:
            axes: Axes along which to compute mean.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with mean values along the specified axes.
        """
        return self.mean[track_grad](IntArray(axes), keepdims, requires_grad)

    fn mean[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute mean along given axes.

        Args:
            axes: Axes along which to compute mean.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with mean values along the specified axes.
        """
        return Mean[Self.dtype].forward[track_grad](
            self, axes, keepdims, requires_grad
        )

    fn reciprocal[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute reciprocal.

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with reciprocal values of the elements.
        """
        return Reciprocal[Self.dtype].forward[track_grad](
            self, requires_grad
        )

    fn variance[
        track_grad: Bool = True
    ](
        self,
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute variance along an axis.

        Args:
            axis: Axis along which to compute variance.
            keepdims: If True, keep reduced axis with size 1.
            unbiased: If True, use n-1 (sample variance). If False, use n (population).
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with variance along the specified axis.
        """
        return Variance[Self.dtype].forward[track_grad](
            self, axis, keepdims, unbiased, requires_grad
        )

    fn std[
        track_grad: Bool = True
    ](
        self,
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute standard deviation along an axis.

        Args:
            axis: Axis along which to compute std.
            keepdims: If True, keep reduced axis with size 1.
            unbiased: If True, use n-1. If False, use n.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with std deviation along the specified axis.
        """
        return StdDev[Self.dtype].forward[track_grad](
            self, axis, keepdims, unbiased, requires_grad
        )

    fn norm(
        self,
        p: Float64 = 2.0,
        axis: Optional[Int] = None,
        keepdims: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Compute Lp norm of the tensor.

        Args:
            p: Power for the norm computation (only p=2.0 supported).
            axis: Axis along which to compute. If None, computes global norm.
            keepdims: If True, keep reduced axes with size 1.

        Returns:
            Tensor with the Lp norm value.

        Raises:
            Panic if p != 2.0 (only L2 norm supported).
        """
        if p == 2.0:
            var squared = self.__mul__[track_grad=False](self)
            var dim = [axis.value()] if axis else List[Int]()
            var sum_sq = squared.sum[track_grad=False](dim, keepdims=keepdims)
            return sum_sq.sqrt[track_grad=False]()
        else:
            panic("Only L2 norm (p=2) currently supported")
            return Tensor[Self.dtype].scalar(0)

    fn __rtruediv__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        return DivideScalar[Self.dtype].forward[track_grad](self, scalar)

    fn __truediv__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        return DivideByScalar[Self.dtype].forward[track_grad](self, scalar)

    fn __truediv__[
        track_grad: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        return Divider[Self.dtype].forward[track_grad](self, other)

    fn update_grad[opcode: Int](mut self, incoming: Gradbox[Self.dtype]):
        """Update gradient using an incoming gradient and operation type.

        Args:
            opcode: Operation mnemonic (MulTensor, AddTensor, etc.)
            incoming: Gradbox containing upstream gradients.
        """
        ref gradbox = self.gradbox[]
        if opcode == MulTensor:
            gradbox *= incoming
        if opcode == AddTensor:
            gradbox += incoming
        if opcode == SubtractTensor:
            gradbox -= incoming
        if opcode == ZeroGrad:
            self.zero_grad()

    fn __iadd__(self, other: Self):
        """In-place addition of another tensor.

        Args:
            other: Tensor to add.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __iadd__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )
        self.buffer.__iadd__(other.buffer)

    fn __isub__(self, other: Self):
        """In-place subtraction of another tensor.

        Args:
            other: Tensor to subtract.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __isub__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.__isub__(other.buffer)

    fn __isub__(self, other: Gradbox[Self.dtype]):
        """In-place subtraction of a Gradbox.

        Args:
            other: Gradbox to subtract.
        """
        self.buffer.__isub__(other.buffer)

    fn __imul__(self, other: Self):
        """In-place multiplication by another tensor.

        Args:
            other: Tensor to multiply by.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __imul__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.__imul__(other.buffer)

    fn __itruediv__(self, other: Self):
        """In-place division by another tensor.

        Args:
            other: Tensor to divide by.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __itruediv__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.__itruediv__(other.buffer)

    fn unique(self) -> Tensor[Self.dtype]:
        """Return a tensor with duplicate elements removed.

        Returns:
            A tensor with the same data but duplicates removed.
        """
        return Tensor[Self.dtype](self.buffer.unique(), requires_grad=False)

    fn count(self, key: Scalar[Self.dtype]) -> Int:
        """Count occurrences of a value in the tensor.

        Args:
            key: Scalar value to count.

        Returns:
            Number of elements equal to key.
        """
        return self.buffer.count(key)

    fn sum_all(self) -> Scalar[Self.dtype]:
        """Sum all elements into a single scalar - CPU only op.

        Returns:
            Sum of all elements in the tensor.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → sum_all is for numeric data types only"
        return self.buffer.sum_all()

    fn product_all(self) -> Scalar[Self.dtype]:
        """Compute the product of all elements - CPU only op.

        Returns:
            Product of all elements in the tensor.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → product_all is for numeric data types only"

        return self.buffer.product_all()

    fn exp[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Element-wise exponential (e^x).

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with exponential of each element.
        """
        return Exponential[Self.dtype].forward[track_grad=track_grad](
            self, requires_grad=requires_grad
        )

    fn __neg__[track_grad: Bool = True](self) -> Tensor[Self.dtype]:
        """Negate all elements.

        Args:
            track_grad: Whether to track gradients.

        Returns:
            Tensor with negated values.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __neg__ is for numeric data types only"

        var zeros = Tensor[Self.dtype].zeros_like(self, requires_grad=False)
        return Subtractor[Self.dtype].forward[track_grad=track_grad](
            zeros, self
        )

    fn __invert__(
        self: Tensor[Self.dtype],
    ) -> Tensor[Self.dtype] where (
        Self.dtype == DType.bool or Self.dtype.is_integral()
    ):
        """Bitwise invert (logical NOT for bool, bitwise NOT for integers).

        Returns:
            Tensor with inverted values.
        """
        return Tensor[Self.dtype](
            self.buffer.unary_ops[INVERT](), requires_grad=False
        )

    fn __abs__(self) -> Tensor[Self.dtype]:
        """Element-wise absolute value.

        Returns:
            Tensor with absolute values.
        """
        return Tensor[Self.dtype](self.buffer.__abs__(), requires_grad=False)

    fn __radd__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Right-side addition (scalar + tensor).

        Args:
            scalar: Scalar value on the left.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with scalar added to all elements.
        """
        return self.__add__[track_grad](scalar)

    fn __add__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Add a scalar to all elements.

        Args:
            scalar: Scalar value to add.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with scalar added to all elements.
        """
        return AddScalar[Self.dtype].forward[track_grad](self, scalar)

    fn max[
        track_grad: Bool = True
    ](
        self, scalar: Scalar[Self.dtype], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Element-wise maximum with a scalar.

        Args:
            scalar: Scalar to compare against.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with max(self, scalar) for each element.
        """
        return MaxScalar[Self.dtype].forward[track_grad](
            self, scalar, requires_grad
        )

    fn min[
        track_grad: Bool = True
    ](
        self, scalar: Scalar[Self.dtype], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Element-wise minimum with a scalar.

        Args:
            scalar: Scalar to compare against.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with min(self, scalar) for each element.
        """
        return MinScalar[Self.dtype].forward[track_grad](
            self, scalar, requires_grad
        )

    fn __add__[
        track_grad: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        """Element-wise addition of two tensors.

        Args:
            other: Tensor to add.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with element-wise sum.
        """
        return Adder[Self.dtype].forward[track_grad](self, other)

    fn __rsub__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Right-side subtraction (scalar - tensor).

        Args:
            scalar: Scalar value on the left.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with scalar minus each element.
        """
        return SubtractFromScalar[Self.dtype].forward[track_grad](self, scalar)

    fn __sub__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Subtract a scalar from all elements.

        Args:
            scalar: Scalar value to subtract.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with scalar subtracted from each element.
        """
        return SubtractScalar[Self.dtype].forward[track_grad](self, scalar)

    fn __sub__[
        track_grad: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        """Element-wise subtraction of two tensors.

        Args:
            other: Tensor to subtract.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with element-wise difference.
        """
        return Subtractor[Self.dtype].forward[track_grad](self, other)

    fn __rmul__[
        track_grad: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Right-side multiplication (scalar * tensor).

        Args:
            scalar: Scalar value on the left.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with each element multiplied by scalar.
        """
        return self.__mul__[track_grad](scalar)

    fn __mul__[
        track_grad: Bool = True
    ](self, factor: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Multiply all elements by a scalar.

        Args:
            factor: Scalar value to multiply by.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with each element multiplied by factor.
        """
        return MultiplyScalar[Self.dtype].forward[track_grad](self, factor)

    fn __mul__[
        track_grad: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        """Element-wise multiplication of two tensors.

        Args:
            other: Tensor to multiply by.
            track_grad: Whether to track gradients.

        Returns:
            Tensor with element-wise product.
        """
        return Multiplicator[Self.dtype].forward[track_grad](self, other)

    fn __mul__(self, other: Gradbox[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise multiplication with a Gradbox.

        Args:
            other: Gradbox to multiply by.

        Returns:
            Gradbox with element-wise product.
        """
        return Multiplicator[Self.dtype].forward(self, other)

    fn __pow__[
        track_grad: Bool = True
    ](
        self, exponent: Scalar[Self.dtype], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Element-wise power: self^exponent.

        Args:
            exponent: Scalar exponent value.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with each element raised to the exponent power.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __pow__ is for numeric data types only"

        return Exponentiator[Self.dtype].forward[track_grad](
            self, exponent, requires_grad
        )

    fn dot[track_grad: Bool = True](self, other: Self) -> Tensor[Self.dtype]:
        """Compute the dot product (matrix multiplication) with another tensor.

        Args:
            other: Tensor to multiply with.
            track_grad: Whether to track gradients.

        Returns:
            Result of matrix multiplication.
        """
        return Dot[Self.dtype].forward[track_grad](self, other)

    fn __iadd__(self, scalar: Scalar[Self.dtype]):
        """In-place addition of a scalar.

        Args:
            scalar: Scalar value to add.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __iadd__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.__iadd__(scalar)

    fn __isub__(self, scalar: Scalar[Self.dtype]):
        """In-place subtraction of a scalar.

        Args:
            scalar: Scalar value to subtract.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __isub__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.__isub__(scalar)

    fn __imul__(self, scalar: Scalar[Self.dtype]):
        """In-place multiplication by a scalar.

        Args:
            scalar: Scalar value to multiply by.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __imul__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.__imul__(scalar)

    fn __itruediv__(self, scalar: Scalar[Self.dtype]):
        """In-place division by a scalar.

        Args:
            scalar: Scalar value to divide by.

        Raises:
            Panic if called on a leaf tensor that requires gradients.
        """
        if self.is_leaf():
            panic(
                "Tensor → __itruediv__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.__itruediv__(scalar)

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        """Print the tensor's metadata and data preview.

        Args:
            num_first: Number of elements to print from the start.
            num_last: Number of elements to print from the end.
        """
        print(
            "\n",
            String(self),
            end="\n",
        )
        empty = List[Int]()
        print_buffer[Self.dtype](
            self.buffer,
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn __del__(deinit self):
        if self.gradbox:
            if (
                self.gradbox[]
                ._refcount[]
                .fetch_sub[ordering=Consistency.RELEASE](1)
                == 1
            ):
                fence[ordering=Consistency.ACQUIRE]()
                self.gradbox.destroy_pointee()
                self.gradbox.free()

    fn mse[
        track_grad: Bool = True
    ](self, target: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        """Mean squared error loss.

        Args:
            target: Target tensor to compare against.
            track_grad: Whether to track gradients.

        Returns:
            Scalar tensor with the MSE loss value.
        """
        var diff = Subtractor[Self.dtype].forward[track_grad](self, target)
        var squared = Multiplicator[Self.dtype].forward[track_grad](diff, diff)
        return squared.mean[track_grad]()

    fn backward[
        graph_size: Int = 50
    ](
        mut output: Tensor[Self.dtype], start_grad: Scalar[Self.dtype] = 1.0
    ) where Self.dtype.is_floating_point():
        """Run backward pass to compute gradients.

        Args:
            start_grad: Initial gradient value (default: 1.0).
            graph_size: Maximum graph size for traversal.
        """
        if not output.requires_grad:
            return
        var shape = output.shape()
        var seed_tensor = Tensor[Self.dtype].full(shape, start_grad)
        output.backward[graph_size](seed_tensor)

    fn requires_grad_(mut self, requires_grad: Bool = True):
        """Enable or disable gradient tracking in-place.

        Args:
            requires_grad: True to enable gradient tracking, False to disable.
        """
        self.requires_grad = requires_grad
        if requires_grad and not self.has_grad():
            self.init_gradbox()

    fn backward[
        graph_size: Int = 50,
    ](
        mut output: Tensor[Self.dtype],
        seed_tensor: Tensor[Self.dtype],
    ) where Self.dtype.is_floating_point():
        """Run backward pass with a specific seed tensor.

        Args:
            seed_tensor: Tensor containing initial gradients.
            graph_size: Maximum graph size for traversal.
        """
        if not output.requires_grad:
            return

        output.seed_grad(seed_tensor)

        try:
            var topo_ids = List[UInt](capacity=graph_size)
            var dfs_stack = List[UInt](capacity=graph_size)
            var node_list = List[Ancestor[Self.dtype]](capacity=graph_size)
            var visited = Set[UInt]()
            var fanin = Dict[UInt, Int]()
            var id_to_index = Dict[UInt, Int]()

            var root = Ancestor[Self.dtype].from_tensor(output)
            dfs_stack.append(root._id)
            node_list.append(root)
            id_to_index[root._id] = 0

            while len(dfs_stack) > 0:
                var node_id = dfs_stack.pop()

                if node_id in visited:
                    continue

                visited.add(node_id)
                topo_ids.append(node_id)

                var node_idx = id_to_index[node_id]
                ref node = node_list[node_idx]

                if node.has_ancestry():
                    for parent in node.ancestry():
                        var parent_id = parent._id
                        fanin[parent_id] = fanin.get(parent_id, 0) + 1

                        if parent_id not in id_to_index:
                            var new_idx = len(node_list)
                            node_list.append(parent)
                            id_to_index[parent_id] = new_idx
                            dfs_stack.append(parent_id)

            var ready_queue = Deque[UInt](capacity=graph_size)
            ready_queue.append(root._id)

            while len(ready_queue) > 0:
                var node_id: UInt = 0
                try:
                    node_id = ready_queue.popleft()
                except key_err:
                    panic(String(key_err))

                var node_idx = id_to_index[node_id]
                ref node = node_list[node_idx]

                if node.has_ancestry():
                    for result in Backward[Self.dtype].invoke(node):
                        ref target_ref = result[0]
                        ref grad = result[1]
                        var op_code = result[2]
                        var target_id = target_ref._id

                        if target_id in id_to_index:
                            var target_idx = id_to_index[target_id]
                            ref target = node_list[target_idx]
                            if target.requires_grad and target.gradbox:
                                if op_code == AddTensor:
                                    target.gradbox[] += grad
                                elif op_code == SubtractTensor:
                                    target.gradbox[] -= grad
                                else:
                                    target.gradbox[].zero_grad()

                        if target_id in fanin:
                            fanin[target_id] -= 1
                            if fanin[target_id] == 0:
                                var target_idx = id_to_index[target_id]
                                if node_list[target_idx].has_ancestry():
                                    ready_queue.append(target_id)

        except e:
            print(e)

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = ElemIterator[Self.dtype, iterable_origin]

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return {Pointer(to=self).get_immutable()}

    fn get(self, index: Int) -> Scalar[Self.dtype]:
        """Get element at a flat index with bounds checking.

        Args:
            index: Flat (linear) index into the tensor's memory.

        Returns:
            The scalar value at that index.

        Raises:
            Panic if index is out of bounds.
        """
        return self.buffer.get(index)

    fn set(self, index: Int, scalar: Scalar[Self.dtype]):
        """Set element at a flat index with bounds checking.

        Args:
            index: Flat (linear) index into the tensor's memory.

            Panic if index is out of bounds.
        """
        self.buffer.set(index, scalar)


    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a view with explicit shape, strides, and offset.

        Args:
            shape: Target shape.
            strides: Memory strides for the view.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.
            validated: If True, skips validation.

        Returns:
            A view tensor over the same buffer.
        """
        return View[Self.dtype].forward[track_grad](
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
    ) -> Tensor[Self.dtype]:
        """Create a view with explicit shape and strides as lists.

        Args:
            shape: Target shape as a list.
            strides: Memory strides as a list.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A view tensor over the same buffer.
        """
        view_shape, view_strides = Shape(shape), Strides(strides)
        return View[Self.dtype].forward[track_grad](
            self, view_shape, view_strides, offset, requires_grad, False
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        *shape_dims: Int,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Create a contiguous view with the given shape.

        Args:
            *shape_dims: Target shape as variadic ints.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A contiguous view tensor.
        """
        shape = Shape(shape_dims)
        strides = Strides.default(shape)
        return View[Self.dtype].forward[track_grad](
            self, shape, strides, offset, requires_grad, False
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Create a contiguous view with the given shape.

        Args:
            shape: Target shape as a list of ints.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A contiguous view tensor.
        """
        view_shape = Shape(shape)
        strides = Strides.default(view_shape)
        return View[Self.dtype].forward[track_grad](
            self, view_shape, strides, offset, requires_grad, False
        )

    fn view[
        track_grad: Bool = True
    ](
        mut self,
        shape: Shape,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Create a contiguous view with the given shape.

        Args:
            shape: Target shape.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A contiguous view tensor.
        """
        return View[Self.dtype].forward[track_grad](
            self, shape, Strides.default(shape), offset, requires_grad, False
        )

    fn into_view[
        track_grad: Bool = True
    ](mut self, requires_grad: Optional[Bool] = None) -> Tensor[Self.dtype]:
        """Convert this tensor into a shared view.

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A view tensor sharing the underlying buffer.
        """
        if self.shared():
            log_warning("Tensor → into_view: already shared")
            return self.copy()
        var shape, strides, offset = self.shape(), self.strides(), self.offset()
        grad_required = requires_grad.or_else(self.requires_grad)
        return View[Self.dtype].forward[track_grad](
            self, shape, strides, offset, grad_required, validated=True
        )

    fn transpose[
        track_grad: Bool = True
    ](mut self, *axes: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Transpose tensor by reversing or permuting axes.

        Args:
            *axes: Axes to permute. If empty, reverses all axes.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A transposed view tensor.
        """
        return self.transpose[track_grad](IntArray(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](
        mut self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Transpose tensor by reversing or permuting axes.

        Args:
            axes: Axes to permute as a list.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A transposed view tensor.
        """
        return self.transpose[track_grad](IntArray(axes), requires_grad)

    fn transpose[
        track_grad: Bool = True
    ](mut self, axes: IntArray, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Transpose tensor by reversing or permuting axes.

        Args:
            axes: Axes to permute as an IntArray.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A transposed view tensor.
        """
        return Transpose.forward[track_grad](self, axes, requires_grad)

    fn slice[
        track_grad: Bool = True
    ](mut self, start: Int, end: Int, step: Int = 1, axis: Int = 0) -> Tensor[
        Self.dtype
    ]:
        """Slice tensor along a single axis with start, end, and step.

        Args:
            start: Start index.
            end: End index.
            step: Step size (default: 1).
            axis: Axis to slice along (default: 0).

        Returns:
            A view tensor over the sliced region.
        """
        var shape, strides, offset = (
            Validator.validate_and_compute_slice_metadata(
                self.shape(), self.strides(), axis, start, end, step
            )
        )

        return View[Self.dtype].forward[track_grad](
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
    ) -> Tensor[Self.dtype]:
        """Slice tensor along multiple axes.

        Args:
            axes: Axes to slice along.
            starts: Start indices per axis.
            ends: End indices per axis.
            steps: Step sizes per axis (default: 1 for all).

        Returns:
            A view tensor over the sliced region.
        """
        step_sizes = IntArray(steps) if steps else IntArray.filled(len(axes), 1)
        if len(step_sizes) != len(axes):
            panic("Tensor → slice: length of steps must match axes length")

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

        return View[Self.dtype].forward[track_grad](
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
        mut self: Tensor[Self.dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Broadcast tensor to a target shape.

        Args:
            target: Shape to broadcast to.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor broadcasted to the target shape.
        """
        return Expand[Self.dtype].forward[track_grad](
            self, target, requires_grad
        )

    fn expand[
        track_grad: Bool = True
    ](
        mut self: Tensor[Self.dtype],
        *target_dims: Int,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Broadcast tensor to a target shape.

        Args:
            *target_dims: Target shape as variadic ints.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor broadcasted to the target shape.
        """
        return Expand[Self.dtype].forward[track_grad](
            self, Shape(target_dims), requires_grad
        )

    fn squeeze[
        track_grad: Bool = True
    ](
        mut self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Remove axes of size 1.

        Args:
            axes: Axes to squeeze. If empty, removes all size-1 axes.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with size-1 axes removed.
        """
        return Squeeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn squeeze[
        track_grad: Bool = True
    ](
        mut self,
        *axes: Int,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        Self.dtype
    ]:
        """Remove axes of size 1.

        Args:
            *axes: Axes to squeeze as variadic ints.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with size-1 axes removed.
        """
        return Squeeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](mut self, *axes: Int, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Insert axes of size 1.

        Args:
            *axes: Axes at which to insert size-1 dimensions.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with size-1 axes inserted.
        """
        return Unsqueeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](
        mut self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Insert axes of size 1.

        Args:
            axes: Axes at which to insert size-1 dimensions.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with size-1 axes inserted.
        """
        return Unsqueeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn unsqueeze[
        track_grad: Bool = True
    ](mut self, axes: IntArray, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Insert axes of size 1.

        Args:
            axes: Axes at which to insert size-1 dimensions.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with size-1 axes inserted.
        """
        return Unsqueeze[Self.dtype].forward[track_grad](
            self, axes, requires_grad
        )

    fn permute[
        track_grad: Bool = True
    ](
        mut self, axes: List[Int], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """Permute axes according to a given ordering.

        Args:
            axes: Permutation order (e.g. [2, 0, 1] for 3D).
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with axes permuted.
        """
        return Permute[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad
        )

    fn permute[
        track_grad: Bool = True
    ](mut self, axes: IntArray, requires_grad: Optional[Bool] = None) -> Tensor[
        Self.dtype
    ]:
        """Permute axes according to a given ordering.

        Args:
            axes: Permutation order as IntArray.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A tensor with axes permuted.
        """
        return Permute[Self.dtype].forward[track_grad](
            self, axes, requires_grad
        )

    fn argmax(
        self, axis: Int = 0, keepdims: Bool = False
    ) -> Tensor[DType.int32]:
        """Index of the maximum value along an axis.

        Args:
            axis: Axis along which to find the argmax.
            keepdims: If True, keep reduced axis with size 1.

        Returns:
            Tensor of indices indicating the position of the max value.
        """
        return Argmax[Self.dtype].argmax(
            tensor=self, axis=axis, keepdims=keepdims
        )

    fn argmin(
        self, axis: Int = 0, keepdims: Bool = False
    ) -> Tensor[DType.int32]:
        """Index of the minimum value along an axis.

        Args:
            axis: Axis along which to find the argmin.
            keepdims: If True, keep reduced axis with size 1.

        Returns:
            Tensor of indices indicating the position of the min value.
        """
        return Argmin[Self.dtype].argmin(
            tensor=self, axis=axis, keepdims=keepdims
        )

    fn max(
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Maximum value along given axes.

        Args:
            axes: Axes along which to find max.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with max values along the specified axes.
        """
        return MinMax[Self.dtype].forward[True](
            self, IntArray(axes), keepdims, requires_grad
        )

    fn max(
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Maximum value along given axes.

        Args:
            axes: Axes along which to find max.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with max values along the specified axes.
        """
        return MinMax[Self.dtype].forward[True](
            self, axes, keepdims, requires_grad
        )

    fn min(
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Minimum value along given axes.

        Args:
            axes: Axes along which to find min.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with min values along the specified axes.
        """
        return MinMax[Self.dtype].forward[False](
            self, IntArray(axes), keepdims, requires_grad
        )

    fn min(
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Minimum value along given axes.

        Args:
            axes: Axes along which to find min.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with min values along the specified axes.
        """
        return MinMax[Self.dtype].forward[False](
            self, axes, keepdims, requires_grad
        )

    fn shuffle[
        track_grad: Bool = True
    ](
        self,
        perm: List[Int] = [],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Shuffle tensor elements along an axis according to a permutation.

        Args:
            perm: Permutation indices for the given axis.
            axis: Axis along which to shuffle.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A shuffled tensor.
        """
        return Shuffle[Self.dtype].forward[track_grad](
            self, perm, axis, requires_grad
        )

    fn relu[
        track_grad: Bool = True
    ](self, requires_grad: Optional[Bool] = None,) -> Tensor[Self.dtype]:
        """Rectified Linear Unit: max(0, x).

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with ReLU applied element-wise.
        """
        return ReLU[Self.dtype].forward[track_grad](self, requires_grad)

    fn clip[
        track_grad: Bool = True
    ](
        self,
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Clip tensor values to a range [min_val, max_val].

        Args:
            min_val: Minimum value.
            max_val: Maximum value.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with values clipped to [min_val, max_val].
        """
        return Clip[Self.dtype].forward[track_grad](
            self, min_val, max_val, requires_grad
        )

    fn tanh[
        track_grad: Bool = True
    ](
        self,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Hyperbolic tangent activation.

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with tanh applied element-wise.
        """
        return Tanh[Self.dtype].forward[track_grad](self, requires_grad)

    fn sigmoid[
        track_grad: Bool = True
    ](
        self,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Sigmoid activation: 1 / (1 + exp(-x)).

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with sigmoid applied element-wise.
        """
        return Sigmoid[Self.dtype].forward[track_grad](self, requires_grad)

    fn softmax[
        track_grad: Bool = True, log: Bool = False
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Softmax activation along given axes.

        Args:
            axes: Axes along which to apply softmax.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with softmax probabilities. Use log=True for log-softmax.
        """
        comptime if log:
            return LogSoftmax[Self.dtype].forward[track_grad](
                self, IntArray(axes), requires_grad
            )
        else:
            return Softmax[Self.dtype].forward[track_grad](
                self, IntArray(axes), requires_grad
            )

    fn softmax[
        track_grad: Bool = True, log: Bool = False
    ](
        self,
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Softmax activation along given axes.

        Args:
            axes: Axes along which to apply softmax.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Tensor with softmax probabilities. Use log=True for log-softmax.
        """
        comptime if log:
            return LogSoftmax[Self.dtype].forward[track_grad](
                self, axes, requires_grad
            )
        else:
            return Softmax[Self.dtype].forward[track_grad](
                self, axes, requires_grad
            )

    fn binary_cross_entropy[
        track_grad: Bool = True
    ](
        pred: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Binary cross entropy loss.

        Args:
            pred: Predicted probabilities.
            target: Ground truth labels (0 or 1).
            epsilon: Small value for numerical stability.

        Returns:
            Scalar tensor with the BCE loss.
        """
        return BCELoss[Self.dtype].forward[track_grad](pred, target, epsilon)

    fn binary_cross_entropy_with_logits[
        track_grad: Bool = True
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """BCE loss with logits (sigmoid applied internally).

        Args:
            logits: Raw unnormalized predictions.
            target: Ground truth labels (0 or 1).
            epsilon: Small value for numerical stability.

        Returns:
            Scalar tensor with the BCE with logits loss.
        """
        return BCEWithLogitsLoss[Self.dtype].forward[track_grad](
            logits, target, epsilon
        )

    fn sum_over_broadcasted_axes(
        batch_tensor: Tensor[Self.dtype], target_shape: Shape
    ) -> Tensor[Self.dtype]:
        """Sum broadcasted tensor over axes matching target shape.

        Args:
            batch_tensor: Tensor to sum.
            target_shape: Shape to broadcast to before summing.

        Returns:
            Tensor summed over broadcasted axes.
        """
        var nd_buffer = batch_tensor.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Tensor[Self.dtype](nd_buffer^, requires_grad=False)

    fn matmul[
        track_grad: Bool = True, mode: Int = mnemonics.mm
    ](A: Tensor[Self.dtype], B: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        """Matrix multiplication of two tensors.

        Args:
            A: Left-hand tensor.
            B: Right-hand tensor.
            track_grad: Whether to track gradients.
            mode: Matrix multiplication mode (default: mm).

        Returns:
            Result of matrix multiplication.
        """
        return Matmul[Self.dtype].forward[track_grad=track_grad, mode=mode](
            A, B
        )

    fn matmul(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        """Matrix multiplication with a Gradbox.

        Args:
            A: Left-hand tensor.
            B: Right-hand Gradbox.

        Returns:
            Result of matrix multiplication as Gradbox.
        """
        return Matmul[Self.dtype].forward(A, B)

    @staticmethod
    fn concat[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Concatenate tensors along an axis.

        Args:
            tensors: List of tensors to concatenate.
            axis: Axis along which to concatenate.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A single concatenated tensor.
        """
        return Concate[Self.dtype].forward[track_grad](
            tensors, axis, requires_grad
        )

    @staticmethod
    fn stack[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Stack tensors along a new axis.

        Args:
            tensors: List of tensors to stack.
            axis: Axis at which to insert the new dimension.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A stacked tensor with shape extended by one dimension.
        """
        return Stack[Self.dtype].forward[track_grad](
            tensors, axis, requires_grad
        )

    @staticmethod
    fn vstack[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Vertical stack (concatenate along axis 0, then flatten 2D).

        Args:
            tensors: List of 1D tensors to stack vertically.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Stacked tensor.
        """
        return Stack[Self.dtype].vstack[track_grad](tensors, requires_grad)

    @staticmethod
    fn hstack[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Horizontal stack (concatenate along last axis).

        Args:
            tensors: List of 1D tensors to stack horizontally.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Stacked tensor.
        """
        return Stack[Self.dtype].hstack[track_grad](tensors, requires_grad)

    @staticmethod
    fn pad[
        track_grad: Bool = True
    ](
        x: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        mode: String = "constant",
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Pad tensor along each axis.

        Args:
            x: Input tensor to pad.
            pad: List of (before, after) padding pairs per axis.
            mode: Padding mode ("constant", etc.).
            value: Fill value for constant mode.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Padded tensor.
        """
        return Pad[Self.dtype].forward[track_grad](
            x, pad, mode, value, requires_grad
        )

    @staticmethod
    fn pad_constant[
        track_grad: Bool = True
    ](
        x: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Pad tensor with a constant value.

        Args:
            x: Input tensor to pad.
            pad: List of (before, after) padding pairs per axis.
            value: Fill value (default: 0.0).
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Padded tensor.
        """
        return Pad[Self.dtype].forward[track_grad](
            x, pad, "constant", value, requires_grad
        )

    @staticmethod
    fn pad2d[
        track_grad: Bool = True
    ](
        x: Tensor[Self.dtype],
        pad_left: Int,
        pad_right: Int,
        pad_top: Int,
        pad_bottom: Int,
        mode: String = "constant",
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """2D padding for image tensors (last two axes).

        Args:
            x: Input tensor.
            pad_left: Padding on the left.
            pad_right: Padding on the right.
            pad_top: Padding on the top.
            pad_bottom: Padding on the bottom.
            mode: Padding mode.
            value: Fill value for constant mode.
            requires_grad: If provided, overrides requires_grad.

        Returns:
            2D-padded tensor.
        """
        var pad = List[Tuple[Int, Int]]()
        pad.append((pad_top, pad_bottom))
        pad.append((pad_left, pad_right))
        return Pad[Self.dtype].forward[track_grad](
            x, pad, mode, value, requires_grad
        )

    @staticmethod
    fn pad_for_conv[
        track_grad: Bool = True
    ](
        x: Tensor[Self.dtype],
        pad: Int,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Same-padding for 4D tensors (N, C, H, W), common in CNNs.

        Args:
            x: Input 4D tensor.
            pad: Padding amount for spatial dimensions (H and W).
            requires_grad: If provided, overrides requires_grad.

        Returns:
            Padded 4D tensor.

        Raises:
            Panic if tensor is not 4D.
        """
        var x_shape = x.shape()
        if x_shape.rank() != 4:
            panic("pad_for_conv: expected 4D tensor")

        var pad_spec = List[Tuple[Int, Int]]()
        pad_spec.append((0, 0))  # No padding on batch
        pad_spec.append((0, 0))  # No padding on channels
        pad_spec.append((pad, pad))  # Pad height
        pad_spec.append((pad, pad))  # Pad width

        return Pad[Self.dtype].forward[track_grad](
            x, pad_spec, "constant", 0.0, requires_grad
        )


@fieldwise_init
struct ElemIterator[dtype: DType, origin: ImmutOrigin](
    RegisterPassable & ImplicitlyCopyable & Iterable & Iterator & Sized
):
    """Iterator over (coordinate, value) pairs in a tensor.

    Yields tuples of IntArray coordinates and their corresponding scalar values.
    """

    comptime Element = Tuple[IntArray, Scalar[Self.dtype]]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var src: Pointer[Tensor[Self.dtype], Self.origin]
    var index_itr: ShapeIndexIterator[ImmutAnyOrigin]

    fn __init__(out self, src: Pointer[Tensor[Self.dtype], Self.origin]):
        """Initialize iterator over a tensor.

        Args:
            src: Pointer to the tensor to iterate over.
        """
        self.src = src
        self.index_itr = rebind[ShapeIndexIterator[ImmutAnyOrigin]](
            src[].shape().__iter__()
        )

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    fn __next__(mut self) raises StopIteration -> Self.Element:
        """Return next (coordinate, value) pair.

        Returns:
            Tuple of (IntArray coordinates, scalar value at that position).

        Raises:
            StopIteration: When all elements have been visited.
        """
        next = self.index_itr.__next__()
        return next, self.src[][next]

    fn __len__(self) -> Int:
        """Number of elements remaining.

        Returns:
            Number of elements yet to be visited.
        """
        return self.index_itr.__len__()

    fn __has_next__(self) -> Bool:
        """Check if there are more elements to visit.

        Returns:
            True if more elements remain, False otherwise.
        """
        return self.index_itr.__has_next__()

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        """Get the bounds of the iterator.

        Returns:
            Tuple of (remaining length, Optional of the same length).
        """
        return self.index_itr.bounds()
