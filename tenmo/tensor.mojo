from std.math import exp, log, sqrt
from std.sys import simd_width_of
from std.utils.numerics import min_finite
from std.memory import unsafe_memcpy
from .shared.shapes import Shape, ShapeIndexIterator
from .ancestry import Ancestors, Ancestor
from .shared.strides import Strides

from .shared.mnemonics import (
    AddTensor,
    SubtractTensor,
    ZeroGrad,
    ScatterAddTensor,
    MulTensor,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    Add,
    Subtract,
    Multiply,
    Divide,
    INVERT,
    DEFAULT_INDEX_DTYPE,
)
from .shared.constants import Epsilon, One, CloseTol
from .shared.idgen import IDGen
from .shared.indexhelper import Idx, i, s
from .shared.logging import log_warning
from .shared.panic import panic
from .shared.timing import now
from .shared.indexhelper import IndexIterator
from .backpropagation import Backward, BackwardFn
from .cast import ToDtypeBackward

from .forwards import *
from .shared.buffers import Buffer
from .validators import Validator
from std.collections import Set, Deque
from .gradbox import Gradbox
from .shared.intarray import IntArray
from .shared.broadcasthelper import ShapeBroadcaster
from .ndbuffer import NDBuffer, print_buffer, NDBufferLite
from max.gpu.host import DeviceBuffer, DeviceContext
from .gpu.device import Device, CPU, GPU
from .shared import Reduction
from .sum_mean_reduction import SumMeanReduction
from std.sys.info import has_accelerator

from .shared import mnemonics


struct Tensor[dtype: DType](
    ImplicitlyCopyable & Sized & Writable & Absable & Equatable & Iterable
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
    var gradbox: Optional[Gradbox[Self.dtype]]
    var ancestors: Optional[Ancestors[Self.dtype]]

    def __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        var shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    def __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    def __init__(out self, shape: Shape, requires_grad: Bool = False):
        self._id = IDGen.generate_id()
        self.buffer = NDBuffer[Self.dtype](shape)
        self.requires_grad = requires_grad
        self.gradbox = {}
        self.ancestors = None
        self.init_gradbox()

    def __init__(out self):
        self._id = IDGen.generate_id()
        self.buffer = NDBuffer[Self.dtype].Empty()
        self.requires_grad = False
        self.gradbox = {}
        self.ancestors = None

    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
        requires_grad: Bool = False,
        *,
        copy: Bool = True,
    ):
        self._id = IDGen.generate_id()
        var num_elements = shape.num_elements()
        var ptr_cast = ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        var buffer = Buffer[Self.dtype](
            num_elements, ptr_cast.unsafe_offset(offset), copy=True
        ) if copy else Buffer[Self.dtype](num_elements, ptr_cast, copy=False)
        var offset_adjusted = 0 if copy else offset
        self.buffer = NDBuffer[Self.dtype](
            buffer^, shape, strides, offset_adjusted
        )
        self.requires_grad = requires_grad
        self.gradbox = {}
        self.ancestors = None
        self.init_gradbox()

    def __init__(
        out self,
        var buffer: NDBuffer[Self.dtype],
        requires_grad: Bool = False,
    ):
        self._id = IDGen.generate_id()
        self.buffer = buffer^
        self.requires_grad = requires_grad
        self.gradbox = {}
        self.ancestors = None
        self.init_gradbox()

    def as_gradbox(
        deinit self, *, contiguous: Bool = True
    ) -> Gradbox[Self.dtype]:
        """Convert tensor to a Gradbox for gradient operations.

        Args:
            contiguous: If True, materializes a contiguous copy first.

        Returns:
            A Gradbox containing the tensor's data.
        """
        if contiguous:
            return Gradbox[Self.dtype](self^.buffer.contiguous())
        else:
            return Gradbox[Self.dtype](self^.buffer^)

    def tolist(self) raises -> List[Scalar[Self.dtype]]:
        """Copy tensor data into a Mojo List.

        Returns:
            List containing all tensor elements in row-major order.
        """
        return self.buffer.tolist()

    def __init__(out self, *, deinit move: Self):
        self._id = move._id
        self.buffer = move.buffer^
        self.requires_grad = move.requires_grad
        self.gradbox = move.gradbox
        self.ancestors = move.ancestors^

    def __init__(out self, *, copy: Self):
        """Copy-init — ALIASES the source's storage by design (PyTorch parity).

        ``var b = a`` shares storage with ``a``: the underlying buffer is
        refcounted (shared-from-birth), the ``Gradbox``/``Ancestors`` handles
        are shared, and ``_id`` is inherited. Mutations through one alias are
        visible through the other. To opt out of sharing, use ``clone()``.
        """
        self._id = copy._id
        self.buffer = copy.buffer.copy()
        self.requires_grad = copy.requires_grad
        self.gradbox = copy.gradbox
        self.ancestors = copy.ancestors.copy()

    def clone(self, requires_grad: Optional[Bool] = None) -> Self:
        """Clone this tensor into an independent, deep copy (opt-out of sharing).

        Unlike the copy-init / assignment (``var b = a``), which aliases the
        source's storage, ``clone`` materialises fresh storage — the result
        shares no memory with the source, so mutations through one never affect
        the other. Layout (shape, strides, offset) is preserved. The clone is a
        fresh leaf: new ``_id``, no ancestry, and (if the source tracks
        gradients) a fresh zero gradbox. On GPU the clone is materialised as an
        independent device buffer.

        ``requires_grad`` is preserved on both CPU and GPU (NDBuffer.clone
        handles the GPU round-trip internally).
        """
        var out = Tensor[Self.dtype](
            self.buffer.clone(), requires_grad=requires_grad.or_else(self.requires_grad)
        )
        return out^

    @always_inline
    def id(self) -> UInt:
        """Get the unique identifier for this tensor.

        Returns:
            Unique UInt ID assigned at tensor creation.
        """
        return self._id

    def init_gradbox(mut self):
        """Initialize gradient storage if requires_grad is True.

        Allocates GPU memory for gradients if the tensor is on GPU.
        """
        if self.requires_grad and not self.gradbox:
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
                        gradbox = Gradbox[Self.dtype](ndb^)
                    except e:
                        print(e)
                        panic(
                            "init_gradbox: failed to allocate GPU gradbox: "
                            + String(e)
                        )
                        gradbox = Gradbox[Self.dtype](
                            Shape()
                        )  # unreachable, satisfies compiler
                else:
                    gradbox = Gradbox[Self.dtype].zeros(self.shape())
            else:
                gradbox = Gradbox[Self.dtype].zeros(self.shape())
            self.gradbox = gradbox^

    @always_inline
    def is_contiguous(self) -> Bool:
        """Check if tensor memory layout is contiguous.

        Returns:
            True if the tensor is stored contiguously in row-major order.
        """
        return self.buffer.is_contiguous()

    @always_inline
    def is_shared(self) -> Bool:
        """Check if the underlying buffer is shared by multiple views.

        Returns:
            True if the buffer is reference-counted and shared.
        """
        return self.buffer.is_shared()

    def is_leaf(self) -> Bool:
        """Check if this tensor is a leaf in the autograd graph.

        A leaf tensor is one that requires gradients but has no ancestors
        (operations that produced it). Leaf tensors are the starting points
        of gradient computation.

        Returns:
            True if this is a leaf tensor, False otherwise.
        """
        return self.requires_grad and self.ancestors is None

    @always_inline
    def __len__(self) -> Int:
        """Get the total number of elements in the tensor.

        Returns:
            The total number of elements (product of shape dimensions).
            Returns 1 for scalar tensors with Shape().
        """
        return self.shape().product()

    @always_inline
    def shape(ref self) -> ref[self.buffer.shape] Shape:
        """Get the shape of this tensor.

        Returns:
            Reference to the tensor's shape.
        """
        return self.buffer.shape

    @always_inline
    def strides(ref self) -> ref[self.buffer.strides] Strides:
        """Get the strides of this tensor.

        Returns:
            Reference to the tensor's strides.
        """
        return self.buffer.strides

    @always_inline
    def offset(self) -> Int:
        """Get the base memory offset of this tensor's view.

        Returns:
            The offset into the underlying buffer.
        """
        return self.buffer.offset

    @always_inline
    def numels(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer.numels()

    @always_inline
    def num_elements(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer.numels()

    @always_inline
    def rank(self) -> Int:
        """Get the number of dimensions.

        Returns:
            The number of axes in the tensor's shape.
        """
        return self.buffer.rank()

    @always_inline
    def max_index(self) -> Int:
        """Highest valid LOGICAL flat index (C-order over this view).

        Always `numels() - 1`. Use with `get`/`set`; for the highest
        touched buffer address use `max_storage_index` with
        `storage_get`/`storage_set`.
        """
        return self.numels() - 1

    @always_inline
    def max_storage_index(self) -> Int:
        """Highest touched buffer address (storage space).

        Use with `storage_get`/`storage_set`; for logical indexing use
        `max_index` with `get`/`set`.
        """
        return self.buffer.max_storage_index()

    def detach(self) -> Tensor[Self.dtype]:
        """Create a new tensor sharing the same data but detached from the
        computation graph.

        The returned tensor:
        - shares the same underlying buffer (no copy)
        - has requires_grad=False
        - has no ancestors — gradient stops here
        - has no gradbox

        Use when you want a value in the forward pass but explicitly do not
        want gradients to flow back through that path. Common cases:
        - stopping gradient through attention weights
        - using a tensor as a constant in a computation
        - breaking cycles in the computation graph
        - inference on a subset of a larger graph

        Returns:
            A new Tensor with the same buffer, shape, strides and offset
            but with requires_grad=False and no ancestry.
        """
        var out = Tensor[Self.dtype](
            self.buffer.share(
                shape=self.shape(),
                strides=self.strides(),
                offset=self.offset(),
            ),
            requires_grad=False,
        )
        return out^

    @always_inline
    def index_iterator(
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
    def __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        """Index tensor with a List of integers.

        Args:
            indices: List of axis indices.

        Returns:
            Scalar value at the specified coordinates.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Tensor → __getitem__(List[Int]): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]

    @always_inline
    def __getitem__(ref self, indices: IntArray) -> Scalar[Self.dtype]:
        """Index tensor with an IntArray of indices.

        Args:
            indices: IntArray of axis indices.

        Returns:
            Scalar value at the specified coordinates.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Tensor → __getitem__(IntArray): Scalar tensor expects no"
                " indices"
            )
        return self.buffer[indices]

    @always_inline
    def __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Index tensor with variadic integer indices.

        Args:
            indices: One index per axis.

        Returns:
            Scalar value at the specified coordinates.
        """
        if self.rank() == 0:
            panic(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(List[Int])"
            )

        return self.buffer[indices]

    def __getitem__[
        track_grad: Bool = True,
    ](self, *slices: Slice, sync: Bool = True) -> Tensor[Self.dtype]:
        """Slice tensor using Slice objects along each axis.

        Args:
            slices: One Slice per axis, specifying start, stop, step.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A view tensor with computed shape, strides, and offset.

        Example:
            ```mojo
            var t = Tensor[DType.float32].zeros(10, 10)
            var view = t[1:5, 2:8]
            ```
        """
        return View[Self.dtype].forward[track_grad=track_grad](
            self, *slices, requires_grad=self.requires_grad, sync=sync
        )

    def chunk(
        self, *indices: Idx, requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Extract independent chunks along the first axis by indices.

        Allocates a new buffer — gradients do not flow back to the source.

        Args:
            indices: Indices along the first axis to extract.
            requires_grad: Whether the result tracks gradients.

        Returns:
            A new tensor containing the selected chunks.
        """
        var ndb = self.buffer.chunk(*indices)
        return Tensor[Self.dtype](ndb^, requires_grad=requires_grad)

    def __getitem__[
        track_grad: Bool = True,
    ](self, *indices: Idx, sync: Bool = True) -> Tensor[Self.dtype]:
        """Advanced indexing with Idx objects (integers or slices).

        Args:
            indices: Idx per axis — either an integer or a Slice.
                Missing axes default to full slices.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A view tensor over the indexed region.
        """
        return View[Self.dtype].forward[track_grad=track_grad](
            self, *indices, requires_grad=self.requires_grad, sync=sync
        )

    def gather[
        track_grad: Bool = True,
    ](
        self,
        indices: List[Int],
        axis: Int = 0,
        reduction: Reduction = Reduction(2),
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return self.gather[track_grad=track_grad](
            indices=IntArray(indices),
            axis=axis,
            reduction=reduction,
            requires_grad=requires_grad,
            sync=sync,
        )

    def gather[
        track_grad: Bool = True,
    ](
        self,
        indices: IntArray,
        axis: Int = 0,
        reduction: Reduction = Reduction(2),
        padding_idx: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Gather slices along `axis` at the given indices.

        Always copies data into a fresh contiguous output tensor.
        Output has no grad connection to input — requires_grad is always False.

        Args:
            indices: Indices to gather along `axis`. May contain negative values,
                     which are normalized to axis_dim + index.
            axis:    Axis to gather along. May be negative. Defaults to 0.
            reduction: How to reduce gathered rows (NONE/SUM/MEAN).
            padding_idx: Index to use for padding out-of-bounds indices.
            requires_grad: Whether the result tracks gradients.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A new contiguous tensor with shape identical to self except
            axis dimension is replaced by len(indices).
        """

        return Gather[Self.dtype].forward[track_grad=track_grad](
            self,
            indices,
            axis,
            reduction,
            padding_idx,
            requires_grad,
            sync=sync,
        )

    def outer[
        track_grad: Bool = True,
    ](mut self, mut other: Tensor[Self.dtype], sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        """Compute the outer product of two tensors.

        Both tensors are flattened to 1-D before the product is computed.
        Result shape is (self.numels(), other.numels()).

        Gradient flows back through both inputs if either requires grad —
        implemented entirely via reshape + unsqueeze + multiply so no
        custom backward is needed.

        Args:
            other: Second tensor. Will be flattened to 1-D.
            sync: Whether to synchronize the GPU operation.

        Returns:
            2-D tensor of shape (self.numels(), other.numels()).
        """
        # Flatten both to 1-D — reshape tracks grad if input does
        var a = self.reshape[track_grad](
            Shape(self.numels()), sync=sync
        )  # (m,)
        var b = other.reshape[track_grad](
            Shape(other.numels()), sync=sync
        )  # (n,)

        # Unsqueeze a → column vector (m, 1)
        var col_axes = IntArray()
        col_axes.append(1)
        var a_col = a.unsqueeze[track_grad](col_axes, sync=sync)  # (m, 1)

        # Broadcast multiply (m, 1) * (n,) → (m, n)
        # This is a broadcast multiply — BroadcastBackward handles grad
        return Multiplicator[Self.dtype].forward[track_grad](
            a_col, b, sync=sync
        )  # (m, n)

    @always_inline
    def __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: One index per axis.
            value: Scalar value to write.
        """
        if self.rank() == 0:
            panic(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(List[Int])"
            )
        self.buffer[indices] = value

    @always_inline
    def __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: List of axis indices.
            value: Scalar value to write.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Tensor → __setitem__(List[Int]): Scalar tensor expects no"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    def __setitem__(self, coord: IntArray, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            coord: IntArray of coordinates.
            value: Scalar value to write.
        """
        if self.rank() == 0 and len(coord) != 0:
            panic(
                "Tensor → __setitem__(IntArray): Scalar tensor expects no"
                " indices"
            )

        self.buffer[coord] = value

    def fill(self, value: Scalar[Self.dtype], *indices: Idx):
        """Fill a region with a scalar value using advanced indexing.

        Args:
            value: Scalar value to write.
            indices: Idx objects (integers or slices) defining the region.
        """
        Filler[Self.dtype].fill(self.buffer, value, indices)

    def fill(self, tensor: Tensor[Self.dtype], *indices: Idx):
        """Copy data from another tensor into a region using advanced indexing.

        Args:
            tensor: Source tensor to copy from.
            indices: Idx objects (integers or slices) defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer, tensor.buffer, indices)

    def fill(self, gradbox: Gradbox[Self.dtype], *indices: Idx):
        """Copy data from a Gradbox into a region using advanced indexing.

        Args:
            gradbox: Source Gradbox to copy from.
            indices: Idx objects (integers or slices) defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer, gradbox.buffer(), indices)

    def fill(self, value: Scalar[Self.dtype], indices: List[Idx]):
        """List[Idx]-based scalar fill — non-variadic twin of fill(value, *idx).
        """
        Filler[Self.dtype].fill_list(self.buffer, value, indices)

    def fill(self, tensor: Tensor[Self.dtype], indices: List[Idx]):
        """List[Idx]-based buffer fill — non-variadic twin of fill(tensor, *idx).
        """
        Filler[Self.dtype].fill_list(self.buffer, tensor.buffer, indices)

    def item(self) -> Scalar[Self.dtype]:
        return self.buffer.item()

    @no_inline
    def __str__(self) -> String:
        var rank = self.rank()
        var s = String("[")
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
        if self.buffer.is_shared():
            s += ", strides: " + String(self.strides())
            s += ", offset: " + String(self.offset())
        s += ", Type: " + String(Self.dtype)
        s += ", requires_grad: " + String(self.requires_grad)
        s += ", outstanding: " + String(self.ref_count())
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
    def __repr__(self) -> String:
        return self.__str__()

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @no_inline
    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write(
            "Tensor(shape=",
            self.shape(),
            ", dtype=",
            Self.dtype,
            ", requires_grad=",
            self.requires_grad,
            ", device=",
            "gpu" if self.is_on_gpu() else "cpu",
            ")",
        )

    def write_to(self, buffer: DeviceBuffer[Self.dtype]) raises:
        with buffer.map_to_host() as host_buffer:
            if self.is_contiguous():
                unsafe_memcpy(
                    dest=host_buffer.unsafe_ptr(),
                    src=self.data_ptr().unsafe_offset(self.offset()),
                    count=self.numels(),
                )
            else:
                var ptr = host_buffer.unsafe_ptr()
                ref data_buffer = self.buffer.data_buffer()
                var offset = 0
                for index in self.index_iterator():
                    ptr.unsafe_offset(offset)[] = data_buffer[index]
                    offset += 1

    def to_gpu(
        self,
        gpu: Optional[GPU] = None,
        requires_grad: Optional[Bool] = None,
        stop_grad: Bool = False,
        sync: Bool = True,
    ) raises -> Tensor[Self.dtype]:
        """Transfer tensor to GPU.

        Args:
            gpu: Target GPU. Uses default GPU if None.
            requires_grad: If provided, overrides the requires_grad flag
                on the returned tensor.
            stop_grad: If True, gradient stops at this GPU tensor —
                       no DeviceTransferBackward registered. Use for
                       permanent GPU residents (model weights).
                       Default False preserves existing grad flow behaviour.
            sync: If True, synchronize GPU after transfer.
        """
        comptime if has_accelerator():
            var target_gpu = gpu.or_else(GPU())
            return DeviceTransfer[Self.dtype].forward(
                self,
                target_gpu.into(),
                stop_grad=stop_grad,
                sync=sync,
            )
        else:
            raise Error(
                "Can not move to GPU. System does not have any accelerator"
                " device"
            )

    def to_cpu(
        self,
        requires_grad: Optional[Bool] = None,
        stop_grad: Bool = False,
        sync: Bool = True,
    ) raises -> Tensor[Self.dtype]:
        """Transfer tensor to CPU.

        Args:
            requires_grad: If provided, overrides the requires_grad flag
                on the returned tensor.
            stop_grad: If True, gradient stops at this CPU tensor —
                       no DeviceTransferBackward registered.
                       Default False preserves existing grad flow behaviour.
            sync: If True, synchronize GPU before transfer.
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
                sync=sync,
            )

        raise Error("System does not have any accelerator")

    def device(self) -> Device:
        """Get the device this tensor is on.

        Returns:
            The CPU or GPU device the tensor resides on.
        """
        return self.buffer.device()

    def device_context(self) -> Optional[DeviceContext]:
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
    def has_grad(self) -> Bool:
        """Check if gradient storage has been initialized.

        Returns:
            True if a gradient buffer exists for this tensor, False otherwise.
        """
        return self.gradbox != None

    @always_inline
    def zero_grad(mut self):
        """Zero out accumulated gradients.

        Call this before backward() to reset gradients to zero.
        Only affects tensors that require and have gradients.
        """
        if self.requires_grad and self.has_grad():
            self.gradbox.value().zero_grad()

    @always_inline
    def gradients(ref self) -> ref[self.gradbox.value()] Gradbox[Self.dtype]:
        """Get reference to the gradient buffer.

        Returns:
            Reference to the Gradbox storing accumulated gradients.
        """
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → gradients(self): called on a tensor that does not"
                " require grad or grad not initialized"
            )
        return self.gradbox.value()

    @always_inline
    def grad(self) -> Gradbox[Self.dtype]:
        """Get accumulated gradients as a detached Gradbox.

        Returns:
            The Gradbox containing accumulated gradients.
        """
        if not self.requires_grad or not self.has_grad():
            panic(
                "Tensor → grad(self): called on a tensor that does not require"
                " grad or grad not initialized"
            )
        return self.gradbox.value().detach()

    def rows(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → rows: tensor rank is not 2")
        return self.shape()[0]

    def cols(self) -> Int:
        if not self.rank() == 2:
            panic("Tensor → cols: tensor rank is not 2")
        return self.shape()[1]

    def is_scalar(self) -> Bool:
        """Check if this tensor is a scalar (zero-dimensional).

        Returns:
            True if the tensor has a single element, False otherwise.
        """
        return self.buffer.is_scalar()

    def is_on_gpu(self) -> Bool:
        """Check if this tensor is stored on a GPU.

        Returns:
            True if the tensor is on a GPU, False otherwise.
        """
        return self.buffer.is_on_gpu()

    def is_on_cpu(self) -> Bool:
        """Check if this tensor is stored on the CPU.

        Returns:
            True if the tensor is on the CPU, False otherwise.
        """
        return self.is_on_gpu() == False

    def __eq__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[Equal](scalar))

    def __ne__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[NotEqual](scalar))

    def __lt__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[LessThan](scalar))

    def __le__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[LessThanEqual](scalar)
        )

    def gt(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThan](scalar)
        )

    def lt(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare_scalar[LessThan](scalar))

    def __gt__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThan](scalar)
        )

    def __ge__(self, scalar: Scalar[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare_scalar[GreaterThanEqual](scalar)
        )

    def __eq__(self, other: Tensor[Self.dtype]) -> Bool:
        return self.buffer == other.buffer

    def eq(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[Equal](other.buffer))

    def __ne__(self, other: Tensor[Self.dtype]) -> Bool:
        return self.buffer != other.buffer

    def ne(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[NotEqual](other.buffer))

    def __lt__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](self.buffer.compare[LessThan](other.buffer))

    def __le__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[LessThanEqual](other.buffer)
        )

    def __gt__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[GreaterThan](other.buffer)
        )

    def __ge__(self, other: Tensor[Self.dtype]) -> Tensor[DType.bool]:
        return Tensor[DType.bool](
            self.buffer.compare[GreaterThanEqual](other.buffer)
        )

    def float(self) -> Tensor[DType.float32]:
        """Convert tensor to float32 dtype.

        Returns:
            A new tensor with dtype float32.
        """
        return self.to_dtype[DType.float32]()

    def float64(self) -> Tensor[DType.float64]:
        """Convert tensor to float64 dtype.

        Returns:
            A new tensor with dtype float64.
        """
        return self.to_dtype[DType.float64]()

    def to_dtype[
        NewType: DType
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[NewType]:
        """Convert tensor to a different data type.

        When the dtype actually changes and gradient tracking is on for a
        floating-point conversion, the result registers `ToDtypeBackward`
        (tenmo/cast.mojo) so gradients flow back across the cast. Same-dtype
        calls and non-float targets return a leaf as before.

        Args:
            requires_grad: If provided, overrides the requires_grad flag
                on the returned tensor.

        Returns:
            A new tensor with the specified dtype.
        """
        var new_type_buffer = self.buffer.to_dtype[NewType]()
        # Gradient tracking is only meaningful for float→float conversions;
        # int targets always come back as leaves.
        var trackable = (
            Self.dtype.is_floating_point() and NewType.is_floating_point()
        )
        var grad_required = (
            requires_grad.or_else(self.requires_grad) and trackable
        )
        var out = Tensor[NewType](new_type_buffer^, requires_grad=grad_required)

        comptime if NewType != Self.dtype:
            if grad_required:
                # No requires_grad_(True): the constructor above already
                # set the flag and initialized the gradbox.
                var backwardFn = BackwardFn.null_arg[NewType](
                    ToDtypeBackward[Self.dtype, NewType]()
                )
                out.add_ancestry_erased[Self.dtype](backwardFn^, self)

        return out^

    def add_ancestry_erased[
        pdtype: DType
    ](mut self, var backwardFn: BackwardFn, parent: Tensor[pdtype],):
        """[A]add_ancestry for the one edge where parent dtype ≠ self dtype.

        The cast op and Seq's seams use this: storage is dtype-erased, so
        an f32 parent can live under an f64 output node. Single parent by
        design (a cast consumes exactly one input) and no ndb snapshot —
        dtype backward never needs forward values. See tenmo/cast.mojo.
        """
        if not self.ancestors:
            self.ancestors = Optional(Ancestors[Self.dtype](backwardFn^))
        else:
            self.ancestors.value().set_backward_fn(backwardFn^)

        ref ancestors = self.ancestors.value()
        var ancestor = parent.to_ancestor()
        ancestors.append_erased[pdtype](ancestor^)

    def to_ancestor(ref self) -> Ancestor[Self.dtype]:
        var out = Ancestor[Self.dtype]()
        out._id = self._id
        out.requires_grad = self.requires_grad
        if self.ancestors:
            out.parents = self.ancestors
        if self.gradbox:
            out.gradbox = self.gradbox
        return out^

    def add_ancestry(
        mut self,
        var backwardFn: BackwardFn,
        *parents: Tensor[Self.dtype],
    ):
        var needs_data = backwardFn.needs_parent_data

        if not self.ancestors:
            self.ancestors = Optional(Ancestors[Self.dtype](backwardFn^))
        else:
            self.ancestors.value().set_backward_fn(backwardFn^)

        ref ancestors = self.ancestors.value()

        for parent in parents:
            var ancestor = parent.to_ancestor()
            if needs_data:
                var nd_buffer: NDBuffer[Self.dtype]
                if parent.buffer.is_shared():
                    # Owned buffers are shared-from-birth — alias via copy-init
                    # (refcount bump keeps the forward storage alive in the graph;
                    # no full deep copy, and nothing about the user's tensor changes).
                    nd_buffer = parent.buffer.copy()
                else:
                    # External-backed parent: external memory can't be refcounted,
                    # so clone its data into owned storage for the graph.
                    nd_buffer = parent.buffer.clone()
                ancestor.ndb = NDBufferLite[Self.dtype](nd_buffer^)
            ancestors.append(ancestor^)

    def has_ancestry(self) -> Bool:
        """Check if this tensor has registered parent dependencies.

        Returns:
            True if parent tensors have been registered, False otherwise.
        """
        return self.ancestors != None

    @always_inline
    def ancestry(
        ref self,
    ) -> ref[self.ancestors.value()] Ancestors[Self.dtype]:
        """Get the ancestry graph for backward pass traversal.

        Returns:
            Reference to the Ancestors containing parent dependencies.
        """
        if self.ancestors == None:
            panic("Tensor → ancestry: ancestry not initialized")
        return self.ancestors.value()

    @always_inline
    def broadcastable(self, to: Tensor[Self.dtype]) -> Bool:
        """Check if this tensor can broadcast to a target shape.

        Args:
            to: Target tensor to check broadcasting compatibility with.

        Returns:
            True if this tensor's shape can broadcast to to.shape().
        """
        return ShapeBroadcaster.broadcastable(self.shape(), to.shape())

    def log[
        track_grad: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        self,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        return Logarithm[Self.dtype].forward[track_grad, epsilon](
            self, requires_grad, sync
        )

    def all_close[
        rtol: Scalar[Self.dtype] = CloseTol[Self.dtype].rtol(),
        atol: Scalar[Self.dtype] = CloseTol[Self.dtype].atol(),
    ](self, other: Self,) -> Bool:
        """Check if two tensors are element-wise close within tolerance.

        Args:
            other: Tensor to compare against.

        Returns:
            True if all elements are within tolerance, False otherwise.
        """
        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    def all(self, pred: def(Scalar[Self.dtype]) thin -> Bool) -> Bool:
        """Returns True if pred holds for all elements.
        Uses NDBuffer.map_to_bool — handles GPU via CPU materialisation.
        """
        return self.buffer.map_to_bool(pred).all_true()

    def any(self, pred: def(Scalar[Self.dtype]) thin -> Bool) -> Bool:
        """Returns True if pred holds for any element.
        Uses NDBuffer.map_to_bool — handles GPU via CPU materialisation.
        """
        return self.buffer.map_to_bool(pred).any_true()

    def all_true(self) -> Bool where Self.dtype == DType.bool:
        """Returns True if all elements are True.
        GPU path: NDBuffer.all_true → DeviceState.all_true (maps to host).
        CPU path: NDBuffer.all_true → Buffer.all_true.
        """

        return self.buffer.all_true()

    def any_true(self) -> Bool where Self.dtype == DType.bool:
        """Returns True if any element is True.
        GPU path: NDBuffer.any_true → DeviceState.any_true (maps to host).
        CPU path: NDBuffer.any_true → Buffer.any_true.
        """
        return self.buffer.any_true()

    def unsafe_ptr(ref self) -> Pointer[Tensor[Self.dtype], MutAnyOrigin]:
        return Pointer(to=self).as_unsafe_any_origin().unsafe_mut_cast[True]()

    def ref_count(self) -> Int:
        return self.buffer.ref_count()

    def data_ptr(ref self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.buffer.data_ptr()

    def seed_grad(mut self, with_tensor: Tensor[Self.dtype]):
        """Seed gradient accumulation with a specific tensor.

        Args:
            with_tensor: Tensor containing gradient values to seed with.
        """
        if not self.requires_grad:
            return
        if not self.has_grad():
            self.requires_grad_()
        self.gradbox.value().seed_grad(with_tensor)

    def seed_grad(mut self, value: Scalar[Self.dtype]):
        """Seed gradient accumulation with a scalar value.

        Args:
            value: Scalar gradient value to seed with.
        """
        if not self.requires_grad:
            return
        if not self.has_grad():
            self.requires_grad_()
        self.gradbox.value().seed_grad(value)

    @always_inline
    def fill(self, value: Scalar[Self.dtype]):
        """Fill the entire tensor with a scalar value.

        Args:
            value: Scalar value to write to all elements.
        """
        self.buffer.fill(value)

    def map_where[
        track_grad: Bool = True,
    ](
        self,
        pred: def(Scalar[Self.dtype]) thin -> Bool,
        value: Scalar[Self.dtype],
    ) raises -> Tensor[Self.dtype]:
        """Replace elements matching a predicate with a scalar value.

        Builds a boolean mask via map_to_bool, then delegates to masked_fill
        for fused autograd and GPU support.

        Args:
            pred: Function that returns True for elements to replace.
            value: Scalar value to write where pred returns True.

        Returns:
            A new tensor with replaced values.
        """
        var mask_ndb = self.buffer.map_to_bool(pred)
        var mask = Tensor[DType.bool](mask_ndb^)
        return self.masked_fill[track_grad=track_grad](mask, value)

    @staticmethod
    def full_like(
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
    def full(
        shape: List[Int],
        value: Scalar[Self.dtype],
        requires_grad: Bool = False,
        device: Optional[Device] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a tensor filled with a scalar value.

        Args:
            shape: Dimensions as a list of ints.
            value: Scalar value to fill with.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.
            sync: If True, synchronize GPU after creation.

        Returns:
            A tensor of given shape filled with value.
        """
        return Self.full(
            Shape(shape), value, requires_grad, device=device, sync=sync
        )

    @staticmethod
    def full(
        shape: Shape,
        scalar: Scalar[Self.dtype],
        requires_grad: Bool = False,
        device: Optional[Device] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a tensor filled with a scalar value.

        Args:
            shape: Tensor shape.
            scalar: Scalar value to fill with.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.
            sync: If True, synchronize GPU after creation.

        Returns:
            A tensor of given shape filled with scalar.
        """
        var target_device = device.or_else(CPU().into())
        return Tensor[Self.dtype](
            NDBuffer[Self.dtype].full(shape, scalar, target_device, sync=sync),
            requires_grad=requires_grad,
        )

    @staticmethod
    def rand(
        *dims: Int,
        low: Scalar[Self.dtype] = 0,
        high: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        return Self.rand(
            Shape(dims),
            min=low,
            max=high,
            init_seed=init_seed,
            requires_grad=requires_grad,
            device=device,
        )

    @staticmethod
    def rand(
        shape: List[Int],
        low: Scalar[Self.dtype] = 0,
        high: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        return Self.rand(
            Shape(shape),
            min=low,
            max=high,
            init_seed=init_seed,
            requires_grad=requires_grad,
            device=device,
        )

    @staticmethod
    def rand(
        shape: Shape,
        min: Scalar[Self.dtype] = 0,
        max: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        var target_device = device.or_else(CPU().into())
        var ndb = NDBuffer[Self.dtype].rand(
            shape,
            min=min,
            max=max,
            init_seed=init_seed,
            device=target_device,
        )
        return Tensor[Self.dtype](ndb^, requires_grad=requires_grad)

    @staticmethod
    def randn(
        *dims: Int,
        mean: Float64 = 0.0,
        std: Float64 = 1.0,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        return Self.randn(
            Shape(dims),
            mean=mean,
            std=std,
            init_seed=init_seed,
            requires_grad=requires_grad,
            device=device,
        )

    @staticmethod
    def randn(
        shape: Shape,
        mean: Float64 = 0.0,
        std: Float64 = 1.0,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        var target_device = device.or_else(CPU().into())
        var ndb = NDBuffer[Self.dtype].randn(
            shape,
            mean=mean,
            std=std,
            init_seed=init_seed,
            device=target_device,
        )
        return Tensor[Self.dtype](ndb^, requires_grad=requires_grad)

    @staticmethod
    def arange(
        *args: Scalar[Self.dtype],
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Create a 1D tensor with evenly spaced values.

        Args:
            args: Start, stop, and optionally step values.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with values from start to stop.

        Example:
            ```mojo
            Tensor[DType.float32].arange(0, 5)      # [0, 1, 2, 3, 4]
            Tensor[DType.float32].arange(0, 10, 2)  # [0, 2, 4, 6, 8]
            ```
        """
        var nd_buffer = NDBuffer[Self.dtype].arange(args)
        var tensor = Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)
        return tensor^

    @staticmethod
    def linspace(
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
        var nd_buffer = NDBuffer[Self.dtype].linspace(start, end, steps)
        var tensor = Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)
        return tensor^

    @staticmethod
    def zeros(
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
    def eye(
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
    def empty(
        *axes_spans: Int,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        var shape = Shape(axes_spans)
        return Self.zeros(shape^, requires_grad, device)

    @staticmethod
    def zeros(
        *axes_spans: Int,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of zeros.

        Args:
            axes_spans: Shape dimensions as variadic ints.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with zeros.
        """
        var shape = Shape(axes_spans)
        return Self.zeros(shape^, requires_grad=requires_grad, device=device)

    @staticmethod
    def zeros_like(
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a zeros tensor matching another tensor's shape.

        Args:
            self: Reference tensor whose shape to match.
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
    def empty(
        shape: Shape,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        return Self.zeros(shape, requires_grad, device)

    @staticmethod
    def zeros(
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
    def ones_like(
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a ones tensor matching another tensor's shape.

        Args:
            self: Reference tensor whose shape to match.
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
    def onehot(
        indices: Tensor[Self.dtype],
        num_classes: Int,
        device: Optional[Device] = None,
        ignore_index: Optional[Int] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Convert tensor of class indices to one-hot encoding.
        Args:
            indices: Tensor of shape (...,) containing class indices.
            num_classes: Number of classes.
            device: Target device.
            ignore_index: If provided, rows where index == ignore_index become all zeros.
            sync: If True, synchronize GPU after kernel launch.
        Returns: Tensor of shape (..., num_classes).
        """
        var onehot_ndb = NDBuffer[Self.dtype].onehot(
            indices.buffer, num_classes, device, ignore_index, sync=sync
        )
        return Tensor[Self.dtype](onehot_ndb^, requires_grad=False)

    @staticmethod
    def d1(
        var *row: Scalar[Self.dtype], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 1D tensor from a variadic list of scalar values.

        Args:
            row: Scalar values forming the single dimension.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with len(row) elements.
        """
        var row_list = List[Scalar[Self.dtype]]()
        for v in row:
            row_list.append(v)
        return Tensor[Self.dtype].d1(row_list^)

    @staticmethod
    def d1(row: Self.Row, requires_grad: Bool = False) -> Tensor[Self.dtype]:
        """Create a 1D tensor from a list of scalar values.

        Args:
            row: List of scalar values forming the single dimension.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with len(row) elements.
        """
        if len(row) == 0:
            return Tensor[Self.dtype].scalar(
                min_finite[Self.dtype](), requires_grad=requires_grad
            )
        var numels = len(row)
        var shape = Shape(IntArray(numels))
        var buffer = Buffer[Self.dtype](numels)
        unsafe_memcpy(
            dest=buffer.data.value(), src=row.unsafe_ptr(), count=numels
        )
        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    def from_list[
        src_dtype: DType
    ](
        values: List[Scalar[src_dtype]],
        requires_grad: Bool = False,
    ) -> Tensor[
        Self.dtype
    ]:
        """Create a 1D tensor from a list of scalar values.

        Args:
            values: List of scalar values.
            requires_grad: Whether to track gradients.

        Returns:
            A 1D tensor with ``len(values)`` elements.
        """
        if len(values) == 0:
            return Tensor[Self.dtype].scalar(
                min_finite[Self.dtype](), requires_grad=requires_grad
            )
        var numels = len(values)
        var shape = Shape(IntArray(numels))
        var buffer = Buffer[Self.dtype](numels)
        var data = buffer.data.unsafe_value()
        for i in range(numels):
            data[unsafe_offset=i] = Scalar[Self.dtype](values[i])
        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    def d2(
        rows: List[Self.Row], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 2D tensor from a list of rows.

        Args:
            rows: List of rows, each row must have equal length.
            requires_grad: Whether to track gradients.

        Returns:
            A 2D tensor of shape (len(rows), len(rows[0])).
        """
        var dims = IntArray(len(rows), len(rows[0]))
        var flattened = List[Scalar[Self.dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                panic("Tensor → d2 → not all rows equal in length")
            flattened.extend(row.copy())
        var shape = Shape(dims)
        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)
        unsafe_memcpy(
            dest=buffer.data.unsafe_value(),
            src=flattened.unsafe_ptr(),
            count=numels,
        )
        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad)

    @staticmethod
    def d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 3D tensor from a list of 2D blocks.

        Args:
            blocks: List of 2D matrices, each matrix must have equal dimensions.
            requires_grad: Whether to track gradients.

        Returns:
            A 3D tensor.
        """
        var dims = IntArray(len(blocks), len(blocks[0]), len(blocks[0][0]))
        var flattened = List[Scalar[Self.dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                panic("Tensor → d3 → not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    panic("Tensor → d3 → not all rows equal in length")

                flattened.extend(row.copy())
        var shape = Shape(dims)
        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)
        unsafe_memcpy(
            dest=buffer.data.unsafe_value(),
            src=flattened.unsafe_ptr(),
            count=numels,
        )
        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    def d4(
        blockgrid: List[Self.Block], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 4D tensor from a nested list structure.

        Args:
            blockgrid: List of 3D blocks, each block must have equal dimensions.
            requires_grad: Whether to track gradients.

        Returns:
            A 4D tensor.
        """
        var dims = IntArray(
            len(blockgrid),
            len(blockgrid[0]),
            len(blockgrid[0][0]),
            len(blockgrid[0][0][0]),
        )
        var flattened = List[Scalar[Self.dtype]](capacity=dims.product())
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
        var shape = Shape(dims)
        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)
        unsafe_memcpy(
            dest=buffer.data.unsafe_value(),
            src=flattened.unsafe_ptr(),
            count=numels,
        )
        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    def d5(
        blockhive: List[Self.Blocks], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a 5D tensor from a deeply nested list structure.

        Args:
            blockhive: List of 4D blocks, each block must have equal dimensions.
            requires_grad: Whether to track gradients.

        Returns:
            A 5D tensor.
        """
        var dims = IntArray(
            len(blockhive),
            len(blockhive[0]),
            len(blockhive[0][0]),
            len(blockhive[0][0][0]),
            len(blockhive[0][0][0][0]),
        )
        var flattened = List[Scalar[Self.dtype]](capacity=dims.product())
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
        var shape = Shape(dims)
        var numels = shape.num_elements()
        var buffer = Buffer[Self.dtype](numels)
        unsafe_memcpy(
            dest=buffer.data.unsafe_value(),
            src=flattened.unsafe_ptr(),
            count=numels,
        )
        var nd_buffer = NDBuffer[Self.dtype](buffer^, shape)
        return Tensor[Self.dtype](nd_buffer^, requires_grad=requires_grad)

    @staticmethod
    def scalar(
        val: Scalar[Self.dtype], requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Create a scalar (0D) tensor from a single value.

        Args:
            val: The scalar value.
            requires_grad: Whether to track gradients.

        Returns:
            A scalar tensor containing val.
        """
        var result = Tensor[Self.dtype](Shape(), requires_grad=requires_grad)
        result[IntArray()] = val
        return result^

    @staticmethod
    def ones(
        *axes_spans: Int,
        requires_grad: Bool = False,
        device: Optional[Device] = None,
    ) -> Tensor[Self.dtype]:
        """Create a tensor of ones.

        Args:
            axes_spans: Shape dimensions as variadic ints.
            requires_grad: Whether to track gradients.
            device: Target device. Defaults to CPU.

        Returns:
            A tensor of given shape filled with ones.
        """
        return Self.ones(Shape(axes_spans), requires_grad, device)

    @staticmethod
    def ones(
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

    def broadcast_to[
        track_grad: Bool = True,
    ](
        self,
        target_shape: Shape,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Broadcast tensor to a target shape.

        Args:
            target_shape: Shape to broadcast to.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with the target shape.
        """
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Tensor → broadcast_to: shape "
                + String(self.shape())
                + " not broadcastable to "
                + String(target_shape)
            )

        return Broadcast[Self.dtype].forward[track_grad](
            self, target_shape, sync=sync
        )

    @always_inline
    def load[
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
    def store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
        """SIMD store of a row segment into a 2D Tensor.

        Preconditions:
            - Tensor must be 2D.
            - Columns must be contiguous for SIMD stores (stride[1] == 1).
            - Caller may set validated=True if these checks are already ensured.
        """
        self.buffer.store[simdwidth, validated](row, col, value)

    def flatten[
        track_grad: Bool = True,
    ](
        self,
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Flatten the tensor by collapsing a range of dimensions.

        Args:
            start_dim: First dimension to flatten (default: 0).
            end_dim: Last dimension to flatten (default: last).
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with flattened dimensions.
        """
        return FlattenForward[Self.dtype].forward[track_grad](
            self, start_dim, end_dim, requires_grad, sync=sync
        )

    def repeat[
        track_grad: Bool = True,
    ](
        self,
        repeat: List[Int],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Repeat tensor along each axis.

        Args:
            repeat: Number of repeats per dimension.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with repeated data.
        """
        return Repeat[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad, sync=sync
        )

    def repeat[
        track_grad: Bool = True,
    ](
        self,
        *repeat: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Repeat tensor along each axis.

        Args:
            repeat: Number of repeats per dimension as variadic ints.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with repeated data.
        """
        return Repeat[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad, sync=sync
        )

    def tile[
        track_grad: Bool = True,
    ](
        self,
        repeat: List[Int],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Tile the tensor by repeating it along each axis.

        Args:
            repeat: Number of tiles per dimension.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tiled tensor.
        """
        return Tile[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad, sync=sync
        )

    def tile[
        track_grad: Bool = True,
    ](
        self,
        *repeat: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Tile the tensor by repeating it along each axis.

        Args:
            repeat: Number of tiles per dimension as variadic ints.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tiled tensor.
        """
        return Tile[Self.dtype].forward[track_grad](
            self, IntArray(repeat), requires_grad, sync=sync
        )

    def contiguous[
        track_grad: Bool = True,
    ](self, requires_grad: Optional[Bool] = None, owned: Bool = True, sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        """Return a contiguous copy of the tensor.

        Unlike torch, this ALWAYS materializes an owned copy — even when the
        source is already contiguous. This is deliberate: callers rely on the
        result being isolated (gradbox segmentation, data_buffer() access with
        zero offset). View ops (transpose/reshape/expand) are the way to share.

        Args:
            requires_grad: If provided, overrides requires_grad.
            owned: If False, allow returning an alias when the source is
                already contiguous+shared with unchanged shape (fast path,
                no copy). Default True preserves the isolation guarantee.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A contiguous tensor with the same data.
        """
        return Contiguous[Self.dtype].forward[track_grad](
            self, requires_grad, owned=owned, sync=sync
        )

    def reshape[
        track_grad: Bool = True,
    ](self, requires_grad: Optional[Bool] = None, sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        if self.numels() != 1:
            panic(
                "Tensor → reshape: only tensor with single element can be"
                " reshaped to scalar tensor"
            )
        return self.reshape[track_grad](
            Shape(), requires_grad=requires_grad, validated=True, sync=sync
        )

    def reshape[
        track_grad: Bool = True,
    ](
        self,
        *newdims: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        if len(newdims) == 1 and newdims[0] == 0:
            return self.reshape[track_grad](
                requires_grad=requires_grad, sync=sync
            )
        var shape = Validator.validate_and_construct_new_shape(
            self.shape(), IntArray(newdims)
        )
        return self.reshape[track_grad](
            shape, requires_grad=requires_grad, validated=True, sync=sync
        )

    def reshape[
        track_grad: Bool = True,
    ](
        self,
        shape: List[Int],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var new_shape = Validator.validate_and_construct_new_shape(
            self.shape(), IntArray(shape)
        )
        return self.reshape[track_grad](
            new_shape, requires_grad=requires_grad, validated=True, sync=sync
        )

    def reshape[
        track_grad: Bool = True,
    ](
        self,
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Reshape[Self.dtype].forward[track_grad](
            self, new_shape, requires_grad, validated, sync=sync
        )

    def sum[
        track_grad: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Sum tensor elements along given axes.

        Args:
            axes: Axes along which to sum.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with summed values along the specified axes.
        """
        return self.sum[track_grad](
            IntArray(axes), keepdims, requires_grad, sync
        )

    def sum[
        track_grad: Bool = True,
    ](
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Sum tensor elements along given axes.

        Args:
            axes: Axes along which to sum.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with summed values along the specified axes.
        """

        return Summer[Self.dtype].forward[track_grad](
            self, axes, keepdims, requires_grad, sync
        )

    def product[
        track_grad: Bool = True,
        store_excl_product: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Compute product along given axes.

        Args:
            axes: Axes along which to compute product.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with product values along the specified axes.
        """
        return self.product[track_grad, store_excl_product](
            IntArray(axes), keepdims, requires_grad, sync
        )

    def product[
        track_grad: Bool = True,
        store_excl_product: Bool = True,
    ](
        self,
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Product tensor elements along given axes.

        Args:
            axes: Axes along which to compute product.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with product values along the specified axes.
        """
        return Product[Self.dtype].forward[track_grad, store_excl_product](
            self, axes, keepdims, requires_grad, sync
        )

    def sqrt[
        track_grad: Bool = True,
    ](
        self,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Element-wise square root.

        Args:
            epsilon: Stability floor used by the backward pass (forward
                computes the plain sqrt).
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with square root of each element.
        """
        return Sqrt[Self.dtype].forward[track_grad](
            self, epsilon, requires_grad, sync
        )

    def mean[
        track_grad: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Compute mean along given axes.

        Args:
            axes: Axes along which to compute mean.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with mean values along the specified axes.
        """
        return self.mean[track_grad](
            IntArray(axes), keepdims, requires_grad, sync
        )

    def mean[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Compute mean along given axes.

        Args:
            axes: Axes along which to compute mean.
            keepdims: If True, keep reduced axes with size 1.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with mean values along the specified axes.
        """
        return Mean[Self.dtype].forward[track_grad](
            self, axes, keepdims, requires_grad, sync
        )

    def reciprocal[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Compute reciprocal.

        Args:
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with reciprocal values of the elements.
        """
        return Reciprocal[Self.dtype].forward[track_grad](
            self, requires_grad, sync
        )

    def variance[
        track_grad: Bool = True,
    ](
        self,
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Compute variance along an axis.

        Args:
            axis: Axis along which to compute variance.
            keepdims: If True, keep reduced axis with size 1.
            unbiased: If True, use n-1 (sample variance). If False, use n (population).
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with variance along the specified axis.
        """
        return Variance[Self.dtype].forward[track_grad](
            self, axis, keepdims, unbiased, requires_grad, sync
        )

    def std[
        track_grad: Bool = True,
    ](
        self,
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Compute standard deviation along an axis.

        Args:
            axis: Axis along which to compute std.
            keepdims: If True, keep reduced axis with size 1.
            unbiased: If True, use n-1. If False, use n.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with std deviation along the specified axis.
        """
        return StdDev[Self.dtype].forward[track_grad](
            self, axis, keepdims, unbiased, requires_grad, sync
        )

    def norm(
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
        """
        if p == 2.0:
            var squared = self.__mul__[track_grad=False](self)
            var dim = [axis.value()] if axis else List[Int]()
            var sum_sq = squared.sum[track_grad=False](dim, keepdims=keepdims)
            return sum_sq.sqrt[track_grad=False]()
        else:
            panic("Only L2 norm (p=2) currently supported")
            return Tensor[Self.dtype].scalar(0)

    def __rtruediv__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        return DivideFromScalar[Self.dtype].forward[track_grad](
            self, scalar, sync=sync
        )

    def __truediv__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        return DivideByScalar[Self.dtype].forward[track_grad](
            self, scalar, sync=sync
        )

    def __truediv__[
        track_grad: Bool = True, sync: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        return Divider[Self.dtype].forward[track_grad](self, other, sync=sync)

    def __iadd__[sync: Bool = True](self, other: Self):
        """In-place addition of another tensor.

        Args:
            other: Tensor to add.
        """
        if self.is_leaf():
            panic(
                "Tensor → __iadd__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )
        self.buffer.inplace_ops[Add](other.buffer, sync=sync)

    def __isub__[sync: Bool = True](self, other: Self):
        """In-place subtraction of another tensor.

        Args:
            other: Tensor to subtract.
        """
        if self.is_leaf():
            panic(
                "Tensor → __isub__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.inplace_ops[Subtract](other.buffer, sync=sync)

    def __isub__[sync: Bool = True](self, other: Gradbox[Self.dtype]):
        """In-place subtraction of a Gradbox.

        Args:
            other: Gradbox to subtract.
        """
        self.buffer.inplace_ops[Subtract](other.buffer(), sync=sync)

    def __imul__[sync: Bool = True](self, other: Self):
        """In-place multiplication by another tensor.

        Args:
            other: Tensor to multiply by.
        """
        if self.is_leaf():
            panic(
                "Tensor → __imul__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.inplace_ops[Multiply](other.buffer, sync=sync)

    def __itruediv__[sync: Bool = True](self, other: Self):
        """In-place division by another tensor.

        Args:
            other: Tensor to divide by.
        """
        if self.is_leaf():
            panic(
                "Tensor → __itruediv__(self, other): can not perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        self.buffer.inplace_ops[Divide](other.buffer, sync=sync)

    def unique(self) -> Tensor[Self.dtype]:
        """Return a tensor with duplicate elements removed.

        Returns:
            A tensor with the same data but duplicates removed.
        """
        return Tensor[Self.dtype](self.buffer.unique(), requires_grad=False)

    def count(self, key: Scalar[Self.dtype]) -> Int:
        """Count occurrences of a value in the tensor.

        Args:
            key: Scalar value to count.

        Returns:
            Number of elements equal to key.
        """
        return self.buffer.count(key)

    def sum_all(self) -> Scalar[Self.dtype]:
        """Sum all elements into a single scalar - CPU only op.

        Returns:
            Sum of all elements in the tensor.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → sum_all is for numeric data types only"
        return SumMeanReduction[Self.dtype].sum_all(self.buffer)

    def product_all(self) -> Scalar[Self.dtype]:
        """Compute the product of all elements - CPU only op.

        Returns:
            Product of all elements in the tensor.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → product_all is for numeric data types only"

        return Product.product_all(self.buffer)

    def exp[
        track_grad: Bool = True,
    ](self, requires_grad: Optional[Bool] = None, sync: Bool = True) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Element-wise exponential (e^x).

        Args:
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with exponential of each element.
        """
        return Exponential[Self.dtype].forward[track_grad=track_grad](
            self, requires_grad=requires_grad, sync=sync
        )

    def __neg__[
        track_grad: Bool = True, sync: Bool = True
    ](self) -> Tensor[Self.dtype]:
        """Negate all elements.

        Returns:
            Tensor with negated values.
        """
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __neg__ is for numeric data types only"

        return Negate[Self.dtype].forward[track_grad](self, sync=sync)

    def __invert__[
        sync: Bool = True
    ](
        self: Tensor[Self.dtype],
    ) -> Tensor[Self.dtype] where (
        Self.dtype == DType.bool or Self.dtype.is_integral()
    ):
        """Bitwise invert (logical NOT for bool, bitwise NOT for integers).

        Returns:
            Tensor with inverted values.
        """
        return Tensor[Self.dtype](
            self.buffer.unary_ops[INVERT](sync=sync), requires_grad=False
        )

    def __abs__(self) -> Tensor[Self.dtype]:
        return Absolute[Self.dtype].forward[track_grad=False](self, sync=True)

    def abs[
        track_grad: Bool = True,
    ](
        self,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[
        Self.dtype
    ]:
        return Absolute[Self.dtype].forward[track_grad=track_grad](
            self, requires_grad=requires_grad, sync=sync
        )

    def cumsum[
        track_grad: Bool = True,
    ](
        self,
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Cumsum[Self.dtype].forward[track_grad=track_grad](
            self, axis=axis, requires_grad=requires_grad, sync=sync
        )

    def tril[
        track_grad: Bool = True,
    ](
        self,
        diagonal: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Tril[Self.dtype].forward[track_grad=track_grad](
            self, diagonal=diagonal, requires_grad=requires_grad, sync=sync
        )

    def triu[
        track_grad: Bool = True,
    ](
        self,
        diagonal: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Triu[Self.dtype].forward[track_grad=track_grad](
            self, diagonal=diagonal, requires_grad=requires_grad, sync=sync
        )

    @staticmethod
    def where[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Tensor[Self.dtype],
        b: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Where[Self.dtype].forward[track_grad](
            condition, a, b, requires_grad=requires_grad, sync=sync
        )

    @staticmethod
    def where[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Scalar[Self.dtype],
        b: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Where[Self.dtype].forward[track_grad](
            condition, a, b, requires_grad=requires_grad, sync=sync
        )

    @staticmethod
    def where[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Tensor[Self.dtype],
        b: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Where[Self.dtype].forward[track_grad](
            condition, a, b, requires_grad=requires_grad, sync=sync
        )

    @staticmethod
    def where[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Scalar[Self.dtype],
        b: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return Where[Self.dtype].forward[track_grad](
            condition, a, b, requires_grad=requires_grad, sync=sync
        )

    def multinomial[
        index_dtype: DType = DEFAULT_INDEX_DTYPE,
    ](
        self,
        num_samples: Int = 1,
        replacement: Bool = False,
        temperature: Scalar[Self.dtype] = 1.0,
        init_seed: Optional[Int] = None,
    ) raises -> Tensor[index_dtype] where Self.dtype.is_floating_point():
        return Multinomial[Self.dtype, index_dtype].sample(
            self, num_samples, replacement, temperature, init_seed
        )

    def masked_fill[
        track_grad: Bool = True,
    ](
        self,
        mask: Tensor[DType.bool],
        value: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) raises -> Tensor[Self.dtype]:
        """Fill elements where mask is True with value.
        Equivalent to Tensor.where(mask, value, self).
        Args:
            mask: Boolean tensor of the same shape.
            value: Scalar fill value.
            requires_grad: Whether the result tracks gradients.
            sync: Whether to sync GPU operations.
        Returns:
            A new tensor with masked elements replaced.
        """
        return Where[Self.dtype].forward[track_grad](
            mask, value, self, requires_grad=requires_grad, sync=sync
        )

    def __radd__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Right-side addition (scalar + tensor).

        Args:
            scalar: Scalar value on the left.

        Returns:
            Tensor with scalar added to all elements.
        """
        return self.__add__[track_grad, sync=sync](scalar)

    def __add__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Add a scalar to all elements.

        Args:
            scalar: Scalar value to add.

        Returns:
            Tensor with scalar added to all elements.
        """
        return AddScalar[Self.dtype].forward[track_grad](
            self, scalar, sync=sync
        )

    def max[
        track_grad: Bool = True, sync: Bool = True
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
            self, scalar, requires_grad, sync
        )

    def min[
        track_grad: Bool = True, sync: Bool = True
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
            self, scalar, requires_grad, sync
        )

    def __add__[
        track_grad: Bool = True, sync: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        """Element-wise addition of two tensors.

        Args:
            other: Tensor to add.

        Returns:
            Tensor with element-wise sum.
        """
        return Adder[Self.dtype].forward[track_grad](self, other, sync=sync)

    def __rsub__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Right-side subtraction (scalar - tensor).

        Args:
            scalar: Scalar value on the left.

        Returns:
            Tensor with scalar minus each element.
        """
        return SubtractFromScalar[Self.dtype].forward[track_grad](
            self, scalar, sync=sync
        )

    def __sub__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Subtract a scalar from all elements.

        Args:
            scalar: Scalar value to subtract.

        Returns:
            Tensor with scalar subtracted from each element.
        """
        return SubtractScalar[Self.dtype].forward[track_grad](
            self, scalar, sync=sync
        )

    def __sub__[
        track_grad: Bool = True, sync: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        """Element-wise subtraction of two tensors.

        Args:
            other: Tensor to subtract.

        Returns:
            Tensor with element-wise difference.
        """
        return Subtractor[Self.dtype].forward[track_grad](
            self, other, sync=sync
        )

    def __sub__[
        sync: Bool = True
    ](self, other: Gradbox[Self.dtype]) -> Tensor[Self.dtype]:
        """Element-wise subtraction of two tensors.

        Args:
            other: Gradbox to subtract.

        Returns:
            Tensor with element-wise difference.
        """
        return Subtractor[Self.dtype].forward(self, other, sync=sync)

    def __rmul__[
        track_grad: Bool = True, sync: Bool = True
    ](self, scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Right-side multiplication (scalar * tensor).

        Args:
            scalar: Scalar value on the left.

        Returns:
            Tensor with each element multiplied by scalar.
        """
        return self.__mul__[track_grad, sync=sync](scalar)

    def __mul__[
        track_grad: Bool = True, sync: Bool = True
    ](self, factor: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        """Multiply all elements by a scalar.

        Args:
            factor: Scalar value to multiply by.

        Returns:
            Tensor with each element multiplied by factor.
        """
        return MultiplyScalar[Self.dtype].forward[track_grad](
            self, factor, sync=sync
        )

    def __mul__[
        track_grad: Bool = True, sync: Bool = True
    ](self, other: Self) -> Tensor[Self.dtype]:
        """Element-wise multiplication of two tensors.

        Args:
            other: Tensor to multiply by.

        Returns:
            Tensor with element-wise product.
        """
        return Multiplicator[Self.dtype].forward[track_grad](
            self, other, sync=sync
        )

    def __mul__[
        sync: Bool = True
    ](self, other: Gradbox[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise multiplication with a Gradbox.

        Args:
            other: Gradbox to multiply by.

        Returns:
            Gradbox with element-wise product.
        """
        return Multiplicator[Self.dtype].forward(self, other, sync=sync)

    def __pow__[
        track_grad: Bool = True, sync: Bool = True
    ](
        self,
        exponent: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
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
            self, exponent, requires_grad, sync=sync
        )

    def dot[
        track_grad: Bool = True
    ](self, other: Self, sync: Bool = True) -> Tensor[Self.dtype]:
        """Dot product between two tensors. Either may be a scalar tensor."""
        return Dot[Self.dtype].forward[track_grad](self, other, sync)

    def dot[
        track_grad: Bool = True,
    ](self, scalar: Scalar[Self.dtype], sync: Bool = True) raises -> Tensor[
        Self.dtype
    ]:
        """Dot product: tensor · scalar (scalar is broadcast to self's shape).
        """
        var other = Tensor[Self.dtype].scalar(scalar, requires_grad=False)
        comptime if has_accelerator():
            if self.is_on_gpu():
                var other_gpu = other.to_gpu()
                return Dot[Self.dtype].forward[track_grad](
                    self, other_gpu, sync
                )
        return Dot[Self.dtype].forward[track_grad](self, other, sync)

    def __iadd__[sync: Bool = True](self, scalar: Scalar[Self.dtype]):
        """In-place addition of a scalar.

        Args:
            scalar: Scalar value to add.
        """
        if self.is_leaf():
            panic(
                "Tensor → __iadd__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Add](scalar, sync=sync)

    def __isub__[sync: Bool = True](self, scalar: Scalar[Self.dtype]):
        """In-place subtraction of a scalar.

        Args:
            scalar: Scalar value to subtract.
        """
        if self.is_leaf():
            panic(
                "Tensor → __isub__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Subtract](scalar, sync=sync)

    def __imul__[sync: Bool = True](self, scalar: Scalar[Self.dtype]):
        """In-place multiplication by a scalar.

        Args:
            scalar: Scalar value to multiply by.
        """
        if self.is_leaf():
            panic(
                "Tensor → __imul__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Multiply](scalar, sync=sync)

    def __itruediv__[sync: Bool = True](self, scalar: Scalar[Self.dtype]):
        """In-place division by a scalar.

        Args:
            scalar: Scalar value to divide by.
        """
        if self.is_leaf():
            panic(
                "Tensor → __itruediv__: can not perform in-place operation on a"
                " leaf tensor requiring grad."
            )
        self.buffer.inplace_scalar_ops[Divide](scalar, sync=sync)

    def print(self, num_first: Int = 10, num_last: Int = 10) raises:
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
        var empty = List[Int]()
        print_buffer(
            self.buffer,
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    def __deinit__(deinit self):
        # Gradbox is Optional[Gradbox] — auto-managed by Mojo lifecycle
        pass

    def mse[
        track_grad: Bool = True,
    ](self, target: Tensor[Self.dtype], sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        """Mean squared error loss.

        Args:
            target: Target tensor to compare against.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Scalar tensor with the MSE loss value.
        """
        var diff = Subtractor[Self.dtype].forward[track_grad](
            self, target, sync
        )
        var squared = Multiplicator[Self.dtype].forward[track_grad](
            diff, diff, sync
        )
        return squared.mean[track_grad]()

    def requires_grad_(mut self, requires_grad: Bool = True):
        """Enable or disable gradient tracking in-place.
        Args:
            requires_grad: True to enable gradient tracking, False to disable.
        """
        self.requires_grad = requires_grad
        if requires_grad and not self.has_grad():
            self.init_gradbox()

    def backward[
        graph_size: Int = 50
    ](
        mut output: Tensor[Self.dtype],
        start_grad: Scalar[Self.dtype] = 1.0,
        sync: Bool = True,
    ):
        """Run backward pass to compute gradients.

        The ancestry DAG is never freed by backward: every call re-derives
        the traversal (DFS + fanin + ready queue) from the ancestry stored
        on the tensors, so backward() stays re-runnable and leaf grads
        accumulate across calls. Intermediate (non-leaf) grads are always
        cleared once consumed; view-family nodes (reshape/transpose/permute/
        slice/unsqueeze/squeeze/expand) clear as layout conduits and transfer
        nodes never clear.

        Args:
            start_grad: Initial gradient value (default: 1.0).
            sync: If True, synchronize GPU before backward.
        """
        if not output.requires_grad:
            return
        output.seed_grad(start_grad)
        # We have already seeded, pass None
        output.backward[graph_size](None, sync=sync)

    def backward[
        graph_size: Int = 50,
    ](
        mut output: Tensor[Self.dtype],
        seed_tensor: Optional[Tensor[Self.dtype]],
        *,
        sync: Bool = True,
    ):
        """Run backward pass with a specific seed tensor.

        The ancestry DAG is never freed by backward: every call re-derives
        the traversal from the ancestry stored on the tensors, so backward()
        stays re-runnable and leaf grads accumulate across calls.
        Intermediate grads are always cleared once consumed; view-family
        nodes always clear; transfer nodes never do.

        Args:
            seed_tensor: Tensor containing initial gradients - if None, assumed to be already seeded.
            sync: If True, synchronize GPU before backward.
        """
        if not output.requires_grad:
            return
        comptime if has_accelerator():
            if sync and output.is_on_gpu():
                output.buffer.sync()
        if seed_tensor:
            output.seed_grad(seed_tensor.value())

        try:
            var topo_ids = List[UInt](capacity=graph_size)
            var dfs_stack = List[UInt](capacity=graph_size)
            var node_list = List[Ancestor[Self.dtype]](capacity=graph_size)
            var visited = Set[UInt]()
            var fanin = Dict[UInt, Int]()
            var id_to_index = Dict[UInt, Int]()

            var root = output.to_ancestor()
            var root_ndb: NDBuffer[Self.dtype]
            if output.buffer.is_shared():
                root_ndb = output.buffer.copy()
            else:
                root_ndb = NDBuffer[Self.dtype](
                    output.buffer.buffer.clone(),
                    shape=output.buffer.shape,
                    strides=output.buffer.strides,
                    offset=output.buffer.offset,
                )
            root.ndb = NDBufferLite[Self.dtype](root_ndb^)
            dfs_stack.append(root._id)
            node_list.append(root.copy())
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
                    # Snapshot parents first: `node` borrows `node_list`,
                    # so the list cannot grow while iterating the borrow.
                    var pending = List[Ancestor[Self.dtype]]()
                    for parent in node.ancestry():
                        pending.append(parent.copy())
                    for k in range(len(pending)):
                        var parent_id = pending[k]._id
                        fanin[parent_id] = fanin.get(parent_id, 0) + 1

                        if parent_id not in id_to_index:
                            var new_idx = len(node_list)
                            node_list.append(pending[k].copy())
                            id_to_index[parent_id] = new_idx
                            dfs_stack.append(parent_id)

            var ready_queue = Deque[UInt](capacity=graph_size)
            ready_queue.append(root._id)

            var parent_ids = List[UInt](capacity=3)
            while len(ready_queue) > 0:
                var node_id: UInt = 0
                try:
                    node_id = ready_queue.popleft()
                except key_err:
                    panic(String(key_err))

                var node_idx = id_to_index[node_id]
                ref node = node_list[node_idx]

                if node.has_ancestry():
                    parent_ids.clear()
                    try:
                        Backward[Self.dtype].invoke(node, parent_ids)
                    except e:
                        print("Backward invoke error: ", e)
                    for i in range(len(parent_ids)):
                        var target_id = parent_ids[i]
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

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return {Pointer(to=self).as_imm()}

    def get(self, index: Int) -> Scalar[Self.dtype]:
        """Get element at a logical flat index with bounds checking.

        C-order over this view: respects slice/transpose offsets and
        strides, so `view.get(i)` is the i-th logical element. Negative
        indices wrap from the end.

        Args:
            index: Logical flat index (< `numels()` after wrapping).

        Returns:
            The scalar value at that index.
        """
        return self.buffer.get(index)

    def set(self, index: Int, scalar: Scalar[Self.dtype]):
        """Set element at a logical flat index with bounds checking.

        Mirror of `get`: honors view offset/strides, so writes through a
        view land on the viewed element (aliasing the parent).

        Args:
            index: Logical flat index (< `numels()` after wrapping).
            scalar: Scalar value to write.
        """
        self.buffer.set(index, scalar)

    def storage_get[checked: Bool = True](
        self, index: Int
    ) -> Scalar[Self.dtype]:
        """Raw storage read: `index` is a buffer address, no view mapping.

        Internal use only — mirrors `NDBuffer.storage_get` for call sites
        holding addresses from `index_iterator()` or `flatten_index(...,
        offset)`. Prefer `get` for logical indexing.
        """
        return self.buffer.storage_get[checked](index)

    def storage_set[checked: Bool = True](
        self, index: Int, scalar: Scalar[Self.dtype]
    ):
        """Raw storage write (mirror of `storage_get`)."""
        self.buffer.storage_set[checked](index, scalar)

    def view[
        track_grad: Bool = True,
    ](
        self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a view with explicit shape, strides, and offset.

        Args:
            shape: Target shape.
            strides: Memory strides for the view.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.
            validated: If True, skips validation.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A view tensor over the same buffer.
        """
        return View[Self.dtype].forward[track_grad](
            self, shape, strides, offset, requires_grad, validated, sync=sync
        )

    def view[
        track_grad: Bool = True,
    ](
        self,
        shape: List[Int],
        strides: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a view with explicit shape and strides as lists.

        Args:
            shape: Target shape as a list.
            strides: Memory strides as a list.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A view tensor over the same buffer.
        """
        var view_shape, view_strides = Shape(shape), Strides(strides)
        return View[Self.dtype].forward[track_grad](
            self,
            view_shape,
            view_strides,
            offset,
            requires_grad,
            False,
            sync=sync,
        )

    def view[
        track_grad: Bool = True,
    ](
        self,
        *shape_dims: Int,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a contiguous view with the given shape.

        Args:
            shape_dims: Target shape as variadic ints.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A contiguous view tensor.
        """
        var shape = Shape(shape_dims)
        var strides = Strides.default(shape)
        return View[Self.dtype].forward[track_grad](
            self, shape, strides, offset, requires_grad, False, sync=sync
        )

    def view[
        track_grad: Bool = True,
    ](
        self,
        shape: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a contiguous view with the given shape.

        Args:
            shape: Target shape as a list of ints.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A contiguous view tensor.
        """
        var view_shape = Shape(shape)
        var strides = Strides.default(view_shape)
        return View[Self.dtype].forward[track_grad](
            self, view_shape, strides, offset, requires_grad, False, sync=sync
        )

    def view[
        track_grad: Bool = True,
    ](
        self,
        shape: Shape,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Create a contiguous view with the given shape.

        Args:
            shape: Target shape.
            offset: Base memory offset.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A contiguous view tensor.
        """
        return View[Self.dtype].forward[track_grad](
            self,
            shape,
            Strides.default(shape),
            offset,
            requires_grad,
            False,
            sync=sync,
        )

    def into_view[
        track_grad: Bool = True, sync: Bool = True
    ](self, requires_grad: Optional[Bool] = None) -> Tensor[Self.dtype]:
        """Convert this tensor into a shared view.

        Args:
            requires_grad: If provided, overrides requires_grad.

        Returns:
            A view tensor sharing the underlying buffer.
        """
        var shape, strides, offset = self.shape(), self.strides(), self.offset()
        var grad_required = requires_grad.or_else(self.requires_grad)
        return View[Self.dtype].forward[track_grad](
            self, shape, strides, offset, grad_required, validated=True
        )

    def transpose[
        track_grad: Bool = True,
    ](
        self,
        *axes: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Transpose tensor by reversing or permuting axes.

        Args:
            axes: Axes to permute. If empty, reverses all axes.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A transposed view tensor.
        """
        return self.transpose[track_grad](
            IntArray(axes), requires_grad, sync=sync
        )

    def transpose[
        track_grad: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Transpose tensor by reversing or permuting axes.

        Args:
            axes: Axes to permute as a list.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A transposed view tensor.
        """
        return self.transpose[track_grad](
            IntArray(axes), requires_grad, sync=sync
        )

    def transpose[
        track_grad: Bool = True,
    ](
        self,
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Transpose tensor by reversing or permuting axes.

        Args:
            axes: Axes to permute as an IntArray.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A transposed view tensor.
        """
        return Transpose.forward[track_grad](
            self, axes, requires_grad, sync=sync
        )

    def slice[
        track_grad: Bool = True, sync: Bool = True
    ](self, start: Int, end: Int, step: Int = 1, axis: Int = 0) -> Tensor[
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
        offset += self.buffer.offset

        return View[Self.dtype].forward[track_grad](
            self,
            shape,
            strides,
            offset,
            self.requires_grad,
            True,
        )

    def slice[
        track_grad: Bool = True, sync: Bool = True
    ](
        self,
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
        var step_sizes = IntArray(steps) if steps else IntArray.filled(len(axes), 1)
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
        offset += self.buffer.offset

        return View[Self.dtype].forward[track_grad](
            self,
            shape,
            strides,
            offset,
            self.requires_grad,
            True,
        )

    def expand[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Broadcast tensor to a target shape.

        Args:
            target: Shape to broadcast to.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor broadcasted to the target shape.
        """
        return Expand[Self.dtype].forward[track_grad](
            self, target, requires_grad, sync=sync
        )

    def expand[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        *target_dims: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Broadcast tensor to a target shape.

        Args:
            target_dims: Target shape as variadic ints.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor broadcasted to the target shape.
        """
        return Expand[Self.dtype].forward[track_grad](
            self, Shape(target_dims), requires_grad, sync=sync
        )

    def squeeze[
        track_grad: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Remove axes of size 1.

        Args:
            axes: Axes to squeeze. If empty, removes all size-1 axes.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with size-1 axes removed.
        """
        return Squeeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad, sync=sync
        )

    def squeeze[
        track_grad: Bool = True,
    ](
        self,
        *axes: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Remove axes of size 1.

        Args:
            axes: Axes to squeeze as variadic ints.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with size-1 axes removed.
        """
        return Squeeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad, sync=sync
        )

    def unsqueeze[
        track_grad: Bool = True,
    ](
        self,
        *axes: Int,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Insert axes of size 1.

        Args:
            axes: Axes at which to insert size-1 dimensions.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with size-1 axes inserted.
        """
        return Unsqueeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad, sync=sync
        )

    def unsqueeze[
        track_grad: Bool = True,
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Insert axes of size 1.

        Args:
            axes: Axes at which to insert size-1 dimensions.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with size-1 axes inserted.
        """
        return Unsqueeze[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad, sync=sync
        )

    def unsqueeze[
        track_grad: Bool = True,
    ](
        self,
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Insert axes of size 1.

        Args:
            axes: Axes at which to insert size-1 dimensions.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with size-1 axes inserted.
        """
        return Unsqueeze[Self.dtype].forward[track_grad](
            self, axes, requires_grad, sync=sync
        )

    def permute[
        track_grad: Bool = True,
    ](
        self,
        axes: List[Int],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Permute axes according to a given ordering.

        Args:
            axes: Permutation order (e.g. [2, 0, 1] for 3D).
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with axes permuted.
        """
        return Permute[Self.dtype].forward[track_grad](
            self, IntArray(axes), requires_grad, sync=sync
        )

    def permute[
        track_grad: Bool = True,
    ](
        self,
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Permute axes according to a given ordering.

        Args:
            axes: Permutation order as IntArray.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A tensor with axes permuted.
        """
        return Permute[Self.dtype].forward[track_grad](
            self, axes, requires_grad, sync=sync
        )

    def argmax[
        index_dtype: DType = DEFAULT_INDEX_DTYPE,
    ](self, axis: Int = 0, keepdims: Bool = False) -> Tensor[index_dtype]:
        return Argmax[Self.dtype, index_dtype].argmax(
            tensor=self, axis=axis, keepdims=keepdims
        )

    def argmin[
        index_dtype: DType = DEFAULT_INDEX_DTYPE,
    ](self, axis: Int = 0, keepdims: Bool = False) -> Tensor[index_dtype]:
        return Argmin[Self.dtype, index_dtype].argmin(
            tensor=self, axis=axis, keepdims=keepdims
        )

    def max(
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

    def max(
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

    def min(
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

    def min(
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

    def shuffle[
        track_grad: Bool = True,
    ](
        self,
        perm: List[Int] = [],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Shuffle tensor elements along an axis according to a permutation.

        Args:
            perm: Permutation indices for the given axis.
            axis: Axis along which to shuffle.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A shuffled tensor.
        """
        return Shuffle[Self.dtype].forward[track_grad](
            self, perm, axis, requires_grad, sync=sync
        )

    def relu[
        track_grad: Bool = True,
    ](self, requires_grad: Optional[Bool] = None, sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        """Rectified Linear Unit: max(0, x).

        Args:
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with ReLU applied element-wise.
        """
        return ReLU[Self.dtype].forward[track_grad](self, requires_grad, sync)

    def gelu[
        track_grad: Bool = True,
    ](self, requires_grad: Optional[Bool] = None, sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        """Gaussian Error Linear Unit.

        Args:
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with GeLU applied element-wise.
        """
        comptime if Self.dtype.is_floating_point():
            return GeLU[Self.dtype].forward[track_grad](
                self, requires_grad, sync
            )
        else:
            panic("GeLU requires dtype to be floating point")
            return Tensor[Self.dtype].scalar(42)

    def clip[
        track_grad: Bool = True,
    ](
        self,
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Clip tensor values to a range [min_val, max_val].

        Args:
            min_val: Minimum value.
            max_val: Maximum value.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with values clipped to [min_val, max_val].
        """
        return Clip[Self.dtype].forward[track_grad](
            self, min_val, max_val, requires_grad, sync
        )

    def fake_quant[
        track_grad: Bool = True,
    ](
        self,
        scale: Tensor[Self.dtype],
        qmin: Scalar[Self.dtype],
        qmax: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Fake quantization with a straight-through estimator.

            out = scale * clamp(round(x / scale), qmin, qmax)

        The quantizer is a staircase, so the TRUE derivative w.r.t. the input
        is 0 almost everywhere. This op overrides it to **1** (that is the
        "straight-through" part) while leaving `d/d(scale)` at its true value
        `k = clamp(round(x/scale), qmin, qmax)`. Training through it therefore
        moves, which plain `round`-then-multiply cannot do.

        Must be a single op rather than a composition: wiring ancestry per
        sub-op would hand back zeros, because `round` is non-differentiable by
        design. See `tenmo/fakequant.mojo`.

        Uses half-to-even rounding (Mojo's `round`), not `floor(x + 0.5)` —
        the substitution rounds half *up* and disagrees on every tie, and a
        dyadic scale puts `x / scale` on a `.5` exactly.

        Args:
            scale: Quantization step. A tensor, so it can be a learnable
                parameter (LSQ-style) — its gradient is `upstream * k`.
            qmin: Lowest representable level (e.g. -128.0).
            qmax: Highest representable level (e.g. 127.0).
            requires_grad: Override gradient tracking (default: inherit).
            sync: Whether to synchronize the GPU operation.

        Returns:
            Dequantized tensor of the same dtype. CPU only for now.
        """
        return FakeQuant[Self.dtype].forward[track_grad](
            self, scale, qmin, qmax, requires_grad, sync
        )

    def round(
        self,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Round to the nearest integer level, **ties to even**.

        NOT differentiable: the true derivative is 0 almost everywhere, so the
        result is always a leaf and never carries a gradient. Deliberately has
        no `track_grad` parameter -- a silently-ignored flag is worse than a
        compile error when a caller expects a graph edge.

        Ties are not a corner case: with a dyadic scale, `x / scale` lands on a
        `.5` exactly, which is why quantization must use this and not
        `floor(x + 0.5)` (that rounds half *up* and disagrees on every tie).
        Note `round(-0.5) = -0.0`, so signed zero can reach the output.

        Args:
            sync: Whether to synchronize the GPU operation (CPU-only op; the
                flag is accepted for signature symmetry).

        Returns:
            Tensor with values rounded to integer levels. Always a leaf.
        """
        return Round[Self.dtype].forward(self, sync)

    def floor(
        self,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Truncate toward negative infinity.

        NOT differentiable: the true derivative is 0 almost everywhere, so the
        result is always a leaf and never carries a gradient. No `track_grad`
        parameter, for the same reason as `round`.

        Args:
            sync: Whether to synchronize the GPU operation (CPU-only op; the
                flag is accepted for signature symmetry).

        Returns:
            Tensor with values floored. Always a leaf.
        """
        return Floor[Self.dtype].forward(self, sync)

    def tanh[
        track_grad: Bool = True,
    ](
        self,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Hyperbolic tangent activation.

        Args:
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with tanh applied element-wise.
        """
        return Tanh[Self.dtype].forward[track_grad](self, requires_grad, sync)

    def sigmoid[
        track_grad: Bool = True,
    ](
        self,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """Sigmoid activation: 1 / (1 + exp(-x)).

        Args:
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with sigmoid applied element-wise.
        """
        return Sigmoid[Self.dtype].forward[track_grad](
            self, requires_grad, sync
        )

    def softmax[
        track_grad: Bool = True, log: Bool = False
    ](
        self,
        axes: List[Int] = [],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Softmax activation along given axes.

        Args:
            axes: Axes along which to apply softmax.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with softmax probabilities. Use log=True for log-softmax.
        """
        comptime if log:
            return LogSoftmax[Self.dtype].forward[track_grad](
                self, IntArray(axes), requires_grad, sync
            )
        else:
            return Softmax[Self.dtype].forward[track_grad](
                self, IntArray(axes), requires_grad, sync
            )

    def softmax[
        track_grad: Bool = True, log: Bool = False
    ](
        self,
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Softmax activation along given axes.

        Args:
            axes: Axes along which to apply softmax.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Tensor with softmax probabilities. Use log=True for log-softmax.
        """
        comptime if log:
            return LogSoftmax[Self.dtype].forward[track_grad](
                self, axes, requires_grad, sync
            )
        else:
            return Softmax[Self.dtype].forward[track_grad](
                self, axes, requires_grad, sync
            )

    def binary_cross_entropy[
        track_grad: Bool = True,
    ](
        pred: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
        reduction: Reduction = Reduction("mean"),
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Binary cross entropy loss.

        Args:
            target: Ground truth labels (0 or 1).
            epsilon: Small value for numerical stability.
            reduction: "mean", "sum", or "none".
            sync: Whether to synchronize the GPU operation.

        Returns:
            Scalar tensor with the BCE loss (mean/sum) or per-element (none).
        """
        return BCELoss[Self.dtype].forward[track_grad](
            pred, target, epsilon, reduction, sync
        )

    def binary_cross_entropy_with_logits[
        track_grad: Bool = True,
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        reduction: Reduction = Reduction("mean"),
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """BCE loss with logits (sigmoid applied internally).

        Args:
            target: Ground truth labels (0 or 1).
            epsilon: Small value for numerical stability.
            reduction: "mean", "sum", or "none".
            sync: Whether to synchronize the GPU operation.

        Returns:
            Scalar tensor with the BCE loss (mean/sum) or per-element (none).
        """
        return BCEWithLogitsLoss[Self.dtype].forward[track_grad](
            logits, target, epsilon, reduction, sync
        )

    @staticmethod
    def add[
        A: DType, track_grad: Bool = True
    ](a: Tensor[A], b: Tensor[A], sync: Bool = True) -> Tensor[A]:
        """Element-wise addition."""
        return Adder[A].forward[track_grad](a, b, sync=sync)

    @staticmethod
    def sub[
        A: DType, track_grad: Bool = True
    ](a: Tensor[A], b: Tensor[A], sync: Bool = True) -> Tensor[A]:
        """Element-wise subtraction."""
        return Subtractor[A].forward[track_grad](a, b, sync=sync)

    @staticmethod
    def mul[
        A: DType, track_grad: Bool = True
    ](a: Tensor[A], b: Tensor[A], sync: Bool = True) -> Tensor[A]:
        """Element-wise multiplication."""
        return Multiplicator[A].forward[track_grad](a, b, sync=sync)

    @staticmethod
    def truediv[
        A: DType, track_grad: Bool = True
    ](a: Tensor[A], b: Tensor[A], sync: Bool = True) -> Tensor[A]:
        """Element-wise division."""
        return Divider[A].forward[track_grad](a, b, sync=sync)

    @staticmethod
    def iadd[A: DType](a: Tensor[A], b: Tensor[A], sync: Bool = True):
        """In-place addition."""
        if a.is_leaf():
            panic(
                "Tensor.iadd: can not perform in-place operation on a leaf"
                " tensor requiring grad."
            )
        a.buffer.inplace_ops[Add](b.buffer, sync=sync)

    @staticmethod
    def isub[A: DType](a: Tensor[A], b: Tensor[A], sync: Bool = True):
        """In-place subtraction."""
        if a.is_leaf():
            panic(
                "Tensor.isub: can not perform in-place operation on a leaf"
                " tensor requiring grad."
            )
        a.buffer.inplace_ops[Subtract](b.buffer, sync=sync)

    @staticmethod
    def imul[A: DType](a: Tensor[A], b: Tensor[A], sync: Bool = True):
        """In-place multiplication."""
        if a.is_leaf():
            panic(
                "Tensor.imul: can not perform in-place operation on a leaf"
                " tensor requiring grad."
            )
        a.buffer.inplace_ops[Multiply](b.buffer, sync=sync)

    @staticmethod
    def itruediv[A: DType](a: Tensor[A], b: Tensor[A], sync: Bool = True):
        """In-place division."""
        if a.is_leaf():
            panic(
                "Tensor.itruediv: can not perform in-place operation on a leaf"
                " tensor requiring grad."
            )
        a.buffer.inplace_ops[Divide](b.buffer, sync=sync)

    def matmul[
        track_grad: Bool = True, mode: Int = mnemonics.mm
    ](
        A: Tensor[Self.dtype], B: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        """Matrix multiplication of two tensors.

        Args:
            B: Right-hand tensor.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Result of matrix multiplication.
        """
        return Matmul[Self.dtype].forward[track_grad=track_grad, mode=mode](
            A, B, sync
        )

    def matmul(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        """Matrix multiplication with a Gradbox.

        Args:
            B: Right-hand Gradbox.

        Returns:
            Result of matrix multiplication as Gradbox.
        """
        return Matmul[Self.dtype].forward(A, B)

    @staticmethod
    def concat[
        track_grad: Bool = True,
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Concatenate tensors along an axis.

        Args:
            tensors: List of tensors to concatenate.
            axis: Axis along which to concatenate.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A single concatenated tensor.
        """
        return Concate[Self.dtype].forward[track_grad](
            tensors, axis, requires_grad, sync
        )

    @staticmethod
    def stack[
        track_grad: Bool = True,
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Stack tensors along a new axis.

        Args:
            tensors: List of tensors to stack.
            axis: Axis at which to insert the new dimension.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            A stacked tensor with shape extended by one dimension.
        """
        return Stack[Self.dtype].forward[track_grad](
            tensors, axis, requires_grad, sync
        )

    @staticmethod
    def vstack[
        track_grad: Bool = True,
    ](
        tensors: List[Tensor[Self.dtype]],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Vertical stack (concatenate along axis 0, then flatten 2D).

        Args:
            tensors: List of 1D tensors to stack vertically.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Stacked tensor.
        """
        return Stack[Self.dtype].vstack[track_grad](
            tensors, requires_grad, sync
        )

    @staticmethod
    def hstack[
        track_grad: Bool = True,
    ](
        tensors: List[Tensor[Self.dtype]],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Horizontal stack (concatenate along last axis).

        Args:
            tensors: List of 1D tensors to stack horizontally.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Stacked tensor.
        """
        return Stack[Self.dtype].hstack[track_grad](
            tensors, requires_grad, sync
        )

    @staticmethod
    def pad[
        track_grad: Bool = True,
    ](
        x: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        mode: String = "constant",
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Pad tensor along each axis.

        Args:
            x: Input tensor to pad.
            pad: List of (before, after) padding pairs per axis.
            mode: Padding mode ("constant", etc.).
            value: Fill value for constant mode.
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Padded tensor.
        """
        return Pad[Self.dtype].forward[track_grad](
            x, pad, mode, value, requires_grad, sync=sync
        )

    @staticmethod
    def pad_constant[
        track_grad: Bool = True,
    ](
        x: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Pad tensor with a constant value.

        Args:
            x: Input tensor to pad.
            pad: List of (before, after) padding pairs per axis.
            value: Fill value (default: 0.0).
            requires_grad: If provided, overrides requires_grad.
            sync: Whether to synchronize the GPU operation.

        Returns:
            Padded tensor.
        """
        return Pad[Self.dtype].forward[track_grad](
            x, pad, "constant", value, requires_grad, sync=sync
        )

    @staticmethod
    def pad2d[
        track_grad: Bool = True,
    ](
        x: Tensor[Self.dtype],
        pad_left: Int,
        pad_right: Int,
        pad_top: Int,
        pad_bottom: Int,
        mode: String = "constant",
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
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
            sync: Whether to synchronize the GPU operation.

        Returns:
            2D-padded tensor.
        """
        var pad = List[Tuple[Int, Int]]()
        pad.append((pad_top, pad_bottom))
        pad.append((pad_left, pad_right))
        return Pad[Self.dtype].forward[track_grad](
            x, pad, mode, value, requires_grad, sync=sync
        )

    @staticmethod
    def pad_for_conv[
        track_grad: Bool = True, sync: Bool = True
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

    def slices(ref self) -> SliceIterator[Self.dtype, origin_of(self)]:
        return SliceIterator(Pointer(to=self))


@fieldwise_init
struct ElemIterator[dtype: DType, origin: ImmOrigin](
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
    var index_itr: ShapeIndexIterator[Self.origin]

    def __init__(out self, src: Pointer[Tensor[Self.dtype], Self.origin]):
        """Initialize iterator over a tensor.

        Args:
            src: Pointer to the tensor to iterate over.
        """
        self.src = src
        self.index_itr = rebind[ShapeIndexIterator[Self.origin]](
            src[].shape().__iter__()
        )

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    def __next__(mut self) raises StopIteration -> Self.Element:
        """Return next (coordinate, value) pair.

        Returns:
            Tuple of (IntArray coordinates, scalar value at that position).

        Raises:
            StopIteration: When all elements have been visited.
        """
        var next = self.index_itr.__next__()
        return next, self.src[][next]

    def __len__(self) -> Int:
        """Number of elements remaining.

        Returns:
            Number of elements yet to be visited.
        """
        return self.index_itr.__len__()

    def __has_next__(self) -> Bool:
        """Check if there are more elements to visit.

        Returns:
            True if more elements remain, False otherwise.
        """
        return self.index_itr.__has_next__()

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        """Get the bounds of the iterator.

        Returns:
            Tuple of (remaining length, Optional of the same length).
        """
        return self.index_itr.bounds()


@fieldwise_init
struct SliceIterator[dtype: DType, origin: ImmOrigin](
    RegisterPassable & ImplicitlyCopyable & Iterable & Iterator & Sized
):
    """Iterates over first dimension slices.

    matrix (4,3)   → yields 4 tensors of shape (3,)
    tensor (4,3,2) → yields 4 tensors of shape (3,2)
    """

    comptime Element = Tensor[Self.dtype]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var src: Pointer[Tensor[Self.dtype], Self.origin]
    var current: Int
    var first_dim_size: Int
    var first_dim_stride: Int
    var slice_shape: Shape
    var slice_strides: Strides

    def __init__(out self, src: Pointer[Tensor[Self.dtype], Self.origin]):
        self.src = src
        self.current = 0
        self.first_dim_size = src[].shape()[0]
        self.first_dim_stride = src[].strides()[0]
        self.slice_shape = src[].shape()[1::]
        self.slice_strides = src[].strides()[1::]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    def __next__(mut self) raises StopIteration -> Self.Element:
        if not self.__has_next__():
            raise StopIteration()
        var idx = self.current
        self.current += 1
        var source = self.src[]
        var offset = source.offset() + idx * self.first_dim_stride
        return View[Self.dtype].forward[track_grad=False](
            source,
            self.slice_shape,
            self.slice_strides,
            offset,
            requires_grad=False,
            validated=True,
        )

    def __len__(self) -> Int:
        return self.first_dim_size - self.current

    def __has_next__(self) -> Bool:
        return self.current < self.first_dim_size

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = self.__len__()
        return remaining, Optional(remaining)
