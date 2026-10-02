from .shared.shapes import Shape
from .shared.mnemonics import *
from .validators import Validator
from .tensor import Tensor
from .shared.intarray import IntArray
from .ndbuffer import NDBuffer, NDBufferLite, print_buffer
from .shared.broadcasthelper import ShapeBroadcaster
from .shared.strides import Strides
from std.sys import simd_width_of, has_accelerator
from .shared.indexhelper import IndexIterator
from .shared.constants import CloseTol
from .filler import Filler
from .sum_mean_reduction import SumMeanReduction
from .shared.indexhelper import Idx
from .shared.panic import panic
from .gpu.device import Device, CPU, GPU
from std.memory import Pointer


struct Gradbox[dtype: DType](
    ImplicitlyCopyable & Sized & Writable & Equatable & Absable
):
    """Refcounted-handle wrapper over an NDBuffer.
    The gradient storage lives in the NDBuffer, whose Buffer refcount
    provides the shared lifecycle.

    The NDBuffer descriptor itself (Shape/Strides/offset) is kept behind a
    shared ``NDBufferLite`` handle so copying a Gradbox is an O(1) refcount
    bump instead of a per-copy deep copy of the descriptor (which allocates
    Shape/Strides dims arrays). ``Gradbox(shape)`` allocates own storage
    (shared-from-birth). ``var b = a`` is an alias operator — the copy shares
    the same gradient storage (Buffer refcount bump); ``clone()`` materialises
    an independent deep copy.
    """

    var handle: NDBufferLite[Self.dtype]

    def __init__(out self, shape: Shape):
        """Initialize with given shape (fresh owned, shared-from-birth storage).
        """
        self.handle = NDBufferLite[Self.dtype](NDBuffer[Self.dtype](shape))

    def __init__(out self, var buffer: NDBuffer[Self.dtype]):
        """Initialize from an existing NDBuffer (adopt ownership/move)."""
        self.handle = NDBufferLite[Self.dtype](buffer^)

    def __init__(out self, *, deinit move: Self):
        """Move constructor — transfer ownership without copying."""
        self.handle = move.handle^

    def __init__(out self, *, copy: Self):
        """Copy constructor — share storage (O(1) NDBufferLite refcount bump).
        """
        self.handle = copy.handle

    def clone(self) -> Gradbox[Self.dtype]:
        """Independent deep copy: fresh refcount and fresh storage.

        Unlike copy-init (``var gb2 = gb1``), which aliases the same gradient
        storage, ``clone`` materialises fresh, independent storage — later
        gradient accumulation through one never affects the other. GPU storage
        is cloned as an independent device buffer.
        """
        var buf_clone = self.buffer().clone()
        return Gradbox[Self.dtype](buf_clone^)

    @always_inline
    def buffer(ref self) -> ref[self.handle.value()] NDBuffer[Self.dtype]:
        """Get reference to the underlying NDBuffer."""
        return self.handle.value()

    @always_inline
    def as_tensor(
        deinit self, requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        """Convert Gradbox to a Tensor.

        Args:
            requires_grad: Whether the resulting tensor requires gradients.

        Returns:
            A Tensor wrapping the Gradbox's buffer.
        """
        if self.is_contiguous():
            return Tensor[Self.dtype](
                self.buffer().copy(), requires_grad=requires_grad
            )
        else:
            return Tensor[Self.dtype](
                self.buffer().contiguous(), requires_grad=requires_grad
            )

    def detach(self) -> Gradbox[Self.dtype]:
        """Return an independent deep copy of this Gradbox.

        Handles both CPU and GPU. Always allocates new storage and copies
        data — never shares the underlying buffer.

        Returns:
            A new Gradbox with independently owned data.
        """
        ref ndb = self.buffer()
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var new_state = ndb.contiguous_device_state()
                    var ndb_copy = NDBuffer[Self.dtype].with_device_state(
                        new_state^, ndb.shape
                    )
                    return Gradbox[Self.dtype](ndb_copy^)
                except e:
                    panic(
                        "Gradbox → detach: GPU deep copy failed: " + String(e)
                    )
        var buf = ndb.contiguous_buffer()
        var ndb_copy = NDBuffer[Self.dtype](buf^, ndb.shape)
        return Gradbox[Self.dtype](ndb_copy^)

    def device(self) -> Device:
        """Get the device this Gradbox is on.

        Returns:
            The CPU or GPU device.
        """
        return self.buffer().device()

    def transpose(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Transpose the Gradbox along the given axes.

        Args:
            axes: The permutation of axes.

        Returns:
            A new contiguous Gradbox with transposed axes.
        """
        var view = self.buffer().transpose(axes)
        return Gradbox[Self.dtype](view.contiguous(owned=True))

    def __abs__(self) -> Gradbox[Self.dtype]:
        """Compute element-wise absolute value.

        Returns:
            A new Gradbox with absolute values.
        """
        return Gradbox[Self.dtype](self.buffer().unary_ops[ABS]())

    def sqrt(
        self,
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
    ) -> Gradbox[Self.dtype]:
        """Compute element-wise square root.

        Args:
            epsilon: Small value to prevent sqrt of negative numbers.

        Returns:
            A new Gradbox with square roots.
        """
        var shape = self.shape()
        var ndb = self.buffer().data_buffer().unary_ops[SQRT]()
        return Gradbox[Self.dtype](NDBuffer[Self.dtype](ndb^, shape))

    def norm(
        self,
        p: Float64 = 2.0,
        axis: Optional[Int] = None,
        keepdims: Bool = False,
    ) -> Gradbox[Self.dtype]:
        """Compute the Lp norm.

        Args:
            p: The order of the norm (only p=2.0 supported).
            axis: Axis along which to compute norm. If None, computes global norm.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            A Gradbox containing the norm(s).
        """
        if p == 2.0:
            # L2 norm: sqrt(sum(x²))
            var squared = self.__mul__(self)
            var dim = IntArray(axis.value()) if axis else IntArray()
            var sum_sq = squared.sum(dim, keepdims=keepdims)
            return sum_sq.sqrt()
        else:
            panic("Only L2 norm (p=2) currently supported")
            return Gradbox[Self.dtype](Shape())

    @staticmethod
    def arange(
        *args: Scalar[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        """Create a 1D Gradbox with evenly spaced values.

        Args:
            args: Start, stop, and optionally step values.

        Returns:
            A 1D Gradbox with values from start to stop.
        """
        var nd_buffer = NDBuffer[Self.dtype].arange(args)
        return Gradbox[Self.dtype](nd_buffer^)

    def flatten(
        self, start_dim: Int = 0, end_dim: Optional[Int] = None
    ) -> Gradbox[Self.dtype]:
        """Flatten the Gradbox to 1D.

        Args:
            start_dim: The first dimension to flatten.
            end_dim: The last dimension to flatten. If None, flattens to the end.

        Returns:
            A flattened Gradbox.
        """
        var flattened_buffer = self.buffer().flatten(start_dim, end_dim)
        return Gradbox[Self.dtype](flattened_buffer^)

    def squeeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Remove dimensions of size 1.

        Args:
            axes: The axes to squeeze.

        Returns:
            A new contiguous Gradbox with squeezed dimensions.
        """
        var view = self.buffer().squeeze(axes)
        return Gradbox[Self.dtype](view.contiguous(owned=True))

    def squeeze(self, axes: List[Int] = []) -> Gradbox[Self.dtype]:
        """Remove dimensions of size 1.

        Args:
            axes: The axes to squeeze as a list.

        Returns:
            A new Gradbox with squeezed dimensions.
        """
        return self.squeeze(IntArray(axes))

    def unsqueeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Add dimensions of size 1.

        Args:
            axes: The axes to unsqueeze.

        Returns:
            A new contiguous Gradbox with unsqueezed dimensions.
        """
        var view = self.buffer().unsqueeze(axes)
        return Gradbox[Self.dtype](view.contiguous(owned=True))

    def unsqueeze(self, axes: List[Int]) -> Gradbox[Self.dtype]:
        """Add dimensions of size 1.

        Args:
            axes: The axes to unsqueeze as a list.

        Returns:
            A new Gradbox with unsqueezed dimensions.
        """
        return self.unsqueeze(IntArray(axes))

    def permute(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Permute the axes of this Gradbox.

        Args:
            axes: The new order of axes.

        Returns:
            A new contiguous Gradbox with permuted axes.
        """
        var view = self.buffer().permute(axes)
        return Gradbox[Self.dtype](view.contiguous(owned=True))

    @staticmethod
    @always_inline
    def full(
        shape: Shape,
        scalar: Scalar[Self.dtype],
        device: Device = CPU().into(),
    ) -> Gradbox[Self.dtype]:
        """Create a Gradbox filled with a scalar value.

        Args:
            shape: The tensor shape.
            scalar: The value to fill with.
            device: The target device (CPU or GPU).

        Returns:
            A new Gradbox filled with the scalar value.
        """
        return Gradbox[Self.dtype](NDBuffer.full(shape, scalar, device))

    @staticmethod
    @always_inline
    def zeros(
        shape: Shape, device: Device = CPU().into()
    ) -> Gradbox[Self.dtype]:
        """Create a Gradbox of zeros.

        Args:
            shape: The tensor shape.
            device: The target device (CPU or GPU).

        Returns:
            A new Gradbox of zeros.
        """
        return Gradbox[Self.dtype].full(
            shape, Scalar[Self.dtype](0), device=device
        )

    @staticmethod
    @always_inline
    def rand(
        shape: Shape,
        min: Scalar[Self.dtype] = 0,
        max: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
    ) -> Gradbox[Self.dtype]:
        """Create a Gradbox with uniform random values in [min, max).

        CPU-only (gradboxes are gradient storage; device-side RNG belongs to
        ``Tensor.rand``). Delegates to ``NDBuffer.rand``.

        Args:
            shape: The tensor shape.
            min: Lower bound (inclusive).
            max: Upper bound (exclusive).
            init_seed: Random seed. If None, randomizes each call.

        Returns:
            A new Gradbox with random values in [min, max).
        """
        return Gradbox[Self.dtype](
            NDBuffer[Self.dtype].rand(
                shape, min=min, max=max, init_seed=init_seed
            )
        )

    @always_inline
    def sum(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Compute the sum along the given axes.

        Args:
            axes: The axes to reduce. If empty, reduces all dimensions.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            A new Gradbox with the sum.
        """
        var normalized_axes = Validator.validate_and_normalize_axes(
            self.shape(), axes
        )
        var nd_buffer = SumMeanReduction[Self.dtype].reduce(
            self.buffer(), normalized_axes=normalized_axes, keepdims=keepdims
        )
        return Gradbox[Self.dtype](nd_buffer^)

    @always_inline
    def mean(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Compute the mean along the given axes.

        Args:
            axes: The axes to reduce. If empty, reduces all dimensions.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            A new Gradbox with the mean.
        """
        var normalized_axes = Validator.validate_and_normalize_axes(
            self.shape(), axes
        )
        var ndb = SumMeanReduction[Self.dtype].reduce[op_code=MEAN](
            self.buffer(), normalized_axes, keepdims
        )
        return Gradbox[Self.dtype](ndb^)

    @always_inline
    def sum_over_broadcasted_axes(
        extended_grad: Gradbox[Self.dtype], target_shape: Shape
    ) -> Gradbox[Self.dtype]:
        """Sum over broadcasted axes to match target shape.

        Args:
            target_shape: The shape to expand to.

        Returns:
            A new Gradbox summed to the target shape.
        """
        var nd_buffer = SumMeanReduction[Self.dtype].sum_over_broadcasted_axes(
            extended_grad.buffer(), target_shape
        )
        return Gradbox[Self.dtype](nd_buffer^)

    def broadcast_to(self, target_shape: Shape) -> Gradbox[Self.dtype]:
        """Broadcast this Gradbox to the target shape.

        Args:
            target_shape: The shape to broadcast to.

        Returns:
            A new Gradbox with the target shape.
        """
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Gradbox → broadcast_to: shape "
                + String(self.shape())
                + " not broadcastable to "
                + String(target_shape)
            )

        var broadcasted_buffer = self.buffer().broadcast_to(target_shape)
        comptime if has_accelerator():
            if broadcasted_buffer.is_on_gpu():
                return Gradbox[Self.dtype](broadcasted_buffer^)
        return Gradbox[Self.dtype](broadcasted_buffer.contiguous(owned=True))

    def __getitem__(self, *indices: Idx) -> Gradbox[Self.dtype]:
        """Index the Gradbox with Idx objects (integers or slices).

        Args:
            indices: One index per axis — either an integer or a Slice.

        Returns:
            A new Gradbox view over the indexed region.
        """
        var ndb = self.buffer().__getitem__(*indices)
        return Gradbox[Self.dtype](ndb^)

    def slice(
        self, start: Int, end: Int, step: Int = 1, axis: Int = 0
    ) -> Gradbox[Self.dtype]:
        """Slice Gradbox along a single axis with start, end, and step.

        Args:
            start: Start index.
            end: End index.
            step: Step size (default: 1).
            axis: Axis to slice along (default: 0).

        Returns:
            A gradbox over the sliced region.
        """
        var shape, strides, offset = (
            Validator.validate_and_compute_slice_metadata(
                self.shape(), self.strides(), axis, start, end, step
            )
        )

        # Handle scalar (rank-0) case
        var is_scalar = len(shape) == 0
        var slice_shape = Shape() if is_scalar else shape
        var slice_strides = Strides() if is_scalar else strides
        var abs_offset = self.offset() + offset
        var shared_buffer = self.buffer().buffer.copy()
        var ndb = NDBuffer[Self.dtype](
            shared_buffer^,
            shape=slice_shape^,
            strides=slice_strides^,
            offset=abs_offset,
        )
        # Propagate device_state for GPU gradboxes
        # On GPU the CPU Buffer is empty — the actual data lives in DeviceState.
        # The sliced view must share the same DeviceState so get/set route
        # correctly to GPU memory using abs_offset and slice_strides.
        comptime if has_accelerator():
            ndb.device_state = self.buffer().device_state.copy()

        return Gradbox[Self.dtype](ndb^)

    @always_inline
    def __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        """Index the Gradbox with a List of integers.

        Args:
            indices: List of axis indices.

        Returns:
            The scalar value at the specified coordinates.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(List): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer()[indices]

    @always_inline
    def __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        """Index the Gradbox with an IntArray.

        Args:
            indices: IntArray of axis indices.

        Returns:
            The scalar value at the specified coordinates.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer()[indices]

    @always_inline
    def __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Index the Gradbox with variadic integer indices.

        Args:
            indices: One index per axis.

        Returns:
            The scalar value at the specified coordinates.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __getitem__([])"
            )

        return self.buffer()[indices]

    @always_inline
    def __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: List of axis indices.
            value: The value to write.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )

        self.buffer()[indices] = value

    @always_inline
    def __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: IntArray of axis indices.
            value: The value to write.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        self.buffer()[indices] = value

    @always_inline
    def __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: One index per axis.
            value: The value to write.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __setitem__([], value)"
            )
        self.buffer()[indices] = value

    @always_inline
    def load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous (stride[1] == 1) for SIMD loads.
            - `col + simdwidth` must not exceed the number of columns.
        """
        return self.buffer().load[simdwidth, validated](row, col)

    @always_inline
    def store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
        """SIMD store of a row segment into a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous for SIMD stores (stride[1] == 1).
            - Caller may set validated=True if these checks are already ensured.
        """
        self.buffer().store[simdwidth, validated](row, col, value)

    def item(self) -> Scalar[Self.dtype]:
        """Extract the scalar value from a scalar Gradbox.

        Returns:
            The scalar value.
        """
        return self.buffer().item()

    @always_inline
    def is_scalar(self) -> Bool:
        """Check if this is a scalar (0-dimensional) Gradbox.

        Returns:
            True if the Gradbox has rank 0.
        """
        return self.buffer().is_scalar()

    @always_inline
    def numels(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer().numels()

    @always_inline
    def num_elements(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer().numels()

    @always_inline
    def __len__(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer().numels()

    @always_inline
    def is_contiguous(self) -> Bool:
        """Check if memory layout is contiguous.

        Returns:
            True if stored in row-major order without gaps.
        """
        return self.buffer().is_contiguous()

    @always_inline
    def rank(self) -> Int:
        """Get the number of dimensions.

        Returns:
            The number of axes in the shape.
        """
        return self.buffer().rank()

    @always_inline
    def offset(self) -> Int:
        """Get the base memory offset.

        Returns:
            The offset into the underlying buffer.
        """
        return self.buffer().offset

    @always_inline
    def shape(self) -> Shape:
        """Get the shape of this Gradbox.

        Returns:
            The shape.
        """
        return self.buffer().shape

    @always_inline
    def strides(self) -> Strides:
        """Get the strides of this Gradbox.

        Returns:
            The strides.
        """
        return self.buffer().strides

    def get(self, index: Int) -> Scalar[Self.dtype]:
        """Get element at a logical flat index with bounds checking.

        C-order over this view: `get(i)` is the i-th logical element.

        Args:
            index: Logical flat index (< `numels()` after wrapping).

        Returns:
            The scalar value at that index.
        """
        return self.buffer().get(index)

    @always_inline
    def index_iterator(
        self,
    ) -> IndexIterator[
        origin_of(self.buffer().shape), origin_of(self.buffer().strides)
    ]:
        """Get an iterator over memory offsets.

        Returns:
            IndexIterator for the Gradbox.
        """
        return self.buffer().index_iterator()

    def __eq__(self, other: Gradbox[Self.dtype]) -> Bool:
        """Check equality with another Gradbox.

        Args:
            other: The Gradbox to compare with.

        Returns:
            True if all elements are equal.
        """
        if self.shape() != other.shape():
            panic(
                "Gradbox → __eq__(other): shape mismatch",
                String(self.shape()),
                "≠",
                String(other.shape()),
            )
        return self.buffer().compare[Equal](other.buffer()).buffer.all_true()

    def __ne__(self, other: Gradbox[Self.dtype]) -> Bool:
        """Check inequality with another Gradbox.

        Args:
            other: The Gradbox to compare with.

        Returns:
            True if any elements differ.
        """
        if self.shape() != other.shape():
            panic(
                "Gradbox → __ne__(other): shape mismatch",
                String(self.shape()),
                "≠",
                String(other.shape()),
            )
        return self.buffer().compare[NotEqual](other.buffer()).buffer.all_true()

    def to_dtype[NewType: DType](self) -> Gradbox[NewType]:
        """Convert Gradbox to a different data type.

        Returns:
            A new gradbox with the specified dtype.
        """
        var new_type_buffer = self.buffer().to_dtype[NewType]()
        return Gradbox[NewType](new_type_buffer^)

    def get_gpu(self) raises -> GPU:
        """Get the GPU this Gradbox is on.

        Returns:
            The GPU device.

        Raises:
            Error if the Gradbox is not on GPU.
        """
        if self.is_on_gpu():
            return self.buffer().get_gpu()
        raise "Gradbox get_gpu: gradbox is not on gpu"

    def is_on_gpu(self) -> Bool:
        """Check if this Gradbox is on a GPU.

        Returns:
            True if on GPU, False if on CPU.
        """
        return self.buffer().is_on_gpu()

    def is_on_cpu(self) -> Bool:
        """Check if this Gradbox is on CPU.

        Returns:
            True if on CPU, False if on GPU.
        """
        return self.is_on_gpu() == False

    def to_cpu(self) raises -> Self:
        """Transfer this Gradbox to CPU.

        Returns:
            A new Gradbox on CPU.

        Raises:
            Error if system has no accelerator.
        """
        comptime if has_accelerator():
            var (code, ndb) = self.buffer().to_device(CPU().into(), sync=False)
            if code == -1:
                return self
            return Gradbox[Self.dtype](ndb^)
        raise Error("System does not have any accelerator")

    def to_gpu(
        self,
        gpu: Optional[GPU] = None,
    ) raises -> Self:
        """Transfer this Gradbox to GPU.

        Args:
            gpu: The target GPU. If None, uses the default GPU.

        Returns:
            A new Gradbox on GPU.

        Raises:
            Error if system has no accelerator.
        """
        comptime if has_accelerator():
            var target = gpu.value().into() if gpu else GPU().into()
            var (code, ndb) = self.buffer().to_device(target, sync=False)
            if code == -1:
                return self
            return Gradbox[Self.dtype](ndb^)
        else:
            raise Error(
                "Can not move to GPU. System does not have any accelerator"
                " device"
            )

    def __str__(self) -> String:
        """Get a string representation of this Gradbox.

        Returns:
            A string showing the Gradbox type, shape, dtype, and device.
        """
        var rank = self.rank()
        var s = String("[")
        if rank == 1:
            s += "1D Gradbox"
        elif rank == 2:
            s += "2D Gradbox"
        elif rank == 3:
            s += "3D Gradbox"
        elif rank == 4:
            s += "4D Gradbox"
        elif rank == 5:
            s += "5D Gradbox"
        else:
            s += "Gradbox"
        s += String(self.shape())
        s += ", Type: " + String(Self.dtype)
        s += ", Strides : " + String(self.strides())
        s += ", Offset : " + String(self.offset())
        s += (
            ", Device : "
            + "gpu: "
            + String(
                self.buffer().gpu_id()
            ) if self.is_on_gpu() else ", Device : "
            + "cpu"
        )
        s += "]"
        return s

    def __repr__(self) -> String:
        """Get a string representation.

        Returns:
            Same as __str__().
        """
        return self.__str__()

    def write_to[W: Writer](self, mut writer: W):
        """Write the Gradbox to a writer.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.__str__())

    @always_inline
    def seed_grad(self, value: Scalar[Self.dtype]):
        """Seed gradients with a scalar value.

        Args:
            value: The value to fill the gradient with.
        """
        self.buffer().fill(value)

    @always_inline
    def seed_grad(self, with_tensor: Tensor[Self.dtype]):
        """Seed gradients from a tensor.

        Args:
            with_tensor: The tensor containing gradient values.
        """
        self.buffer().fill(with_tensor.buffer)

    @always_inline
    def zero_grad(self):
        """Zero out all gradients."""
        ref ndb = self.buffer()
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                ndb.zero()
                return
        ndb.buffer.zero()

    def fill(self, value: Scalar[Self.dtype], *indices: Idx):
        """Fill a region with a scalar value.

        Args:
            value: The value to write.
            indices: Idx objects defining the region.
        """
        Filler[Self.dtype].fill(self.buffer(), value, indices)

    def fill(self, tensor: Tensor[Self.dtype], *indices: Idx):
        """Fill a region with tensor data.

        Args:
            tensor: The tensor to copy from.
            indices: Idx objects defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer(), tensor.buffer, indices)

    def fill(self, gradbox: Gradbox[Self.dtype], *indices: Idx):
        """Fill a region with Gradbox data.

        Args:
            gradbox: The Gradbox to copy from.
            indices: Idx objects defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer(), gradbox.buffer(), indices)

    @always_inline
    def clamp_in_place(
        self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]
    ):
        """Clamp values in place to [lower_bound, upper_bound].

        Args:
            lower_bound: Minimum value.
            upper_bound: Maximum value.
        """
        self.buffer().clamp_in_place(lower_bound, upper_bound)

    def max(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise maximum with a scalar.

        Args:
            scalar: The scalar to compare with.

        Returns:
            A new Gradbox with max values.
        """
        return Gradbox[Self.dtype](self.buffer().scalar_ops[MAX](scalar))

    def min(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise minimum with a scalar.

        Args:
            scalar: The scalar to compare with.

        Returns:
            A new Gradbox with min values.
        """
        return Gradbox[Self.dtype](self.buffer().scalar_ops[MIN](scalar))

    def __mul__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Multiply by a scalar.

        Args:
            scalar: The scalar to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().scalar_ops[Multiply](scalar, sync=sync)
        )

    def __rmul__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right multiply by a scalar.

        Args:
            scalar: The scalar to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return self.__mul__[sync=sync](scalar)

    def __add__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Add a scalar.

        Args:
            scalar: The scalar to add.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().scalar_ops[Add](scalar, sync=sync)
        )

    def __radd__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right add a scalar.

        Args:
            scalar: The scalar to add.

        Returns:
            A new Gradbox with the result.
        """
        return self.__add__[sync=sync](scalar)

    def __sub__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Subtract a scalar.

        Args:
            scalar: The scalar to subtract.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().scalar_ops[Subtract](scalar, sync=sync)
        )

    def __rsub__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right subtract (scalar - self).

        Args:
            scalar: The scalar to subtract from.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().scalar_ops[ReverseSubtract](scalar, sync=sync)
        )

    def __truediv__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Divide by a scalar.

        Args:
            scalar: The scalar to divide by.

        Returns:
            A new Gradbox with the result.
        """
        if scalar == Scalar[Self.dtype](0):
            panic("Gradbox → __truediv__(scalar): can not divide by zero")
        return Gradbox[Self.dtype](
            self.buffer().scalar_ops[Divide](scalar, sync=sync)
        )

    def __rtruediv__[
        sync: Bool = False
    ](self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right divide (scalar / self).

        Args:
            scalar: The scalar to divide by self.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().scalar_ops[ReverseDivide](scalar, sync=sync)
        )

    def __mul__[sync: Bool = False](self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise multiply with another Gradbox.

        Args:
            other: The Gradbox to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Multiply](other.buffer(), sync=sync)
        )

    def __mul__[
        sync: Bool = False
    ](self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise multiply with a Tensor.

        Args:
            other: The Tensor to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Multiply](other.buffer, sync=sync)
        )

    def matmul(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        """Matrix multiplication.

        Args:
            B: Right operand.

        Returns:
            A new Gradbox with the matrix product.
        """
        var ndb = NDBuffer[Self.dtype].matmul_nd(A.buffer(), B.buffer)
        return Gradbox[Self.dtype](ndb^)

    def __add__[sync: Bool = False](self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise add another Gradbox.

        Args:
            other: The Gradbox to add.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Add](other.buffer(), sync=sync)
        )

    def __add__[
        sync: Bool = False
    ](self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise add a Tensor.

        Args:
            other: The Tensor to add.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Add](other.buffer, sync=sync)
        )

    def __sub__[sync: Bool = False](self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise subtract another Gradbox.

        Args:
            other: The Gradbox to subtract.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Subtract](other.buffer(), sync=sync)
        )

    def __sub__[
        sync: Bool = False
    ](self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise subtract a Tensor.

        Args:
            other: The Tensor to subtract.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Subtract](other.buffer, sync=sync)
        )

    def __truediv__[
        sync: Bool = False
    ](self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise divide by another Gradbox.

        Args:
            other: The Gradbox to divide by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer().arithmetic_ops[Divide](other.buffer(), sync=sync)
        )

    def __imul__[sync: Bool = False](self, scalar: Scalar[Self.dtype]):
        """In-place multiply by a scalar.

        Args:
            scalar: The scalar to multiply by.
        """
        self.buffer().inplace_scalar_ops[Multiply](scalar, sync=sync)

    def __iadd__[sync: Bool = False](self, scalar: Scalar[Self.dtype]):
        """In-place add a scalar.

        Args:
            scalar: The scalar to add.
        """
        self.buffer().inplace_scalar_ops[Add](scalar, sync=sync)

    def __isub__[sync: Bool = False](self, scalar: Scalar[Self.dtype]):
        """In-place subtract a scalar.

        Args:
            scalar: The scalar to subtract.
        """
        self.buffer().inplace_scalar_ops[Subtract](scalar, sync=sync)

    def __itruediv__[sync: Bool = False](self, scalar: Scalar[Self.dtype]):
        """In-place divide by a scalar.

        Args:
            scalar: The scalar to divide by.
        """
        self.buffer().inplace_scalar_ops[Divide](scalar, sync=sync)

    @always_inline
    def __imul__[sync: Bool = False](self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise multiply.

        Args:
            incoming: The Gradbox to multiply by.
        """
        self.buffer().inplace_ops[Multiply](incoming.buffer(), sync=sync)

    @always_inline
    def __iadd__[sync: Bool = False](self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise add.

        Args:
            incoming: The Gradbox to add.
        """
        self.buffer().inplace_ops[Add](incoming.buffer(), sync=sync)

    @always_inline
    def __isub__[sync: Bool = False](self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise subtract.

        Args:
            incoming: The Gradbox to subtract.
        """
        self.buffer().inplace_ops[Subtract](incoming.buffer(), sync=sync)

    @always_inline
    def __itruediv__[sync: Bool = False](self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise divide.

        Args:
            incoming: The Gradbox to divide by.
        """
        self.buffer().inplace_ops[Divide](incoming.buffer(), sync=sync)

    def all_close[
        rtol: Scalar[Self.dtype] = Scalar[Self.dtype](1e-5),
        atol: Scalar[Self.dtype] = Scalar[Self.dtype](1e-8) if Self.dtype
        == DType.float64 else Scalar[Self.dtype](1e-5),
    ](self, other: Self) -> Bool:
        """Check if all elements are close to another Gradbox.

        Args:
            other: The Gradbox to compare with.

        Returns:
            True if all elements are within tolerance.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "Gradbox → all_close(Self): is for floating point data types only"
        if self.shape() != other.shape():
            panic(
                "Gradbox → all_close(Self): expects same shaped gradboxes: "
                + String(self.shape())
                + ", "
                + String(other.shape())
            )

        return self.buffer().all_close[rtol=rtol, atol=atol](other.buffer())

    def all_close[
        rtol: Scalar[Self.dtype] = CloseTol[Self.dtype].rtol(),
        atol: Scalar[Self.dtype] = CloseTol[Self.dtype].atol(),
    ](self, other: Tensor[Self.dtype]) -> Bool:
        """Check if all elements are close to a Tensor.

        Args:
            other: The Tensor to compare with.

        Returns:
            True if all elements are within tolerance.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "Gradbox → all_close(Tensor): is for floating point data types only"
        if self.shape() != other.shape():
            panic(
                "Gradbox → all_close(Tensor): expects same shaped tensor: "
                + String(self.shape())
                + ", "
                + String(other.shape())
            )

        return self.buffer().all_close[rtol=rtol, atol=atol](other.buffer)

    @always_inline
    def reshape(self) -> Gradbox[Self.dtype]:
        """Reshape to a scalar Gradbox.

        Returns:
            A scalar Gradbox.
        """
        if self.numels() != 1:
            panic(
                "Gradbox → reshape: only gradbox with single element can be"
                " reshaped to scalar gradbox"
            )
        return self.reshape(Shape(), validated=True)

    @always_inline
    def reshape(
        self, new_shape: Shape, validated: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Reshape to a new shape.

        Args:
            new_shape: The target shape.
            validated: If True, skip validation (caller guarantees validity).

        Returns:
            A new contiguous Gradbox with the reshaped buffer.

            NOTE: reshape forwards NDBuffer.reshape as-is — a view when the
            source is contiguous (torch-like write-through; see
            test_gradbox_reshape), freshly materialised otherwise (GPU and
            non-contiguous / unshared sources). This is the documented
            exception to the Gradbox independence rule. Need a decoupled
            buffer? reshape then clone().
        """
        var nd_buffer = self.buffer().reshape(new_shape, validated)
        return Gradbox[Self.dtype](nd_buffer^)

    def __eq__(self, tensor: Tensor[Self.dtype]) -> Bool:
        """Check equality with a Tensor.

        Args:
            tensor: The Tensor to compare with.

        Returns:
            True if all elements are equal.
        """
        if self.shape() != tensor.shape():
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                String(self.shape()),
                ",",
                String(tensor.shape()),
            )
        return self.buffer().compare[Equal](tensor.buffer).buffer.all_true()

    def print(self, num_first: Int = 10, num_last: Int = 10) raises:
        """Print the Gradbox contents.

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
            self.buffer(),
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    def data_ptr(ref self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Get a pointer to the data.

        Returns:
            Pointer to the underlying data.
        """
        return self.buffer().data_ptr()
