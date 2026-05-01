from .shapes import Shape
from .mnemonics import *
from .validators import Validator
from .tensor import Tensor
from .intarray import IntArray
from .ndbuffer import NDBuffer
from .broadcasthelper import ShapeBroadcaster
from .strides import Strides
from std.sys import simd_width_of, has_accelerator
from .matmul import Matmul
from std.random import seed, random_float64
from .buffers import Buffer
from .forwards import Mean, Sqrt
from .indexhelper import IndexIterator
from .filler import Filler
from .common_utils import Idx, panic, print_buffer
from .device import Device, CPU, GPU
from .device_transfer import DeviceTransfer
from std.os.atomic import Atomic, Consistency, fence


struct Gradbox[dtype: DType](
    ImplicitlyCopyable & Movable & Sized & Writable & Equatable & Absable
):
    """Gradient storage container with reference counting for efficient gradient accumulation.

    Gradbox wraps an NDBuffer with reference-counted ownership, optimized for gradient
    computation in the autograd system. It provides arithmetic and tensor operations
    that are used during backward pass for gradient accumulation.

    Example:
    ```mojo
    var gradbox = Gradbox[DType.float32].zeros(Shape(3, 4))
    gradbox.zero_grad()
    ```
    """

    var buffer: NDBuffer[Self.dtype]
    var _refcount: UnsafePointer[Atomic[DType.uint64], MutExternalOrigin]
    comptime Empty = Gradbox[Self.dtype].zeros(Shape())

    fn __init__(out self, shape: Shape, share: Bool = True):
        """Initialize a Gradbox with the given shape.

        Args:
            shape: The tensor shape.
            share: If True, share the underlying buffer. If False, create own copy.
        """
        var ndb = NDBuffer[Self.dtype](shape)
        if share:
            self.buffer = ndb.share()
            _ = ndb^
        else:
            self.buffer = ndb^
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)

    fn __init__(out self, var buffer: NDBuffer[Self.dtype], share: Bool = True):
        """Initialize a Gradbox from an existing NDBuffer.

        Args:
            buffer: The NDBuffer to wrap.
            share: If True, share the buffer. If False, take ownership.
        """
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.buffer = buffer.share() if share else buffer^

    fn __moveinit__(out self, deinit take: Self):
        """Move constructor — transfer ownership without copying."""
        self._refcount = take._refcount
        self.buffer = take.buffer^

    fn __copyinit__(out self, copy: Self):
        """Copy constructor — increments reference count."""
        self.buffer = copy.buffer.copy()
        self._refcount = copy._refcount
        _ = self._refcount[].fetch_add[ordering=Consistency.MONOTONIC](1)

    fn __del__(deinit self):
        """Destructor — decrements refcount and frees if last owner."""
        if not self._refcount:
            return
        if self._refcount[].fetch_sub[ordering=Consistency.RELEASE](1) != 1:
            return  # other owners exist — do nothing
        fence[ordering=Consistency.ACQUIRE]()
        # Last owner — free refcount allocation
        # buffer.__del__ handles its own cleanup via NDBuffer/Buffer refcount
        self._refcount.destroy_pointee()
        self._refcount.free()

    fn ref_count(self) -> UInt64:
        """Get the current reference count.

        Returns:
            Number of references to this Gradbox.
        """
        return self._refcount[].load[ordering=Consistency.MONOTONIC]()

    fn is_shared(self) -> Bool:
        """Check if this Gradbox is shared (ref_count > 1).

        Returns:
            True if shared by multiple owners.
        """
        return self.ref_count() > 1

    @always_inline
    fn as_tensor(
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
                self^.buffer^, requires_grad=requires_grad
            )
        else:
            return Tensor[Self.dtype](
                self^.buffer.contiguous(), requires_grad=requires_grad
            )

    fn device(self) -> Device:
        """Get the device this Gradbox is on.

        Returns:
            The CPU or GPU device.
        """
        return self.buffer.device()

    fn transpose(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Transpose the Gradbox along the given axes.

        Args:
            axes: The permutation of axes.

        Returns:
            A new contiguous Gradbox with transposed axes.
        """
        var owned_buffer = self.buffer.copy()
        var nd_buffer = owned_buffer.transpose(axes, shared=False)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn __abs__(self) -> Gradbox[Self.dtype]:
        """Compute element-wise absolute value.

        Returns:
            A new Gradbox with absolute values.
        """
        return Gradbox[Self.dtype](self.buffer.__abs__(), share=False)

    fn sqrt(
        self,
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
    ) -> Gradbox[Self.dtype]:
        """Compute element-wise square root.

        Args:
            epsilon: Small value to prevent sqrt of negative numbers.

        Returns:
            A new Gradbox with square roots.
        """
        return Sqrt[Self.dtype].forward(self, epsilon)

    fn norm(
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

        Raises:
            Panic if p is not 2.0.
        """
        if p == 2.0:
            # L2 norm: sqrt(sum(x²))
            var squared = self.__mul__(self)
            var dim = IntArray(axis.value()) if axis else IntArray()
            var sum_sq = squared.sum(dim, keepdims=keepdims)
            return sum_sq.sqrt()
        else:
            panic("Only L2 norm (p=2) currently supported")
            return Gradbox[Self.dtype](Shape(), share=False)

    @staticmethod
    fn arange(
        *args: Scalar[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        """Create a 1D Gradbox with evenly spaced values.

        Args:
            *args: Start, stop, and optionally step values.

        Returns:
            A 1D Gradbox with values from start to stop.
        """
        nd_buffer = NDBuffer[Self.dtype].arange(args)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn flatten(
        self, start_dim: Int = 0, end_dim: Optional[Int] = None
    ) -> Gradbox[Self.dtype]:
        """Flatten the Gradbox to 1D.

        Args:
            start_dim: The first dimension to flatten.
            end_dim: The last dimension to flatten. If None, flattens to the end.

        Returns:
            A flattened Gradbox.
        """
        flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        return Gradbox[Self.dtype](flattened_buffer^, share=False)

    fn squeeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Remove dimensions of size 1.

        Args:
            axes: The axes to squeeze.

        Returns:
            A new Gradbox with squeezed dimensions.
        """
        var buffer = self.buffer.copy()
        var ndb = buffer.squeeze(axes, shared=False)
        return Gradbox[Self.dtype](ndb^, share=False)

    fn squeeze(self, axes: List[Int] = []) -> Gradbox[Self.dtype]:
        """Remove dimensions of size 1.

        Args:
            axes: The axes to squeeze as a list.

        Returns:
            A new Gradbox with squeezed dimensions.
        """
        return self.squeeze(IntArray(axes))

    fn unsqueeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Add dimensions of size 1.

        Args:
            axes: The axes to unsqueeze.

        Returns:
            A new Gradbox with unsqueezed dimensions.
        """
        var buffer = self.buffer.copy()
        var ndb = buffer.unsqueeze(axes, shared=False)
        return Gradbox[Self.dtype](ndb^, share=False)

    fn unsqueeze(self, axes: List[Int]) -> Gradbox[Self.dtype]:
        """Add dimensions of size 1.

        Args:
            axes: The axes to unsqueeze as a list.

        Returns:
            A new Gradbox with unsqueezed dimensions.
        """
        return self.unsqueeze(IntArray(axes))

    fn permute(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Permute the axes of this Gradbox.

        Args:
            axes: The new order of axes.

        Returns:
            A new owned contiguous Gradbox with permuted axes.
        """
        var self_buffer = self.buffer.copy()
        var result_ndb = self_buffer.permute(axes, shared=False)
        return Gradbox[Self.dtype](result_ndb^, share=False)

    @staticmethod
    @always_inline
    fn full(
        shape: Shape,
        scalar: Scalar[Self.dtype],
        share: Bool = False,
        device: Device = CPU().into(),
    ) -> Gradbox[Self.dtype]:
        """Create a Gradbox filled with a scalar value.

        Args:
            shape: The tensor shape.
            scalar: The value to fill with.
            share: If True, share the buffer. If False, create owned copy.
            device: The target device (CPU or GPU).

        Returns:
            A new Gradbox filled with the scalar value.
        """
        return Gradbox[Self.dtype](
            NDBuffer.full(shape, scalar, device), share=share
        )

    @staticmethod
    @always_inline
    fn zeros(
        shape: Shape, share: Bool = False, device: Device = CPU().into()
    ) -> Gradbox[Self.dtype]:
        """Create a Gradbox of zeros.

        Args:
            shape: The tensor shape.
            share: If True, share the buffer. If False, create owned copy.
            device: The target device (CPU or GPU).

        Returns:
            A new Gradbox filled with zeros.
        """
        return Gradbox[Self.dtype](
            NDBuffer.full(shape, Scalar[Self.dtype](0), device=device),
            share=share,
        )

    @staticmethod
    fn rand(
        shape: Shape,
        min: Scalar[Self.dtype] = 0,
        max: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        share: Bool = False,
    ) -> Gradbox[Self.dtype]:
        """Create a Gradbox with uniform random values.

        Args:
            shape: The tensor shape.
            min: Lower bound (inclusive).
            max: Upper bound (exclusive).
            init_seed: Random seed. If None, randomizes each call.
            requires_grad: Whether to track gradients (unused, for API compat).
            share: If True, share the buffer. If False, create owned copy.

        Returns:
            A new Gradbox with random values in [min, max).
        """
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        numels = shape.num_elements()
        buffer = Buffer[Self.dtype](numels)
        for i in range(numels):
            buffer[i] = random_float64(
                min.cast[Self.dtype.float64](), max.cast[DType.float64]()
            ).cast[Self.dtype]()

        return Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape), share=share
        )

    @always_inline
    fn detach(self, share: Bool = False) -> Gradbox[Self.dtype]:
        """Create a detached (contiguous) copy of this Gradbox.

        Args:
            share: If True, share the buffer. If False, create owned copy.

        Returns:
            A new contiguous Gradbox.
        """
        return Gradbox[Self.dtype](self.buffer.contiguous(), share=share)

    fn shared(self) -> Bool:
        """Check if the underlying buffer is shared.

        Returns:
            True if the buffer is shared.
        """
        return self.buffer.shared()

    @always_inline
    fn sum(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Compute the sum along the given axes.

        Args:
            axes: The axes to reduce. If empty, reduces all dimensions.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            A new Gradbox with the sum.
        """
        var nd_buffer = self.buffer.reduce(
            normalized_axes=axes, keepdims=keepdims
        )
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    @always_inline
    fn mean(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Compute the mean along the given axes.

        Args:
            axes: The axes to reduce. If empty, reduces all dimensions.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            A new Gradbox with the mean.
        """
        return Mean[Self.dtype].forward(self, axes=axes, keepdims=keepdims)

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_grad: Gradbox[Self.dtype], target_shape: Shape
    ) -> Gradbox[Self.dtype]:
        """Sum over broadcasted axes to match target shape.

        Args:
            extended_grad: The gradient that was broadcasted.
            target_shape: The shape to expand to.

        Returns:
            A new Gradbox summed to the target shape.
        """
        var nd_buffer = extended_grad.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn broadcast_to(
        self, target_shape: Shape, share: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Broadcast this Gradbox to the target shape.

        Args:
            target_shape: The shape to broadcast to.
            share: If True, share the buffer. If False, create owned copy.

        Returns:
            A new Gradbox with the target shape.

        Raises:
            Panic if the shapes are not broadcastable.
        """
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Gradbox → broadcast_to: shape "
                + String(self.shape())
                + " not broadcastable to "
                + String(target_shape)
            )

        var broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        var out = Gradbox[Self.dtype](broadcasted_buffer^, share=share)
        return out^

    fn __getitem__(self, *indices: Idx) -> Gradbox[Self.dtype]:
        """Index the Gradbox with Idx objects (integers or slices).

        Args:
            *indices: One index per axis — either an integer or a Slice.

        Returns:
            A new Gradbox view over the indexed region.

        Raises:
            Panic if called on an unshared Gradbox.
        """
        if not self.shared():
            panic(
                "Gradbox -> __getitem__(self, *indices: Idx): can not call on"
                " an unshared gradbox"
            )
        # Compute view metadata
        view_shape, view_strides, relative_offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape(), self.strides(), indices
            )
        )

        # Handle scalar (rank-0) case
        is_scalar = len(view_shape) == 0
        shape = Shape() if is_scalar else view_shape
        strides = Strides() if is_scalar else view_strides
        abs_offset = self.offset() + relative_offset
        shared_buffer = self.buffer.buffer.copy()
        ndb = NDBuffer[Self.dtype](
            shared_buffer^, shape=shape^, strides=strides^, offset=abs_offset
        )

        return Gradbox[Self.dtype](ndb^, share=False)

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        """Index the Gradbox with a List of integers.

        Args:
            indices: List of axis indices.

        Returns:
            The scalar value at the specified coordinates.

        Raises:
            Panic if called on a scalar Gradbox with non-empty indices.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(List): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        """Index the Gradbox with an IntArray.

        Args:
            indices: IntArray of axis indices.

        Returns:
            The scalar value at the specified coordinates.

        Raises:
            Panic if called on a scalar Gradbox with non-empty indices.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Index the Gradbox with variadic integer indices.

        Args:
            *indices: One index per axis.

        Returns:
            The scalar value at the specified coordinates.

        Raises:
            Panic if called on a scalar Gradbox.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __getitem__([])"
            )

        return self.buffer[indices]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: List of axis indices.
            value: The value to write.

        Raises:
            Panic if called on a scalar Gradbox with non-empty indices.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )

        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            indices: IntArray of axis indices.
            value: The value to write.

        Raises:
            Panic if called on a scalar Gradbox with non-empty indices.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
        """Set a scalar value at given coordinates.

        Args:
            *indices: One index per axis.
            value: The value to write.

        Raises:
            Panic if called on a scalar Gradbox with non-empty indices.
        """
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __setitem__([], value)"
            )
        self.buffer[indices] = value

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous (stride[1] == 1) for SIMD loads.
            - `col + simdwidth` must not exceed the number of columns.
        """
        return self.buffer.load[simdwidth, validated](row, col)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
        """SIMD store of a row segment into a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous for SIMD stores (stride[1] == 1).
            - Caller may set validated=True if these checks are already ensured.
        """
        self.buffer.store[simdwidth, validated](row, col, value)

    fn item(self) -> Scalar[Self.dtype]:
        """Extract the scalar value from a scalar Gradbox.

        Returns:
            The scalar value.

        Raises:
            Panic if the Gradbox is not a scalar.
        """
        return self.buffer.item()

    @always_inline
    fn is_scalar(self) -> Bool:
        """Check if this is a scalar (0-dimensional) Gradbox.

        Returns:
            True if the Gradbox has rank 0.
        """
        return self.buffer.is_scalar()

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
    fn __len__(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        return self.buffer.numels()

    @always_inline
    fn is_contiguous(self) -> Bool:
        """Check if memory layout is contiguous.

        Returns:
            True if stored in row-major order without gaps.
        """
        return self.strides().is_contiguous(self.shape())

    @always_inline
    fn rank(self) -> Int:
        """Get the number of dimensions.

        Returns:
            The number of axes in the shape.
        """
        return self.buffer.rank()

    @always_inline
    fn offset(self) -> Int:
        """Get the base memory offset.

        Returns:
            The offset into the underlying buffer.
        """
        return self.buffer.offset

    @always_inline
    fn shape(ref self) -> ref[self.buffer.shape] Shape:
        """Get the shape of this Gradbox.

        Returns:
            Reference to the shape.
        """
        return self.buffer.shape

    @always_inline
    fn strides(ref self) -> ref[self.buffer.strides] Strides:
        """Get the strides of this Gradbox.

        Returns:
            Reference to the strides.
        """
        return self.buffer.strides

    fn data_buffer(ref self) -> ref[self.buffer] NDBuffer[Self.dtype]:
        """Get the underlying NDBuffer.

        Returns:
            Reference to the NDBuffer.
        """
        return self.buffer

    fn get(self, index: Int) -> Scalar[Self.dtype]:
        """Get element at a flat index with bounds checking.

        Args:
            index: Flat (linear) index into the gradbox's memory.

        Returns:
            The scalar value at that index.

        Raises:
            Panic if index is out of bounds.
        """
        return self.buffer.get(index)

    @always_inline
    fn index_iterator(
        self,
    ) -> IndexIterator[
        origin_of(self.buffer.shape), origin_of(self.buffer.strides)
    ]:
        """Get an iterator over memory offsets.

        Returns:
            IndexIterator for the Gradbox.
        """
        return self.buffer.index_iterator()

    fn __eq__(self, other: Gradbox[Self.dtype]) -> Bool:
        """Check equality with another Gradbox.

        Args:
            other: The Gradbox to compare with.

        Returns:
            True if all elements are equal.

        Raises:
            Panic if shapes don't match.
        """
        if self.shape() != other.shape():
            panic(
                "Gradbox → __eq__(other): shape mismatch",
                String(self.shape()),
                "≠",
                String(other.shape()),
            )
        return self.buffer.compare[Equal](other.buffer).buffer.all_true()

    fn __ne__(self, other: Gradbox[Self.dtype]) -> Bool:
        """Check inequality with another Gradbox.

        Args:
            other: The Gradbox to compare with.

        Returns:
            True if any elements differ.

        Raises:
            Panic if shapes don't match.
        """
        if self.shape() != other.shape():
            panic(
                "Gradbox → __ne__(other): shape mismatch",
                String(self.shape()),
                "≠",
                String(other.shape()),
            )
        return self.buffer.compare[NotEqual](other.buffer).buffer.all_true()

    fn get_gpu(self) raises -> GPU:
        """Get the GPU this Gradbox is on.

        Returns:
            The GPU device.

        Raises:
            Error if the Gradbox is not on GPU.
        """
        if self.is_on_gpu():
            return self.buffer.get_gpu()
        raise "Gradbox get_gpu: gradbox is not on gpu"

    fn is_on_gpu(self) -> Bool:
        """Check if this Gradbox is on a GPU.

        Returns:
            True if on GPU, False if on CPU.
        """
        return self.buffer.is_on_gpu()

    fn is_on_cpu(self) -> Bool:
        """Check if this Gradbox is on CPU.

        Returns:
            True if on CPU, False if on GPU.
        """
        return self.is_on_gpu() == False

    fn to_cpu(self) raises -> Self:
        """Transfer this Gradbox to CPU.

        Returns:
            A new Gradbox on CPU.

        Raises:
            Error if system has no accelerator.
        """
        comptime if has_accelerator():
            return DeviceTransfer[Self.dtype].forward(self, CPU().into())
        raise Error("System does not have any accelerator")

    fn to_gpu(
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
            if gpu:
                return DeviceTransfer[Self.dtype].forward(
                    self, gpu.value().into()
                )
            else:
                return DeviceTransfer[Self.dtype].forward(self, GPU().into())
        else:
            raise Error(
                "Can not move to GPU. System does not have any accelerator"
                " device"
            )

    fn __str__(self) -> String:
        """Get a string representation of this Gradbox.

        Returns:
            A string showing the Gradbox type, shape, dtype, and device.
        """
        rank = self.rank()
        s = String("[")
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
        s += ", Shared : " + String(self.shared())
        s += ", Strides : " + String(self.strides())
        s += ", Offset : " + String(self.offset())
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

    fn __repr__(self) -> String:
        """Get a string representation.

        Returns:
            Same as __str__().
        """
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        """Write the Gradbox to a writer.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.__str__())

    @always_inline
    fn seed_grad(self, value: Scalar[Self.dtype]):
        """Seed gradients with a scalar value.

        Args:
            value: The value to fill the gradient with.
        """
        self.buffer.fill(value)

    @always_inline
    fn seed_grad(self, with_tensor: Tensor[Self.dtype]):
        """Seed gradients from a tensor.

        Args:
            with_tensor: The tensor containing gradient values.
        """
        self.buffer.fill(with_tensor.buffer)

    @always_inline
    fn zero_grad(self):
        """Zero out all gradients."""
        self.buffer.zero()

    fn fill(self, value: Scalar[Self.dtype], *indices: Idx):
        """Fill a region with a scalar value.

        Args:
            value: The value to write.
            *indices: Idx objects defining the region.
        """
        Filler[Self.dtype].fill(self.buffer, value, indices)

    fn fill(self, tensor: Tensor[Self.dtype], *indices: Idx):
        """Fill a region with tensor data.

        Args:
            tensor: The tensor to copy from.
            *indices: Idx objects defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer, tensor.buffer, indices)

    fn fill(self, gradbox: Gradbox[Self.dtype], *indices: Idx):
        """Fill a region with Gradbox data.

        Args:
            gradbox: The Gradbox to copy from.
            *indices: Idx objects defining the destination region.
        """
        Filler[Self.dtype].fill(self.buffer, gradbox.buffer, indices)

    @always_inline
    fn clamp_in_place(
        self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]
    ):
        """Clamp values in place to [lower_bound, upper_bound].

        Args:
            lower_bound: Minimum value.
            upper_bound: Maximum value.
        """
        self.buffer.clamp_in_place(lower_bound, upper_bound)

    fn max(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise maximum with a scalar.

        Args:
            scalar: The scalar to compare with.

        Returns:
            A new Gradbox with max values.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[MAX](scalar), share=False
        )

    fn min(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise minimum with a scalar.

        Args:
            scalar: The scalar to compare with.

        Returns:
            A new Gradbox with min values.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[MIN](scalar), share=False
        )

    fn __mul__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Multiply by a scalar.

        Args:
            scalar: The scalar to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Multiply](scalar), share=False
        )

    fn __rmul__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right multiply by a scalar.

        Args:
            scalar: The scalar to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return self.__mul__(scalar)

    fn __add__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Add a scalar.

        Args:
            scalar: The scalar to add.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Add](scalar), share=False
        )

    fn __radd__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right add a scalar.

        Args:
            scalar: The scalar to add.

        Returns:
            A new Gradbox with the result.
        """
        return self.__add__(scalar)

    fn __sub__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Subtract a scalar.

        Args:
            scalar: The scalar to subtract.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar), share=False
        )

    fn __rsub__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right subtract (scalar - self).

        Args:
            scalar: The scalar to subtract from.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), share=False
        )

    fn __truediv__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Divide by a scalar.

        Args:
            scalar: The scalar to divide by.

        Returns:
            A new Gradbox with the result.

        Raises:
            Panic if scalar is zero.
        """
        if scalar == Scalar[Self.dtype](0):
            panic("Gradbox → __truediv__(scalar): can not divide by zero")
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Divide](scalar), share=False
        )

    fn __rtruediv__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        """Right divide (scalar / self).

        Args:
            scalar: The scalar to divide by self.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), share=False
        )

    fn __mul__(self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise multiply with another Gradbox.

        Args:
            other: The Gradbox to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn __mul__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise multiply with a Tensor.

        Args:
            other: The Tensor to multiply by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn matmul(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        """Matrix multiplication.

        Args:
            A: Left operand.
            B: Right operand.

        Returns:
            A new Gradbox with the matrix product.
        """
        return Matmul[Self.dtype].forward(A, B)

    fn __add__(self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise add another Gradbox.

        Args:
            other: The Gradbox to add.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __add__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise add a Tensor.

        Args:
            other: The Tensor to add.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __sub__(self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise subtract another Gradbox.

        Args:
            other: The Gradbox to subtract.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __sub__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        """Element-wise subtract a Tensor.

        Args:
            other: The Tensor to subtract.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __truediv__(self, other: Self) -> Gradbox[Self.dtype]:
        """Element-wise divide by another Gradbox.

        Args:
            other: The Gradbox to divide by.

        Returns:
            A new Gradbox with the result.
        """
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer), share=False
        )

    fn __imul__(self, scalar: Scalar[Self.dtype]):
        """In-place multiply by a scalar.

        Args:
            scalar: The scalar to multiply by.
        """
        self.buffer.inplace_scalar_ops[Multiply](scalar)

    fn __iadd__(self, scalar: Scalar[Self.dtype]):
        """In-place add a scalar.

        Args:
            scalar: The scalar to add.
        """
        self.buffer.inplace_scalar_ops[Add](scalar)

    fn __isub__(self, scalar: Scalar[Self.dtype]):
        """In-place subtract a scalar.

        Args:
            scalar: The scalar to subtract.
        """
        self.buffer.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[Self.dtype]):
        """In-place divide by a scalar.

        Args:
            scalar: The scalar to divide by.
        """
        self.buffer.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise multiply.

        Args:
            incoming: The Gradbox to multiply by.
        """
        self.buffer.inplace_ops[Multiply](incoming.buffer)

    @always_inline
    fn __iadd__(self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise add.

        Args:
            incoming: The Gradbox to add.
        """
        self.buffer.inplace_ops[Add](incoming.buffer)

    @always_inline
    fn __isub__(self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise subtract.

        Args:
            incoming: The Gradbox to subtract.
        """
        self.buffer.inplace_ops[Subtract](incoming.buffer)

    @always_inline
    fn __itruediv__(self, incoming: Gradbox[Self.dtype]):
        """In-place element-wise divide.

        Args:
            incoming: The Gradbox to divide by.
        """
        self.buffer.inplace_ops[Divide](incoming.buffer)

    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        """Check if all elements are close to another Gradbox.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            other: The Gradbox to compare with.

        Returns:
            True if all elements are within tolerance.

        Raises:
            Panic if shapes differ or dtype is not floating point.
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

        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Tensor[Self.dtype]) -> Bool:
        """Check if all elements are close to a Tensor.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            other: The Tensor to compare with.

        Returns:
            True if all elements are within tolerance.

        Raises:
            Panic if shapes differ or dtype is not floating point.
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

        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    @always_inline
    fn reshape(self, share: Bool = False) -> Gradbox[Self.dtype]:
        """Reshape to a scalar Gradbox.

        Args:
            share: If True, share the buffer. If False, create owned copy.

        Returns:
            A scalar Gradbox.

        Raises:
            Panic if the Gradbox has more than one element.
        """
        if self.numels() != 1:
            panic(
                "Gradbox → reshape: only gradbox with single element can be"
                " reshaped to scalar gradbox"
            )
        return self.reshape(Shape(), validated=True, share=share)

    @always_inline
    fn reshape(
        self, new_shape: Shape, validated: Bool = False, share: Bool = False
    ) -> Gradbox[Self.dtype]:
        """Reshape to a new shape.

        Args:
            new_shape: The target shape.
            validated: If True, skip validation (caller guarantees validity).
            share: If True, share the buffer. If False, create owned copy.

        Returns:
            A new Gradbox with the reshaped buffer.
        """
        var nd_buffer = self.buffer.reshape(new_shape, validated)

        return Gradbox[Self.dtype](nd_buffer^, share=share)

    fn __eq__(self, tensor: Tensor[Self.dtype]) -> Bool:
        """Check equality with a Tensor.

        Args:
            tensor: The Tensor to compare with.

        Returns:
            True if all elements are equal.

        Raises:
            Panic if shapes differ.
        """
        if self.shape() != tensor.shape():
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                String(self.shape()),
                ",",
                String(tensor.shape()),
            )
        return self.buffer.compare[Equal](tensor.buffer).buffer.all_true()

    fn print(self, num_first: Int = 10, num_last: Int = 10):
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
        empty = List[Int]()
        print_buffer[Self.dtype](
            self.buffer,
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn data_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Scalar[Self.dtype], origin, address_space=address_space
    ]:
        """Get a pointer to the data.

        Returns:
            Pointer to the underlying data.
        """
        return (
            self.buffer.data_ptr()
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )
