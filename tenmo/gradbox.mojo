### Mojo Tensor Gradbox
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
from .utilities import Utils
from .indexhelper import IndexIterator
from .filler import Filler
from .common_utils import Idx, panic, print_buffer
from .device import Device, CPU, GPU
from .device_transfer import DeviceTransfer
from std.os.atomic import Atomic, Consistency, fence

struct Gradbox[dtype: DType](
    ImplicitlyCopyable
    & Movable
    & Sized
    & Writable
    & Equatable
    & Absable
):
    var buffer: NDBuffer[Self.dtype]
    var _refcount: UnsafePointer[Atomic[DType.uint64], MutExternalOrigin]
    comptime Empty = Gradbox[Self.dtype].zeros(Shape())

    fn __init__(out self, shape: Shape, share: Bool = True):
        var ndb = NDBuffer[Self.dtype](shape)
        if share:
            self.buffer = ndb.share()
            _= ndb^
        else:
            self.buffer = ndb^
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)

        _="""fn __init__1(out self: Self, shape: Shape, share: Bool = True):
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        buffer = NDBuffer[Self.dtype](shape)
        self.buffer = buffer.share() if share else buffer^"""

    fn __init__(out self, var buffer: NDBuffer[Self.dtype], share: Bool = True):
        self._refcount = alloc[Atomic[DType.uint64]](1)
        self._refcount[] = Atomic[DType.uint64](1)
        self.buffer = buffer.share() if share else buffer^


    fn __moveinit__(out self, deinit take: Self):
        self._refcount = take._refcount
        self.buffer = take.buffer^

    fn __copyinit__(out self, copy: Self):
        self.buffer = copy.buffer.copy()
        self._refcount = copy._refcount
        _ = self._refcount[].fetch_add[ordering=Consistency.MONOTONIC](1)


    fn __del__(deinit self):
        if not self._refcount:
            return
        if self._refcount[].fetch_sub[ordering=Consistency.RELEASE](1) != 1:
            return  # other owners exist — do nothing
        fence[ordering=Consistency.ACQUIRE]()
        # Last owner — free refcount allocation
        # buffer.__del__ handles its own cleanup via NDBuffer/Buffer refcount
        self._refcount.destroy_pointee()
        self._refcount.free()

    # ========================================
    # Refcount inspection
    # ========================================

    fn ref_count(self) -> UInt64:
        return self._refcount[].load[ordering=Consistency.MONOTONIC]()

    fn is_shared(self) -> Bool:
        return self.ref_count() > 1

    @always_inline
    fn as_tensor(
        deinit self, requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        if self.is_contiguous():
            return Tensor[Self.dtype](
                self^.buffer^, requires_grad=requires_grad
            )
        else:
            return Tensor[Self.dtype](
                self^.buffer.contiguous(), requires_grad=requires_grad
            )

    fn device(self) -> Device:
        return self.buffer.device()

    fn transpose(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Fused transpose then contiguous - GPU aware via NDBuffer."""
        var owned_buffer = self.buffer.copy()
        var nd_buffer = owned_buffer.transpose(axes, shared=False)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn __abs__(self) -> Gradbox[Self.dtype]:
        var buffer = self.buffer.map[
            Utils[Self.dtype].abs_buffer, Utils[Self.dtype].abs_scalar
        ]()
        var nd_buffer = NDBuffer[Self.dtype](buffer^, self.buffer.shape)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn sqrt(
        self,
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
    ) -> Gradbox[Self.dtype]:
        return Sqrt[Self.dtype].forward(self, epsilon)

    fn norm(
        self,
        p: Float64 = 2.0,
        axis: Optional[Int] = None,
        keepdims: Bool = False,
    ) -> Gradbox[Self.dtype]:
        """Compute Lp norm. Supports L2 (p=2) only."""
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
        nd_buffer = NDBuffer[Self.dtype].arange(args)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn flatten(
        self, start_dim: Int = 0, end_dim: Optional[Int] = None
    ) -> Gradbox[Self.dtype]:
        flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        return Gradbox[Self.dtype](flattened_buffer^, share=False)

    # shared=False → always owned, contiguous, GPU-aware
    fn squeeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
        var buffer = self.buffer.copy()
        var ndb = buffer.squeeze(axes, shared=False)
        return Gradbox[Self.dtype](ndb^, share=False)

    fn squeeze(self, axes: List[Int] = []) -> Gradbox[Self.dtype]:
        return self.squeeze(IntArray(axes))

    fn unsqueeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
        var buffer = self.buffer.copy()
        var ndb = buffer.unsqueeze(axes, shared=False)
        return Gradbox[Self.dtype](ndb^, share=False)

    fn unsqueeze(self, axes: List[Int]) -> Gradbox[Self.dtype]:
        return self.unsqueeze(IntArray(axes))

    fn permute(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """
        Permute axes of this Gradbox.
        Returns owned contiguous copy — never a view.
        GPU safe: delegates to NDBuffer.permute(shared=False)
        which uses contiguous() → contiguous_device_state() on GPU.
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
        return Gradbox[Self.dtype](
            NDBuffer.full(shape, scalar, device), share=share
        )

    @staticmethod
    @always_inline
    fn zeros(
        shape: Shape, share: Bool = False, device: Device = CPU().into()
    ) -> Gradbox[Self.dtype]:
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
        return Gradbox[Self.dtype](self.buffer.contiguous(), share=share)

    fn shared(self) -> Bool:
        return self.buffer.shared()

    @always_inline
    fn sum(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = self.buffer.reduce(
            normalized_axes=axes, keepdims=keepdims
        )
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    @always_inline
    fn mean(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        return Mean[Self.dtype].forward(self, axes=axes, keepdims=keepdims)

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_grad: Gradbox[Self.dtype], target_shape: Shape
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = extended_grad.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn broadcast_to(
        self, target_shape: Shape, share: Bool = False
    ) -> Gradbox[Self.dtype]:
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
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(List): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __getitem__([])"
            )

        return self.buffer[indices]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )

        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
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
        return self.buffer.item()

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.buffer.is_scalar()

    @always_inline
    fn numels(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn num_elements(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn __len__(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.strides().is_contiguous(self.shape())

    @always_inline
    fn rank(self) -> Int:
        return self.buffer.rank()

    @always_inline
    fn offset(self) -> Int:
        return self.buffer.offset

    @always_inline
    fn shape(ref self) -> ref [self.buffer.shape] Shape:
        return self.buffer.shape

    @always_inline
    fn strides(ref self) -> ref [self.buffer.strides] Strides:
        return self.buffer.strides

    fn data_buffer(ref self) -> ref[self.buffer]NDBuffer[Self.dtype]:
        return self.buffer

    @always_inline
    fn index_iterator(
        self,
    ) -> IndexIterator[
        origin_of(self.buffer.shape), origin_of(self.buffer.strides)
    ]:
        return self.buffer.index_iterator()

    fn __eq__(self, other: Gradbox[Self.dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __eq__(other): shape mismatch",
                String(self.shape()),
                "≠",
                String(other.shape()),
            )
        return self.buffer.compare[Equal](other.buffer).buffer.all_true()

    fn __ne__(self, other: Gradbox[Self.dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __ne__(other): shape mismatch",
                String(self.shape()),
                "≠",
                String(other.shape()),
            )
        return self.buffer.compare[NotEqual](other.buffer).buffer.all_true()

    fn get_gpu(self) raises -> GPU:
        if self.is_on_gpu():
            return self.buffer.get_gpu()
        raise "Gradbox get_gpu: gradbox is not on gpu"

    fn is_on_gpu(self) -> Bool:
        return self.buffer.is_on_gpu()

    fn is_on_cpu(self) -> Bool:
        return self.is_on_gpu() == False

    fn to_cpu(self) raises -> Self:
        comptime if has_accelerator():
            return DeviceTransfer[Self.dtype].forward(self, CPU().into())
        raise Error("System does not have any accelerator")

    fn to_gpu(
        self,
        gpu: Optional[GPU] = None,
    ) raises -> Self:
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
            + String(self.buffer.gpu_id()) if self.is_on_gpu() else ", Device : "
            + "cpu"
        )
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    fn seed_grad(self, value: Scalar[Self.dtype]):
        self.buffer.fill(value)

    @always_inline
    fn seed_grad(self, with_tensor: Tensor[Self.dtype]):
        self.buffer.fill(with_tensor.buffer)

    @always_inline
    fn zero_grad(self):
        self.buffer.zero()

    fn fill(self, value: Scalar[Self.dtype], *indices: Idx):
        Filler[Self.dtype].fill(self.buffer, value, indices)

    fn fill(self, tensor: Tensor[Self.dtype], *indices: Idx):
        Filler[Self.dtype].fill(self.buffer, tensor.buffer, indices)

    fn fill(self, gradbox: Gradbox[Self.dtype], *indices: Idx):
        Filler[Self.dtype].fill(self.buffer, gradbox.buffer, indices)

    @always_inline
    fn clamp_in_place(
        self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]
    ):
        self.buffer.clamp_in_place(lower_bound, upper_bound)

    fn max(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[MAX](scalar), share=False
        )

    fn min(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[MIN](scalar), share=False
        )

    fn __mul__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Multiply](scalar), share=False
        )

    fn __rmul__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return self.__mul__(scalar)

    fn __add__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Add](scalar), share=False
        )

    fn __radd__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return self.__add__(scalar)

    fn __sub__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar), share=False
        )

    fn __rsub__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), share=False
        )

    fn __truediv__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        if scalar == Scalar[Self.dtype](0):
            panic("Gradbox → __truediv__(scalar): can not divide by zero")
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Divide](scalar), share=False
        )

    fn __rtruediv__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), share=False
        )

    fn __mul__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn __mul__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn matmul(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return Matmul[Self.dtype].forward(A, B)

    fn __add__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __add__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __sub__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __sub__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __truediv__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer), share=False
        )

    fn __imul__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Multiply](scalar)

    fn __iadd__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Add](scalar)

    fn __isub__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, incoming: Gradbox[Self.dtype]):
        self.buffer.inplace_ops[Multiply](incoming.buffer)

    @always_inline
    fn __iadd__(self, incoming: Gradbox[Self.dtype]):
        self.buffer.inplace_ops[Add](incoming.buffer)

    @always_inline
    fn __isub__(self, incoming: Gradbox[Self.dtype]):
        self.buffer.inplace_ops[Subtract](incoming.buffer)

    @always_inline
    fn __itruediv__(self, incoming: Gradbox[Self.dtype]):
        self.buffer.inplace_ops[Divide](incoming.buffer)

    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        comptime assert Self.dtype.is_floating_point(), "Gradbox → all_close(Self): is for floating point data types only"
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
        comptime assert
            Self.dtype.is_floating_point(),
            (
                "Gradbox → all_close(Tensor): is for floating point data types"
                " only"
            )
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
        var nd_buffer = self.buffer.reshape(new_shape, validated)

        return Gradbox[Self.dtype](nd_buffer^, share=share)

    fn __eq__(self, tensor: Tensor[Self.dtype]) -> Bool:
        if self.shape() != tensor.shape():
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                String(self.shape()),
                ",",
                String(tensor.shape()),
            )
        return self.buffer.compare[Equal](tensor.buffer).buffer.all_true()

    fn print(self, num_first: Int = 10, num_last: Int = 10):
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
    ](ref [origin, address_space]self) -> UnsafePointer[
        Scalar[Self.dtype], origin, address_space=address_space
    ]:
        return (
            self.buffer.data_ptr()
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )


fn main() raises:
   print("passes")
