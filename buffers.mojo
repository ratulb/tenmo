from algorithm import vectorize
from sys import simd_width_of, size_of
from memory import memset_zero, memcpy
from math import exp, log, ceil, tanh, sqrt
from common_utils import log_debug, panic
from utils.numerics import max_finite
from os.atomic import Atomic, Consistency, fence
from operators import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    Overwrite,
    SigmoidOp,
    TanhForwardOp,
    TanhBackwardOp,
    Log,
    Exp,
    ReLUBackwardOp,
    ReLUForwardOp,
    ReverseDivide,
    SqrtForwardOp,
    SqrtBackwardOp,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)

alias _volatile = False


struct Buffer[dtype: DType = DType.float32](
    ImplicitlyCopyable
    & Movable
    & Sized
    & Stringable
    & Writable
    & Representable
    & Absable
):

    """
    Unified buffer with optional reference counting built-in.

    When unshared: Just manages data pointer.
    When shared: Adds atomic refcount in same allocation as data.
    """

    var size: Int
    var data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var _refcount: UnsafePointer[
        Atomic[DType.uint64], MutAnyOrigin
    ]  # Null if not shared!
    var external: Bool

    fn __init__(out self):
        self.size = 0
        self.data = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]()
        self._refcount = UnsafePointer[
            Atomic[DType.uint64], MutAnyOrigin
        ]()  # Null
        self.external = False

    fn __init__(out self, size: Int, external: Bool = False):
        if size < 0:
            panic("Buffer size must be >= 0")
        self.size = size
        self.external = external
        self._refcount = UnsafePointer[
            Atomic[DType.uint64], MutAnyOrigin
        ]()  # Null (not shared yet)

        if size == 0:
            self.data = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]()
        else:
            if external:
                self.data = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]()
            else:
                self.data = alloc[Scalar[Self.dtype]](size)

    fn __init__(out self, elems: List[Scalar[Self.dtype]]):
        var length = len(elems)
        self.data = alloc[Scalar[Self.dtype]](length)
        self.size = length
        memcpy(dest=self.data, src=elems._data, count=length)
        self.external = False
        self._refcount = UnsafePointer[
            Atomic[DType.uint64], MutAnyOrigin
        ]()  # Null

    fn __init__(
        out self,
        size: Int,
        data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        copy: Bool = False,
    ):
        self.size = size
        self._refcount = UnsafePointer[
            Atomic[DType.uint64], MutAnyOrigin
        ]()  # Null
        if copy:
            self.data = alloc[Scalar[Self.dtype]](size)
            memcpy(dest=self.data, src=data, count=size)
            self.external = False
        else:
            self.data = data
            self.external = True

    # Shared state management

    fn is_shared(self) -> Bool:
        """Check if this buffer has ref counting enabled."""
        return (
            self._refcount
            != UnsafePointer[Atomic[DType.uint64], MutAnyOrigin]()
        )

    fn shared(mut self):
        """
        Convert this buffer to shared mode (enable ref counting).

        Memory layout transformation:
        Before: [data array]
        After:  [refcount: 8 bytes][data array]
        """
        if self.is_shared():
            return  # Already shared

        if self.external:
            panic("Cannot share external buffer")

        if self.size == 0:
            return  # Nothing to share

        # Allocate new memory: [refcount][data]
        var refcount_size = size_of[Atomic[DType.uint64]]()
        var data_size = self.size * size_of[Scalar[Self.dtype]]()
        var total_size = refcount_size + data_size
        var new_alloc = alloc[UInt8](total_size)

        # Initialize refcount at start
        var refcount_ptr = new_alloc.bitcast[Atomic[DType.uint64]]()
        refcount_ptr[] = Atomic[DType.uint64](1)

        # Copy data after refcount
        var new_data = new_alloc.offset(refcount_size).bitcast[
            Scalar[Self.dtype]
        ]()
        memcpy(dest=new_data, src=self.data, count=self.size)

        # Free old allocation

        self.data.free()
        log_debug("Buffer__del__ → freed unshared data pointer")

        # Update pointers
        self.data = new_data
        self._refcount = refcount_ptr

    fn ref_count(self) -> UInt64:
        """Count the amount of current references.

        Returns:
            The current amount of references to the pointee.
        """
        return self._refcount[].load[ordering = Consistency.MONOTONIC]()

    # ========================================
    # Copy/Move semantics
    # ========================================

    fn __copyinit__(out self, other: Self):
        """Copy buffer - increment refcount if shared."""
        self.size = other.size
        self.data = other.data
        self.external = other.external
        self._refcount = other._refcount

        if self.is_shared():
            # Atomic increment (only for shared buffers)
            _ = self._refcount[].fetch_add[ordering = Consistency.MONOTONIC](1)
        else:
            # Not shared - deep copy data
            if self.size > 0 and not self.external:
                self.data = alloc[Scalar[Self.dtype]](self.size)
                memcpy(dest=self.data, src=other.data, count=self.size)

    fn __moveinit__(out self, deinit other: Self):
        """Move buffer - no refcount change."""
        self.size = other.size
        self.data = other.data
        self.external = other.external
        self._refcount = other._refcount

    fn __del__(deinit self):
        """Destroy buffer - handle both shared and unshared cases."""
        if self.size == 0 or not self.data.__bool__():
            return

        if self.external:
            return  # Don't free external data

        if self.is_shared():
            # Shared buffer - atomic decrement
            if (
                self._refcount[].fetch_sub[ordering = Consistency.RELEASE](1)
                != 1
            ):
                return  # Other references exist

            # Last reference - free everything
            fence[ordering = Consistency.ACQUIRE]()

            # Destroy data elements
            for i in range(self.size):
                (self.data + i).destroy_pointee()

            # Free allocation (starts at refcount, not data)
            var refcount_size = size_of[Atomic[DType.uint64]]()
            var alloc_start = self._refcount.bitcast[UInt8]()
            alloc_start.free()

        else:
            # Unshared buffer - direct free
            for i in range(self.size):
                (self.data + i).destroy_pointee()
            self.data.free()
            log_debug("Buffer__del__ → freed unshared buffer")

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    fn __iter__(ref self) -> ElementIterator[Self.dtype, origin_of(self)]:
        return ElementIterator(Pointer(to=self))

    fn __getitem__(self, slice: Slice) -> Buffer[Self.dtype]:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)
        var result_size = len(spread)

        if result_size == 0:
            return Buffer[Self.dtype]()

        var result = Buffer[Self.dtype](result_size)

        # Fast path: contiguous (step == 1)
        if step == 1:
            memcpy(dest=result.data, src=self.data + start, count=result_size)
            return result^

        # Strided path: use SIMD if beneficial
        alias simd_width = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        # Only use SIMD if we have enough elements
        if result_size >= simd_width:
            var num_chunks = result_size // simd_width
            var remainder = result_size % simd_width

            # SIMD strided loads
            for chunk in range(num_chunks):
                var src_idx = start + chunk * simd_width * step
                var dst_idx = chunk * simd_width

                # Strided load from source
                var values = (self.data + src_idx).strided_load[
                    width=simd_width
                ](step)

                # Contiguous store to result
                result.data.store[width=simd_width](dst_idx, values)

            # Handle remainder scalars
            var start_remainder = num_chunks * simd_width
            for i in range(remainder):
                result.data[start_remainder + i] = self.data[
                    start + (start_remainder + i) * step
                ]
        else:
            # Too small for SIMD - just use scalar loop
            for i in range(result_size):
                result.data[i] = self.data[start + i * step]

        return result^

    @always_inline
    fn __getitem__old(self, slice: Slice) -> Buffer[Self.dtype]:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)
        var result_size = len(spread)

        if result_size == 0:
            return Buffer[Self.dtype]()

        var result = Buffer[Self.dtype](result_size)

        # Fast path: contiguous slice (step == 1)
        if step == 1:
            memcpy(dest=result.data, src=self.data + start, count=result_size)
        else:
            # Slow path: non-contiguous slice
            for i in range(result_size):
                result.data[i] = self.data[start + i * step]

        return result^

    @always_inline
    fn __getitem__(self, index: Int) -> Scalar[Self.dtype]:
        debug_assert(
            index >= 0 and index < self.size,
            "Buffer -> __getitem__: index out of bounds",
            self.size,
            index,
        )
        return self.data.load[width=1, volatile=_volatile](index)

    @always_inline
    fn __setitem__(self, index: Int, scalar: Scalar[Self.dtype]):
        debug_assert(
            index >= 0 and index < self.size,
            "Buffer -> __setitem__: index out of bounds",
            self.size,
            index,
        )
        self.data.store[width=1, volatile=_volatile](index, scalar)

    @always_inline
    fn load[
        simdwidth: Int = 1
    ](self, offset: Int) -> SIMD[Self.dtype, simdwidth]:
        debug_assert(
            offset >= 0 and offset + simdwidth <= self.size,
            "Buffer -> load : offset out of bounds",
            self.size,
            offset,
            simdwidth,
        )
        return self.data.load[width=simdwidth, volatile=_volatile](offset)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, offset: Int, values: SIMD[Self.dtype, simdwidth]):
        debug_assert(
            offset >= 0 and offset + simdwidth <= self.size,
            "Buffer -> store : offset out of bounds",
            self.size,
            offset,
            simdwidth,
        )

        self.data.store[width=simdwidth, volatile=_volatile](offset, values)

    @always_inline
    fn __add__(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __add__(other) is for numeric data types only",
        ]()

        if not self.size == other.size:
            panic(
                "Buffer → __add__(other): buffer size does not match -> self:",
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )
        if self.size == 0:
            panic("Buffer → __add__(other): buffer size 0")

        return self.arithmetic_ops[Add, False](other)

    @always_inline
    fn __iadd__(self, other: Buffer[Self.dtype]):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __iadd__(other) is for numeric data types only",
        ]()

        if self.size != other.size:
            panic(
                "Buffer __iadd__: buffer sizes must match: self.size ->",
                self.size.__str__(),
                ", and other size:",
                other.size.__str__(),
            )

        if self.size == 0:
            return

        self.inplace_ops[Add, False](other)

    @always_inline
    fn __isub__(self, other: Buffer[Self.dtype]):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __isub__(other) is for numeric data types only",
        ]()

        if self.size != other.size:
            panic(
                "Buffer __isub__(other): buffer sizes must match: self.size: ",
                self.size.__str__(),
                "and other size: ",
                other.size.__str__(),
            )

        if self.size == 0:
            return

        self.inplace_ops[Subtract, False](other)

    @always_inline
    fn __sub__(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __sub__(other) is for numeric data types only",
        ]()

        if not self.size == other.size:
            panic(
                "Buffer → __sub__(other): buffer size does not match -> self:",
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )
        if self.size == 0:
            panic("Buffer → __sub__(other): buffer size 0")

        return self.arithmetic_ops[Subtract, False](other)

    @always_inline
    fn __mul__(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[Self.dtype]:
        # No constraint checking for Self.dtype - DType.bool multplication allowed
        if not self.size == other.size:
            panic(
                "Buffer → __mul__(other): buffer size does not match -> self:",
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )
        if self.size == 0:
            panic("Buffer → __mul__(other): buffer size 0")

        return self.arithmetic_ops[Multiply, False](other)

    @always_inline
    fn __imul__(self, other: Buffer[Self.dtype]):
        # No constraint checking for Self.dtype - DType.bool multplication allowed
        if self.size != other.size:
            panic(
                (
                    "Buffer → __imul__(Buffer[Self.dtype]: buffer size does not"
                    " match -> self:"
                ),
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )

        if self.size == 0:
            return

        self.inplace_ops[Multiply, False](other)

    @always_inline
    fn inplace_ops_scalar[
        op_code: Int
    ](
        self: Buffer[Self.dtype],
        scalar: Scalar[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        var actual_end = end_index.or_else(self.size)
        var extent = actual_end - start_index

        if self.size == 0 or extent <= 0:
            return

        @parameter
        if Self.dtype == DType.bool:
            # Special handling for bool - process element by element
            for i in range(start_index, actual_end):

                @parameter
                if op_code == Multiply:
                    self[i] = self[i] * scalar
                elif op_code == Add:
                    self[i] = self[i] + scalar
                elif op_code == Subtract:
                    self[i] = self[i] - scalar
                else:  # Divide
                    self[i] = self[i] / scalar
            return

        # SIMD vectorization for non-bool types
        alias simd_width = simd_width_of[Self.dtype]()

        var idx = start_index
        var remaining = extent

        # Process full SIMD chunks
        while remaining >= simd_width:
            var block = self.load[simdwidth=simd_width](idx)
            var op_result: SIMD[Self.dtype, simd_width]

            @parameter
            if op_code == Multiply:
                op_result = block * scalar
            elif op_code == Add:
                op_result = block + scalar
            elif op_code == Subtract:
                op_result = block - scalar
            else:  # Divide
                op_result = block / scalar

            self.store[simdwidth=simd_width](idx, op_result)

            idx += simd_width
            remaining -= simd_width

        # Process remaining elements (scalar tail)
        for i in range(idx, actual_end):

            @parameter
            if op_code == Multiply:
                self[i] = self[i] * scalar
            elif op_code == Add:
                self[i] = self[i] + scalar
            elif op_code == Subtract:
                self[i] = self[i] - scalar
            else:  # Divide
                self[i] = self[i] / scalar

    @always_inline
    fn inplace_ops[
        op_code: Int, validate: Bool = True
    ](
        self: Buffer[Self.dtype],
        other: Buffer[Self.dtype],
        self_start: Int = 0,
        self_end: Optional[Int] = None,
        other_start: Int = 0,
        other_end: Optional[Int] = None,
    ):
        var self_extent = self_end.or_else(self.size) - self_start
        var other_extent = other_end.or_else(other.size) - other_start

        @parameter
        if validate:
            if (
                self_extent <= 0
                or other_extent <= 0
                or self_extent != other_extent
            ):
                log_debug(
                    "Buffer -> inplace_ops: range mismatch: self range -> "
                    + self_extent.__str__()
                    + ", other range: "
                    + other_extent.__str__(),
                )
                return

        # Determine SIMD width based on dtype
        @parameter
        if Self.dtype == DType.bool:
            # Special handling for bool (bit-packed)

            # Scalar loop for booleans
            for idx in range(self_extent):
                var result: Scalar[Self.dtype]

                @parameter
                if op_code == Multiply:
                    result = self[self_start + idx] * other[other_start + idx]

                else:  # Overwrite
                    result = other[other_start + idx]

                self[self_start + idx] = result

        else:
            # SIMD vectorization for non-bool types
            alias simd_width = simd_width_of[Self.dtype]()

            # Calculate loop bounds
            var vectorized_end = (self_extent // simd_width) * simd_width

            # Vectorized loop (manual unrolling)
            for idx in range(0, vectorized_end, simd_width):
                var op_result: SIMD[Self.dtype, simd_width]

                @parameter
                if op_code == Multiply:
                    op_result = self.load[simdwidth=simd_width](
                        self_start + idx
                    ) * other.load[simdwidth=simd_width](other_start + idx)

                elif op_code == Add:
                    op_result = self.load[simdwidth=simd_width](
                        self_start + idx
                    ) + other.load[simdwidth=simd_width](other_start + idx)

                elif op_code == Subtract:
                    op_result = self.load[simdwidth=simd_width](
                        self_start + idx
                    ) - other.load[simdwidth=simd_width](other_start + idx)

                elif op_code == Divide:
                    op_result = self.load[simdwidth=simd_width](
                        self_start + idx
                    ) / other.load[simdwidth=simd_width](other_start + idx)

                else:  # Overwrite
                    op_result = other.load[simdwidth=simd_width](
                        other_start + idx
                    )

                self.store[simdwidth=simd_width](self_start + idx, op_result)

            # Scalar tail loop for remaining elements
            for idx in range(vectorized_end, self_extent):
                var result: Scalar[Self.dtype]

                @parameter
                if op_code == Multiply:
                    result = self[self_start + idx] * other[other_start + idx]
                elif op_code == Add:
                    result = self[self_start + idx] + other[other_start + idx]
                elif op_code == Subtract:
                    result = self[self_start + idx] - other[other_start + idx]
                elif op_code == Divide:
                    result = self[self_start + idx] / other[other_start + idx]
                else:  # Overwrite
                    result = other[other_start + idx]

                self[self_start + idx] = result

    @always_inline
    fn arithmetic_ops[
        op_code: Int, validate: Bool = True
    ](
        self: Buffer[Self.dtype],
        other: Buffer[Self.dtype],
        self_start: Int = 0,
        self_end: Optional[Int] = None,
        other_start: Int = 0,
        other_end: Optional[Int] = None,
    ) -> Buffer[Self.dtype]:
        var self_actual_end = self_end.or_else(self.size)
        var other_actual_end = other_end.or_else(other.size)
        var self_extent = self_actual_end - self_start
        var other_extent = other_actual_end - other_start

        @parameter
        if validate:
            if (
                self_extent <= 0
                or other_extent <= 0
                or self_extent != other_extent
            ):
                panic(
                    "Buffer -> arithmetic_ops: range mismatch: self range -> "
                    + self_extent.__str__()
                    + ", other range: "
                    + other_extent.__str__()
                )

        var out = Buffer[Self.dtype](self_extent)

        @parameter
        if Self.dtype == DType.bool:
            # Special handling for bool - process element by element
            for i in range(self_extent):

                @parameter
                if op_code == Multiply:
                    out[i] = self[self_start + i] * other[other_start + i]
                else:
                    panic(
                        "Buffer -> arithmetic_ops: invalid operation for"
                        " DType.bool"
                    )
            return out^

        # SIMD vectorization for non-bool types
        alias simd_width = simd_width_of[Self.dtype]()

        var idx = 0
        var remaining = self_extent

        # Process full SIMD chunks
        while remaining >= simd_width:
            var self_block = self.load[simdwidth=simd_width](self_start + idx)
            var other_block = other.load[simdwidth=simd_width](
                other_start + idx
            )
            var op_result: SIMD[Self.dtype, simd_width]

            @parameter
            if op_code == Multiply:
                op_result = self_block * other_block
            elif op_code == Add:
                op_result = self_block + other_block
            elif op_code == Subtract:
                op_result = self_block - other_block
            else:  # Divide
                op_result = self_block / other_block

            out.store[simdwidth=simd_width](idx, op_result)

            idx += simd_width
            remaining -= simd_width

        # Process remaining elements (scalar tail)
        for i in range(idx, self_extent):

            @parameter
            if op_code == Multiply:
                out[i] = self[self_start + i] * other[other_start + i]
            elif op_code == Add:
                out[i] = self[self_start + i] + other[other_start + i]
            elif op_code == Subtract:
                out[i] = self[self_start + i] - other[other_start + i]
            else:  # Divide
                out[i] = self[self_start + i] / other[other_start + i]

        return out^

    @always_inline
    fn arithmetic_ops_scalar[
        op_code: Int
    ](
        self: Buffer[Self.dtype],
        scalar: Scalar[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Buffer[Self.dtype]:
        var actual_end = end_index.or_else(self.size)
        var extent = actual_end - start_index

        if self.size == 0 or extent <= 0:
            panic("Buffer -> arithmetic_ops_scalar: buffer size 0")

        var out = Buffer[Self.dtype](extent)

        @parameter
        if Self.dtype == DType.bool:
            # Special handling for bool - process element by element
            for i in range(extent):

                @parameter
                if op_code == Multiply:
                    out[i] = self[start_index + i] * scalar
                else:
                    out[i] = self[start_index + i].__rtruediv__(scalar)

                    panic(
                        "Buffer -> arithmetic_ops_scalar: invalid operation for"
                        " DType.bool"
                    )
            return out^

        # SIMD vectorization for non-bool types
        alias simd_width = simd_width_of[Self.dtype]()

        var idx = 0
        var remaining = extent

        # Process full SIMD chunks
        while remaining >= simd_width:
            var block = self.load[simdwidth=simd_width](start_index + idx)
            var op_result: SIMD[Self.dtype, simd_width]

            @parameter
            if op_code == Multiply:
                op_result = block * scalar
            elif op_code == Add:
                op_result = block + scalar
            elif op_code == Subtract:
                op_result = block - scalar
            elif op_code == ReverseSubtract:
                op_result = scalar - block
            elif op_code == Divide:
                op_result = block / scalar
            else:  # ReverseDivide
                op_result = block.__rtruediv__(scalar)

            out.store[simdwidth=simd_width](idx, op_result)

            idx += simd_width
            remaining -= simd_width

        # Process remaining elements (scalar tail)
        for i in range(idx, extent):

            @parameter
            if op_code == Multiply:
                out[i] = self[start_index + i] * scalar
            elif op_code == Add:
                out[i] = self[start_index + i] + scalar
            elif op_code == Subtract:
                out[i] = self[start_index + i] - scalar
            elif op_code == ReverseSubtract:
                out[i] = scalar - self[start_index + i]
            elif op_code == Divide:
                out[i] = self[start_index + i] / scalar
            else:  # ReverseDivide
                out[i] = self[start_index + i].__rtruediv__(scalar)

        return out^

    @always_inline
    fn __truediv__(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __truediv__(other) is for numeric data types only",
        ]()

        if not self.size == other.size:
            panic(
                (
                    "Buffer → __truediv__(other): buffer size does not match ->"
                    " self:"
                ),
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )

        if self.size == 0:
            panic("Buffer → __truediv__(other): buffer size 0")

        return self.arithmetic_ops[Divide, False](other)

    @always_inline
    fn __itruediv__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __itruediv__(other) is for numeric data types only",
        ]()

        if not self.size == other.size:
            panic(
                (
                    "Buffer → __itruediv__(other): buffer size does not"
                    " match -> self:"
                ),
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )
        if self.size == 0:
            return

        self.inplace_ops[Divide, False](other)

    @always_inline
    fn __iadd__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __iadd__(scalar) is for numeric data types only",
        ]()

        self.inplace_ops_scalar[Add](scalar)

    @always_inline
    fn __isub__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __isub__(scalar) is for numeric data types only",
        ]()

        self.inplace_ops_scalar[Subtract](scalar)

    @always_inline
    fn __imul__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]):
        # No constraint checking for Self.dtype - DType.bool multplication allowed
        self.inplace_ops_scalar[Multiply](scalar)

    @always_inline
    fn __itruediv__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __itruediv__(scalar) is for numeric data types only",
        ]()

        if scalar == Scalar[Self.dtype](0):
            panic("Buffer → __itruediv__(scalar): can not divide by zero")

        self.inplace_ops_scalar[Divide](scalar)

    @always_inline
    fn __rsub__(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __rsub__(scalar) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        return self.arithmetic_ops_scalar[ReverseSubtract](scalar)

    @always_inline
    fn __sub__(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __sub__(scalar) -> Buffer is for numeric data types only",
        ]()

        return self.arithmetic_ops_scalar[Subtract](scalar)

    @always_inline
    fn __rmul__(
        self: Buffer[Self.dtype], factor: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __rmul__(scalar) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        return self.__mul__(factor)

    @always_inline
    fn __mul__(
        self: Buffer[Self.dtype], factor: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        # No constraint checking for Self.dtype - DType.bool multplication allowed
        return self.arithmetic_ops_scalar[Multiply](factor)

    @always_inline
    fn __radd__(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __radd__(scalar) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        return self.__add__(scalar)

    @always_inline
    fn __add__(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __add__(scalar) -> Buffer  is for numeric data types"
                " only"
            ),
        ]()

        return self.arithmetic_ops_scalar[Add](scalar)

    @always_inline
    fn __truediv__(
        self: Buffer[Self.dtype], divisor: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __truediv__(scalar) -> Buffer  is for numeric data"
                " types only"
            ),
        ]()

        return self.arithmetic_ops_scalar[Divide](divisor)

    @always_inline
    fn __rtruediv__(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __rtruediv__(scalar) -> Buffer  is for numeric data"
                " types only"
            ),
        ]()

        return self.arithmetic_ops_scalar[ReverseDivide](scalar)

    # Helper for compile-time dispatch
    @always_inline
    @staticmethod
    fn unary_ops_helper[
        op_code: Int,
        smdwidth: Int,
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
    ](block: SIMD[Self.dtype, smdwidth]) -> SIMD[Self.dtype, smdwidth]:
        @parameter
        if op_code == SigmoidOp:
            return 1.0 / (1.0 + exp(-block))
        elif op_code == ReLUForwardOp:
            return max(block, 0.0)
        # tanh(x) = 2 * σ(2x) - 1  # where σ is sigmoid
        elif op_code == TanhForwardOp:
            # return 2 * (1.0 / (1.0 + exp(-2 * block))) - 1
            return tanh(block)
        elif op_code == TanhBackwardOp:
            return 1 - (tanh(block) * tanh(block))
        elif op_code == Exp:
            return exp(block)
        elif op_code == Log:
            return log(block)
        elif op_code == SqrtForwardOp:
            return sqrt(block)
        elif op_code == SqrtBackwardOp:
            return 1 / (epsilon + 2 * sqrt(block))
        else:
            return block

    @always_inline
    fn compare_buffer_full[
        op_code: Int
    ](self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Buffer[
        DType.bool
    ]:
        if not self.size == other.size:
            panic(
                (
                    "Buffer → compare_buffer_full: buffer size does not match"
                    " -> self:"
                ),
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )

        var total = self.size
        if total == 0:
            panic("Buffer -> compare_buffer_full: buffer sizes are zero")

        var out = Buffer[DType.bool](total)

        alias smdwidth = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        # Manual vectorization: process chunks of smdwidth elements
        var num_full_chunks = total // smdwidth
        var remainder = total % smdwidth

        # Process full SIMD chunks
        for chunk in range(num_full_chunks):
            var idx = chunk * smdwidth
            var self_block = self.load[simdwidth=smdwidth](idx)
            var other_block = other.load[simdwidth=smdwidth](idx)
            var result: SIMD[DType.bool, smdwidth]

            @parameter
            if op_code == Equal:
                result = self_block.eq(other_block)
            elif op_code == NotEqual:
                result = self_block.ne(other_block)
            elif op_code == GreaterThan:
                result = self_block.gt(other_block)
            elif op_code == GreaterThanEqual:
                result = self_block.ge(other_block)
            elif op_code == LessThan:
                result = self_block.lt(other_block)
            else:  # LessThanEqual
                result = self_block.le(other_block)

            # Store results element-by-element (required for bool bit-packing)
            for i in range(smdwidth):
                out[idx + i] = result[i]

        # Process remaining elements
        if remainder > 0:
            var start_idx = num_full_chunks * smdwidth
            for i in range(remainder):
                var idx = start_idx + i
                var self_val = self[idx]
                var other_val = other[idx]
                var result: Bool

                @parameter
                if op_code == Equal:
                    result = self_val == other_val
                elif op_code == NotEqual:
                    result = self_val != other_val
                elif op_code == GreaterThan:
                    result = self_val > other_val
                elif op_code == GreaterThanEqual:
                    result = self_val >= other_val
                elif op_code == LessThan:
                    result = self_val < other_val
                else:  # LessThanEqual
                    result = self_val <= other_val

                out[idx] = result

        return out^

    @always_inline
    fn compare_scalar_full[
        op_code: Int
    ](self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Buffer[
        DType.bool
    ]:
        var total = self.size
        if total == 0:
            panic("Buffer -> compare_scalar_full: buffer size is zero")

        var out = Buffer[DType.bool](total)

        alias smdwidth = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        # Manual vectorization: process chunks of smdwidth elements
        var num_full_chunks = total // smdwidth
        var remainder = total % smdwidth

        # Process full SIMD chunks
        for chunk in range(num_full_chunks):
            var idx = chunk * smdwidth
            var block = self.load[simdwidth=smdwidth](idx)
            var result: SIMD[DType.bool, smdwidth]

            @parameter
            if op_code == Equal:
                result = block.eq(scalar)
            elif op_code == NotEqual:
                result = block.ne(scalar)
            elif op_code == GreaterThan:
                result = block.gt(scalar)
            elif op_code == GreaterThanEqual:
                result = block.ge(scalar)
            elif op_code == LessThan:
                result = block.lt(scalar)
            else:  # LessThanEqual
                result = block.le(scalar)

            # Store results element-by-element (required for bool bit-packing)
            for i in range(smdwidth):
                out[idx + i] = result[i]

        # Process remaining elements
        if remainder > 0:
            var start_idx = num_full_chunks * smdwidth
            for i in range(remainder):
                var idx = start_idx + i
                var val = self[idx]
                var result: Bool

                @parameter
                if op_code == Equal:
                    result = val == scalar
                elif op_code == NotEqual:
                    result = val != scalar
                elif op_code == GreaterThan:
                    result = val > scalar
                elif op_code == GreaterThanEqual:
                    result = val >= scalar
                elif op_code == LessThan:
                    result = val < scalar
                else:  # LessThanEqual
                    result = val <= scalar

                out[idx] = result

        return out^

    @always_inline
    fn select[
        op_code: Int, validate: Bool = True
    ](
        self: Buffer[Self.dtype],
        other: Buffer[Self.dtype],
        self_start: Int = 0,
        self_end: Optional[Int] = None,
        other_start: Int = 0,
        other_end: Optional[Int] = None,
    ) -> Buffer[Self.dtype]:
        var self_extent = self_end.or_else(self.size) - self_start
        var other_extent = other_end.or_else(other.size) - other_start

        @parameter
        if validate:
            if (
                self_extent <= 0
                or other_extent <= 0
                or self_extent != other_extent
            ):
                panic(
                    "Buffer -> select: range mismatch: self range -> "
                    + self_extent.__str__()
                    + ", other range: ",
                    other_extent.__str__(),
                )

        var out = Buffer[Self.dtype].zeros(self_extent)
        var zero = Scalar[Self.dtype](0)

        alias smdwidth = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        # Manual vectorization
        var num_full_chunks = self_extent // smdwidth
        var remainder = self_extent % smdwidth

        # Process full SIMD chunks
        for chunk in range(num_full_chunks):
            var idx = chunk * smdwidth
            var selected: SIMD[Self.dtype, smdwidth]

            @parameter
            if op_code == ReLUBackwardOp:
                var cond_block = self.load[simdwidth=smdwidth](self_start + idx)
                var true_block = other.load[simdwidth=smdwidth](
                    other_start + idx
                )
                var false_block = out.load[simdwidth=smdwidth](idx)
                selected = (cond_block.gt(zero)).select(true_block, false_block)
            else:  # Bogus
                selected = out.load[simdwidth=smdwidth](idx)

            out.store[simdwidth=smdwidth](idx, selected)

        # Process remaining elements
        if remainder > 0:
            var start_idx = num_full_chunks * smdwidth
            for i in range(remainder):
                var idx = start_idx + i
                var selected: Scalar[Self.dtype]

                @parameter
                if op_code == ReLUBackwardOp:
                    var cond_val = self[self_start + idx]
                    var true_val = other[other_start + idx]
                    var false_val = out[idx]
                    selected = true_val if cond_val > zero else false_val
                else:  # Bogus
                    selected = out[idx]

                out[idx] = selected

        return out^

    @always_inline
    fn unary_ops[
        op_code: Int
    ](
        self: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Buffer[Self.dtype]:
        var extent = end_index.or_else(self.size) - start_index
        var out = Buffer[Self.dtype](extent)

        alias smdwidth = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        # Manual vectorization
        var num_full_chunks = extent // smdwidth
        var remainder = extent % smdwidth

        # Process full SIMD chunks
        for chunk in range(num_full_chunks):
            var idx = chunk * smdwidth
            var block = self.load[simdwidth=smdwidth](start_index + idx)
            var result = Self.unary_ops_helper[op_code, smdwidth](block)
            out.store[simdwidth=smdwidth](idx, result)

        # Process remaining elements
        if remainder > 0:
            var start_idx = num_full_chunks * smdwidth
            for i in range(remainder):
                var idx = start_idx + i
                var val = self[start_index + idx]
                var result = Self.unary_ops_helper[op_code, 1](val)
                out[idx] = result

        return out^

    @always_inline
    fn unary_ops_with_mask[
        op_code: Int
    ](
        self: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Tuple[Buffer[Self.dtype], Buffer[Self.dtype]]:
        """Compute unary operation and mask simultaneously.

        For ReLU: output = max(0, x), mask = (x > 0) ? 1.0 : 0.0

        Returns:
            Tuple of (output_buffer, mask_buffer).
        """
        var extent = end_index.or_else(self.size) - start_index
        var out = Buffer[Self.dtype](extent)
        var mask = Buffer[Self.dtype](extent)

        alias simd_width = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        # Manual vectorization
        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var zero = SIMD[Self.dtype, simd_width](0)
        var one = SIMD[Self.dtype, simd_width](1)

        # Process full SIMD chunks
        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var block = self.load[simdwidth=simd_width](start_index + idx)

            # Compute output
            var result = Self.unary_ops_helper[op_code, simd_width](block)

            # Compute mask based on operation
            var mask_block: SIMD[Self.dtype, simd_width]

            @parameter
            if op_code == ReLUForwardOp:
                # Mask is 1.0 where input > 0, else 0.0
                mask_block = block.gt(SIMD[Self.dtype, simd_width](0)).select(
                    one, zero
                )
            else:
                # For other ops, no masking needed (can extend later)
                mask_block = one

            out.store[simdwidth=simd_width](idx, result)
            mask.store[simdwidth=simd_width](idx, mask_block)

        # Process remaining elements
        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var zero_scalar = Scalar[Self.dtype](0)
            var one_scalar = Scalar[Self.dtype](1)

            for i in range(remainder):
                var idx = start_idx + i
                var val = self.load[simdwidth=1](start_index + idx)
                var result = Self.unary_ops_helper[op_code, 1](val)

                var mask_val: Scalar[Self.dtype]

                @parameter
                if op_code == ReLUForwardOp:
                    mask_val = (
                        one_scalar if val[0] > zero_scalar else zero_scalar
                    )
                else:
                    mask_val = one_scalar

                out.store[simdwidth=1](idx, result)
                mask.store[simdwidth=1](idx, SIMD[Self.dtype, 1](mask_val))

        return (out^, mask^)

    @always_inline
    fn exp(self) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → exp is for numeric data types only",
        ]()

        return self.unary_ops[Exp]()

    @staticmethod
    @always_inline
    fn full(value: Scalar[Self.dtype], size: Int) -> Buffer[Self.dtype]:
        buffer = Buffer[Self.dtype](size)
        buffer.fill(value)
        return buffer^

    @always_inline
    @staticmethod
    fn arange[
        max_arange_elements: Int = 1000000  # Safety limit to prevent infinite loops with very small steps
    ](*args: Scalar[Self.dtype]) -> Buffer[Self.dtype]:
        return Self.arange[max_arange_elements](args)

    @always_inline
    @staticmethod
    fn arange[
        max_arange_elements: Int = 1000000  # Safety limit to prevent infinite loops with very small steps
    ](args: VariadicList[Scalar[Self.dtype]]) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → arange is for numeric data types only",
        ]()

        start: Scalar[Self.dtype] = 0
        end: Scalar[Self.dtype] = max_finite[Self.dtype]()
        step: Scalar[Self.dtype] = 1

        n = len(args)
        if n == 1:
            end = args[0]
        elif n == 2:
            start = args[0]
            end = args[1]
        elif n == 3:
            start = args[0]
            end = args[1]
            step = args[2]
        else:
            panic("Buffer → arange: expected 1 to 3 arguments")

        if step == 0:
            panic("Buffer → arange: step can not be zero")

        # Estimate size to avoid frequent reallocations
        # Add 2 as safety margin for floating-point precision
        delta = abs(end - start)
        step_abs = abs(step)
        est_size = ceil(delta / step_abs).__int__() + 2
        var data = List[Scalar[Self.dtype]](capacity=Int(est_size))
        var value = start

        if step > 0:
            while value < end:
                data.append(value)
                value += step
                if len(data) > max_arange_elements:
                    panic(
                        "Buffer → arange: too many elements, possible infinite"
                        " loop"
                    )
        else:
            while value > end:
                data.append(value)
                value += step
                if len(data) > max_arange_elements:
                    panic(
                        "Buffer → arange: too many elements, possible infinite"
                        " loop"
                    )

        if len(data) == 0:
            panic("Buffer → arange: computed arange size is zero")

        return Buffer[Self.dtype](data^)

    @staticmethod
    @always_inline
    fn zeros(size: Int) -> Buffer[Self.dtype]:
        buffer = Buffer[Self.dtype](size)
        memset_zero(buffer.data, size)
        return buffer^

    @staticmethod
    fn linspace(
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        steps: Int,
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → linspace is for numeric data types only",
        ]()

        if steps < 1:
            panic("Buffer → linspace: steps must be at least 1")

        if steps == 1:
            var buffer = Buffer[Self.dtype](1)
            buffer[0] = start
            return buffer^

        var step_size = (end - start) / Scalar[Self.dtype](steps - 1)
        var buffer = Buffer[Self.dtype](steps)

        for i in range(steps):
            buffer[i] = start + Scalar[Self.dtype](i) * step_size

        return buffer^

    @always_inline
    fn zero(self: Buffer[Self.dtype]):
        memset_zero(self.data, self.size)

    @always_inline
    fn sum(
        self: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → sum is for numeric data types only",
        ]()

        var accum = Scalar[Self.dtype](0)
        if self.size == 0:
            return accum

        var extent = end_index.or_else(self.size) - start_index
        if extent <= 0:
            return accum

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (extent // simd_width) * simd_width

        # Vectorized accumulation
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx + start_index)
            accum += chunk.reduce_add()

        # Scalar tail
        for idx in range(vectorized_end, extent):
            accum += self[idx + start_index]

        return accum

    @always_inline
    fn product(
        self: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → product is for numeric data types only",
        ]()

        var result = Scalar[Self.dtype](1)
        var extent = end_index.or_else(self.size) - start_index

        if extent <= 0:
            return result  # Empty product = multiplicative identity

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (extent // simd_width) * simd_width

        # Vectorized multiplication
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx + start_index)
            result *= chunk.reduce_mul()

        # Scalar tail
        for idx in range(vectorized_end, extent):
            result *= self[idx + start_index]

        return result

    @always_inline
    fn __pow__(
        self: Buffer[Self.dtype], exponent: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            (
                "Buffer → __pow__(exponent) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        var out = Buffer[Self.dtype](self.size)

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (self.size // simd_width) * simd_width

        # Vectorized power
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx)
            out.store[simdwidth=simd_width](idx, chunk.__pow__(exponent))

        # Scalar tail
        for idx in range(vectorized_end, self.size):
            out[idx] = self[idx].__pow__(exponent)

        return out^

    @always_inline
    fn __abs__(self) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __abs__ is for numeric data types only",
        ]()

        var out = Buffer[Self.dtype](self.size)

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (self.size // simd_width) * simd_width

        # Vectorized absolute value
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx)
            out.store[simdwidth=simd_width](idx, chunk.__abs__())

        # Scalar tail
        for idx in range(vectorized_end, self.size):
            out[idx] = self[idx].__abs__()

        return out^

    @always_inline
    fn clamp(
        self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → clamp is for numeric data types only",
        ]()

        var out = Buffer[Self.dtype](self.size)

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (self.size // simd_width) * simd_width

        # Vectorized absolute value
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx)
            out.store[simdwidth=simd_width](
                idx, chunk.clamp(lower_bound, upper_bound)
            )

        # Scalar tail
        for idx in range(vectorized_end, self.size):
            out[idx] = self[idx].clamp(lower_bound, upper_bound)

        return out^

    @always_inline
    fn clamp_in_place(
        self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]
    ):
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → clamp is for numeric data types only",
        ]()

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (self.size // simd_width) * simd_width

        # Vectorized absolute value
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx)
            self.store[simdwidth=simd_width](
                idx, chunk.clamp(lower_bound, upper_bound)
            )

        # Scalar tail
        for idx in range(vectorized_end, self.size):
            self[idx] = self[idx].clamp(lower_bound, upper_bound)

    @always_inline
    fn fill(
        self: Buffer[Self.dtype],
        value: Scalar[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        """Fill a segment or the whole buffer with a value."""
        var extent = end_index.or_else(self.size) - start_index

        # Safety checks
        if extent <= 0:
            panic("Buffer → fill: segment size must be greater than zero")

        @parameter
        if Self.dtype == DType.bool:
            # Scalar loop for booleans
            for idx in range(extent):
                self[idx + start_index] = value
        else:
            alias simd_width = simd_width_of[Self.dtype]()
            var vectorized_end = (extent // simd_width) * simd_width

            # Vectorized fill
            for idx in range(0, vectorized_end, simd_width):
                self.store[simdwidth=simd_width](idx + start_index, value)

            # Scalar tail
            for idx in range(vectorized_end, extent):
                self[idx + start_index] = value

    @always_inline
    fn __neg__(self: Buffer[Self.dtype]) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __neg__ is for numeric data types only",
        ]()

        var out = Buffer[Self.dtype](self.size)

        alias simd_width = simd_width_of[Self.dtype]()
        var vectorized_end = (self.size // simd_width) * simd_width

        # Vectorized negation
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx)
            out.store[simdwidth=simd_width](idx, chunk.__neg__())

        # Scalar tail
        for idx in range(vectorized_end, self.size):
            out[idx] = self[idx].__neg__()

        return out^

    @always_inline
    fn log(self: Buffer[Self.dtype]) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → log is for numeric data types only",
        ]()

        return self.unary_ops[Log]()

    @always_inline
    fn __invert__(self) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_integral() or DType.bool == Self.dtype,
            "Buffer → __invert__ is for Bool or integral data types only",
        ]()

        var out = Buffer[Self.dtype](self.size)

        alias simd_width = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        var vectorized_end = (self.size // simd_width) * simd_width

        # Vectorized invert
        for idx in range(0, vectorized_end, simd_width):
            var chunk = self.load[simdwidth=simd_width](idx)
            out.store[simdwidth=simd_width](idx, chunk.__invert__())

        # Scalar tail
        for idx in range(vectorized_end, self.size):
            out[idx] = self[idx].__invert__()

        return out^

    @always_inline
    fn compare_scalar[
        op_code: Int
    ](self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        total = self.size
        if total == 0:
            return False

        alias smdwidth = simd_width_of[Self.dtype]()

        var simd_blocks = total // smdwidth
        for block in range(simd_blocks):
            var idx = block * smdwidth

            @parameter
            if op_code == Equal:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .eq(scalar)
                    .reduce_and()
                ):
                    return False
            elif op_code == NotEqual:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .ne(scalar)
                    .reduce_and()
                ):
                    return False
            elif op_code == GreaterThanEqual:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .ge(scalar)
                    .reduce_and()
                ):
                    return False
            elif op_code == GreaterThan:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .gt(scalar)
                    .reduce_and()
                ):
                    return False
            elif op_code == LessThanEqual:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .le(scalar)
                    .reduce_and()
                ):
                    return False
            else:  # op_code == LessThan
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .lt(scalar)
                    .reduce_and()
                ):
                    return False

        i = simd_blocks * smdwidth

        for k in range(i, total):
            if not Self.compare_pair[op_code](
                self.load[simdwidth=1](k), scalar
            ):
                return False
        return True

    @always_inline
    fn __eq__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        return self.compare_scalar[Equal](scalar)

    @always_inline
    fn __ne__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        return self.compare_scalar[NotEqual](scalar)

    @always_inline
    fn __lt__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        return self.compare_scalar[LessThan](scalar)

    @always_inline
    fn __le__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        return self.compare_scalar[LessThanEqual](scalar)

    @always_inline
    fn __gt__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        return self.compare_scalar[GreaterThan](scalar)

    @always_inline
    fn __ge__(self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]) -> Bool:
        return self.compare_scalar[GreaterThanEqual](scalar)

    @always_inline
    fn eq(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_scalar_full[Equal](scalar)

    @always_inline
    fn ne(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_scalar_full[NotEqual](scalar)

    @always_inline
    fn ge(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_scalar_full[GreaterThanEqual](scalar)

    @always_inline
    fn gt(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_scalar_full[GreaterThan](scalar)

    @always_inline
    fn le(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_scalar_full[LessThanEqual](scalar)

    @always_inline
    fn lt(
        self: Buffer[Self.dtype], scalar: Scalar[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_scalar_full[LessThan](scalar)

    @always_inline
    fn eq(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_buffer_full[Equal](other)

    @always_inline
    fn ne(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_buffer_full[NotEqual](other)

    @always_inline
    fn lt(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_buffer_full[LessThan](other)

    @always_inline
    fn le(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_buffer_full[LessThanEqual](other)

    @always_inline
    fn gt(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_buffer_full[GreaterThan](other)

    @always_inline
    fn ge(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[DType.bool]:
        return self.compare_buffer_full[GreaterThanEqual](other)

    @always_inline
    fn compare_buffer[
        op_code: Int
    ](self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        if not self.size == other.size:
            panic(
                "Buffer → compare_buffer: buffer sizes do not match -> self:",
                self.size.__str__(),
                "vs. other:",
                other.size.__str__(),
            )

        total = self.size
        if total == 0:
            return False

        alias smdwidth = simd_width_of[Self.dtype]()

        var simd_blocks = total // smdwidth
        for block in range(simd_blocks):
            var idx = block * smdwidth

            @parameter
            if op_code == Equal:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .eq(other.load[simdwidth=smdwidth](idx))
                    .reduce_and()
                ):
                    return False
            elif op_code == NotEqual:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .ne(other.load[simdwidth=smdwidth](idx))
                    .reduce_and()
                ):
                    return False
            elif op_code == GreaterThanEqual:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .ge(other.load[simdwidth=smdwidth](idx))
                    .reduce_and()
                ):
                    return False
            elif op_code == GreaterThan:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .gt(other.load[simdwidth=smdwidth](idx))
                    .reduce_and()
                ):
                    return False
            elif op_code == LessThanEqual:
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .le(other.load[simdwidth=smdwidth](idx))
                    .reduce_and()
                ):
                    return False
            else:  # op_code == LessThan
                if (
                    not self.load[simdwidth=smdwidth](idx)
                    .lt(other.load[simdwidth=smdwidth](idx))
                    .reduce_and()
                ):
                    return False

        i = simd_blocks * smdwidth

        for k in range(i, total):
            if not Self.compare_pair[op_code](
                self.load[simdwidth=1](k), other.load[simdwidth=1](k)
            ):
                return False
        return True

    @always_inline
    fn __eq__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        return self.compare_buffer[Equal](other)

    @always_inline
    fn __ne__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        return self.compare_buffer[NotEqual](other)

    @always_inline
    fn __lt__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        return self.compare_buffer[LessThan](other)

    @always_inline
    fn __le__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        return self.compare_buffer[LessThanEqual](other)

    @always_inline
    fn __gt__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        return self.compare_buffer[GreaterThan](other)

    @always_inline
    fn __ge__(self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        return self.compare_buffer[GreaterThanEqual](other)

    @always_inline
    @staticmethod
    fn compare_pair[
        op_code: Int
    ](left: Scalar[Self.dtype], right: Scalar[Self.dtype]) -> Bool:
        @parameter
        if op_code == Equal:
            return left == right
        elif op_code == NotEqual:
            return left != right
        elif op_code == GreaterThanEqual:
            return left >= right
        elif op_code == GreaterThan:
            return left > right
        elif op_code == LessThanEqual:
            return left <= right
        else:  # op_code == LessThan
            return left < right

    @always_inline
    fn float(self) -> Buffer[DType.float32]:
        return self.to_dtype[DType.float32]()

    @always_inline
    fn float64(self) -> Buffer[DType.float64]:
        return self.to_dtype[DType.float64]()

    @always_inline
    fn to_dtype[
        NewType: DType, simdwidth: Int = simd_width_of[NewType]()
    ](self) -> Buffer[NewType]:
        total = self.size
        out = Buffer[NewType](total)

        @parameter
        if (
            (Self.dtype == NewType)
            or (Self.dtype == DType.bool)
            or (NewType == DType.bool)
        ):
            # Handle element-by-element cases:
            # - Same type conversion
            # - Converting FROM bool (bit-packed source)
            # - Converting TO bool (bit-packed destination)
            for i in range(total):
                out[i] = self[i].cast[NewType]()
        else:
            # Both types are non-bool and different: use efficient SIMD vectorization
            alias input_simd_width = simd_width_of[Self.dtype]()

            # Use the smaller of the two SIMD widths for safe vectorization
            alias min_simd_width = input_simd_width if input_simd_width < simdwidth else simdwidth

            var vectorized_end = (self.size // min_simd_width) * min_simd_width

            for idx in range(0, vectorized_end, min_simd_width):
                var chunk = self.load[simdwidth=min_simd_width](idx)
                out.store[simdwidth=min_simd_width](idx, chunk.cast[NewType]())

            # Scalar tail
            for idx in range(vectorized_end, self.size):
                out[idx] = self[idx].cast[NewType]()

        return out^

    @always_inline
    fn dot(
        lhs: Buffer[Self.dtype], rhs: Buffer[Self.dtype]
    ) -> Scalar[Self.dtype]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → dot: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        return (lhs * rhs).sum()

    @always_inline
    fn overwrite(
        self,
        other: Buffer[Self.dtype],
        self_start: Int = 0,
        self_end: Optional[Int] = None,
        other_start: Int = 0,
        other_end: Optional[Int] = None,
    ):
        self.inplace_ops[Overwrite, True](
            other, self_start, self_end, other_start, other_end
        )

    fn count(
        self: Buffer[Self.dtype],
        key: Scalar[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Int:
        if self.size == 0:
            return 0

        var actual_end = end_index.or_else(self.size)
        var extent = actual_end - start_index

        if extent <= 0:
            return 0

        var total = 0

        @parameter
        if Self.dtype == DType.bool:
            # Special handling for bool due to bit packing
            # Process element by element (no SIMD for packed bools)
            for i in range(start_index, actual_end):
                if self[i] == key:
                    total += 1
            return total

        # SIMD vectorization for non-bool types
        alias simd_width = simd_width_of[Self.dtype]()

        var idx = start_index
        var remaining = extent

        # Process full SIMD chunks
        while remaining >= simd_width:
            var block = self.load[simdwidth=simd_width](idx)
            var result = block.eq(key)

            # Check if all match
            if result.reduce_and():
                total += simd_width
            elif result.reduce_or():
                # Some match - count individually
                for i in range(simd_width):
                    if result[i]:
                        total += 1
            # If reduce_or() is false, no matches - skip

            idx += simd_width
            remaining -= simd_width

        # Process remaining elements (scalar tail)
        for i in range(idx, actual_end):
            if self[i] == key:
                total += 1

        return total

    fn tolist(self: Buffer[Self.dtype]) -> List[Scalar[Self.dtype]]:
        var result = List[Scalar[Self.dtype]](capacity=Int(len(self)))
        for i in range(len(self)):
            result.append(self[i])
        return result^

    @always_inline
    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self: Buffer[Self.dtype], other: Buffer[Self.dtype]) -> Bool:
        """Check if all elements are close within tolerance: |a - b| <= atol + rtol * |b|.
        """
        constrained[
            Self.dtype.is_floating_point(),
            "Buffer → all_close is for floating point data types only",
        ]()

        if self.size != other.size:
            return False

        if self.size == 0:
            return True

        @parameter
        fn check_close[smdwidth: Int](idx: Int) -> Bool:
            vec1 = self.load[simdwidth=smdwidth](idx)
            vec2 = other.load[simdwidth=smdwidth](idx)
            diff = abs(vec1 - vec2)
            tolerance = atol + rtol * abs(vec2)
            return diff.le(tolerance).reduce_and()

        alias simd_width = simd_width_of[Self.dtype]()
        num_elems = self.size
        simd_blocks = num_elems // simd_width

        for i in range(simd_blocks):
            if not check_close[simd_width](i * simd_width):
                return False

        # Handle tail
        for k in range(simd_blocks * simd_width, num_elems):
            if abs(self[k] - other[k]) > atol + rtol * abs(other[k]):
                return False

        return True

    @always_inline
    fn any(
        self: Buffer[Self.dtype], pred: fn (Scalar[Self.dtype]) -> Bool
    ) -> Bool:
        """Check if any element satisfies the predicate."""
        for i in range(self.size):
            if pred(self[i]):
                return True
        return False

    @always_inline
    fn all(
        self: Buffer[Self.dtype], pred: fn (Scalar[Self.dtype]) -> Bool
    ) -> Bool:
        """Check if all elements satisfy the predicate."""
        for i in range(self.size):
            if not pred(self[i]):
                return False
        return True

    @always_inline
    fn map_to_bool(
        self: Buffer[Self.dtype], pred: fn (Scalar[Self.dtype]) -> Bool
    ) -> Buffer[DType.bool]:
        """Apply predicate to each element, returning a boolean buffer."""
        var out = Buffer[DType.bool](self.size)
        for i in range(self.size):
            out[i] = pred(self[i])
        return out^

    @always_inline
    fn all_true(self: Buffer[DType.bool]) -> Bool:
        """Check if all elements in a boolean buffer are True."""
        if self.size == 0:
            return True

        for i in range(self.size):
            if not self[i]:
                return False
        return True

    @always_inline
    fn any_true(self: Buffer[DType.bool]) -> Bool:
        """Check if any element in a boolean buffer is True."""
        for i in range(self.size):
            if self[i]:
                return True
        return False

    fn string(self) -> String:
        var result = "Buffer["

        # Use self.size instead of len(self) - self.size has actual count
        if self.size <= 60:
            for i in range(self.size):
                result += self.load(i).__str__()
                if i < self.size - 1:
                    result += ", "
        else:
            for i in range(15):
                result += self.load(i).__str__()
                result += ", "

            result += "..., "

            for i in range(self.size - 15, self.size):
                result += self.load(i).__str__()
                if i < self.size - 1:
                    result += ", "

        result += (
            "], dtype="
            + self.dtype.__str__()
            + ", size="
            + self.size.__str__()
            + "]"
        )
        return result

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        length = len(self)
        writer.write("Buffer[")
        if length <= 60:
            for i in range(length):
                writer.write(self.load(i))
                if i < length - 1:
                    writer.write(", ")
        else:
            for i in range(15):
                writer.write(self.load(i))
                writer.write(", ")

            writer.write("..., ")
            for i in range(length - 15, length):
                writer.write(self.load(i))
                if i < length - 1:
                    writer.write(", ")

        writer.write(", dtype=", self.dtype, ", size=", length, "]")

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()


@register_passable
struct ElementIterator[
    dtype: DType,
    origin: Origin[False],
](Sized & Copyable):
    var index: Int
    var src: Pointer[Buffer[Self.dtype], Self.origin]

    fn __init__(out self, src: Pointer[Buffer[Self.dtype], Self.origin]):
        self.src = src
        self.index = 0

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(mut self) -> Scalar[Self.dtype]:
        self.index += 1
        return self.src[][self.index - 1]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index
