from algorithm import vectorize
from sys import simd_width_of,  size_of
from memory import memset_zero, memcpy, ArcPointer
from math import exp, log, ceil
from common_utils import log_debug, panic
from utils.numerics import max_finite
from os.atomic import Atomic, Consistency, fence

struct Buffer[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Writable & Representable & Absable
):

    """
    Unified buffer with optional reference counting built-in.

    When unshared: Just manages data pointer (like before).
    When shared: Adds atomic refcount in same allocation as data.
    """
    var size: Int
    var data: UnsafePointer[Scalar[dtype]]
    var _refcount: UnsafePointer[Atomic[DType.uint64]]  # Null if not shared!
    var external: Bool

    # ========================================
    # Constructors
    # ========================================


    fn __init__(out self):
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]]()
        self._refcount = UnsafePointer[Atomic[DType.uint64]]()  # Null
        self.external = False

    fn __init__(out self, size: Int, external: Bool = False):
        if size < 0:
            panic("Buffer size must be >= 0")
        self.size = size
        self.external = external
        self._refcount = UnsafePointer[Atomic[DType.uint64]]()  # Null (not shared yet)

        if size == 0:
            self.data = UnsafePointer[Scalar[dtype]]()
        else:
            if external:
                self.data = UnsafePointer[Scalar[dtype]]()
            else:
                self.data = UnsafePointer[Scalar[dtype]].alloc(size)

    fn __init__(out self, elems: List[Scalar[dtype]]):
        var length = len(elems)
        self.data = UnsafePointer[Scalar[dtype]].alloc(length)
        self.size = length
        memcpy(dest=self.data, src=elems._data, count=length)
        self.external = False
        self._refcount = UnsafePointer[Atomic[DType.uint64]]()  # Null

    fn __init__(
        out self,
        size: Int,
        data: UnsafePointer[Scalar[dtype]],
        copy: Bool = False,
    ):
        self.size = size
        self._refcount = UnsafePointer[Atomic[DType.uint64]]()  # Null
        if copy:
            self.data = UnsafePointer[Scalar[dtype]].alloc(size)
            memcpy(dest=self.data, src=data, count=size)
            self.external = False
        else:
            self.data = data
            self.external = True

    # ========================================
    # Shared state management (NEW!)
    # ========================================

    fn is_shared(self) -> Bool:
        """Check if this buffer has ref counting enabled."""
        return self._refcount != UnsafePointer[Atomic[DType.uint64]]()


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
        var data_size = self.size *  size_of[Scalar[dtype]]()
        var total_size = refcount_size + data_size
        var new_alloc = UnsafePointer[UInt8].alloc(total_size)

        # Initialize refcount at start
        var refcount_ptr = new_alloc.bitcast[Atomic[DType.uint64]]()
        refcount_ptr[] = Atomic[DType.uint64](1)

        # Copy data after refcount
        var new_data = new_alloc.offset(refcount_size).bitcast[Scalar[dtype]]()
        memcpy(dest=new_data, src=self.data, count=self.size)

        # Free old allocation

        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.data.free()
        log_debug("Buffer__del__ → freed data pointees")

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
            _ = self._refcount[].fetch_add[ordering=Consistency.MONOTONIC](1)
        else:
            # Not shared - deep copy data
            if self.size > 0 and not self.external:
                self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
                memcpy(dest=self.data, src=other.data, count=self.size)

    fn __moveinit__(out self, deinit other: Self):
        """Move buffer - no refcount change."""
        self.size = other.size
        self.data = other.data
        self.external = other.external
        self._refcount = other._refcount

    fn __del__(deinit self):
        """Destroy buffer - handle both shared and unshared cases."""
        if self.size == 0 or not self.data.__as_bool__():
            return

        if self.external:
            return  # Don't free external data

        if self.is_shared():
            # Shared buffer - atomic decrement
            if self._refcount[].fetch_sub[ordering=Consistency.RELEASE](1) != 1:
                return  # Other references exist

            # Last reference - free everything
            fence[ordering=Consistency.ACQUIRE]()

            # Destroy data elements
            for i in range(self.size):
                (self.data + i).destroy_pointee()

            # Free allocation (starts at refcount, not data)
            var refcount_size = size_of[Atomic[DType.uint64]]()
            var alloc_start = self._refcount.bitcast[Int64]()
            alloc_start.free()

        else:
            # Unshared buffer - direct free
            for i in range(self.size):
                (self.data + i).destroy_pointee()
            self.data.free()
            log_debug("Buffer__del__ → freed unshared buffer")

    @staticmethod
    fn Empty() -> Buffer[dtype]:
        return Buffer[dtype]()


    fn __len__(self) -> Int:
        return self.size

    fn __iter__(ref self) -> ElementIterator[dtype, origin_of(self)]:
        return ElementIterator(Pointer(to=self))

    fn __getitem__(self, slice: Slice) -> Buffer[dtype]:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)

        if not len(spread):
            return Buffer[dtype]()

        # Calculate the correct size based on the actual number of elements
        var result_size = len(spread)
        var result = Buffer[dtype](result_size)

        # Use consecutive indices for the result buffer
        var result_index = 0
        for i in spread:
            result[result_index] = self[
                i
            ]  # Copy the element from source to result
            result_index += 1

        return result^

    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        return self.data.load[width=1, volatile=True](index)

    fn __setitem__(self, index: Int, scalar: Scalar[dtype]):
        self.data.store[width=1, volatile=True](index, scalar)

    @always_inline
    fn load[simdwidth: Int = 1](self, offset: Int) -> SIMD[dtype, simdwidth]:
        return self.data.load[
            width=simdwidth,
            volatile=True,
        ](offset)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, offset: Int, values: SIMD[dtype, simdwidth]):
        self.data.store[width=simdwidth, volatile=True](offset, values)

    @always_inline
    fn __add__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn add_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) + rhs.load[simdwidth](idx)),
            )

        vectorize[add_elems, simd_width](lhs.size)
        return out^

    @always_inline
    fn __iadd__[
        simd_width: Int = simd_width_of[dtype]()
    ](self, rhs: Buffer[dtype]):
        """In-place addition with another buffer."""

        # Safety checks
        if self.size != rhs.size:
            panic("Buffer __iadd__: buffer sizes must match")

        if self.size == 0:
            return

        # Use parameterized function for vectorization
        @parameter
        fn inplace_add_elems[simdwidth: Int](idx: Int):
            # Load SIMD vectors from both buffers
            var lhs_vec = self.load[simdwidth](idx)
            lhs_vec += rhs.load[simdwidth](idx)
            # Store result back to self
            self.store[simdwidth](idx, lhs_vec)

        # Vectorize the operation
        vectorize[inplace_add_elems, simd_width](self.size)

    @always_inline
    fn __isub__[
        simd_width: Int = simd_width_of[dtype]()
    ](self, rhs: Buffer[dtype]):
        if self.size != rhs.size:
            panic(
                "Buffer __isub__: buffer sizes must match: self.size: ",
                self.size.__str__(),
                "and right hand size: ",
                rhs.size.__str__(),
            )

        if self.size == 0:
            return

        @parameter
        fn inplace_sub_elems[simdwidth: Int](idx: Int):
            var lhs_vec = self.load[simdwidth](idx)
            lhs_vec -= rhs.load[simdwidth](idx)
            self.store[simdwidth](idx, lhs_vec)

        vectorize[inplace_sub_elems, simd_width](self.size)

    @always_inline
    fn __sub__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn subtract_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) - rhs.load[simdwidth](idx)),
            )

        vectorize[subtract_elems, simd_width](lhs.size)
        return out^

    fn __mul__(
        lhs: Buffer[DType.bool], rhs: Buffer[DType.bool]
    ) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __mul__(Buffer[DType.bool]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        # vectorize[mul_elems, simd_width_of[DType.bool]()](lhs.size)
        vectorize[mul_elems, 1](lhs.size)
        _ = """alias simd_width = simd_width_of[DType.bool]()
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) * rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k) == rhs.load(k)))"""
        return out^

    @always_inline
    fn __mul__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __mul__(Buffer[dtype]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        out = Buffer[dtype](lhs.size)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simd_width](lhs.size)
        return out^

    @always_inline
    fn __imul__[
        simd_width: Int = simd_width_of[dtype]()
    ](self, rhs: Buffer[dtype]):
        """SIMD-optimized in-place addition."""
        if self.size != rhs.size:
            panic(
                (
                    "Buffer → __imul__(Buffer[dtype]: buffer size does not"
                    " match -> lhs:"
                ),
                self.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        if self.size == 0:
            return

        total_size = self.size
        vectorized_size = (total_size // simd_width) * simd_width

        # Process SIMD chunks
        for i in range(0, vectorized_size, simd_width):
            var lhs_vec = self.load[simd_width](i)
            var rhs_vec = rhs.load[simd_width](i)
            self.store[simd_width](i, lhs_vec * rhs_vec)

        # Process remaining elements
        for i in range(vectorized_size, total_size):
            self[i] = self[i] * rhs[i]

    @always_inline
    fn __imul__(lhs: Buffer[DType.bool], rhs: Buffer[DType.bool]):
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __imul__(Buffer[DType.bool]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        total = lhs.size
        alias simd_width: Int = simd_width_of[dtype]()
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) * rhs.load[simd_width](idx)
            for k in range(simd_width):
                lhs.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            lhs.store(k, Scalar[DType.bool](lhs.load(k) == rhs.load(k)))

    fn __truediv__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __truediv__(rhs) is for numeric data types only",
        ]()

        if not lhs.size == rhs.size:
            panic(
                "Buffer → __trudiv__(rhs): buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        out = Buffer[dtype](lhs.size)

        @parameter
        fn div_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) / rhs.load[simdwidth](idx)),
            )

        vectorize[div_elems, simd_width](lhs.size)
        return out^

    fn __itruediv__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __itruediv__(rhs) is for numeric data types only",
        ]()

        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __itruediv__(rhs): buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        @parameter
        fn div_elems[simdwidth: Int](idx: Int):
            vec_lhs = lhs.load[simdwidth](idx)
            vec_rhs = rhs.load[simdwidth](idx)
            vec_lhs.__itruediv__(vec_rhs)
            lhs.store[simdwidth](idx, vec_lhs)

        vectorize[div_elems, simd_width](lhs.size)

    fn __radd__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __radd__(scalar) is for numeric data types only",
        ]()

        return this.__add__(scalar)

    fn __add__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __add__(scalar) is for numeric data types only",
        ]()

        var out = Buffer[dtype](this.size)

        @parameter
        fn add_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[add_scalar, simd_width](this.size)
        return out^

    fn __iadd__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __iadd__(scalar) is for numeric data types only",
        ]()

        @parameter
        fn inplace_add_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[inplace_add_scalar, simd_width](this.size)

    fn __isub__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __isub__(scalar) is for numeric data types only",
        ]()

        @parameter
        fn inplace_sub_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, this.load[simdwidth](idx) - scalar)

        vectorize[inplace_sub_scalar, simd_width](this.size)

    fn __imul__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __imul__(scalar) is for numeric data types only",
        ]()

        @parameter
        fn inplace_mul_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, this.load[simdwidth](idx) * scalar)

        vectorize[inplace_mul_scalar, simd_width](this.size)

    fn __itruediv__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __itruediv__(scalar) is for numeric data types only",
        ]()

        if scalar == Scalar[dtype](0):
            panic("Buffer → __itruediv__(scalar): can not divide by zero")

        @parameter
        fn inplace_div_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, this.load[simdwidth](idx) / scalar)

        vectorize[inplace_div_scalar, simd_width](this.size)

    fn __rsub__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn sub_from_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, scalar - this.load[simdwidth](idx))

        vectorize[sub_from_scalar, simd_width](this.size)
        return out^

    fn __sub__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn sub_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) - scalar)

        vectorize[sub_scalar, simd_width](this.size)
        return out^

    fn __rmul__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        return this.__mul__(factor)

    fn __mul__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn mul_by_factor[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) * factor)

        vectorize[mul_by_factor, simd_width](this.size)
        return out^

    fn __pow__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], exponent: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn raise_to_power[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__pow__(exponent)
            )

        vectorize[raise_to_power, simd_width](this.size)
        return out^

    fn __truediv__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], divisor: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn divide_by_divisor[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__truediv__(divisor)
            )

        vectorize[divide_by_divisor, simd_width](this.size)
        return out^

    fn __rtruediv__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn divide_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__rtruediv__(scalar)
            )

        vectorize[divide_scalar, simd_width](this.size)
        return out^

    fn __abs__(self) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __abs__ is for numeric data types only",
        ]()
        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn absolute_value[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, self.load[simdwidth](idx).__abs__())

        vectorize[absolute_value, simd_width_of[dtype]()](total)
        return out^

    fn exp(self) -> Buffer[dtype]:
        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn exp_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, exp(self.load[simdwidth](idx)))

        vectorize[exp_elems, simd_width_of[dtype]()](total)
        return out^

    @staticmethod
    @always_inline
    fn full[
        simd_width: Int = simd_width_of[dtype]()
    ](value: Scalar[dtype], size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        buffer.fill[simd_width](value)
        return buffer^

    @always_inline
    @staticmethod
    fn arange(args: VariadicList[Scalar[dtype]]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → arange is for numeric data types only",
        ]()

        start: Scalar[dtype] = 0
        end: Scalar[dtype] = max_finite[dtype]()
        step: Scalar[dtype] = 1

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
        var data = List[Scalar[dtype]](capacity=UInt(est_size))
        var value = start

        # Safety limit to prevent infinite loops with very small steps
        alias MAX_ARANGE_ELEMENTS = 1000000

        if step > 0:
            while value < end:
                data.append(value)
                value += step
                if len(data) > MAX_ARANGE_ELEMENTS:
                    panic(
                        "Buffer → arange: too many elements, possible infinite"
                        " loop"
                    )
        else:
            while value > end:
                data.append(value)
                value += step
                if len(data) > MAX_ARANGE_ELEMENTS:
                    panic(
                        "Buffer → arange: too many elements, possible infinite"
                        " loop"
                    )

        if len(data) == 0:
            panic("Buffer → arange: computed arange size is zero")

        return Buffer[dtype](data^)

    @staticmethod
    @always_inline
    fn zeros[
        simd_width: Int = simd_width_of[dtype]()
    ](size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        memset_zero(buffer.data, size)
        return buffer^

    @staticmethod
    fn linspace(
        start: Scalar[dtype],
        end: Scalar[dtype],
        steps: Int,
    ) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → linspace is for numeric data types only",
        ]()

        if steps < 1:
            panic("Buffer → linspace: steps must be at least 1")

        if steps == 1:
            var buffer = Buffer[dtype](1)
            buffer[0] = start
            return buffer^

        var step_size = (end - start) / Scalar[dtype](steps - 1)
        var buffer = Buffer[dtype](steps)

        for i in range(steps):
            buffer[i] = start + Scalar[dtype](i) * step_size

        return buffer^

    @always_inline
    fn fill[
        simd_width: Int = simd_width_of[dtype]()
    ](
        this: Buffer[dtype],
        value: Scalar[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        """Fill a segment or the whole buffer with a value."""
        actual_end = end_index.value() if end_index else this.size
        segment_size = actual_end - start_index

        # Safety checks
        if segment_size < 1:
            panic("Buffer → fill: segment size must be greater than zero")

        @parameter
        fn set_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx + start_index, value)

        if dtype == DType.bool:
            vectorize[set_scalar, 1](segment_size)
        else:
            vectorize[set_scalar, simd_width](segment_size)

    @always_inline
    fn zero(this: Buffer[dtype]):
        memset_zero(this.data, this.size)

    fn __neg__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __neg__ is for numeric data types only",
        ]()

        var out = Buffer[dtype](this.size)

        @parameter
        fn negate_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__neg__())

        vectorize[negate_elems, simd_width](this.size)
        return out^

    fn log[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → log is for numeric data types only",
        ]()

        total = this.size
        out = Buffer[dtype](total)

        @parameter
        fn log_of[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, log(this.load[simdwidth](idx)))

        vectorize[log_of, simd_width](total)
        return out^

    fn __invert__[
        simd_width: Int = simd_width_of[DType.bool]()
    ](this: Buffer[DType.bool]) -> Buffer[DType.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        _ = """@parameter
        fn invert_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__invert__())

        vectorize[invert_elems, simd_width](total)"""

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).__invert__()
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k).__invert__())
        return out^

    fn __eq__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not this.load[simd_width](idx) == scalar:
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k) == scalar:
                return False
        return True

    fn eq[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).eq(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).eq(scalar)))
        return out^

    fn __ne__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not this.load[simd_width](idx) != scalar:
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k) != scalar:
                return False
        return True

    fn ne[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).ne(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).ne(scalar)))
        return out^

    fn lt[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).lt(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k).lt(scalar))
        return out^

    fn __lt__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).lt(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).lt(scalar):
                return False
        return True

    fn __le__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).le(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).le(scalar):
                return False
        return True

    fn le[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).le(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).le(scalar)))
        return out^

    fn __gt__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).gt(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).gt(scalar):
                return False
        return True

    fn gt[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).gt(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).gt(scalar)))
        return out^

    fn __ge__[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).ge(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).ge(scalar):
                return False
        return True

    fn ge[
        simd_width: Int = simd_width_of[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).ge(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).ge(scalar)))
        return out^

    fn eq[
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → eq: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).eq(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).eq(rhs.load(k))))
        return out^

    @always_inline
    fn __eq__[
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __eq__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not lhs.load[simd_width](idx) == rhs.load[simd_width](idx):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k) == rhs.load(k):
                return False
        return True

    fn ne[
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ne__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ne(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).ne(rhs.load(k))))
        return out^

    fn __ne__[
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ne__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ne(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).ne(rhs.load(k)):
                return False
        return True

    fn lt[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → lt(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).lt(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).lt(rhs.load(k))))
        return out^

    fn __lt__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __lt__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).lt(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).lt(rhs.load(k)):
                return False
        return True

    fn le[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → le(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).le(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).le(rhs.load(k))))
        return out^

    fn __le__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __le__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not lhs.load[simd_width](idx).le(rhs.load[simd_width](idx)):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k) <= rhs.load(k):
                return False
        return True

    fn __gt__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __gt__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).gt(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).gt(rhs.load(k)):
                return False
        return True

    fn gt[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → gt(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).gt(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).gt(rhs.load(k))))
        return out^

    fn __ge__[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ge__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ge(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).ge(rhs.load(k)):
                return False
        return True

    fn ge[
        simd_width: Int = simd_width_of[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → ge: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ge(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).ge(rhs.load(k))))
        return out^

    fn float(self) -> Buffer[DType.float32]:
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Buffer[DType.float64]:
        return self.to_dtype[DType.float64]()

    fn to_dtype[
        NewType: DType, simdwidth: Int = simd_width_of[NewType]()
    ](self) -> Buffer[NewType]:
        total = self.size
        out = Buffer[NewType](total)

        @parameter
        if NewType != DType.bool:

            @parameter
            fn cast_values[simd_width: Int](idx: Int):
                out.store[simd_width](
                    idx, self.load[simd_width](idx).cast[NewType]()
                )

            vectorize[cast_values, simdwidth](self.size)
        else:
            simd_blocks = total // simdwidth
            for block in range(simd_blocks):
                idx = block * simdwidth
                cmp = self.load[simdwidth](idx).cast[NewType]()
                for k in range(simdwidth):
                    out.store[simdwidth](idx + k, cmp[idx + k])
            i = simd_blocks * simdwidth

            for k in range(i, total):
                out.store(k, self.load(k).cast[NewType]())

        return out^

    @always_inline
    fn sum[
        simd_width: Int = simd_width_of[dtype]()
    ](
        this: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → sum is for numeric data types only",
        ]()
        var summ = Scalar[dtype](0)
        total = (end_index.value() if end_index else this.size) - start_index
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = start_index + block * simd_width
            summ += this.load[simd_width](idx).reduce_add()
        i = simd_blocks * simd_width

        for k in range(i, total):
            summ += this.load(k + start_index)
        return summ

    fn dot(lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Scalar[dtype]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → dot: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        return (lhs * rhs).sum()

    fn product[
        simd_width: Int = simd_width_of[dtype](),
    ](
        this: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → product is for numeric data types only",
        ]()

        var product = Scalar[dtype](1)
        total = (end_index.value() if end_index else this.size) - start_index
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            product *= this.load[simd_width](idx).reduce_mul()
        i = simd_blocks * simd_width

        for k in range(i, total):
            product *= this.load(k)
        return product

    @always_inline
    fn overwrite[
        simd_width: Int = simd_width_of[dtype]()
    ](
        self: Buffer[dtype],
        result: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        """Overwrite a segment of this buffer with result data."""

        actual_end = end_index.value() if end_index else self.size
        segment_size = actual_end - start_index

        # Safety checks
        if segment_size != result.size:
            panic(
                (
                    "Buffer → overwrite: segment size must match result buffer"
                    " size."
                ),
                "start_index:",
                start_index.__str__(),
                "end_index:",
                actual_end.__str__(),
                "segment_size:",
                segment_size.__str__(),
                "buffer size:",
                self.size.__str__(),
                "result size:",
                result.size.__str__(),
            )

        if segment_size == 0:
            return

        # Special handling for bool due to bit packing
        if dtype == DType.bool:
            for i in range(segment_size):
                self[start_index + i] = result[i]
            return

        @parameter
        fn overwrite_elems[simdwidth: Int](idx: Int):
            result_vec = result.load[simdwidth](idx)
            self.store[simdwidth](start_index + idx, result_vec)

        vectorize[overwrite_elems, simd_width](segment_size)

    fn count[
        simd_width: Int = simd_width_of[dtype]()
    ](
        this: Buffer[dtype],
        key: Scalar[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Int:
        actual_end = end_index.value() if end_index else this.size
        total_elements = actual_end - start_index

        total = 0

        @parameter
        fn matches[simdwidth: Int](idx: Int):
            block = this.load[simdwidth](idx + start_index)
            result = block == key
            if result:
                total += simdwidth
            elif not result and simdwidth > 1:
                for i in range(simdwidth):
                    if block[i] == key:
                        total += 1

        @parameter
        if dtype == DType.bool:
            vectorize[matches, 1](total_elements)
        else:
            vectorize[matches, simd_width](total_elements)

        return total

    fn all_close[
        simd_width: Int = simd_width_of[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Buffer → all_close is for floating point data types only",
        ]()

        num_elems = len(lhs)
        simd_blocks = num_elems // simd_width
        tail_start = simd_blocks * simd_width

        for i in range(simd_blocks):
            vector1 = lhs.load[simd_width](i * simd_width)
            vector2 = rhs.load[simd_width](i * simd_width)

            diff = abs(vector1 - vector2)
            tolerance = atol + rtol * abs(vector2)
            if not diff.le(tolerance).reduce_and():
                return False

        # Handle tail (non-SIMD leftover)
        for k in range(tail_start, num_elems):
            value1 = lhs[k]
            value2 = rhs[k]
            if abs(value1 - value2).gt(atol + rtol * abs(value2)):
                return False

        return True

    fn any[
        simd_width: Int = simd_width_of[dtype](),
    ](
        this,
        scalar_pred: fn (Scalar[dtype]) -> Bool,
        simd_pred: Optional[
            fn (SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]
        ] = None,
    ) -> Bool:
        num_elems = len(this)
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = this.load[simd_width](i * simd_width)
            if simd_pred:
                any_true = simd_pred.value()(vector)
                if any_true.reduce_or():
                    return True
            else:
                for j in range(simd_width):
                    if scalar_pred(vector[j]):
                        return True

        for k in range(remaining):
            scalar = this.load(simd_blocks * simd_width + k)
            if scalar_pred(scalar):
                return True

        return False

    fn for_all[
        simd_width: Int = simd_width_of[dtype](),
    ](this: Buffer[dtype], pred: fn (Scalar[dtype]) -> Bool) -> Buffer[
        DType.bool
    ]:
        out = Buffer[DType.bool](len(this))

        for i in range(len(this)):
            out[i] = pred(this[i])

        return out^

    fn all_true[
        simd_width: Int = simd_width_of[dtype](),
    ](self: Buffer[dtype], pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        total = self.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            vals = self.load[simd_width](idx)
            for k in range(simd_width):
                if not pred(vals[k]):
                    return False

        i = simd_blocks * simd_width
        for k in range(i, total):
            if not pred(self.load(k)):
                return False

        return True

    fn all_true(buf: Buffer[DType.bool]) -> Bool:
        fn pred(scalar: Scalar[DType.bool]) -> Bool:
            return scalar == Scalar[DType.bool](True)

        return buf.all_true[1](pred)

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
    var src: Pointer[Buffer[dtype], origin]

    fn __init__(out self, src: Pointer[Buffer[dtype], origin]):
        self.src = src
        self.index = 0

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(mut self) -> Scalar[dtype]:
        self.index += 1
        return self.src[][self.index - 1]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


fn main() raises:
    alias dtype = DType.float32
    l = List[Scalar[dtype]](0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    b = Buffer[dtype](l)
    r = b[3:8]
    r.shared()
    r1 = r.copy()
    print(r.is_shared(), r.ref_count())
    _= r^

    print("done")

from testing import assert_true, assert_false
