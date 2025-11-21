from algorithm import vectorize
from sys import simd_width_of, size_of
from memory import memset_zero, memcpy, ArcPointer
from math import exp, log, ceil
from common_utils import log_debug, panic
from utils.numerics import max_finite
from os.atomic import Atomic, Consistency, fence
from operators import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    ReverseDivide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)


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
        self._refcount = UnsafePointer[
            Atomic[DType.uint64]
        ]()  # Null (not shared yet)

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
        var data_size = self.size * size_of[Scalar[dtype]]()
        var total_size = refcount_size + data_size
        var new_alloc = UnsafePointer[UInt8].alloc(total_size)

        # Initialize refcount at start
        var refcount_ptr = new_alloc.bitcast[Atomic[DType.uint64]]()
        refcount_ptr[] = Atomic[DType.uint64](1)

        # Copy data after refcount
        var new_data = new_alloc.offset(refcount_size).bitcast[Scalar[dtype]]()
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

    @staticmethod
    fn Empty() -> Buffer[dtype]:
        return Buffer[dtype]()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    fn __iter__(ref self) -> ElementIterator[dtype, origin_of(self)]:
        return ElementIterator(Pointer(to=self))

    @always_inline
    fn __getitem__(self, slice: Slice) -> Buffer[dtype]:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)
        var result_size = len(spread)

        if result_size == 0:
            return Buffer[dtype]()

        var result = Buffer[dtype](result_size)

        # Fast path: contiguous slice (step == 1)
        if step == 1:
            memcpy(dest=result.data, src=self.data + start, count=result_size)
        else:
            # Slow path: non-contiguous slice
            for i in range(result_size):
                result.data[i] = self.data[start + i * step]

        return result^

    @always_inline
    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        debug_assert(
            index >= 0 and index < self.size,
            "Buffer -> __getitem__: index out of bounds",
            self.size,
            index,
        )
        return self.data.load[width=1](index)

    @always_inline
    fn __setitem__(self, index: Int, scalar: Scalar[dtype]):
        debug_assert(
            index >= 0 and index < self.size,
            "Buffer -> __setitem__: index out of bounds",
            self.size,
            index,
        )
        self.data.store[width=1](index, scalar)

    @always_inline
    fn load[simdwidth: Int = 1](self, offset: Int) -> SIMD[dtype, simdwidth]:
        debug_assert(
            offset >= 0 and offset + simdwidth <= self.size,
            "Buffer -> load : offset out of bounds",
            self.size,
            offset,
            simdwidth,
        )
        return self.data.load[width=simdwidth,](offset)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, offset: Int, values: SIMD[dtype, simdwidth]):
        debug_assert(
            offset >= 0 and offset + simdwidth <= self.size,
            "Buffer -> store : offset out of bounds",
            self.size,
            offset,
            simdwidth,
        )

        self.data.store[width=simdwidth](offset, values)

    @always_inline
    fn __add__(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
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
    fn __iadd__(self, other: Buffer[dtype]):
        constrained[
            dtype.is_numeric(),
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
    fn __isub__(self, other: Buffer[dtype]):
        constrained[
            dtype.is_numeric(),
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
    fn __sub__(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
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
    fn __mul__(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[dtype]:
        # No constraint checking for dtype - DType.bool multplication allowed
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
    fn __imul__(self, other: Buffer[dtype]):
        # No constraint checking for dtype - DType.bool multplication allowed
        if self.size != other.size:
            panic(
                (
                    "Buffer → __imul__(Buffer[dtype]: buffer size does not"
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
        self: Buffer[dtype],
        scalar: Scalar[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        var extent = end_index.or_else(self.size) - start_index
        if self.size == 0 or extent <= 0:
            return

        @parameter
        fn inplace_scalar_op[smdwidth: Int](idx: Int):
            var op_result: SIMD[dtype, smdwidth]

            @parameter
            if op_code == Multiply:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) * scalar
                )

            elif op_code == Add:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) + scalar
                )

            elif op_code == Subtract:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) - scalar
                )

            else:  # Divide
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) / scalar
                )

            self.store[simdwidth=smdwidth](start_index + idx, op_result)

        alias smdwidth = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[inplace_scalar_op, smdwidth](extent)

    @always_inline
    fn inplace_ops[
        op_code: Int, validate: Bool = True
    ](
        self: Buffer[dtype],
        other: Buffer[dtype],
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
                    "Buffer -> inplace_scalar_ops: range mismatch: self"
                    " range -> "
                    + self_extent.__str__()
                    + ", other range: ",
                    other_extent.__str__(),
                )
                return

        @parameter
        fn inplace_elems_op[smdwidth: Int](idx: Int):
            var op_result: SIMD[dtype, smdwidth]

            @parameter
            if op_code == Multiply:
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) * other.load[simdwidth=smdwidth](other_start + idx)

            elif op_code == Add:
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) + other.load[simdwidth=smdwidth](other_start + idx)

            elif op_code == Subtract:
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) - other.load[simdwidth=smdwidth](other_start + idx)

            else:  # Divide
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) / other.load[simdwidth=smdwidth](other_start + idx)

            self.store[simdwidth=smdwidth](self_start + idx, op_result)

        alias smdwidth = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[inplace_elems_op, smdwidth](self_extent)

    @always_inline
    fn arithmetic_ops[
        op_code: Int, validate: Bool = True
    ](
        self: Buffer[dtype],
        other: Buffer[dtype],
        self_start: Int = 0,
        self_end: Optional[Int] = None,
        other_start: Int = 0,
        other_end: Optional[Int] = None,
    ) -> Buffer[dtype]:
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
                    "Buffer -> arithmetic_ops: range mismatch: self range -> "
                    + self_extent.__str__()
                    + ", other range: ",
                    other_extent.__str__(),
                )

        var out = Buffer[dtype](self_extent)

        @parameter
        fn arithmetic_op[smdwidth: Int](idx: Int):
            var op_result: SIMD[dtype, smdwidth]

            @parameter
            if op_code == Multiply:
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) * other.load[simdwidth=smdwidth](other_start + idx)

            elif op_code == Add:
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) + other.load[simdwidth=smdwidth](other_start + idx)

            elif op_code == Subtract:
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) - other.load[simdwidth=smdwidth](other_start + idx)

            else:  # Divide
                op_result = self.load[simdwidth=smdwidth](
                    self_start + idx
                ) / other.load[simdwidth=smdwidth](other_start + idx)

            out.store[simdwidth=smdwidth](idx, op_result)

        alias smdwidth = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[arithmetic_op, smdwidth](self_extent)

        return out^

    @always_inline
    fn arithmetic_ops_scalar[
        op_code: Int
    ](
        self: Buffer[dtype],
        scalar: Scalar[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Buffer[dtype]:
        var extent = end_index.or_else(self.size) - start_index
        if self.size == 0 or extent <= 0:
            panic("Buffer -> arithmetic_ops_scalar: buffer size 0")

        var out = Buffer[dtype](extent)

        @parameter
        fn arithmetic_op_scalar[smdwidth: Int](idx: Int):
            var op_result: SIMD[dtype, smdwidth]

            @parameter
            if op_code == Multiply:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) * scalar
                )

            elif op_code == Add:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) + scalar
                )

            elif op_code == Subtract:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) - scalar
                )

            elif op_code == ReverseSubtract:
                op_result = scalar - self.load[simdwidth=smdwidth](
                    start_index + idx
                )
            elif op_code == Divide:
                op_result = (
                    self.load[simdwidth=smdwidth](start_index + idx) / scalar
                )

            else:  # ReverseDivide
                op_result = self.load[simdwidth=smdwidth](
                    start_index + idx
                ).__rtruediv__(scalar)

            out.store[simdwidth=smdwidth](idx, op_result)

        alias smdwidth = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[arithmetic_op_scalar, smdwidth](extent)

        return out^

    @always_inline
    fn __truediv__(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
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
    fn __itruediv__(self: Buffer[dtype], other: Buffer[dtype]):
        constrained[
            dtype.is_numeric(),
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
    fn __iadd__(self: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __iadd__(scalar) is for numeric data types only",
        ]()

        self.inplace_ops_scalar[Add](scalar)

    @always_inline
    fn __isub__(self: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __isub__(scalar) is for numeric data types only",
        ]()

        self.inplace_ops_scalar[Subtract](scalar)

    @always_inline
    fn __imul__(self: Buffer[dtype], scalar: Scalar[dtype]):
        # No constraint checking for dtype - DType.bool multplication allowed
        self.inplace_ops_scalar[Multiply](scalar)

    @always_inline
    fn __itruediv__(self: Buffer[dtype], scalar: Scalar[dtype]):
        constrained[
            dtype.is_numeric(),
            "Buffer → __itruediv__(scalar) is for numeric data types only",
        ]()

        if scalar == Scalar[dtype](0):
            panic("Buffer → __itruediv__(scalar): can not divide by zero")

        self.inplace_ops_scalar[Divide](scalar)

    @always_inline
    fn __rsub__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __rsub__(scalar) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        return self.arithmetic_ops_scalar[ReverseSubtract](scalar)

    @always_inline
    fn __sub__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __sub__(scalar) -> Buffer is for numeric data types only",
        ]()

        return self.arithmetic_ops_scalar[Subtract](scalar)

    @always_inline
    fn __rmul__(self: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __rmul__(scalar) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        return self.__mul__(factor)

    @always_inline
    fn __mul__(self: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        # No constraint checking for dtype - DType.bool multplication allowed
        return self.arithmetic_ops_scalar[Multiply](factor)

    @always_inline
    fn __radd__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __radd__(scalar) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        return self.__add__(scalar)

    @always_inline
    fn __add__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __add__(scalar) -> Buffer  is for numeric data types"
                " only"
            ),
        ]()

        return self.arithmetic_ops_scalar[Add](scalar)

    @always_inline
    fn __truediv__(
        self: Buffer[dtype], divisor: Scalar[dtype]
    ) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __truediv__(scalar) -> Buffer  is for numeric data"
                " types only"
            ),
        ]()

        return self.arithmetic_ops_scalar[Divide](divisor)

    @always_inline
    fn __rtruediv__(
        self: Buffer[dtype], scalar: Scalar[dtype]
    ) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __rtruediv__(scalar) -> Buffer  is for numeric data"
                " types only"
            ),
        ]()

        return self.arithmetic_ops_scalar[ReverseDivide](scalar)

    @always_inline
    fn __pow__(self: Buffer[dtype], exponent: Scalar[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            (
                "Buffer → __pow__(exponent) -> Buffer is for numeric data types"
                " only"
            ),
        ]()

        var out = Buffer[dtype](self.size)

        @parameter
        fn raise_to_power[smdwidth: Int](idx: Int):
            out.store[simdwidth=smdwidth](
                idx, self.load[simdwidth=smdwidth](idx).__pow__(exponent)
            )

        vectorize[raise_to_power, simd_width_of[dtype]()](self.size)
        return out^

    @always_inline
    fn __abs__(self) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __abs__ is for numeric data types only",
        ]()
        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn absolute_value[smdwidth: Int](idx: Int):
            out.store[simdwidth=smdwidth](
                idx, self.load[simdwidth=smdwidth](idx).__abs__()
            )

        vectorize[absolute_value, simd_width_of[dtype]()](total)
        return out^

    @always_inline
    fn exp(self) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → exp is for numeric data types only",
        ]()

        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn exp_elems[smdwidth: Int](idx: Int):
            out.store[simdwidth=smdwidth](
                idx, exp(self.load[simdwidth=smdwidth](idx))
            )

        vectorize[exp_elems, simd_width_of[dtype]()](total)
        return out^

    @staticmethod
    @always_inline
    fn full(value: Scalar[dtype], size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        buffer.fill(value)
        return buffer^

    @always_inline
    @staticmethod
    fn arange[
        max_arange_elements: Int = 1000000  # Safety limit to prevent infinite loops with very small steps
    ](*args: Scalar[dtype]) -> Buffer[dtype]:
        return Self.arange[max_arange_elements](args)

    @always_inline
    @staticmethod
    fn arange[
        max_arange_elements: Int = 1000000  # Safety limit to prevent infinite loops with very small steps
    ](args: VariadicList[Scalar[dtype]]) -> Buffer[dtype]:
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

        return Buffer[dtype](data^)

    @staticmethod
    @always_inline
    fn zeros(size: Int) -> Buffer[dtype]:
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
    fn fill(
        self: Buffer[dtype],
        value: Scalar[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        """Fill a segment or the whole buffer with a value."""
        extent = end_index.or_else(self.size) - start_index

        # Safety checks
        if extent <= 0:
            panic("Buffer → fill: segment size must be greater than zero")

        @parameter
        fn set_scalar[smdwidth: Int](idx: Int):
            self.store[simdwidth=smdwidth](idx + start_index, value)

        alias simd_width = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[set_scalar, simd_width](extent)

    @always_inline
    fn zero(self: Buffer[dtype]):
        memset_zero(self.data, self.size)

    @always_inline
    fn __neg__(self: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __neg__ is for numeric data types only",
        ]()

        var out = Buffer[dtype](self.size)

        @parameter
        fn negate_elems[smdwidth: Int](idx: Int):
            out.store[simdwidth=smdwidth](
                idx, self.load[simdwidth=smdwidth](idx).__neg__()
            )

        vectorize[negate_elems, simd_width_of[dtype]()](self.size)
        return out^

    @always_inline
    fn log(self: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → log is for numeric data types only",
        ]()

        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn log_of[smdwidth: Int](idx: Int):
            out.store[simdwidth=smdwidth](
                idx, log(self.load[simdwidth=smdwidth](idx))
            )

        vectorize[log_of, simd_width_of[dtype]()](total)
        return out^

    @always_inline
    fn __invert__(self: Buffer[DType.bool]) -> Buffer[DType.bool]:
        total = self.size
        out = Buffer[DType.bool](total)

        @parameter
        fn invert_elems[smdwidth: Int](idx: Int):
            out.store[simdwidth=smdwidth](
                idx, self.load[simdwidth=smdwidth](idx).__invert__()
            )

        vectorize[invert_elems, 1](total)
        return out^

    @always_inline
    fn compare_scalar[
        op_code: Int
    ](self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = self.size
        if total == 0:
            return False

        alias smdwidth = simd_width_of[dtype]()

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

            @parameter
            if op_code == Equal:
                if not self.load[simdwidth=1](k) == scalar:
                    return False
            elif op_code == NotEqual:
                if self.load[simdwidth=1](k) == scalar:
                    return False
            elif op_code == GreaterThanEqual:
                if not self.load[simdwidth=1](k) >= scalar:
                    return False
            elif op_code == GreaterThan:
                if not self.load[simdwidth=1](k) > scalar:
                    return False
            elif op_code == LessThanEqual:
                if not self.load[simdwidth=1](k) <= scalar:
                    return False
            else:  # op_code == LessThan
                if not self.load[simdwidth=1](k) < scalar:
                    return False

        return True

    @always_inline
    fn __eq__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        return self.compare_scalar[Equal](scalar)

    @always_inline
    fn __ne__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        return self.compare_scalar[NotEqual](scalar)

    @always_inline
    fn __lt__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        return self.compare_scalar[LessThan](scalar)

    @always_inline
    fn __le__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        return self.compare_scalar[LessThanEqual](scalar)

    @always_inline
    fn __gt__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        return self.compare_scalar[GreaterThan](scalar)

    @always_inline
    fn __ge__(self: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        return self.compare_scalar[GreaterThanEqual](scalar)

    @always_inline
    fn compare_scalar_full[
        op_code: Int
    ](self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[DType.bool]:
        total = self.size
        if total == 0:
            panic("Buffer -> compare_scalar_full: buffer size is zero")

        var out = Buffer[DType.bool](total)

        @parameter
        fn compare[smdwidth: Int](idx: Int):
            block = self.load[simdwidth=smdwidth](idx)
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
                out.store(idx + i, result[i])

        alias simd_width = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[compare, simd_width](total)

        return out^

    @always_inline
    fn eq(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        return self.compare_scalar_full[Equal](scalar)

    @always_inline
    fn ne(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        return self.compare_scalar_full[NotEqual](scalar)

    @always_inline
    fn ge(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        return self.compare_scalar_full[GreaterThanEqual](scalar)

    @always_inline
    fn gt(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        return self.compare_scalar_full[GreaterThan](scalar)

    @always_inline
    fn le(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        return self.compare_scalar_full[LessThanEqual](scalar)

    @always_inline
    fn lt(self: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        return self.compare_scalar_full[LessThan](scalar)

    @always_inline
    fn eq(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
        return self.compare_buffer_full[Equal](other)

    @always_inline
    fn ne(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
        return self.compare_buffer_full[NotEqual](other)

    @always_inline
    fn lt(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
        return self.compare_buffer_full[LessThan](other)

    @always_inline
    fn le(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
        return self.compare_buffer_full[LessThanEqual](other)

    @always_inline
    fn gt(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
        return self.compare_buffer_full[GreaterThan](other)

    @always_inline
    fn ge(self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
        return self.compare_buffer_full[GreaterThanEqual](other)

    @always_inline
    fn compare_buffer_full[
        op_code: Int
    ](self: Buffer[dtype], other: Buffer[dtype]) -> Buffer[DType.bool]:
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

        total = self.size
        if total == 0:
            panic("Buffer -> compare_scalar_full: buffer sizes are zero")

        var out = Buffer[DType.bool](total)

        @parameter
        fn compare[smdwidth: Int](idx: Int):
            self_block = self.load[simdwidth=smdwidth](idx)
            other_block = other.load[simdwidth=smdwidth](idx)
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
                out.store(idx + i, result[i])

        alias simd_width = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[compare, simd_width](total)

        return out^

    @always_inline
    fn compare_buffer[
        op_code: Int
    ](self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
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

        alias smdwidth = simd_width_of[dtype]()

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

            @parameter
            if op_code == Equal:
                if not self.load[simdwidth=1](k) == other.load[simdwidth=1](k):
                    return False
            elif op_code == NotEqual:
                if self.load[simdwidth=1](k) == other.load[simdwidth=1](k):
                    return False
            elif op_code == GreaterThanEqual:
                if not self.load[simdwidth=1](k) >= other.load[simdwidth=1](k):
                    return False
            elif op_code == GreaterThan:
                if not self.load[simdwidth=1](k) > other.load[simdwidth=1](k):
                    return False
            elif op_code == LessThanEqual:
                if not self.load[simdwidth=1](k) <= other.load[simdwidth=1](k):
                    return False
            else:  # op_code == LessThan
                if not self.load[simdwidth=1](k) < other.load[simdwidth=1](k):
                    return False

        return True

    @always_inline
    fn __eq__(self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
        return self.compare_buffer[Equal](other)

    @always_inline
    fn __ne__(self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
        return self.compare_buffer[NotEqual](other)

    @always_inline
    fn __lt__(self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
        return self.compare_buffer[LessThan](other)

    @always_inline
    fn __le__(self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
        return self.compare_buffer[LessThanEqual](other)

    @always_inline
    fn __gt__(self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
        return self.compare_buffer[GreaterThan](other)

    @always_inline
    fn __ge__(self: Buffer[dtype], other: Buffer[dtype]) -> Bool:
        return self.compare_buffer[GreaterThanEqual](other)

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
            (dtype == NewType)
            or (dtype == DType.bool)
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
            alias input_simd_width = simd_width_of[dtype]()

            # Use the smaller of the two SIMD widths for safe vectorization
            alias working_simd_width = input_simd_width if input_simd_width < simdwidth else simdwidth

            @parameter
            fn cast_values[simd_width: Int](idx: Int):
                out.store[simd_width](
                    idx, self.load[simd_width](idx).cast[NewType]()
                )

            vectorize[cast_values, working_simd_width](self.size)

        return out^

    @always_inline
    fn sum(
        self: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → sum is for numeric data types only",
        ]()
        var accum = Scalar[dtype](0)
        extent = end_index.or_else(self.size) - start_index
        if extent == 0:
            return accum

        @parameter
        fn sum_up[smdwidth: Int](idx: Int):
            accum += self.load[simdwidth=smdwidth](
                idx + start_index
            ).reduce_add()

        vectorize[sum_up, simd_width_of[dtype]()](extent)
        return accum

    @always_inline
    fn dot(lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Scalar[dtype]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → dot: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        return (lhs * rhs).sum()

    @always_inline
    fn product(
        self: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → product is for numeric data types only",
        ]()

        var result = Scalar[dtype](1)
        extent = end_index.or_else(self.size) - start_index

        @parameter
        fn multiply[smdwidth: Int](idx: Int):
            result *= self.load[simdwidth=smdwidth](
                idx + start_index
            ).reduce_mul()

        vectorize[multiply, simd_width_of[dtype]()](extent)

        return result

    @always_inline
    fn overwrite(
        self: Buffer[dtype],
        buffer: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ):
        """Overwrite a segment of this buffer with result data."""

        extent = end_index.or_else(self.size) - start_index

        # Safety checks
        if extent != buffer.size:
            panic(
                "Buffer → overwrite: write extent must match buffer size.",
                "start_index:",
                start_index.__str__(),
                "end_index:",
                end_index.__str__(),
                "write extent:",
                extent.__str__(),
                "self buffer size:",
                self.size.__str__(),
                "buffer size:",
                buffer.size.__str__(),
            )

        if extent == 0:
            return

        @parameter
        fn overwrite_elems[smdwidth: Int](idx: Int):
            result_vec = buffer.load[simdwidth=smdwidth](idx)
            self.store[simdwidth=smdwidth](start_index + idx, result_vec)

        alias simd_width = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[overwrite_elems, simd_width](extent)

    @always_inline
    fn count(
        self: Buffer[dtype],
        key: Scalar[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Int:
        extent = end_index.or_else(self.size) - start_index
        total = 0

        @parameter
        fn matches[smdwidth: Int](idx: Int):
            block = self.load[simdwidth=smdwidth](idx + start_index)
            result = block.eq(key)

            if result.reduce_and():
                # All elements match
                total += smdwidth
            elif not result.reduce_or():
                # No elements match - do nothing
                pass
            else:
                # Some elements match - count individually
                for i in range(smdwidth):
                    if result[i]:
                        total += 1

        alias simd_width = 1 if dtype == DType.bool else simd_width_of[dtype]()
        vectorize[matches, simd_width](extent)
        return total

    @always_inline
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
    alias dtype = DType.bool
    b = Buffer[dtype](42)
    b.fill(True, 10, 20)
    b.fill(True, 30)
    print(b)
    print()
    # c = Buffer[dtype].full(True, 42)
    # print(c)

    r = b.eq(True)

    print()
    print(r)

    _ = """c *= b
    print(c)"""

    _ = """alias dtype = DType.float32
    l = List[Scalar[dtype]](0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    b = Buffer[dtype](l)
    r = b[3:8]
    r.shared()
    r1 = r.copy()
    print(r.is_shared(), r.ref_count())
    print("************")
    print(r1, r)
    r1[0] = 78787

    print(r1, r)

    r.zero()

    print(r1, r)
    _ = r^

    print("done")

    buff = Buffer[dtype](l)
    ref is_copy = UnsafePointer(to=buff)[]

    print("Now check: ", is_copy.data == buff.data)
    is_copy[0] = 999
    print(is_copy, buff)"""


from testing import assert_true, assert_false
