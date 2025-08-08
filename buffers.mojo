from algorithm import vectorize
from sys import simdwidthof
from memory import memset_zero, memcpy

# from runtime.asyncrt import num_physical_cores
from sys import num_logical_cores, num_physical_cores
from common_utils import log_debug


@fieldwise_init
struct Buffer[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Writable & Representable
):
    var num_elements: Int
    var data: UnsafePointer[Scalar[dtype]]

    fn __init__(out self, num_elements: Int):
        self.data = UnsafePointer[Scalar[dtype]].alloc(num_elements)
        self.num_elements = num_elements

        _="""fn __init__(out self, num_elements: Int, data: UnsafePointer[Scalar[dtype]]):
        self.num_elements = num_elements
        self.data = data"""

    fn __len__(self) -> Int:
        return self.num_elements

    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        return self.data.load[width=1, volatile=True](index)

    fn __setitem__(self, index: Int, scalar: Scalar[dtype]):
        self.data.store[width=1, volatile=True](index, scalar)

    fn load[simdwidth: Int = 1](self, offset: Int) -> SIMD[dtype, simdwidth]:
        return self.data.load[width=simdwidth, volatile=True](offset)

    fn clone(self) -> Buffer[dtype]:
        data = UnsafePointer[Scalar[dtype]].alloc(self.num_elements)
        memcpy(data, self.data, self.num_elements)
        cloned = Buffer[dtype](self.num_elements, data)
        return cloned

    fn store[
        simdwidth: Int = 1
    ](self, offset: Int, values: SIMD[dtype, simdwidth]):
        self.data.store[width=simdwidth, volatile=True](offset, values)

    fn __add__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.num_elements)

        @parameter
        fn add_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) + rhs.load[simdwidth](idx)),
            )

        vectorize[add_elems, simd_width](lhs.num_elements)
        return out

    fn __iadd__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        @parameter
        fn inplace_add_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) + rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_add_elems, simd_width](lhs.num_elements)

    fn __isub__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        @parameter
        fn inplace_sub_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) - rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_sub_elems, simd_width](lhs.num_elements)

    fn __sub__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.num_elements)

        @parameter
        fn subtract_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) - rhs.load[simdwidth](idx)),
            )

        vectorize[subtract_elems, simd_width](lhs.num_elements)
        return out

    fn __mul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.num_elements)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simd_width](lhs.num_elements)
        return out

    fn __imul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        @parameter
        fn inplace_mul_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_mul_elems, simd_width](lhs.num_elements)

    fn __radd__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        return this.__add__(scalar)

    fn __add__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn add_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[add_scalar, simd_width](this.num_elements)
        return out

    fn __iadd__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        @parameter
        fn inplace_add_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[inplace_add_scalar, simd_width](this.num_elements)

    fn __rsub__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn sub_from_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, scalar - this.load[simdwidth](idx))

        vectorize[sub_from_scalar, simd_width](this.num_elements)
        return out

    fn __sub__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn sub_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) - scalar)

        vectorize[sub_scalar, simd_width](this.num_elements)
        return out

    fn __rmul__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        return this.__mul__(factor)

    fn __mul__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn mul_by_factor[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) * factor)

        vectorize[mul_by_factor, simd_width](this.num_elements)
        return out

    fn __pow__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], exponent: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn raise_to_power[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__pow__(exponent)
            )

        vectorize[raise_to_power, simd_width](this.num_elements)
        return out

    fn __truediv__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], divisor: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn divide_by_divisor[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__truediv__(divisor)
            )

        vectorize[divide_by_divisor, simd_width](this.num_elements)
        return out

    fn __rtruediv__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn divide_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__rtruediv__(scalar)
            )

        vectorize[divide_scalar, simd_width](this.num_elements)
        return out

    fn fill[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        @parameter
        fn set_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, scalar)

        vectorize[set_scalar, simd_width](this.num_elements)

    fn zero(this: Buffer[dtype]):
        memset_zero(this.data, this.num_elements)

    fn __neg__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __neg__ is for numeric data types only",
        ]()

        var out = Buffer[dtype](this.num_elements)

        @parameter
        fn negate_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__neg__())

        vectorize[negate_elems, simd_width](this.num_elements)
        return out

    fn __invert__[
        simd_width: Int = simdwidthof[DType.bool]()
    ](this: Buffer[DType.bool]) -> Buffer[DType.bool]:
        var out = Buffer[DType.bool](this.num_elements)

        @parameter
        fn invert_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__invert__())

        vectorize[invert_elems, simd_width](this.num_elements)
        return out

    fn __eq__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        var out = Buffer[dtype.bool](this.num_elements)

        @parameter
        fn compare_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__eq__(scalar))

        vectorize[compare_scalar, simd_width](this.num_elements)
        return out

    fn __ne__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        var out = Buffer[dtype.bool](this.num_elements)

        @parameter
        fn compare_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__ne__(scalar))

        vectorize[compare_scalar, simd_width](this.num_elements)
        return out

    fn __lt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        var out = Buffer[dtype.bool](this.num_elements)

        @parameter
        fn compare_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__lt__(scalar))

        vectorize[compare_scalar, simd_width](this.num_elements)
        return out

    fn __le__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        var out = Buffer[dtype.bool](this.num_elements)

        @parameter
        fn compare_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__le__(scalar))

        vectorize[compare_scalar, simd_width](this.num_elements)
        return out

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        var out = Buffer[dtype.bool](this.num_elements)

        @parameter
        fn compare_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__gt__(scalar))

        vectorize[compare_scalar, simd_width](this.num_elements)
        return out

    fn __ge__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        var out = Buffer[dtype.bool](this.num_elements)

        @parameter
        fn compare_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__ge__(scalar))

        vectorize[compare_scalar, simd_width](this.num_elements)
        return out

    fn __eq__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.num_elements)

        @parameter
        fn compare_buffer[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx).__eq__(rhs.load[simdwidth](idx))),
            )

        vectorize[compare_buffer, simd_width](lhs.num_elements)
        return out

    fn __ne__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.num_elements)

        @parameter
        fn compare_buffer[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx).__ne__(rhs.load[simdwidth](idx))),
            )

        vectorize[compare_buffer, simd_width](lhs.num_elements)
        return out

    fn __lt__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.num_elements)

        @parameter
        fn compare_buffer[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx).__lt__(rhs.load[simdwidth](idx))),
            )

        vectorize[compare_buffer, simd_width](lhs.num_elements)
        return out

    fn __le__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.num_elements)

        @parameter
        fn compare_buffer[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx).__le__(rhs.load[simdwidth](idx))),
            )

        vectorize[compare_buffer, simd_width](lhs.num_elements)
        return out

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.num_elements)

        @parameter
        fn compare_buffer[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx).__gt__(rhs.load[simdwidth](idx))),
            )

        vectorize[compare_buffer, simd_width](lhs.num_elements)
        return out

    fn __ge__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.num_elements)

        @parameter
        fn compare_buffer[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx).__ge__(rhs.load[simdwidth](idx))),
            )

        vectorize[compare_buffer, simd_width](lhs.num_elements)
        return out

    fn sum[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype]) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → sum is for numeric data types only",
        ]()

        var summ = Scalar[dtype](0)

        @parameter
        fn sum_elems[simdwidth: Int](idx: Int):
            summ += this.load[simdwidth](idx).reduce_add()

        vectorize[sum_elems, simd_width](this.num_elements)
        return summ

    fn dot(lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Scalar[dtype]:
        return (lhs * rhs).sum()

    fn product[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype]) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → sum is for numeric data types only",
        ]()

        var product = Scalar[dtype](1)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            product *= this.load[simdwidth](idx).reduce_mul()

        vectorize[mul_elems, simd_width](this.num_elements)
        return product

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
    ](
        lhs: Buffer[dtype],
        rhs: Buffer[dtype],
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ) -> Bool:
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
            if (diff > tolerance).reduce_or():
                return False
        for k in range(tail_start, num_elems):
            value1 = lhs[simd_blocks * simd_width + k]
            value2 = rhs[simd_blocks * simd_width + k]
            if abs(value1 - value2) > atol + rtol * abs(value2):
                return False

        return True

    fn any[
        simd_width: Int = simdwidthof[dtype](),
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
        simd_width: Int = simdwidthof[dtype](),
    ](
        this,
        scalar_pred: fn (Scalar[dtype]) -> Bool,  # Mandatory scalar predicate
        simd_pred: Optional[
            fn (SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]
        ] = None,  # Optional SIMD predicate
    ) -> Bool:
        num_elems = len(this)
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = this.load[simd_width](i * simd_width)
            if simd_pred:
                all_true = simd_pred.value()(vector)
                if not all_true.reduce_and():
                    return False
            else:
                for j in range(simd_width):
                    if not scalar_pred(vector[j]):
                        return False

        for k in range(remaining):
            scalar = this.load(simd_blocks * simd_width + k)
            if not scalar_pred(scalar):
                return False

        return True

    fn all_true(self: Buffer[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.for_all(all_truthy)

    fn any_true(self: Buffer[DType.bool]) -> Bool:
        fn any_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.any(any_truthy)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        length = len(self)
        writer.write("Buffer[")
        if length <= 30:
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

    fn free(owned this):
        for i in range(len(this)):
            (this.data + i).destroy_pointee()
        this.data.free()
        log_debug("Buffer__del__ → freed data pointees")
        _ = this^


fn main() raises:
    buff = Buffer(2)
    buff1 = Buffer(2)
    buff.fill(5)
    buff1.fill(2)
    r = buff.dot(buff1)
    print(r)
    cloned = buff.clone()
    print(cloned)
