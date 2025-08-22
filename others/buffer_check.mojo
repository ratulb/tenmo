from testing import assert_true, assert_false
from os import abort
from algorithm import vectorize
from sys import simdwidthof
from memory import memset_zero, memcpy
from common_utils import log_debug, panic


@fieldwise_init
struct Buffer[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Writable & Representable
):
    var size: Int
    var data: UnsafePointer[Scalar[dtype]]
    alias Empty = Buffer[dtype]()

    fn __init__(out self):
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]]()

    fn __init__(out self, size: Int):
        self.data = UnsafePointer[Scalar[dtype]].alloc(size)
        self.size = size

        _ = """fn __init__(out self, size: Int, data: UnsafePointer[Scalar[dtype]]):
        self.size = size
        self.data = data"""

    fn __len__(self) -> Int:
        return self.size

    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        return self.data.load[width=1, volatile=True](index)

    fn __setitem__(self, index: Int, scalar: Scalar[dtype]):
        self.data.store[width=1, volatile=True](index, scalar)

    @always_inline
    fn load[simdwidth: Int = 1](self, offset: Int) -> SIMD[dtype, simdwidth]:
        return self.data.load[width=simdwidth, volatile=True](offset)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, offset: Int, values: SIMD[dtype, simdwidth]):
        self.data.store[width=simdwidth, volatile=True](offset, values)

    fn __add__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn add_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) + rhs.load[simdwidth](idx)),
            )

        vectorize[add_elems, simd_width](lhs.size)
        return out

    fn __sub__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn subtract_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) - rhs.load[simdwidth](idx)),
            )

        vectorize[subtract_elems, simd_width](lhs.size)
        return out

    fn __mul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            loaded = lhs.load[simdwidth](idx)
            print("left hand side loaded: ", loaded.dtype, loaded.size, loaded)
            out.data.store[width=simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simd_width](lhs.size)
        print("output of mul")
        print(out)
        return out

    fn __mul__(lhs: Buffer[DType.bool], rhs: Buffer[DType.bool]) -> Buffer[DType.bool]:
        out = Buffer[DType.bool](lhs.size)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            lhs_loaded = lhs.data.load[width=simdwidth](idx)
            rhs_loaded = rhs.data.load[width=simdwidth](idx)
            cmp = lhs_loaded * rhs_loaded
            out.data.store[width=simdwidth](
                idx, lhs_loaded * rhs_loaded
            )

        vectorize[mul_elems, simdwidthof[DType.bool]()](lhs.size)
        print("output of mul ****")
        print(out)
        return out


    fn __add__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn add_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[add_scalar, simd_width](this.size)
        return out

    @staticmethod
    fn full[
        simd_width: Int = simdwidthof[dtype]()
    ](value: Scalar[dtype], size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        buffer.fill[simd_width](value)
        return buffer

    fn fill[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], value: Scalar[dtype]):
        @parameter
        fn set_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, value)

        vectorize[set_scalar, simd_width](this.size)

    fn zero(this: Buffer[dtype]):
        memset_zero(this.data, this.size)

    fn __eq__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) == scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) == scalar)
        return out

    fn __eq__[
        dtype: DType,
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __eq__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) == rhs.load[simd_width](idx)
            # for k in range(simd_width):
            # out.store[simd_width](idx + k, cmp[idx + k])
            out.data.store[width=simd_width](
                idx, cmp.slice[simd_width, offset=0]()
            )
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) == rhs.load(k))
        print("out: ", out)
        return out

    fn float(self) -> Buffer[DType.float32]:
        if dtype == DType.float32:
            return rebind[Buffer[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Buffer[DType.float64]:
        if dtype == DType.float64:
            return rebind[Buffer[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Buffer[NewType]:
        out = Buffer[NewType](self.size)

        @parameter
        fn cast_values[simd_width: Int](idx: Int):
            out.store[simd_width](
                idx, self.load[simd_width](idx).cast[NewType]()
            )

        vectorize[cast_values, simdwidthof[NewType]()](self.size)
        return out

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
    ](this: Buffer[dtype], pred: fn (Scalar[dtype]) -> Bool) -> Buffer[
        DType.bool
    ]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            vector = this.load[simd_width](idx)

            for k in range(simd_width):
                out.store[simd_width](idx + k, pred(vector[k]))

        i = simd_blocks * simd_width
        for k in range(i, total):
            out.store(k, pred(this.load(k)))

        return out

    fn all_true[
        simd_width: Int = simdwidthof[dtype](),
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

    fn all_true[
        simd_width: Int = simdwidthof[DType.bool](),
    ](buf: Buffer[DType.bool]) -> Bool:
        fn pred(scalar: Scalar[DType.bool]) -> Bool:
            return scalar == True

        return buf.all_true[simd_width](pred)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        length = len(self)
        writer.write("Buffer[")
        if length <= 1000:
            for i in range(length):
                writer.write(self.load(i))
                if i < length - 1:
                    writer.write(", ")
        else:
            for i in range(15):
                writer.write(self.data.load(i))
                writer.write(", ")

            writer.write("..., ")
            for i in range(length - 15, length):
                writer.write(self[i])
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

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[DType.bool]:
        var out = Buffer[DType.bool](this.size)

        @parameter
        fn cmp_scalar[simdwidth: Int](idx: Int):
            print("idx and simdwidth: ", idx, simdwidth)
            out.store[simdwidth](idx, this.load[simdwidth](idx) > scalar)

        #vectorize[cmp_scalar, simdwidthof[DType.bool]()](this.size)
        vectorize[cmp_scalar, simdwidthof[DType.bool]()](this.size)
        return out

fn test_buffer_greater_than_scalar() raises:
    x = Buffer[DType.float32](66)
    x.fill(42)
    cmp_result = x > 41
    print("cmp_result")
    print()
    print()
    print(cmp_result)

fn test_buffer_buffer_mul() raises:
    x = Buffer[DType.bool](65)
    x.fill(True)
    y = Buffer[DType.bool](65)
    y.fill(True)
    expect = Buffer[DType.bool](65)
    expect.fill(True)
    mul_result = x * y
    print("mul_result")
    print()
    print()
    print(mul_result)
    _="""cmp_result = mul_result == expect
    print("cmp_result")
    print()
    print()
    print(cmp_result)
    assert_true(
        cmp_result.all_true(),
        "Buffer buffer mul for boolean - assertion failed",
    )"""
    _ = """y.fill(False)
    expect.fill(True)
    mul_result = x * y
    cmp_result = mul_result == expect
    assert_false(
        cmp_result.all_true(),
        "Buffer buffer mul for boolean(True * False) - assertion failed",
    )"""


fn test_buffer_buffer_add() raises:
    a = Buffer(72)
    a.fill(43.0)
    b = Buffer(72)
    b.fill(43.0)
    expected = Buffer(72)
    expected.fill(86)
    added = a + b
    print("added")
    print(added)
    result = added == expected
    print(result)
    _ = """assert_true(
        result.all_true(),
        "Buffer buffer add assertion failed",
    )"""


fn main() raises:
    #test_buffer_greater_than_scalar()
    #test_buffer_buffer_mul()
    test_buffer_buffer_add()

    _="""alias length = 16
    bool_store = UnsafePointer[Scalar[DType.bool]].alloc(length)
    data1 = SIMD[DType.float32, length](
        10, 20, 20, 30, 1, 1, 1, 1, 10, 20, 20, 30, 1, 1, 1, 1
    )
    data2 = SIMD[DType.float32, length](
        10, 20, 21, 31, 1, 1, 1, 1, 10, 20, 20, 30, 1, 1, 2, 2
    )"""
    _="""cmp = data1 == data2
    # bool_data = SIMD[DType.bool, length](True)
    # bool_data = SIMD[DType.bool, length](True, False, True, False)
    bool_store.store[width=length](0, cmp)"""

    _="""#loaded_data = bool_store.load[width=length](0)  # Reload as SIMD
    loaded_data = compare_scalars[length](data1, data2)
    #for i in range(length):
        #print(loaded_data[i])
    print("printing loaded data")
    print(loaded_data.data.load[width=8](0))
    print(loaded_data.load[simdwidth=8](0))
    print(loaded_data)"""


fn compare_scalars[
    simdwidth: Int, dtype: DType = DType.float32
#](lhs: SIMD[dtype, simdwidth], rhs: SIMD[dtype, simdwidth]) -> SIMD[DType.bool, simdwidth]:
](lhs: SIMD[dtype, simdwidth], rhs: SIMD[dtype, simdwidth]) -> Buffer[DType.bool]:
    write_index = 0
    var out = Buffer[DType.bool](30)
    @parameter
    fn compare[size: Int](idx: Int):
        print(size, idx, simdwidth)

        out.store[simdwidth](write_index, lhs == rhs)
        write_index += idx
    #vectorize[compare, simdwidthof[DType.bool]()](simdwidth)
    vectorize[compare, simdwidth](simdwidth)
    #var out = UnsafePointer[Scalar[DType.bool]].alloc(simdwidth)

    #out.store[simdwidth](0, lhs == rhs)
    print(out.data.load[width=simdwidth](0))
    #return out.data.load[width=simdwidth](0)
    return out
