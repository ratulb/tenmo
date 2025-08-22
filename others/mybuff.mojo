from algorithm import vectorize
from sys import simdwidthof
from os import abort

@fieldwise_init
struct Buffer[usr_dtype: DType, dtype: DType = DType.uint8 if usr_dtype == DType.bool else usr_dtype](
#struct Buffer[usr_dtype: DType = DType.float32, dtype: DType = DType.uint8 if usr_dtype == DType.bool else usr_dtype](
    Copyable & Movable & Sized & Stringable & Writable & Representable
):
    var size: Int
    var data: UnsafePointer[Scalar[dtype]]
    var is_packed_bool: Bool

    fn __init__(out self, size: Int):
        self.size = size
        self.is_packed_bool = (usr_dtype == DType.bool)
        alloc_size = size // (8 if usr_dtype == DType.bool else 1)
        self.data = UnsafePointer[Scalar[dtype]].alloc(alloc_size)

    fn __init__(out self):
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]]()
        self.is_packed_bool = (dtype == DType.bool)

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

    fn store_bool[width: Int](self, idx: Int, val: SIMD[DType.bool, width]):
        if not self.is_packed_bool:
            abort("Buffer not configured for packed bool storage")

        # Pack 8 bools per byte
        var packed = Scalar[dtype](0)
        for i in range(min(width, 8)):
            if val[i]:
                packed |= (1 << i)
        self.data.store[width=1](idx // 8, packed)

    fn fill[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], value: Scalar[dtype]):
        @parameter
        fn set_scalar[simdwidth: Int](idx: Int):
            if dtype != DType.bool:
                this.store[simdwidth](idx, value)

        vectorize[set_scalar, simd_width](this.size)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        length = len(self)
        writer.write("Buffer[")
        if length <= 100:
            for i in range(length):
                writer.write(self[i])
                if i < length - 1:
                    writer.write(", ")
        else:
            for i in range(15):
                writer.write(self[i])
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

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[DType.bool]:
        var out = Buffer[DType.bool](this.size)

        @parameter
        fn cmp_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) > scalar)

        vectorize[cmp_scalar, simdwidthof[DType.bool]()](this.size)
        return out


fn test_buffer_greater_than_scalar() raises:
    _="""x = Buffer[DType.float32](66)
    x.fill(42)
    cmp_result = x > 41
    print("cmp_result")
    print(cmp_result)"""
    pass


fn main() raises:
    #test_buffer_greater_than_scalar()
    pass
