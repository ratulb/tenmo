
from algorithm import vectorize
from sys import simdwidthof
from os import abort

@fieldwise_init
struct Buffer[usr_dtype: DType, storage_dtype: DType = DType.uint8 if usr_dtype == DType.bool else usr_dtype](Copyable & Movable):
    var size: Int
    var data: UnsafePointer[Scalar[storage_dtype]]
    var is_packed_bool: Bool

    fn __init__(out self, size: Int):
        self.size = size
        self.is_packed_bool = usr_dtype == DType.bool
        var alloc_size = size
        if usr_dtype == DType.bool:
            alloc_size = (size + 7) // 8  # Calculate bytes needed for packed bools
        self.data = UnsafePointer[Scalar[storage_dtype]].alloc(alloc_size)

    fn load[width: Int](self, idx: Int) -> SIMD[usr_dtype, width]:
        if self.is_packed_bool:
            var packed = self.data.load[width=1](idx // 8)
            var result = SIMD[DType.bool, width](False)
            for i in range(width):
                var bit_pos = (idx % 8) + i
                result[i] = (packed & (1 << (bit_pos % 8))) != 0
            return rebind[SIMD[usr_dtype, 1]](result)
        else:
            return rebind[SIMD[usr_dtype, width]](self.data.load[width=width](idx))

    fn store[width: Int](self, idx: Int, val: SIMD[usr_dtype, width]):
        if self.is_packed_bool:
            var bool_val = rebind[Scalar[DType.bool]](val)
            for i in range(width):
                var byte_idx = (idx + i) // 8
                var bit_pos = (idx + i) % 8
                var current = self.data.load[width=1](byte_idx)
                if bool_val[i]:
                    current = current | (1 << bit_pos)
                else:
                    current = current & ~(1 << bit_pos)
                self.data.store[width=1](byte_idx, current)
        else:
            self.data.store[width=width](idx, rebind[SIMD[storage_dtype, width]](val))

    fn fill[simd_width: Int = simdwidthof[usr_dtype]()](self, value: Scalar[usr_dtype]):
        @parameter
        fn set_scalar[width: Int](idx: Int):
            if usr_dtype == DType.bool:
                var bool_val = value.__bool__()
                var val = SIMD[DType.bool, width](bool_val)
                self.store[width=width](idx, rebind[Scalar[usr_dtype]](val))
            else:
                var val = SIMD[usr_dtype, width](value)
                self.store[width=width](idx, val)

        vectorize[set_scalar, simd_width](self.size)

    fn __gt__[simd_width: Int = simdwidthof[usr_dtype]()](
        self, 
        scalar: Scalar[usr_dtype]
    ) -> Buffer[DType.bool]:
        var out = Buffer[DType.bool](self.size)

        @parameter
        fn cmp_scalar[width: Int](idx: Int):
            var loaded = self.load[width=width](idx)
            var cmp = loaded > SIMD[usr_dtype, width](scalar)
            out.store[width=width](idx, rebind[SIMD[DType.bool, width]](cmp))

        vectorize[cmp_scalar, simd_width](self.size)
        return out

fn main() raises:
    # Test case
    var buf = Buffer[DType.float32](64)
    buf.fill(42.0)
    var result = buf > 40.0
    print(result.size)
