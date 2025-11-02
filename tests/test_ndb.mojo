from testing import assert_true, assert_false
from buffers import Buffer
from layout.int_tuple import IntArray
from shapes import Shape
from strides import Strides
from ndbuffer import NDBuffer
from operators import *
from intlist import IntList


fn main() raises:
    var runs = 1
    alias _dtype = DType.float32

    for _ in range(runs):
        test_ndbuffer_set_get()
        test_scalar_buffer()
        test_fill_2()
        test_broadcast_fill()
        test_zero()
        test_add()
        test_equal()
        test_dtype_conversion()
        test_element_at()
        test_ndbuffer_inplace_ops()
        test_count()
        test_unique()
        test_inplace_operations()
        test_inplace_broadcast_operations()
        test_ndbuffer_broadcast_ops()
        test_scalar_ops()
        test_compare_scalar()
        test_compare_buffer()
        test_buffer_overwrite()
        test_scalar_inplace_update()
        test_ndbuffer_fill()
        test_buffer_sum_all()
        test_buffer_sum()
    pass


fn test_buffer_sum() raises:
    print("test_buffer_sum")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))
    result = ndb.sum(IntList(0), True)
    assert_true(result.data() == Buffer[dtype]([21, 24, 27, 30, 33, 36, 39]))

    result = ndb.sum(IntList(0), False)
    assert_true(result.data() == Buffer[dtype]([21, 24, 27, 30, 33, 36, 39]))

    result = ndb.sum(IntList(0, 1), True)
    assert_true(result.data() == Buffer[dtype]([210]))

    result = ndb.sum(IntList(1), True)
    assert_true(result.data() == Buffer[dtype]([21, 70, 119]))


fn test_buffer_sum_all() raises:
    print("test_buffer_sum_all")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))

    assert_true(ndb.sum_all() == 210)
    shared = ndb.share(Shape(5, 2), offset=1, strides=Strides(2, 2))
    assert_true(shared.sum_all() == 60)
    # Scalar
    ndb = NDBuffer[dtype](Shape())
    ndb.fill(42)
    assert_true(ndb.sum_all() == 42)
    shared = ndb.share()
    assert_true(
        shared.sum_all() == 42 and shared.item() == 42 and ndb.item() == 42
    )
    # Shape(1)
    ndb = NDBuffer[dtype](Shape(1))
    ndb.fill(39)
    assert_true(ndb.sum_all() == 39 and ndb[IntList(0)] == 39)
    shared = ndb.share()
    assert_true(
        shared.sum_all() == 39 and shared.item() == 39 and ndb.item() == 39
    )


fn test_buffer_overwrite() raises:
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))
    result = Buffer[dtype]([42, 42, 42])
    ndb.data().overwrite(result, 3, 6)


fn test_compare_buffer() raises:
    print("test_compare_buffer")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb2 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 3, 4, 6]), Shape(2, 3))
    result = ndb.compare[GreaterThan](ndb2)
    assert_true(
        result.data()
        == Buffer[DType.bool]([False, False, False, True, True, False])
    )


fn test_compare_scalar() raises:
    print("test_compare_scalar")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    result = ndb.compare_scalar[GreaterThan](3)
    assert_true(
        result.data()
        == Buffer[DType.bool]([False, False, False, True, True, True])
    )

    shared = ndb.share(Shape(1, 3), strides=Strides(1, 2), offset=1)
    result = shared.compare_scalar[Equal](4)
    assert_true(result.data() == Buffer[DType.bool]([False, True, False]))


fn test_inplace_broadcast_operations() raises:
    print("test_inplace_broadcast_operations")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb2 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3]), Shape(3))
    ndb += ndb2
    assert_true(ndb.data() == Buffer[dtype]([2, 4, 6, 5, 7, 9]))

    ndb -= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6]))

    ndb_shared = ndb.share()
    ndb2_shared = ndb2.share()

    ndb_shared += ndb2_shared
    assert_true(ndb.data() == Buffer[dtype]([2, 4, 6, 5, 7, 9]))

    ndb_shared -= ndb2_shared
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6]))


fn test_inplace_operations() raises:
    print("test_inplace_operations")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](
        Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]), Shape(3, 3)
    )
    ndb2 = NDBuffer[dtype](
        Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]), Shape(3, 3)
    )
    ndb += ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2)
    ndb -= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    ndb *= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2)
    ndb /= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared = ndb.share()

    ndb += ndb2
    assert_true(shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2)
    ndb -= ndb2
    assert_true(shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    ndb *= ndb2
    assert_true(
        shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2
    )
    ndb /= ndb2
    assert_true(shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared2 = ndb.share(Shape(2, 3), offset=3)
    ndb2_shared = ndb2.share(Shape(2, 3))

    shared2 += ndb2_shared

    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 5, 7, 9, 11, 13, 15]))
    shared2 -= ndb2_shared

    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared3 = ndb.share(Shape(1, 3), offset=3, strides=Strides(1, 2))
    shared4 = ndb2.share(Shape(1, 3), strides=Strides(1, 3))

    shared3 += shared4

    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 5, 5, 10, 7, 15, 9]))


fn test_unique() raises:
    print("test_unique")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([2, 2, 3, 4, 2, 6]), Shape(2, 3))
    assert_true(ndb.unique().data() == Buffer[dtype]([2, 3, 4, 6]))


fn test_count() raises:
    print("test_count")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([2, 2, 3, 4, 2, 6]), Shape(2, 3))
    assert_true(ndb.count(2) == 3)
    shared = ndb.share()
    assert_true(
        shared.count(2) == 3 and ndb.count(2) == 3 and ndb.count(3) == 1
    )
    share2 = shared.share(Shape(5, 1), offset=1)
    assert_true(share2.count(2) == 2)
    share3 = ndb.share(Shape(2))
    assert_true(share3.count(2) == 2)
    share4 = ndb.share(Shape(1))
    assert_true(share4.count(2) == 1)


fn test_scalar_inplace_update() raises:
    print("test_scalar_inplace_update")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb.inplace_scalar_ops[Add](99)
    assert_true(ndb.data() == Buffer[dtype]([100, 101, 102, 103, 104, 105]))
    shared = ndb.share(Shape(3, 1), offset=3)
    shared.inplace_scalar_ops[Add](10)
    assert_true(ndb.data() == Buffer[dtype]([100, 101, 102, 113, 114, 115]))

    shared2 = ndb.share(Shape(1, 3), offset=0, strides=Strides(1, 2))
    shared2.inplace_scalar_ops[Add](100)
    assert_true(ndb.data() == Buffer[dtype]([200, 101, 202, 113, 214, 115]))


fn test_element_at() raises:
    print("test_element_at")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    shared = ndb.share(Shape(3, 1), offset=3)
    assert_true(
        shared.max_index() == 5 and shared.element_at(shared.max_index()) == 6
    )


fn test_scalar_ops() raises:
    print("test_scalar_ops")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb_shared = ndb.share(Shape(1, 3), offset=3)
    result = ndb_shared.scalar_ops[Add](42)
    assert_true(result.data() == Buffer[dtype]([46, 47, 48]))


fn test_dtype_conversion() raises:
    print("test_dtype_conversion")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb_shared = ndb.share(Shape(1, 3), offset=3)
    converted = ndb_shared.to_dtype[DType.float64]()

    assert_true(
        converted.data() == Buffer[DType.float64]([4, 5, 6])
        and not converted.shared()
        and converted.strides == Strides(3, 1)
        and converted._contiguous
    )


fn test_equal() raises:
    print("test_equal")
    alias dtype = DType.float32
    ndb1 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb1_shared = ndb1.share(Shape(1, 3), offset=3)
    ndb2 = NDBuffer[dtype](Buffer[dtype]([4, 10, 6]), Shape(1, 3))
    result = ndb1_shared.compare[Equal](ndb2)
    assert_true(result.data() == Buffer[DType.bool]([True, False, True]))


fn test_add() raises:
    print("test_add")
    alias dtype = DType.float32
    ndb1 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb1_shared = ndb1.share(Shape(1, 3), offset=3)
    ndb2 = NDBuffer[dtype](Buffer[dtype]([10, 20, 30]), Shape(1, 3))

    result = ndb1_shared + ndb2
    assert_true(
        result.data() == Buffer[dtype]([14, 25, 36])
        and result.shared() == False
    )


fn test_zero() raises:
    print("test_zero")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    ndb.fill(42)
    shared = ndb.share(Shape(3), offset=3)
    shared.zero()
    assert_true(ndb.data() == Buffer[dtype]([42, 42, 42, 0, 0, 0]))


fn test_broadcast_fill() raises:
    print("test_broadcast_fill")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    filler = NDBuffer[dtype](Shape(2, 1))
    filler.fill(42)
    ndb.fill(filler)
    assert_true(ndb.data() == Buffer[dtype]([42, 42, 42, 42, 42, 42]))

    filler.fill(89)
    shared = filler.share()
    ndb.fill(shared)
    assert_true(ndb.data() == Buffer[dtype]([89, 89, 89, 89, 89, 89]))


fn test_fill_2() raises:
    print("test_fill_2")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    filler = NDBuffer[dtype](Shape(2, 3))
    filler.fill(91)
    ndb.fill(filler)

    assert_true(ndb.data() == Buffer[dtype].full(91, 6))

    shared1 = ndb.share(Shape(3), offset=3)
    filler = NDBuffer[dtype](Shape(3))
    filler.fill(92)
    shared1.fill(filler)

    assert_true(shared1.data() == Buffer[dtype]([91, 91, 91, 92, 92, 92]))
    assert_true(ndb.data() == Buffer[dtype]([91, 91, 91, 92, 92, 92]))

    # Left contiguous, right non-contiguous
    ndb = NDBuffer[dtype](Shape(2, 2))
    filler = NDBuffer[dtype](Shape(2, 1, 4))
    filler.fill(102)
    filler_shared = filler.share(Shape(2, 2), offset=4)
    ndb.fill(filler_shared)

    assert_true(ndb.data() == Buffer[dtype]([102, 102, 102, 102]))
    # Both shared
    ndb = NDBuffer[dtype](Shape(2, 2))
    filler_shared.fill(31)
    ndb_shared = ndb.share()
    ndb_shared.fill(filler_shared)
    assert_true(ndb.data() == Buffer[dtype]([31, 31, 31, 31]))

    filler = NDBuffer[dtype](Shape(2, 1, 4))
    filler.fill(1919)
    filler_shared = filler.share(Shape(2, 2), strides=Strides(1, 2))
    ndb_shared.fill(filler_shared)

    assert_true(
        ndb.data() == Buffer[dtype]([1919, 1919, 1919, 1919])
        and not filler_shared._contiguous,
    )
    # Left non-contiguous and right contiguous
    filler1 = NDBuffer[dtype](Shape(2, 2))
    filler1.fill(47)

    ndb1 = NDBuffer[dtype](Shape(2, 1, 4))
    ndb1.fill(1)
    ndb_shared1 = ndb1.share(Shape(2, 2), strides=Strides(1, 2), offset=1)
    ndb_shared1.fill(filler1)

    assert_true(ndb1.data() == Buffer[dtype]([1, 47, 47, 47, 47, 1, 1, 1]))

    # left and right No contiguous

    ndb1 = NDBuffer[dtype](
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    )
    ndb1_shared = ndb1.share(Shape(2, 3), strides=Strides(1, 2), offset=12)

    ndb2 = NDBuffer[dtype](
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160
    )
    ndb2_shared = ndb2.share(Shape(2, 3), strides=Strides(1, 3), offset=0)

    ndb1_shared.fill(ndb2_shared)

    assert_true(
        ndb1.data()
        == Buffer[dtype](
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                10,
                20,
                40,
                50,
                70,
                80,
                18,
                19,
                20,
                21,
                22,
                23,
            ]
        )
    )

    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8]))
    ndb_shared = ndb.share(Shape(3), offset=2)
    ndb_shared.fill(42)
    assert_true(ndb_shared.data() == Buffer[dtype]([1, 2, 42, 42, 42, 6, 7, 8]))


fn test_scalar_buffer() raises:
    print("test_scalar_buffer")
    ndb = NDBuffer[DType.bool]()
    assert_true(ndb.is_scalar())
    ndb.fill(True)
    assert_true(ndb.item() == True)
    assert_true(ndb[IntArray()] == True)


fn test_ndbuffer_fill() raises:
    print("test_ndbuffer_fill")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(8))
    ndb.fill(42)
    expected = Buffer[dtype].full(42, 8)
    assert_true(ndb.data() == expected, "NDBuffer fill assertion 1 failed")
    assert_false(ndb.shared(), "NDBuffer not shared assertion failed")
    shared = ndb.share()
    assert_true(ndb.shared(), "NDBuffer shared assertion failed - post sharing")
    shared.fill(91)
    expected = Buffer[dtype].full(91, 8)
    assert_true(ndb.data() == expected, "NDBuffer fill assertion 2 failed")
    share2 = ndb.share(Shape(3), Strides(2), offset=2)
    share2.fill(81)
    var l: List[Scalar[dtype]] = [91, 91, 81, 91, 81, 91, 81, 91]

    expected = Buffer[dtype](l)
    assert_true(
        share2.data() == expected
        and ndb.data() == expected
        and shared.data() == expected,
        "Fill via shape, strides and offset failed",
    )
    ndb = NDBuffer[dtype]()
    filler = NDBuffer[dtype]()
    filler.fill(39)
    ndb.fill(filler)
    assert_true(ndb.item() == 39)

    filler = NDBuffer[dtype](Shape(1))
    filler.fill(42)
    ndb.fill(filler)
    assert_true(ndb.item() == 42)
    shared = ndb.share()

    filler.fill(101)
    shared.fill(filler)

    assert_true(ndb.item() == 101)

    alias _Bool = Scalar[DType.bool]

    _list = List[Scalar[DType.bool]](
        [
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
        ]
    )
    buff = Buffer[DType.bool](_list.copy())
    ndb_bool = NDBuffer[DType.bool](buff.copy())
    ndb_bool_shared = ndb_bool.share(Shape(5), offset=1)
    ndb_bool_shared.fill(False)
    assert_true(
        ndb_bool.data()
        == Buffer[DType.bool](
            [True, False, False, False, False, False, True, True, True]
        )
    )


fn test_ndbuffer_broadcast_ops() raises:
    print("test_ndbuffer_broadcast_ops")

    alias dtype = DType.float32
    buffer1 = Buffer[dtype]([42, 42, 42, 42, 42, 42])
    shape1 = Shape(2, 3)
    ndbuffer1 = NDBuffer[dtype](buffer1^, shape1)

    buffer2 = Buffer[dtype]([3, 3, 3])
    shape2 = Shape(3)
    ndbuffer2 = NDBuffer[dtype](buffer2^, shape2)

    result = ndbuffer1.arithmetic_ops[Add](ndbuffer2)
    assert_true(result.data() == (Buffer[dtype]([42, 42, 42, 42, 42, 42]) + 3))

    result = result.arithmetic_ops[Subtract](ndbuffer2)
    assert_true(result.data() == Buffer[dtype]([42, 42, 42, 42, 42, 42]))


fn test_ndbuffer_inplace_ops() raises:
    print("test_ndbuffer_inplace_ops")

    alias dtype = DType.float32
    buffer1 = Buffer[dtype](30)
    buffer1.fill(42)
    shape = Shape(5, 6)
    ndbuffer1 = NDBuffer[dtype](buffer1^, shape, None)
    index1 = IntArray(2)
    index1[0] = 4
    index1[1] = 5
    assert_true(ndbuffer1[index1] == 42, "NDBuffer get failed")

    buffer2 = Buffer[dtype](30)
    buffer2.fill(24)
    shape1 = Shape(5, 6)
    ndbuffer2 = NDBuffer[dtype](buffer2^, shape1, None)

    _shared = ndbuffer1.share(shape1)
    ndbuffer1 += ndbuffer2
    # ndbuffer1.__iadd__[check_contiguity=False](ndbuffer2)

    expected = Buffer[dtype].full(66, 30)

    assert_true(
        ndbuffer1.data() == expected, "In place add failed for NDBuffer"
    )

    shared_buffer = ndbuffer1.share(shape1)
    assert_true(shared_buffer.data() == expected, "NDBuffer sharing failed")
    assert_true(ndbuffer1.shared(), "NDBuffer buffer nullification failed")


fn test_ndbuffer_set_get() raises:
    print("test_ndbuffer_set_get")

    alias dtype = DType.float32
    buffer = Buffer[dtype](1)
    buffer[0] = 42
    shape = Shape()
    ndbuffer = NDBuffer[dtype](buffer.copy(), shape, None)
    assert_true(ndbuffer[IntArray()] == 42, "NDBuffer get failed")
    ndbuffer[IntArray()] = 97
    assert_true(ndbuffer[IntArray()] == 97, "NDBuffer get failed post update")
    assert_true(ndbuffer.item() == 97, "NDBuffer item() failed post update")
    assert_true(ndbuffer.is_scalar(), "NDBuffer is_scalar check failed")
