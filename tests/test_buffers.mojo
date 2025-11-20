from buffers import Buffer


fn test_lt() raises:
    print("test_lt")
    size = 68
    ll = List[Scalar[DType.int64]](capacity=size)
    for _ in range(size):
        ll.append(Scalar[DType.int64](2))
    ll[0] = 5
    ll[65] = 42
    buffer = Buffer[DType.int64](ll)
    cmp_result = buffer.lt(5)

    # Assertions
    assert_true(
        cmp_result[0] == False,
        "buffer.lt: index 0 should be False (5 is not < 5)",
    )
    assert_true(
        cmp_result[1] == True, "buffer.lt: index 1 should be True (2 < 5)"
    )
    assert_true(
        cmp_result[65] == False,
        "buffer.lt: index 65 should be False (42 is not < 5)",
    )
    assert_true(
        cmp_result[67] == True, "buffer.lt: index 67 should be True (2 < 5)"
    )
    print("test_lt passed")


fn test_lt_breaks_original() raises:
    print("test_lt_breaks_original")
    size = 20
    ll = List[Scalar[DType.float32]](capacity=size)

    # Create a pattern that will show the bug clearly
    for i in range(size):
        ll.append(Scalar[DType.float32](i))

    buffer = Buffer[DType.float32](ll)

    cmp_result = buffer.lt(10.0)

    # Assertions
    assert_true(cmp_result[0] == True, "buffer.lt: 0 < 10 should be True")
    assert_true(cmp_result[9] == True, "buffer.lt: 9 < 10 should be True")
    assert_true(cmp_result[10] == False, "buffer.lt: 10 < 10 should be False")
    assert_true(cmp_result[15] == False, "buffer.lt: 15 < 10 should be False")
    assert_true(cmp_result[19] == False, "buffer.lt: 19 < 10 should be False")
    print("test_lt_breaks_original passed")


fn test_simd_indexing() raises:
    print("test_simd_indexing")
    alias simd_width = 8

    # Create a SIMD vector with distinct values
    var vec = SIMD[DType.int32, simd_width](0, 1, 2, 3, 4, 5, 6, 7)

    print("Direct indexing:")
    for i in range(simd_width):
        assert_true(vec[i] == i, "SIMD vector indexing failed")

    print("\nAttempting out of bounds indexing (expecting crash):")
    # This will segfault - commenting out for safety
    # var should_crash = vec[8]
    print("test_simd_indexing passed (in-bounds checks only)")


fn test_what_values() raises:
    print("test_what_values")
    alias simd_width = 8

    var buffer = Buffer[DType.float32](16)
    for i in range(16):
        buffer[i] = i

    for block in range(2):
        idx = block * simd_width
        cmp = buffer.load[simd_width](idx).lt(10.0)

        # Verify expected values for Block 0
        if block == 0:
            for k in range(simd_width):
                assert_true(
                    cmp[k] == True, "Block 0: all values 0-7 should be < 10"
                )

        # Verify expected values for Block 1
        if block == 1:
            assert_true(
                cmp[0] == True, "Block 1: cmp[0] (value 8) should be < 10"
            )
            assert_true(
                cmp[1] == True, "Block 1: cmp[1] (value 9) should be < 10"
            )
            for k in range(2, simd_width):
                assert_true(
                    cmp[k] == False,
                    "Block 1: cmp["
                    + k.__str__()
                    + "] (value "
                    + (8 + k).__str__()
                    + ") should be >= 10",
                )

    print("test_what_values passed")


fn test_to_dtype_same_type() raises:
    print("test_to_dtype_same_type")
    var buffer = Buffer[DType.int32](5)
    for i in range(5):
        buffer[i] = i * 10

    var result = buffer.to_dtype[DType.int32]()

    for i in range(5):
        assert_true(
            result[i] == buffer[i],
            "Same type conversion should preserve values",
        )

    print("test_to_dtype_same_type passed")


fn test_to_dtype_non_bool() raises:
    print("test_to_dtype_non_bool")
    var buffer = Buffer[DType.int32](5)
    for i in range(5):
        buffer[i] = i * 10

    var result = buffer.to_dtype[DType.float32]()

    for i in range(5):
        assert_true(
            result[i] == Scalar[DType.float32](i * 10),
            "int32 to float32 conversion failed",
        )

    print("test_to_dtype_non_bool passed")


fn test_to_dtype_to_bool() raises:
    print("test_to_dtype_to_bool")
    var buffer = Buffer[DType.int32](5)
    buffer[0] = 0
    buffer[1] = 1
    buffer[2] = 42
    buffer[3] = 0
    buffer[4] = -5

    var result = buffer.to_dtype[DType.bool]()

    assert_true(result[0] == False, "0 should cast to False")
    assert_true(result[1] == True, "1 should cast to True")
    assert_true(result[2] == True, "42 should cast to True")
    assert_true(result[3] == False, "0 should cast to False")
    assert_true(result[4] == True, "-5 should cast to True")

    print("test_to_dtype_to_bool passed")


fn test_to_dtype_from_bool() raises:
    print("test_to_dtype_from_bool")
    var buffer = Buffer[DType.bool](5)
    buffer[0] = False
    buffer[1] = True
    buffer[2] = True
    buffer[3] = False
    buffer[4] = True

    var result = buffer.to_dtype[DType.int32]()

    assert_true(result[0] == 0, "False should cast to 0")
    assert_true(result[1] == 1, "True should cast to 1")
    assert_true(result[2] == 1, "True should cast to 1")
    assert_true(result[3] == 0, "False should cast to 0")
    assert_true(result[4] == 1, "True should cast to 1")

    print("test_to_dtype_from_bool passed")


fn test_to_dtype_large_buffer() raises:
    print("test_to_dtype_large_buffer")
    size = 100
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = i * 0.5

    var result = buffer.to_dtype[DType.int32]()

    for i in range(size):
        expected = Int32(i * 0.5)
        assert_true(
            result[i] == expected,
            "Large buffer conversion failed at index " + i.__str__(),
        )

    print("test_to_dtype_large_buffer passed")


fn run_all_tests() raises:
    print("=" * 50)
    print("Running all buffer tests")
    print("=" * 50)

    test_lt()
    test_lt_breaks_original()
    test_simd_indexing()
    test_what_values()
    test_to_dtype_same_type()
    test_to_dtype_non_bool()
    test_to_dtype_to_bool()
    test_to_dtype_from_bool()
    test_to_dtype_large_buffer()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


fn main() raises:
    print("Running buffer tests")
    test_overwrite()
    test_fill_segment()
    test_buffer_iter()
    test_buffer_slice()
    test_buffer_buffer_add()
    test_buffer_buffer_mul()
    test_buffer_scalar_float_greater_than()
    test_buffer_scalar_float_less_than_eq()
    test_buffer_scalar_float_greater_than_eq()
    test_buffer_scalar_float_less_than()
    test_buffer_scalar_float_equality()
    test_buffer_scalar_float_inequality()
    test_buffer_float_equality()
    test_buffer_dot()
    test_buffer_prod()
    test_buffer_sum()
    test_buffer_float_greater_than_eq()
    test_buffer_float_greater_than()
    test_buffer_float_less_than()
    test_buffer_float_inequality()
    test_buffer_float_less_eq_than()
    test_count()
    test_log()
    run_all_tests()
    print("Done running buffer tests")


from testing import assert_true, assert_false


fn test_count() raises:
    print("test_count")
    size = 135
    ll = List[Scalar[DType.int64]](capacity=size)
    for _ in range(size):
        ll.append(Scalar[DType.int64](2))
    buffer = Buffer[DType.int64](ll)
    assert_true(
        buffer.count(Scalar[DType.int64](2)) == size, "count assertion 1 failed"
    )

    buffer[0] = 3
    assert_true(
        buffer.count(2) == size - 1,
        "count assertion 2 failed",
    )
    buffer[10] = 3
    assert_true(
        buffer.count(2) == size - 2,
        "count assertion 3 failed",
    )

    assert_true(
        buffer.count(3) == 2,
        "count assertion 4 failed",
    )

    assert_true(
        buffer.count(42) == 0,
        "count assertion 5 failed",
    )

    lb = List[Scalar[DType.bool]](capacity=size)
    for _ in range(size):
        lb.append(Scalar[DType.bool](True))
    buffer_b = Buffer[DType.bool](lb)
    assert_true(buffer_b.count(True) == size, "count assertion 6 failed")

    buffer_b[0] = False
    assert_true(
        buffer_b.count(True) == size - 1,
        "count assertion 7 failed",
    )
    buffer_b[10] = False
    assert_true(
        buffer_b.count(True) == size - 2,
        "count assertion 8 failed",
    )

    assert_true(
        buffer_b.count(False) == 2,
        "count assertion 9 failed",
    )


fn test_log() raises:
    print("test_log")
    ll = List[Scalar[DType.float32]](capacity=100)
    for i in range(1, 100):
        ll.insert(0, Scalar[DType.float32](i))
    buf = Buffer[DType.float32](ll)
    logs = buf.log()

    assert_true(logs[len(logs) - 1] == 0, "Buffer log zero assertion failed")
    assert_true(
        logs[0] == 4.59512, "Buffer log assertion failed for value at index 0"
    )


fn test_buffer_iter() raises:
    print("test_buffer_iter")
    buff = Buffer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sliced = buff[4:1:-1]
    var expect = 5
    for elem in sliced:
        assert_true(elem == expect, "Buffer iter assertion failed")
        expect -= 1


fn test_buffer_slice() raises:
    print("test_buffer_slice")
    buff = Buffer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sliced = buff[4:1:-2]
    assert_true(
        (sliced == Buffer([5, 3])),
        "Buffer slicing assertion failed",
    )


fn test_buffer_buffer_mul() raises:
    print("test_buffer_buffer_mul")
    x = Buffer[DType.bool](129)
    x.fill(Scalar[DType.bool](True))
    y = Buffer[DType.bool](129)
    y.fill(Scalar[DType.bool](True))
    expect = Buffer[DType.bool](129)
    expect.fill(Scalar[DType.bool](True))
    mul_result = x * y
    cmp_result = mul_result == expect
    assert_true(
        cmp_result,
        "Buffer buffer mul for boolean - assertion failed",
    )
    y.fill(Scalar[DType.bool](False))
    expect.fill(Scalar[DType.bool](True))
    mul_result = x * y
    cmp_result = mul_result == expect
    assert_false(
        cmp_result,
        "Buffer buffer mul for boolean(True * False) - assertion failed",
    )


fn test_buffer_buffer_add() raises:
    print("test_buffer_buffer_add")
    a = Buffer(72)
    a.fill(43.0)
    b = Buffer(72)
    b.fill(43.0)
    expected = Buffer(72)
    expected.fill(86)
    added = a + b
    result = added == expected
    assert_true(
        result,
        "Buffer buffer add assertion failed",
    )


fn test_buffer_scalar_float_greater_than_eq() raises:
    print("test_buffer_scalar_float_greater_than_eq")
    a = Buffer(72)
    a.fill(43.0)
    result1 = a >= 42
    a.fill(42.0)
    result2 = a >= 42
    assert_true(
        result1 and result2,
        "Buffer scalar float greater than eq assertion failed",
    )


fn test_buffer_scalar_float_less_than_eq() raises:
    print("test_buffer_scalar_float_less_than_eq")
    a = Buffer(72)
    a.fill(42.0)
    result1 = a <= 43
    result2 = a <= 42
    assert_true(
        result1 and result2,
        "Buffer scalar float less than eq assertion failed",
    )


fn test_buffer_scalar_float_greater_than() raises:
    print("test_buffer_scalar_float_greater_than")
    a = Buffer(72)
    a.fill(42.0)
    result = a > 41
    assert_true(result, "Buffer scalar float greater than assertion failed")


fn test_buffer_scalar_float_less_than() raises:
    print("test_buffer_scalar_float_less_than")
    a = Buffer(72)
    a.fill(42.0)
    result = a < 43
    assert_true(result, "Buffer scalar float less than assertion failed")


fn test_buffer_scalar_float_inequality() raises:
    print("test_buffer_scalar_float_inequality")
    a = Buffer(72)
    a.fill(42.0)
    result = a != 43
    assert_true(result, "Buffer scalar float inequality assertion failed")


fn test_buffer_scalar_float_equality() raises:
    print("test_buffer_scalar_float_equality")
    a = Buffer(72)
    a.fill(42.0)
    result = a == 42
    assert_true(result, "Buffer scalar float equality assertion failed")


fn test_buffer_dot() raises:
    print("test_buffer_dot")
    a = Buffer(33)
    a.fill(42.0)
    b = Buffer(33)
    b.fill(2.0)
    assert_true(
        a.dot(b) == b.dot(a) and a.dot(b) == 2772, "dot assertion failed"
    )


fn test_buffer_prod() raises:
    print("test_buffer_prod")
    a = Buffer(2)
    a.fill(42.0)
    result = a.product()
    assert_true(result == 1764, "prod assertion failed")


fn test_buffer_sum() raises:
    print("test_buffer_sum")
    a = Buffer(72)
    a.fill(42.0)
    result = a.sum()
    assert_true(result == 3024, "Sum assertion failed")


fn test_buffer_float_greater_than_eq() raises:
    print("test_buffer_float_greater_than_eq")
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = b >= a
    assert_true(result, "72 float greater than eq assertion failed")

    a = Buffer(31)
    a.fill(42.0)
    b = Buffer(31)
    b.fill(42)
    result = b >= a
    assert_true(result, "31 float greater than eq assertion failed")


fn test_buffer_float_greater_than() raises:
    print("test_buffer_float_greater_than")
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = b > a
    assert_true(result, "72 float greater than assertion failed")


fn test_buffer_float_less_eq_than() raises:
    print("test_buffer_float_less_eq_than")
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = a <= b
    assert_true(result, "72 float less than eq assertion failed")

    a = Buffer(65)
    a.fill(42.0)
    b = Buffer(65)
    b.fill(42)
    result = a <= b
    assert_true(result, "65 float less than eq assertion failed")


fn test_buffer_float_less_than() raises:
    print("test_buffer_float_less_than")
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = a < b
    assert_true(result, "72 float less than assertion failed")


fn test_buffer_float_equality() raises:
    print("test_buffer_float_equality")
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(42)
    result = a == b
    assert_true(result, "72 float equality assertion failed")

    a = Buffer(1)
    a.fill(42.0)
    b = Buffer(1)
    b.fill(42)
    result = a == b
    assert_true(result, "1 float equality assertion failed")

    a = Buffer(1024)
    a.fill(42.0)
    b = Buffer(1024)
    b.fill(42)
    result = a == b
    assert_true(result, "1024 float equality assertion failed")


fn test_buffer_float_inequality() raises:
    print("test_buffer_float_inequality")
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = a != b
    assert_true(result, "72 float inequality assertion failed")

    a = Buffer(1)
    a.fill(42.0)
    b = Buffer(1)
    b.fill(420)
    result = a != b
    assert_true(result, "1 float inequality assertion failed")

    a = Buffer(1024)
    a.fill(42.0)
    b = Buffer(1024)
    b.fill(420)
    result = a != b
    assert_true(result, "1024 float inequality assertion failed")


fn test_fill_segment() raises:
    print("test_fill_segment")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)

    buffer.fill(42, 3, 6)
    assert_true(
        buffer
        == Buffer[dtype](
            [
                0,
                1,
                2,
                42,
                42,
                42,
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
            ]
        )
    )
    bool_buff = Buffer[DType.bool](
        [True, True, True, False, False, False, False, False, False]
    )
    bool_buff.fill(False, 0, 3)
    assert_true(
        bool_buff
        == Buffer[DType.bool](
            [False, False, False, False, False, False, False, False, False]
        )
    )


fn test_overwrite() raises:
    print("test_overwrite")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    result = Buffer[dtype]([42, 42, 42])

    buffer.overwrite(result, 3, 6)
    assert_true(
        buffer
        == Buffer[dtype](
            [
                0,
                1,
                2,
                42,
                42,
                42,
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
            ]
        )
    )
