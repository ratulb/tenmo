from buffers import Buffer

# ============================================
# Test Constants
# ============================================
alias SMALL_SIZE = 7  # Less than typical SIMD width
alias MEDIUM_SIZE = 68  # Multiple SIMD blocks + remainder
alias LARGE_SIZE = 1000  # Stress test


# ============================================
# __eq__ Tests
# ============================================
fn test_dunder_eq_all_equal_int32() raises:
    print("test_dunder_eq_all_equal_int32")
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer == 42, "All elements are 42, should return True")
    assert_true(not (buffer == 0), "No elements are 0, should return False")
    print("test_dunder_eq_all_equal_int32 passed")


fn test_dunder_eq_one_different() raises:
    print("test_dunder_eq_one_different")

    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[0] = 99
    assert_true(
        not (buffer == 100), "First element differs, should return False"
    )

    buffer[0] = 100
    buffer[MEDIUM_SIZE - 1] = 99
    assert_true(
        not (buffer == 100), "Last element differs, should return False"
    )

    buffer[MEDIUM_SIZE - 1] = 100
    buffer[MEDIUM_SIZE // 2] = 99
    assert_true(
        not (buffer == 100), "Middle element differs, should return False"
    )
    print("test_dunder_eq_one_different passed")


fn test_dunder_eq_small_buffer() raises:
    print("test_dunder_eq_small_buffer")
    var buffer = Buffer[DType.int32].full(5, SMALL_SIZE)
    assert_true(buffer == 5, "Small buffer: all equal")

    buffer[SMALL_SIZE - 1] = 6
    assert_true(not (buffer == 5), "Small buffer: one different")
    print("test_dunder_eq_small_buffer passed")


# ============================================
# __ne__ Tests
# ============================================
fn test_dunder_ne_all_different() raises:
    print("test_dunder_ne_all_different")
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer != 0, "All != 0, should return True")
    assert_true(not (buffer != 42), "None != 42, should return False")
    print("test_dunder_ne_all_different passed")


fn test_dunder_ne_one_equal() raises:
    print("test_dunder_ne_one_equal")
    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[MEDIUM_SIZE // 2] = 50

    assert_true(
        not (buffer != 50), "One element equals 50, should return False"
    )
    assert_true(
        not (buffer != 100), "Most elements equal 100, should return False"
    )
    print("test_dunder_ne_one_equal passed")


# ============================================
# __gt__ Tests
# ============================================
fn test_dunder_gt_all_greater() raises:
    print("test_dunder_gt_all_greater")
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer > 99, "All > 99")
    assert_true(not (buffer > 100), "None > 100 (equal)")
    assert_true(not (buffer > 101), "None > 101")
    print("test_dunder_gt_all_greater passed")


fn test_dunder_gt_one_not_greater() raises:
    print("test_dunder_gt_one_not_greater")
    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[0] = 50

    assert_true(not (buffer > 99), "First element 50 not > 99")
    print("test_dunder_gt_one_not_greater passed")


fn test_dunder_gt_negative() raises:
    print("test_dunder_gt_negative")
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = i - 30

    assert_true(buffer > -31, "All > -31")
    assert_true(not (buffer > -30), "First element is -30, not > -30")
    print("test_dunder_gt_negative passed")


# ============================================
# __ge__ Tests
# ============================================
fn test_dunder_ge_all_greater_equal() raises:
    print("test_dunder_ge_all_greater_equal")
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer >= 100, "All >= 100 (equal)")
    assert_true(buffer >= 50, "All >= 50")
    assert_true(not (buffer >= 101), "None >= 101")
    print("test_dunder_ge_all_greater_equal passed")


fn test_dunder_ge_one_less() raises:
    print("test_dunder_ge_one_less")
    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[MEDIUM_SIZE - 1] = 99

    assert_true(not (buffer >= 100), "Last element 99 not >= 100")
    print("test_dunder_ge_one_less passed")


# ============================================
# __lt__ Tests
# ============================================
fn test_dunder_lt_all_less() raises:
    print("test_dunder_lt_all_less")
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer < 51, "All < 51")
    assert_true(not (buffer < 50), "None < 50 (equal)")
    assert_true(not (buffer < 49), "None < 49")
    print("test_dunder_lt_all_less passed")


fn test_dunder_lt_one_not_less() raises:
    print("test_dunder_lt_one_not_less")
    var buffer = Buffer[DType.int64].full(50, MEDIUM_SIZE)
    buffer[0] = 100

    assert_true(not (buffer < 51), "First element 100 not < 51")
    print("test_dunder_lt_one_not_less passed")


fn test_dunder_lt_float32() raises:
    print("test_dunder_lt_float32")
    var buffer = Buffer[DType.float32](20)
    for i in range(20):
        buffer[i] = Scalar[DType.float32](i)

    assert_true(buffer < 20.0, "All < 20.0")
    assert_true(not (buffer < 19.0), "Last element 19.0 not < 19.0")
    print("test_dunder_lt_float32 passed")


# ============================================
# __le__ Tests
# ============================================
fn test_dunder_le_all_less_equal() raises:
    print("test_dunder_le_all_less_equal")
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer <= 50, "All <= 50 (equal)")
    assert_true(buffer <= 100, "All <= 100")
    assert_true(not (buffer <= 49), "None <= 49")
    print("test_dunder_le_all_less_equal passed")


fn test_dunder_le_one_greater() raises:
    print("test_dunder_le_one_greater")
    var buffer = Buffer[DType.int64].full(50, MEDIUM_SIZE)
    buffer[0] = 51

    assert_true(not (buffer <= 50), "First element 51 not <= 50")
    print("test_dunder_le_one_greater passed")


# ============================================
# Edge Cases
# ============================================
fn test_dunder_compare_single_element() raises:
    print("test_dunder_compare_single_element")
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 42

    assert_true(buffer == 42, "Single: == 42")
    assert_true(buffer != 0, "Single: != 0")
    assert_true(buffer > 41, "Single: > 41")
    assert_true(buffer >= 42, "Single: >= 42")
    assert_true(buffer < 43, "Single: < 43")
    assert_true(buffer <= 42, "Single: <= 42")
    print("test_dunder_compare_single_element passed")


fn test_dunder_compare_empty() raises:
    print("test_dunder_compare_empty")
    var buffer = Buffer[DType.int32]()

    assert_true(not (buffer == 0), "Empty: == returns False")
    assert_true(not (buffer != 0), "Empty: != returns False")
    assert_true(not (buffer > 0), "Empty: > returns False")
    assert_true(not (buffer >= 0), "Empty: >= returns False")
    assert_true(not (buffer < 0), "Empty: < returns False")
    assert_true(not (buffer <= 0), "Empty: <= returns False")
    print("test_dunder_compare_empty passed")


fn test_dunder_compare_simd_boundary() raises:
    print("test_dunder_compare_simd_boundary")
    # Test sizes around SIMD width (8 for int32)
    alias sizes = VariadicList[Int](1, 7, 8, 9, 15, 16, 17, 31, 32, 33)

    @parameter
    for idx in range(len(sizes)):
        alias size = sizes[idx]
        var buffer = Buffer[DType.int32].full(10, size)

        assert_true(buffer == 10, "Size " + size.__str__() + ": ==")
        assert_true(buffer >= 10, "Size " + size.__str__() + ": >=")
        assert_true(buffer <= 10, "Size " + size.__str__() + ": <=")

        buffer[size - 1] = 11
        assert_true(
            not (buffer == 10), "Size " + size.__str__() + ": != after change"
        )
        assert_true(
            not (buffer <= 10),
            "Size " + size.__str__() + ": not <= after change",
        )

    print("test_dunder_compare_simd_boundary passed")


fn test_dunder_compare_position_sensitivity() raises:
    print("test_dunder_compare_position_sensitivity")
    alias positions = VariadicList[Int](0, 7, 8, 9, 16, 33, 64, 67)

    @parameter
    for idx in range(len(positions)):
        alias pos = positions[idx]
        if pos < MEDIUM_SIZE:
            var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)
            buffer[pos] = 99

            assert_true(
                not (buffer >= 100), "Position " + pos.__str__() + ": >= fails"
            )
            assert_true(
                not (buffer == 100), "Position " + pos.__str__() + ": == fails"
            )

    print("test_dunder_compare_position_sensitivity passed")


fn test_dunder_compare_uint8() raises:
    print("test_dunder_compare_uint8")
    var buffer = Buffer[DType.uint8].full(100, MEDIUM_SIZE)

    assert_true(buffer == 100, "uint8: ==")
    assert_true(buffer > 99, "uint8: >")
    assert_true(buffer < 101, "uint8: <")
    print("test_dunder_compare_uint8 passed")


fn test_dunder_compare_float64() raises:
    print("test_dunder_compare_float64")
    var buffer = Buffer[DType.float64].full(3.14159, MEDIUM_SIZE)

    assert_true(buffer == 3.14159, "float64: ==")
    assert_true(buffer > 3.14, "float64: >")
    assert_true(buffer < 3.15, "float64: <")
    print("test_dunder_compare_float64 passed")


# ============================================
# Test Runner
# ============================================
fn run_all_dunder_comparison_tests() raises:
    print("=" * 60)
    print("Running dunder comparison tests")
    print("=" * 60)

    test_dunder_eq_all_equal_int32()
    test_dunder_eq_one_different()
    test_dunder_eq_small_buffer()
    test_dunder_ne_all_different()
    test_dunder_ne_one_equal()
    test_dunder_gt_all_greater()
    test_dunder_gt_one_not_greater()
    test_dunder_gt_negative()
    test_dunder_ge_all_greater_equal()
    test_dunder_ge_one_less()
    test_dunder_lt_all_less()
    test_dunder_lt_one_not_less()
    test_dunder_lt_float32()
    test_dunder_le_all_less_equal()
    test_dunder_le_one_greater()
    test_dunder_compare_single_element()
    test_dunder_compare_empty()
    test_dunder_compare_simd_boundary()
    test_dunder_compare_position_sensitivity()
    test_dunder_compare_uint8()
    test_dunder_compare_float64()

    print("=" * 60)
    print("All dunder comparison tests passed!")
    print("=" * 60)


# ============================================
# EQ Tests
# ============================================
fn test_eq_int32() raises:
    print("test_eq_int32")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i % 10

    var result = buffer.eq(5)

    # Check indices where i % 10 == 5: 5, 15, 25, 35, 45, 55, 65
    for i in range(size):
        expected = (i % 10) == 5
        assert_true(
            result[i] == expected,
            "test_eq_int32: index "
            + i.__str__()
            + " expected "
            + expected.__str__(),
        )
    print("test_eq_int32 passed")


fn test_eq_float64() raises:
    print("test_eq_float64")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float64](i % 5)

    var result = buffer.eq(3.0)

    for i in range(size):
        expected = (i % 5) == 3
        assert_true(
            result[i] == expected,
            "test_eq_float64: index " + i.__str__() + " failed",
        )
    print("test_eq_float64 passed")


fn test_eq_all_same() raises:
    print("test_eq_all_same")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(42, size)

    var result_true = buffer.eq(42)
    var result_false = buffer.eq(0)

    for i in range(size):
        assert_true(
            result_true[i] == True, "test_eq_all_same: all should be True"
        )
        assert_true(
            result_false[i] == False, "test_eq_all_same: all should be False"
        )
    print("test_eq_all_same passed")


fn test_eq_small_buffer() raises:
    print("test_eq_small_buffer")
    size = SMALL_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i

    var result = buffer.eq(3)

    for i in range(size):
        expected = i == 3
        assert_true(
            result[i] == expected,
            "test_eq_small_buffer: index " + i.__str__() + " failed",
        )
    print("test_eq_small_buffer passed")


# ============================================
# NE Tests
# ============================================
fn test_ne_int32() raises:
    print("test_ne_int32")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i % 10

    var result = buffer.ne(5)

    for i in range(size):
        expected = (i % 10) != 5
        assert_true(
            result[i] == expected,
            "test_ne_int32: index " + i.__str__() + " failed",
        )
    print("test_ne_int32 passed")


fn test_ne_float32() raises:
    print("test_ne_float32")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float32](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float32](i)

    var result = buffer.ne(10.0)

    for i in range(size):
        expected = i != 10
        assert_true(
            result[i] == expected,
            "test_ne_float32: index " + i.__str__() + " failed",
        )
    print("test_ne_float32 passed")


fn test_ne_all_different() raises:
    print("test_ne_all_different")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(100, size)

    var result = buffer.ne(0)

    for i in range(size):
        assert_true(
            result[i] == True, "test_ne_all_different: all should be True"
        )
    print("test_ne_all_different passed")


# ============================================
# GT Tests
# ============================================
fn test_gt_int64() raises:
    print("test_gt_int64")
    size = MEDIUM_SIZE
    var ll = List[Scalar[DType.int64]](capacity=size)
    for i in range(size):
        ll.append(Scalar[DType.int64](i))
    var buffer = Buffer[DType.int64](ll)

    var result = buffer.gt(50)

    for i in range(size):
        expected = i > 50
        assert_true(
            result[i] == expected,
            "test_gt_int64: index " + i.__str__() + " failed",
        )
    print("test_gt_int64 passed")


fn test_gt_float32() raises:
    print("test_gt_float32")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float32](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float32](i) * 0.5

    var result = buffer.gt(15.0)

    for i in range(size):
        expected = (Scalar[DType.float32](i) * 0.5) > 15.0
        assert_true(
            result[i] == expected,
            "test_gt_float32: index " + i.__str__() + " failed",
        )
    print("test_gt_float32 passed")


fn test_gt_negative_values() raises:
    print("test_gt_negative_values")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i - 30  # Values from -30 to 37

    var result = buffer.gt(0)

    for i in range(size):
        expected = (i - 30) > 0
        assert_true(
            result[i] == expected,
            "test_gt_negative_values: index " + i.__str__() + " failed",
        )
    print("test_gt_negative_values passed")


fn test_gt_boundary() raises:
    print("test_gt_boundary")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = 10
    buffer[0] = 11
    buffer[size - 1] = 9

    var result = buffer.gt(10)

    assert_true(
        result[0] == True, "test_gt_boundary: first element should be > 10"
    )
    assert_true(
        result[size - 1] == False,
        "test_gt_boundary: last element should not be > 10",
    )
    for i in range(1, size - 1):
        assert_true(
            result[i] == False,
            "test_gt_boundary: middle elements should not be > 10",
        )
    print("test_gt_boundary passed")


# ============================================
# GE Tests
# ============================================
fn test_ge_int32() raises:
    print("test_ge_int32")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i

    var result = buffer.ge(50)

    for i in range(size):
        expected = i >= 50
        assert_true(
            result[i] == expected,
            "test_ge_int32: index " + i.__str__() + " failed",
        )
    print("test_ge_int32 passed")


fn test_ge_float64() raises:
    print("test_ge_float64")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float64](i)

    var result = buffer.ge(33.0)

    for i in range(size):
        expected = i >= 33
        assert_true(
            result[i] == expected,
            "test_ge_float64: index " + i.__str__() + " failed",
        )
    print("test_ge_float64 passed")


fn test_ge_equal_value() raises:
    print("test_ge_equal_value")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(100, size)

    var result = buffer.ge(100)

    for i in range(size):
        assert_true(
            result[i] == True, "test_ge_equal_value: all should be >= 100"
        )
    print("test_ge_equal_value passed")


# ============================================
# LT Tests
# ============================================
fn test_lt_int64() raises:
    print("test_lt_int64")
    size = MEDIUM_SIZE
    var ll = List[Scalar[DType.int64]](capacity=size)
    for _ in range(size):
        ll.append(Scalar[DType.int64](2))
    ll[0] = 5
    ll[65] = 42
    var buffer = Buffer[DType.int64](ll)

    var result = buffer.lt(5)

    assert_true(result[0] == False, "test_lt_int64: 5 is not < 5")
    assert_true(result[1] == True, "test_lt_int64: 2 < 5")
    assert_true(result[65] == False, "test_lt_int64: 42 is not < 5")
    assert_true(result[67] == True, "test_lt_int64: 2 < 5")
    print("test_lt_int64 passed")


fn test_lt_float32() raises:
    print("test_lt_float32")
    size = 20
    var ll = List[Scalar[DType.float32]](capacity=size)
    for i in range(size):
        ll.append(Scalar[DType.float32](i))
    var buffer = Buffer[DType.float32](ll)

    var result = buffer.lt(10.0)

    for i in range(size):
        expected = i < 10
        assert_true(
            result[i] == expected,
            "test_lt_float32: index " + i.__str__() + " failed",
        )
    print("test_lt_float32 passed")


fn test_lt_all_less() raises:
    print("test_lt_all_less")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32].full(0, size)

    var result = buffer.lt(100)

    for i in range(size):
        assert_true(result[i] == True, "test_lt_all_less: all should be < 100")
    print("test_lt_all_less passed")


fn test_lt_none_less() raises:
    print("test_lt_none_less")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32].full(100, size)

    var result = buffer.lt(50)

    for i in range(size):
        assert_true(
            result[i] == False, "test_lt_none_less: none should be < 50"
        )
    print("test_lt_none_less passed")


# ============================================
# LE Tests
# ============================================
fn test_le_int32() raises:
    print("test_le_int32")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i

    var result = buffer.le(50)

    for i in range(size):
        expected = i <= 50
        assert_true(
            result[i] == expected,
            "test_le_int32: index " + i.__str__() + " failed",
        )
    print("test_le_int32 passed")


fn test_le_float64() raises:
    print("test_le_float64")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float64](i)

    var result = buffer.le(33.0)

    for i in range(size):
        expected = i <= 33
        assert_true(
            result[i] == expected,
            "test_le_float64: index " + i.__str__() + " failed",
        )
    print("test_le_float64 passed")


fn test_le_equal_value() raises:
    print("test_le_equal_value")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(100, size)

    var result = buffer.le(100)

    for i in range(size):
        assert_true(
            result[i] == True, "test_le_equal_value: all should be <= 100"
        )
    print("test_le_equal_value passed")


# ============================================
# Edge Case Tests
# ============================================
fn test_compare_single_element() raises:
    print("test_compare_single_element")
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 5

    assert_true(buffer.eq(5)[0] == True, "Single element eq failed")
    assert_true(buffer.ne(5)[0] == False, "Single element ne failed")
    assert_true(buffer.gt(4)[0] == True, "Single element gt failed")
    assert_true(buffer.ge(5)[0] == True, "Single element ge failed")
    assert_true(buffer.lt(6)[0] == True, "Single element lt failed")
    assert_true(buffer.le(5)[0] == True, "Single element le failed")
    print("test_compare_single_element passed")


fn test_compare_simd_boundary() raises:
    print("test_compare_simd_boundary")
    # Test sizes around typical SIMD widths
    alias sizes = VariadicList[Int](
        1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65
    )

    @parameter
    for s_idx in range(len(sizes)):
        alias size = sizes[s_idx]
        var buffer = Buffer[DType.int32](size)
        for i in range(size):
            buffer[i] = i

        var result = buffer.lt(size // 2)

        for i in range(size):
            expected = i < (size // 2)
            assert_true(
                result[i] == expected,
                "test_compare_simd_boundary: size "
                + size.__str__()
                + " index "
                + i.__str__()
                + " failed",
            )

    print("test_compare_simd_boundary passed")


fn test_compare_large_buffer() raises:
    print("test_compare_large_buffer")
    size = LARGE_SIZE
    var buffer = Buffer[DType.float32](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float32](i)

    var result = buffer.ge(500.0)

    for i in range(size):
        expected = i >= 500
        assert_true(
            result[i] == expected,
            "test_compare_large_buffer: index " + i.__str__() + " failed",
        )
    print("test_compare_large_buffer passed")


fn test_compare_uint8() raises:
    print("test_compare_uint8")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.uint8](size)
    for i in range(size):
        buffer[i] = Scalar[DType.uint8](i % 256)

    var result = buffer.gt(100)

    for i in range(size):
        expected = (i % 256) > 100
        assert_true(
            result[i] == expected,
            "test_compare_uint8: index " + i.__str__() + " failed",
        )
    print("test_compare_uint8 passed")


fn test_compare_int8() raises:
    print("test_compare_int8")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int8](size)
    for i in range(size):
        buffer[i] = Scalar[DType.int8](i - 30)  # -30 to 37

    var result = buffer.ge(0)

    for i in range(size):
        expected = (i - 30) >= 0
        assert_true(
            result[i] == expected,
            "test_compare_int8: index " + i.__str__() + " failed",
        )
    print("test_compare_int8 passed")


fn test_compare_int16() raises:
    print("test_compare_int16")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int16](size)
    for i in range(size):
        buffer[i] = Scalar[DType.int16](i * 100)

    var result = buffer.lt(3000)

    for i in range(size):
        expected = (i * 100) < 3000
        assert_true(
            result[i] == expected,
            "test_compare_int16: index " + i.__str__() + " failed",
        )
    print("test_compare_int16 passed")


fn test_compare_uint64() raises:
    print("test_compare_uint64")
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.uint64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.uint64](i * 1000)

    var result = buffer.le(50000)

    for i in range(size):
        expected = (i * 1000) <= 50000
        assert_true(
            result[i] == expected,
            "test_compare_uint64: index " + i.__str__() + " failed",
        )
    print("test_compare_uint64 passed")


# ============================================
# Consolidated Test Runner
# ============================================
fn run_all_comparison_tests() raises:
    print("=" * 60)
    print("Running all comparison tests")
    print("=" * 60)

    # EQ tests
    test_eq_int32()
    test_eq_float64()
    test_eq_all_same()
    test_eq_small_buffer()

    # NE tests
    test_ne_int32()
    test_ne_float32()
    test_ne_all_different()

    # GT tests
    test_gt_int64()
    test_gt_float32()
    test_gt_negative_values()
    test_gt_boundary()

    # GE tests
    test_ge_int32()
    test_ge_float64()
    test_ge_equal_value()

    # LT tests
    test_lt_int64()
    test_lt_float32()
    test_lt_all_less()
    test_lt_none_less()

    # LE tests
    test_le_int32()
    test_le_float64()
    test_le_equal_value()

    # Edge cases
    test_compare_single_element()
    test_compare_simd_boundary()
    test_compare_large_buffer()

    # Different DTypes
    test_compare_uint8()
    test_compare_int8()
    test_compare_int16()
    test_compare_uint64()

    print("=" * 60)
    print("All comparison tests passed!")
    print("=" * 60)


# ====================================="
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
    run_all_dunder_comparison_tests()
    run_all_comparison_tests()
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
