from buffers import Buffer
from operators import *

# ============================================
# Test Constants
# ============================================
alias SMALL_SIZE = 7  # Less than typical SIMD width
alias MEDIUM_SIZE = 68  # Multiple SIMD blocks + remainder
alias LARGE_SIZE = 1000  # Stress test


# ============================================
# Constructor Tests
# ============================================
fn test_constructor_empty() raises:
    print("test_constructor_empty")
    var buffer = Buffer[DType.int32]()
    assert_true(buffer.size == 0, "Empty buffer size should be 0")
    assert_true(len(buffer) == 0, "Empty buffer len should be 0")
    print("test_constructor_empty passed")


fn test_constructor_size() raises:
    print("test_constructor_size")
    var buffer = Buffer[DType.float32](MEDIUM_SIZE)
    assert_true(buffer.size == MEDIUM_SIZE, "Buffer size mismatch")
    assert_true(len(buffer) == MEDIUM_SIZE, "Buffer len mismatch")
    print("test_constructor_size passed")


fn test_constructor_from_list() raises:
    print("test_constructor_from_list")
    var ll = List[Scalar[DType.int32]](capacity=5)
    for i in range(5):
        ll.append(Scalar[DType.int32](i * 10))

    var buffer = Buffer[DType.int32](ll)
    assert_true(buffer.size == 5, "Buffer size from list mismatch")
    for i in range(5):
        assert_true(
            buffer[i] == i * 10,
            "Buffer value from list mismatch at " + i.__str__(),
        )
    print("test_constructor_from_list passed")


# ============================================
# Static Factory Tests
# ============================================
fn test_full() raises:
    print("test_full")
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)
    assert_true(buffer.size == MEDIUM_SIZE, "full: size mismatch")
    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 42, "full: value mismatch at " + i.__str__())
    print("test_full passed")


fn test_zeros() raises:
    print("test_zeros")
    var buffer = Buffer[DType.float32].zeros(MEDIUM_SIZE)
    assert_true(buffer.size == MEDIUM_SIZE, "zeros: size mismatch")
    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 0.0, "zeros: value mismatch at " + i.__str__())
    print("test_zeros passed")


fn test_arange_single_arg() raises:
    print("test_arange_single_arg")
    var buffer = Buffer[DType.int32].arange(10)
    assert_true(buffer.size == 10, "arange: size mismatch")
    for i in range(10):
        assert_true(buffer[i] == i, "arange: value mismatch at " + i.__str__())
    print("test_arange_single_arg passed")


fn test_arange_two_args() raises:
    print("test_arange_two_args")
    var buffer = Buffer[DType.int32].arange(5, 10)
    assert_true(buffer.size == 5, "arange: size mismatch")
    for i in range(5):
        assert_true(buffer[i] == i + 5, "arange: value mismatch")
    print("test_arange_two_args passed")


fn test_arange_three_args() raises:
    print("test_arange_three_args")
    var buffer = Buffer[DType.int32].arange(0, 10, 2)
    assert_true(buffer.size == 5, "arange step 2: size mismatch")
    for i in range(5):
        assert_true(buffer[i] == i * 2, "arange step 2: value mismatch")
    print("test_arange_three_args passed")


fn test_linspace() raises:
    print("test_linspace")
    var buffer = Buffer[DType.float32].linspace(0.0, 10.0, 11)
    assert_true(buffer.size == 11, "linspace: size mismatch")
    for i in range(11):
        assert_true(
            abs(buffer[i] - Scalar[DType.float32](i)) < 0.001,
            "linspace: value mismatch at " + i.__str__(),
        )
    print("test_linspace passed")


# ============================================
# Indexing and Slicing Tests
# ============================================
fn test_getitem_setitem() raises:
    print("test_getitem_setitem")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i * 5

    for i in range(10):
        assert_true(buffer[i] == i * 5, "getitem/setitem mismatch")
    print("test_getitem_setitem passed")


fn test_slice_contiguous() raises:
    print("test_slice_contiguous")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i

    var sliced = buffer[2:7]
    assert_true(sliced.size == 5, "slice: size mismatch")
    for i in range(5):
        assert_true(sliced[i] == i + 2, "slice: value mismatch")
    print("test_slice_contiguous passed")


fn test_slice_with_step() raises:
    print("test_slice_with_step")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i

    var sliced = buffer[0:10:2]
    assert_true(sliced.size == 5, "slice step: size mismatch")
    for i in range(5):
        assert_true(sliced[i] == i * 2, "slice step: value mismatch")
    print("test_slice_with_step passed")


fn test_slice_empty() raises:
    print("test_slice_empty")
    var buffer = Buffer[DType.int32](10)
    var sliced = buffer[5:5]
    assert_true(sliced.size == 0, "empty slice: size should be 0")
    print("test_slice_empty passed")


# ============================================
# Arithmetic Operations (Buffer + Buffer)
# ============================================
fn test_add_buffers() raises:
    print("test_add_buffers")
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    var result = a + b

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 15, "add buffers: value mismatch")
    print("test_add_buffers passed")


fn test_sub_buffers() raises:
    print("test_sub_buffers")
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    var result = a - b

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 7, "sub buffers: value mismatch")
    print("test_sub_buffers passed")


fn test_mul_buffers() raises:
    print("test_mul_buffers")
    var a = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    var result = a * b

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 12, "mul buffers: value mismatch")
    print("test_mul_buffers passed")


fn test_div_buffers() raises:
    print("test_div_buffers")
    var a = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var b = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    var result = a / b

    for i in range(MEDIUM_SIZE):
        assert_true(abs(result[i] - 5.0) < 0.001, "div buffers: value mismatch")
    print("test_div_buffers passed")


# ============================================
# In-place Arithmetic (Buffer += Buffer)
# ============================================
fn test_iadd_buffers() raises:
    print("test_iadd_buffers")
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    a += b

    for i in range(MEDIUM_SIZE):
        assert_true(a[i] == 15, "iadd buffers: value mismatch")
    print("test_iadd_buffers passed")


fn test_isub_buffers() raises:
    print("test_isub_buffers")
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    a -= b

    for i in range(MEDIUM_SIZE):
        assert_true(a[i] == 7, "isub buffers: value mismatch")
    print("test_isub_buffers passed")


fn test_imul_buffers() raises:
    print("test_imul_buffers")
    var a = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    a *= b

    for i in range(MEDIUM_SIZE):
        assert_true(a[i] == 12, "imul buffers: value mismatch")
    print("test_imul_buffers passed")


fn test_itruediv_buffers() raises:
    print("test_itruediv_buffers")
    var a = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var b = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    a /= b

    for i in range(MEDIUM_SIZE):
        assert_true(abs(a[i] - 5.0) < 0.001, "itruediv buffers: value mismatch")
    print("test_itruediv_buffers passed")


# ============================================
# Arithmetic Operations (Buffer + Scalar)
# ============================================
fn test_add_scalar() raises:
    print("test_add_scalar")
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var result = buffer + 5

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 15, "add scalar: value mismatch")
    print("test_add_scalar passed")


fn test_radd_scalar() raises:
    print("test_radd_scalar")
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var result = 5 + buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 15, "radd scalar: value mismatch")
    print("test_radd_scalar passed")


fn test_sub_scalar() raises:
    print("test_sub_scalar")
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var result = buffer - 3

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 7, "sub scalar: value mismatch")
    print("test_sub_scalar passed")


fn test_rsub_scalar() raises:
    print("test_rsub_scalar")
    var buffer = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    var result = 10 - buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 7, "rsub scalar: value mismatch")
    print("test_rsub_scalar passed")


fn test_mul_scalar() raises:
    print("test_mul_scalar")
    var buffer = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var result = buffer * 3

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 12, "mul scalar: value mismatch")
    print("test_mul_scalar passed")


fn test_rmul_scalar() raises:
    print("test_rmul_scalar")
    var buffer = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var result = 3 * buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 12, "rmul scalar: value mismatch")
    print("test_rmul_scalar passed")


fn test_truediv_scalar() raises:
    print("test_truediv_scalar")
    var buffer = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var result = buffer / 2.0

    for i in range(MEDIUM_SIZE):
        assert_true(
            abs(result[i] - 5.0) < 0.001, "truediv scalar: value mismatch"
        )
    print("test_truediv_scalar passed")


fn test_rtruediv_scalar() raises:
    print("test_rtruediv_scalar")
    var buffer = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    var result = 10.0 / buffer

    for i in range(MEDIUM_SIZE):
        assert_true(
            abs(result[i] - 5.0) < 0.001, "rtruediv scalar: value mismatch"
        )
    print("test_rtruediv_scalar passed")


# ============================================
# In-place Scalar Operations
# ============================================
fn test_iadd_scalar() raises:
    print("test_iadd_scalar")
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    buffer += 5

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 15, "iadd scalar: value mismatch")
    print("test_iadd_scalar passed")


fn test_isub_scalar() raises:
    print("test_isub_scalar")
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    buffer -= 3

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 7, "isub scalar: value mismatch")
    print("test_isub_scalar passed")


fn test_imul_scalar() raises:
    print("test_imul_scalar")
    var buffer = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    buffer *= 3

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 12, "imul scalar: value mismatch")
    print("test_imul_scalar passed")


fn test_itruediv_scalar() raises:
    print("test_itruediv_scalar")
    var buffer = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    buffer /= 2.0

    for i in range(MEDIUM_SIZE):
        assert_true(
            abs(buffer[i] - 5.0) < 0.001, "itruediv scalar: value mismatch"
        )
    print("test_itruediv_scalar passed")


# ============================================
# Unary Operations
# ============================================
fn test_neg() raises:
    print("test_neg")
    var buffer = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    var result = -buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == -5, "neg: value mismatch")
    print("test_neg passed")


fn test_abs() raises:
    print("test_abs")
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = i - 30  # Mix of negative and positive

    var result = abs(buffer)
    for i in range(MEDIUM_SIZE):
        assert_true(result[i] >= 0, "abs: should be non-negative")
        assert_true(result[i] == abs(i - 30), "abs: value mismatch")
    print("test_abs passed")


fn test_pow() raises:
    print("test_pow")
    var buffer = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    var result = buffer**3.0

    for i in range(MEDIUM_SIZE):
        assert_true(abs(result[i] - 8.0) < 0.001, "pow: value mismatch")
    print("test_pow passed")


fn test_exp() raises:
    print("test_exp")
    var buffer = Buffer[DType.float32].full(0.0, 10)
    var result = buffer.exp()

    for i in range(10):
        assert_true(abs(result[i] - 1.0) < 0.001, "exp(0) should be 1")
    print("test_exp passed")


fn test_log() raises:
    print("test_log")
    var buffer = Buffer[DType.float32].full(1.0, 10)
    var result = buffer.log()

    for i in range(10):
        assert_true(abs(result[i] - 0.0) < 0.001, "log(1) should be 0")
    print("test_log passed")


fn test_invert_bool() raises:
    print("test_invert_bool")
    var buffer = Buffer[DType.bool](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Scalar[DType.bool](i % 2 == 0)  # Alternating True/False

    var result = ~buffer
    for i in range(MEDIUM_SIZE):
        expected = i % 2 != 0
        assert_true(
            result[i] == expected, "invert: value mismatch at " + i.__str__()
        )
    print("test_invert_bool passed")


# ============================================
# Reduction Operations
# ============================================
fn test_sum() raises:
    print("test_sum")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i + 1  # 1 to 10

    var result = buffer.sum()
    assert_true(result == 55, "sum: expected 55, got " + result.__str__())
    print("test_sum passed")


fn test_sum_with_range() raises:
    print("test_sum_with_range")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i + 1  # 1 to 10

    var result = buffer.sum(2, 5)  # Sum indices 2,3,4 -> 3+4+5 = 12
    assert_true(result == 12, "sum range: expected 12, got " + result.__str__())
    print("test_sum_with_range passed")


fn test_product() raises:
    print("test_product")
    var buffer = Buffer[DType.int32](5)
    for i in range(5):
        buffer[i] = i + 1  # 1 to 5

    var result = buffer.product()
    assert_true(
        result == 120, "product: expected 120 (5!), got " + result.__str__()
    )
    print("test_product passed")


fn test_product_with_range() raises:
    print("test_product_with_range")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i + 1

    var result = buffer.product(0, 3)  # 1*2*3 = 6
    assert_true(
        result == 6, "product range: expected 6, got " + result.__str__()
    )
    print("test_product_with_range passed")


fn test_dot() raises:
    print("test_dot")
    var a = Buffer[DType.int32].full(2, 5)
    var b = Buffer[DType.int32].full(3, 5)

    var result = a.dot(b)
    assert_true(
        result == 30, "dot: expected 30 (2*3*5), got " + result.__str__()
    )
    print("test_dot passed")


# ============================================
# Comparison Operations (Scalar)
# ============================================
fn test_eq_scalar_all_equal() raises:
    print("test_eq_scalar_all_equal")
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer == 42, "eq scalar: all equal should be True")
    assert_true(not (buffer == 0), "eq scalar: none equal should be False")
    print("test_eq_scalar_all_equal passed")


fn test_ne_scalar() raises:
    print("test_ne_scalar")
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer != 0, "ne scalar: all != should be True")
    assert_true(not (buffer != 42), "ne scalar: none != should be False")
    print("test_ne_scalar passed")


fn test_gt_scalar() raises:
    print("test_gt_scalar")
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer > 99, "gt scalar: all > 99")
    assert_true(not (buffer > 100), "gt scalar: none > 100")
    print("test_gt_scalar passed")


fn test_ge_scalar() raises:
    print("test_ge_scalar")
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer >= 100, "ge scalar: all >= 100")
    assert_true(not (buffer >= 101), "ge scalar: none >= 101")
    print("test_ge_scalar passed")


fn test_lt_scalar() raises:
    print("test_lt_scalar")
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer < 51, "lt scalar: all < 51")
    assert_true(not (buffer < 50), "lt scalar: none < 50")
    print("test_lt_scalar passed")


fn test_le_scalar() raises:
    print("test_le_scalar")
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer <= 50, "le scalar: all <= 50")
    assert_true(not (buffer <= 49), "le scalar: none <= 49")
    print("test_le_scalar passed")


# ============================================
# Element-wise Comparison (Scalar) -> Buffer[bool]
# ============================================
fn test_eq_full_scalar() raises:
    print("test_eq_full_scalar")
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = i % 10

    var result = buffer.eq(5)
    for i in range(MEDIUM_SIZE):
        expected = (i % 10) == 5
        assert_true(
            result[i] == expected, "eq full: mismatch at " + i.__str__()
        )
    print("test_eq_full_scalar passed")


fn test_ne_full_scalar() raises:
    print("test_ne_full_scalar")
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = i % 10

    var result = buffer.ne(5)
    for i in range(MEDIUM_SIZE):
        expected = (i % 10) != 5
        assert_true(
            result[i] == expected, "ne full: mismatch at " + i.__str__()
        )
    print("test_ne_full_scalar passed")


fn test_lt_full_scalar() raises:
    print("test_lt_full_scalar")
    var buffer = Buffer[DType.int32](20)
    for i in range(20):
        buffer[i] = i

    var result = buffer.lt(10)
    for i in range(20):
        expected = i < 10
        assert_true(
            result[i] == expected, "lt full: mismatch at " + i.__str__()
        )
    print("test_lt_full_scalar passed")


fn test_gt_full_scalar() raises:
    print("test_gt_full_scalar")
    var buffer = Buffer[DType.int32](20)
    for i in range(20):
        buffer[i] = i

    var result = buffer.gt(10)
    for i in range(20):
        expected = i > 10
        assert_true(
            result[i] == expected, "gt full: mismatch at " + i.__str__()
        )
    print("test_gt_full_scalar passed")


# ============================================
# Comparison Operations (Buffer)
# ============================================
fn test_eq_buffers() raises:
    print("test_eq_buffers")
    var a = Buffer[DType.int32].full(42, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(a == b, "eq buffers: identical should be True")

    b[0] = 0
    assert_true(not (a == b), "eq buffers: one different should be False")
    print("test_eq_buffers passed")


fn test_ne_buffers() raises:
    print("test_ne_buffers")
    var a = Buffer[DType.int32](MEDIUM_SIZE)
    var b = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = i
        b[i] = i + 1  # All different

    assert_true(a != b, "ne buffers: all different should be True")

    a[0] = b[0]
    assert_true(not (a != b), "ne buffers: one equal should be False")
    print("test_ne_buffers passed")


fn test_lt_buffers() raises:
    print("test_lt_buffers")
    var a = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(10, MEDIUM_SIZE)

    assert_true(a < b, "lt buffers: all a < b")

    a[0] = 10
    assert_true(not (a < b), "lt buffers: one not < should be False")
    print("test_lt_buffers passed")


# ============================================
# Element-wise Comparison (Buffer) -> Buffer[bool]
# ============================================
fn test_eq_full_buffers() raises:
    print("test_eq_full_buffers")
    var a = Buffer[DType.int32](MEDIUM_SIZE)
    var b = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = i
        b[i] = i % 10

    var result = a.eq(b)
    for i in range(MEDIUM_SIZE):
        expected = a[i] == b[i]
        assert_true(
            result[i] == expected, "eq full buffers: mismatch at " + i.__str__()
        )
    print("test_eq_full_buffers passed")


fn test_lt_full_buffers() raises:
    print("test_lt_full_buffers")
    var a = Buffer[DType.int32](MEDIUM_SIZE)
    var b = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = i
        b[i] = 30

    var result = a.lt(b)
    for i in range(MEDIUM_SIZE):
        expected = a[i] < b[i]
        assert_true(
            result[i] == expected, "lt full buffers: mismatch at " + i.__str__()
        )
    print("test_lt_full_buffers passed")


# ============================================
# Utility Methods
# ============================================
fn test_fill() raises:
    print("test_fill")
    var buffer = Buffer[DType.int32].zeros(MEDIUM_SIZE)
    buffer.fill(99, 10, 20)

    for i in range(MEDIUM_SIZE):
        if i >= 10 and i < 20:
            assert_true(buffer[i] == 99, "fill: should be 99 in range")
        else:
            assert_true(buffer[i] == 0, "fill: should be 0 outside range")
    print("test_fill passed")


fn test_zero() raises:
    print("test_zero")
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)
    buffer.zero()

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 0, "zero: all should be 0")
    print("test_zero passed")


fn test_overwrite() raises:
    print("test_overwrite")
    var buffer = Buffer[DType.int32].zeros(MEDIUM_SIZE)
    var source = Buffer[DType.int32].full(99, 10)
    buffer.overwrite(source, 10, 20)

    for i in range(MEDIUM_SIZE):
        if i >= 10 and i < 20:
            assert_true(buffer[i] == 99, "overwrite: should be 99 in range")
        else:
            assert_true(buffer[i] == 0, "overwrite: should be 0 outside range")
    print("test_overwrite passed")


fn test_count() raises:
    print("test_count")
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = i % 10

    var count_5 = buffer.count(5)
    # Values 0-67, with i % 10 == 5 at indices: 5, 15, 25, 35, 45, 55, 65 = 7 occurrences
    assert_true(
        count_5 == 7, "count: expected 7 fives, got " + count_5.__str__()
    )

    var count_0 = buffer.count(0)
    # i % 10 == 0 at indices: 0, 10, 20, 30, 40, 50, 60 = 7 occurrences
    assert_true(
        count_0 == 7, "count: expected 7 zeros, got " + count_0.__str__()
    )
    print("test_count passed")


fn test_count_with_range() raises:
    print("test_count_with_range")
    var buffer = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    buffer[10] = 99
    buffer[20] = 99
    buffer[30] = 99

    var count_all = buffer.count(99)
    assert_true(count_all == 3, "count all: expected 3")

    var count_range = buffer.count(99, 15, 35)
    assert_true(count_range == 2, "count range: expected 2 (at 20 and 30)")
    print("test_count_with_range passed")


# ============================================
# Type Conversion Tests
# ============================================
fn test_to_dtype_int_to_float() raises:
    print("test_to_dtype_int_to_float")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i * 10

    var result = buffer.to_dtype[DType.float32]()
    for i in range(10):
        assert_true(
            abs(result[i] - Scalar[DType.float32](i * 10)) < 0.001,
            "to_dtype int->float: mismatch at " + i.__str__(),
        )
    print("test_to_dtype_int_to_float passed")


fn test_to_dtype_float_to_int() raises:
    print("test_to_dtype_float_to_int")
    var buffer = Buffer[DType.float32](10)
    for i in range(10):
        buffer[i] = Scalar[DType.float32](i) + 0.7

    var result = buffer.to_dtype[DType.int32]()
    for i in range(10):
        # Float to int truncates
        assert_true(
            result[i] == i, "to_dtype float->int: mismatch at " + i.__str__()
        )
    print("test_to_dtype_float_to_int passed")


fn test_to_dtype_to_bool() raises:
    print("test_to_dtype_to_bool")
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = i  # 0, 1, 2, ...

    var result = buffer.to_dtype[DType.bool]()
    assert_true(result[0] == False, "to_dtype to bool: 0 should be False")
    for i in range(1, 10):
        assert_true(
            result[i] == True, "to_dtype to bool: non-zero should be True"
        )
    print("test_to_dtype_to_bool passed")


fn test_to_dtype_from_bool() raises:
    print("test_to_dtype_from_bool")
    var buffer = Buffer[DType.bool](10)
    for i in range(10):
        buffer[i] = i % 2 == 0  # Alternating

    var result = buffer.to_dtype[DType.int32]()
    for i in range(10):
        expected = 1 if i % 2 == 0 else 0
        assert_true(
            result[i] == expected,
            "to_dtype from bool: mismatch at " + i.__str__(),
        )
    print("test_to_dtype_from_bool passed")


fn test_float_convenience() raises:
    print("test_float_convenience")
    var buffer = Buffer[DType.int32].full(42, 10)
    var result = buffer.float()

    assert_true(result[0] == 42.0, "float() convenience: value mismatch")
    print("test_float_convenience passed")


fn test_float64_convenience() raises:
    print("test_float64_convenience")
    var buffer = Buffer[DType.int32].full(42, 10)
    var result = buffer.float64()

    assert_true(result[0] == 42.0, "float64() convenience: value mismatch")
    print("test_float64_convenience passed")


# ============================================
# Boolean Buffer Operations
# ============================================
fn test_mul_bool_buffers() raises:
    print("test_mul_bool_buffers")
    var a = Buffer[DType.bool](MEDIUM_SIZE)
    var b = Buffer[DType.bool](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = i % 2 == 0  # Even indices True
        b[i] = i % 3 == 0  # Divisible by 3 True

    var result = a * b  # Logical AND
    for i in range(MEDIUM_SIZE):
        expected = (i % 2 == 0) and (i % 3 == 0)  # Divisible by 6
        assert_true(
            result[i] == expected, "mul bool: mismatch at " + i.__str__()
        )
    print("test_mul_bool_buffers passed")


fn test_imul_bool_scalar() raises:
    print("test_imul_bool_scalar")
    var buffer = Buffer[DType.bool](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = True

    buffer *= False
    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == False, "imul bool scalar: all should be False")
    print("test_imul_bool_scalar passed")


# ============================================
# Edge Cases and SIMD Boundary Tests
# ============================================
fn test_simd_boundary_sizes() raises:
    print("test_simd_boundary_sizes")
    alias sizes = VariadicList[Int](
        1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65
    )

    @parameter
    for idx in range(len(sizes)):
        alias size = sizes[idx]
        var a = Buffer[DType.int32].full(10, size)
        var b = Buffer[DType.int32].full(5, size)

        var sum_result = a + b
        var mul_result = a * b

        for i in range(size):
            assert_true(
                sum_result[i] == 15, "SIMD boundary add: size " + size.__str__()
            )
            assert_true(
                mul_result[i] == 50, "SIMD boundary mul: size " + size.__str__()
            )

    print("test_simd_boundary_sizes passed")


fn test_arithmetic_ops_with_range() raises:
    print("test_arithmetic_ops_with_range")
    var a = Buffer[DType.int32].full(10, 20)
    var b = Buffer[DType.int32].full(3, 10)

    # Test arithmetic_ops with ranges
    var result = a.arithmetic_ops[Add, True](b, 5, 15, 0, 10)
    assert_true(result.size == 10, "arithmetic_ops range: size mismatch")
    for i in range(10):
        assert_true(result[i] == 13, "arithmetic_ops range: value mismatch")
    print("test_arithmetic_ops_with_range passed")


fn test_inplace_ops_with_range() raises:
    print("test_inplace_ops_with_range")
    var a = Buffer[DType.int32].full(10, 20)
    var b = Buffer[DType.int32].full(5, 10)

    a.inplace_ops[Add, True](b, 5, 15, 0, 10)

    for i in range(20):
        if i >= 5 and i < 15:
            assert_true(
                a[i] == 15, "inplace_ops range: modified region should be 15"
            )
        else:
            assert_true(
                a[i] == 10, "inplace_ops range: unmodified region should be 10"
            )
    print("test_inplace_ops_with_range passed")


fn test_inplace_ops_scalar_with_range() raises:
    print("test_inplace_ops_scalar_with_range")
    var buffer = Buffer[DType.int32].full(10, 20)

    buffer.inplace_ops_scalar[Multiply](2, 5, 15)

    for i in range(20):
        if i >= 5 and i < 15:
            assert_true(
                buffer[i] == 20,
                "inplace_ops_scalar range: modified should be 20",
            )
        else:
            assert_true(
                buffer[i] == 10,
                "inplace_ops_scalar range: unmodified should be 10",
            )
    print("test_inplace_ops_scalar_with_range passed")


# ============================================
# Different DType Tests
# ============================================
fn test_operations_uint8() raises:
    print("test_operations_uint8")
    var a = Buffer[DType.uint8].full(100, MEDIUM_SIZE)
    var b = Buffer[DType.uint8].full(50, MEDIUM_SIZE)

    var sum_result = a + b
    # var mul_result = a * b

    assert_true(sum_result[0] == 150, "uint8 add")
    # Note: 100 * 50 = 5000, but uint8 max is 255, so overflow
    print("test_operations_uint8 passed")


fn test_operations_int16() raises:
    print("test_operations_int16")
    var a = Buffer[DType.int16].full(1000, MEDIUM_SIZE)
    var b = Buffer[DType.int16].full(500, MEDIUM_SIZE)

    var sum_result = a + b
    assert_true(sum_result[0] == 1500, "int16 add")
    print("test_operations_int16 passed")


fn test_operations_float64() raises:
    print("test_operations_float64")
    var a = Buffer[DType.float64].full(1.5, MEDIUM_SIZE)
    var b = Buffer[DType.float64].full(2.5, MEDIUM_SIZE)

    var sum_result = a + b
    var mul_result = a * b

    assert_true(abs(sum_result[0] - 4.0) < 0.001, "float64 add")
    assert_true(abs(mul_result[0] - 3.75) < 0.001, "float64 mul")
    print("test_operations_float64 passed")


# ============================================
# Consolidated Test Runner
# ============================================
fn run_all_buffer_tests() raises:
    print("=" * 60)
    print("Running all Buffer tests")
    print("=" * 60)

    # Constructors
    test_constructor_empty()
    test_constructor_size()
    test_constructor_from_list()

    # Static factories
    test_full()
    test_zeros()
    test_arange_single_arg()
    test_arange_two_args()
    test_arange_three_args()
    test_linspace()

    # Indexing
    test_getitem_setitem()
    test_slice_contiguous()
    test_slice_with_step()
    test_slice_empty()

    # Buffer arithmetic
    test_add_buffers()
    test_sub_buffers()
    test_mul_buffers()
    test_div_buffers()

    # In-place buffer arithmetic
    test_iadd_buffers()
    test_isub_buffers()
    test_imul_buffers()
    test_itruediv_buffers()

    # Scalar arithmetic
    test_add_scalar()
    test_radd_scalar()
    test_sub_scalar()
    test_rsub_scalar()
    test_mul_scalar()
    test_rmul_scalar()
    test_truediv_scalar()
    test_rtruediv_scalar()

    # In-place scalar arithmetic
    test_iadd_scalar()
    test_isub_scalar()
    test_imul_scalar()
    test_itruediv_scalar()

    # Unary ops
    test_neg()
    test_abs()
    test_pow()
    test_exp()
    test_log_orig()
    test_invert_bool()

    # Reductions
    test_sum()
    test_sum_with_range()
    test_product()
    test_product_with_range()
    test_dot()

    # Scalar comparisons
    test_eq_scalar_all_equal()
    test_ne_scalar()
    test_gt_scalar()
    test_ge_scalar()
    test_lt_scalar()
    test_le_scalar()

    # Element-wise scalar comparisons
    test_eq_full_scalar()
    test_ne_full_scalar()
    test_lt_full_scalar()
    test_gt_full_scalar()

    # Buffer comparisons
    test_eq_buffers()
    test_ne_buffers()
    test_lt_buffers()

    # Element-wise buffer comparisons
    test_eq_full_buffers()
    test_lt_full_buffers()

    # Utility methods
    test_fill()
    test_zero()
    test_overwrite()
    test_count()
    test_count_with_range()

    # Type conversion
    test_to_dtype_int_to_float()
    test_to_dtype_float_to_int()
    test_to_dtype_to_bool()
    test_to_dtype_from_bool()
    test_float_convenience()
    test_float64_convenience()

    # Boolean operations
    test_mul_bool_buffers()
    test_imul_bool_scalar()

    # Edge cases
    test_simd_boundary_sizes()
    test_arithmetic_ops_with_range()
    test_inplace_ops_with_range()
    test_inplace_ops_scalar_with_range()

    # Different DTypes
    test_operations_uint8()
    test_operations_int16()
    test_operations_float64()

    print("=" * 60)
    print("All Buffer tests passed!")
    print("=" * 60)


# ============================================
# __eq__ Tests - new tests above
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


fn test_to_dtype_to_bool_orig() raises:
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


fn test_to_dtype_from_bool_orig() raises:
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
    test_to_dtype_to_bool_orig()
    test_to_dtype_from_bool_orig()
    test_to_dtype_large_buffer()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


fn main() raises:
    print("Running buffer tests")
    run_all_dunder_comparison_tests()
    run_all_comparison_tests()
    test_overwrite_orig()
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
    test_count_orig()
    test_log()
    run_all_tests()
    run_all_buffer_tests()
    print("Done running buffer tests")


from testing import assert_true, assert_false


fn test_count_orig() raises:
    print("test_count_orig")
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


fn test_log_orig() raises:
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


fn test_overwrite_orig() raises:
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
