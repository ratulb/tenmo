from buffers import Buffer
from operators import *
from sys import simd_width_of
from time import perf_counter_ns

# ============================================
# Test Constants
# ============================================
alias SMALL_SIZE = 7  # Less than typical SIMD width
alias MEDIUM_SIZE = 68  # Multiple SIMD blocks + remainder
alias LARGE_SIZE = 1000  # Stress test
alias SMALL_SIZE_NEW = 17  # Prime number to test tail loop


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
    # test_mul_bool_buffers()
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
    # test_buffer_buffer_mul()
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
    run_all_count_tests()
    run_all_inplace_scalar_tests()
    run_all_arithmetic_ops_tests()
    run_all_manual_vectorization_tests()
    run_all_buffer_ops_tests()
    run_all_manual_vectorization_tests_2()
    run_relu_optimization_tests()
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
    l = List[Scalar[dtype]](capacity=Int(size))
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
    l = List[Scalar[dtype]](capacity=Int(size))
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


fn run_all_count_tests() raises:
    """Run all buffer count tests."""
    print("\n=== Running Buffer Count Test Suite ===\n")

    # Basic tests
    test_count_basic_cnt()
    test_count_with_range_cnt()
    test_count_empty_buffer_cnt()
    test_count_single_element_cnt()
    test_count_all_match_cnt()
    test_count_none_match_cnt()

    # Pattern tests
    test_count_alternating_pattern_cnt()
    test_count_simd_boundary_cnt()
    test_count_range_partial_simd_cnt()

    # Dtype-specific tests
    test_count_float32_cnt()
    test_count_float64_cnt()
    test_count_int8_cnt()
    test_count_int64_cnt()
    test_count_uint8_cnt()

    # Boolean tests
    test_count_bool_true_cnt()
    test_count_bool_false_cnt()
    test_count_bool_all_true_cnt()
    test_count_bool_all_false_cnt()
    test_count_bool_range_cnt()

    # Edge case tests
    test_count_large_buffer_cnt()
    test_count_negative_values_cnt()
    test_count_zero_cnt()
    test_count_start_equals_end_cnt()
    test_count_invalid_range_cnt()
    test_count_large_buffer_cnt_777_with_default()
    test_count_large_buffer_cnt_777()

    print("\n=== All Buffer Count Tests Passed! ===\n")


# ============================================================================
# Buffer Count Tests
# ============================================================================


fn test_count_basic_cnt() raises:
    """Test basic counting functionality."""
    print("test_count_basic_cnt")
    var buffer = Buffer[DType.int32](68)
    for i in range(68):
        buffer[i] = i % 10

    var count_5 = buffer.count(5)
    # Values 0-67, with i % 10 == 5 at indices: 5, 15, 25, 35, 45, 55, 65 = 7 occurrences
    assert_true(count_5 == 7)

    var count_0 = buffer.count(0)
    # i % 10 == 0 at indices: 0, 10, 20, 30, 40, 50, 60 = 7 occurrences
    assert_true(count_0 == 7)

    var count_9 = buffer.count(9)
    # i % 10 == 9 at indices: 9, 19, 29, 39, 49, 59 = 6 occurrences
    assert_true(count_9 == 6)


fn test_count_with_range_cnt() raises:
    """Test counting within a specified range."""
    print("test_count_with_range_cnt")
    var buffer = Buffer[DType.int32].full(5, 68)
    buffer[10] = 99
    buffer[20] = 99
    buffer[30] = 99

    var count_all = buffer.count(99)
    assert_true(count_all == 3)

    var count_range = buffer.count(99, 15, 35)
    assert_true(count_range == 2)  # At indices 20 and 30

    var count_outside = buffer.count(99, 0, 10)
    assert_true(count_outside == 0)


fn test_count_empty_buffer_cnt() raises:
    """Test counting on empty buffer."""
    print("test_count_empty_buffer_cnt")
    var buffer = Buffer[DType.int32](0)
    var count = buffer.count(42)
    assert_true(count == 0)


fn test_count_single_element_cnt() raises:
    """Test counting in single-element buffer."""
    print("test_count_single_element_cnt")
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 42

    var count_match = buffer.count(42)
    assert_true(count_match == 1)

    var count_no_match = buffer.count(99)
    assert_true(count_no_match == 0)


fn test_count_all_match_cnt() raises:
    """Test counting when all elements match."""
    print("test_count_all_match_cnt")
    var buffer = Buffer[DType.int32].full(7, 100)
    var count = buffer.count(7)
    assert_true(count == 100)


fn test_count_none_match_cnt() raises:
    """Test counting when no elements match."""
    print("test_count_none_match_cnt")
    var buffer = Buffer[DType.int32].full(5, 100)
    var count = buffer.count(999)
    assert_true(count == 0)


fn test_count_alternating_pattern_cnt() raises:
    """Test counting with alternating pattern."""
    print("test_count_alternating_pattern_cnt")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = 1 if i % 2 == 0 else 0

    var count_ones = buffer.count(1)
    assert_true(count_ones == 50)

    var count_zeros = buffer.count(0)
    assert_true(count_zeros == 50)


fn test_count_simd_boundary_cnt() raises:
    """Test counting across SIMD boundaries."""
    print("test_count_simd_boundary_cnt")
    alias simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 3 + 5  # Not aligned to SIMD width
    var buffer = Buffer[DType.int32](size)

    # Fill with pattern
    for i in range(size):
        buffer[i] = 42 if i % 7 == 0 else 0

    var count = buffer.count(42)
    # Count manually
    var expected = 0
    for i in range(size):
        if i % 7 == 0:
            expected += 1
    assert_true(count == expected)


fn test_count_range_partial_simd_cnt() raises:
    """Test counting with range that doesn't align to SIMD width."""
    print("test_count_range_partial_simd_cnt")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    # Set specific values
    buffer[10] = 999
    buffer[15] = 999
    buffer[20] = 999
    buffer[25] = 999

    # Count in range [12, 23) - should find at 15 and 20
    var count = buffer.count(999, 12, 23)
    assert_true(count == 2)


fn test_count_float32_cnt() raises:
    """Test counting with float32 dtype."""
    print("test_count_float32_cnt")
    var buffer = Buffer[DType.float64](50)
    for i in range(50):
        buffer[i] = 3.14 if i % 5 == 0 else 2.71

    var count_pi = buffer.count(3.14)
    assert_true(count_pi == 10)  # Indices: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45

    var count_e = buffer.count(2.71)
    assert_true(count_e == 40)


fn test_count_float64_cnt() raises:
    """Test counting with float64 dtype."""
    print("test_count_float64_cnt")
    var buffer = Buffer[DType.float64](64)
    for i in range(64):
        buffer[i] = 1.5 if i < 32 else 2.5

    var count_1_5 = buffer.count(1.5)
    assert_true(count_1_5 == 32)

    var count_2_5 = buffer.count(2.5)
    assert_true(count_2_5 == 32)


fn test_count_int8_cnt() raises:
    """Test counting with int8 dtype."""
    print("test_count_int8_cnt")
    var buffer = Buffer[DType.int8](100)
    for i in range(100):
        buffer[i] = Int8(i % 128)

    var count_42 = buffer.count(Int8(42))
    assert_true(count_42 == 1)  # Only at index 42

    var count_0 = buffer.count(Int8(0))
    assert_true(count_0 == 1)  # Only at index 0


fn test_count_int64_cnt() raises:
    """Test counting with int64 dtype."""
    print("test_count_int64_cnt")
    var buffer = Buffer[DType.int64](80)
    for i in range(80):
        buffer[i] = 1000000 if i % 10 == 0 else i

    var count = buffer.count(1000000)
    assert_true(count == 8)  # At indices: 0, 10, 20, 30, 40, 50, 60, 70


fn test_count_bool_true_cnt() raises:
    """Test counting True in boolean buffer."""
    print("test_count_bool_true_cnt")
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = True if i % 3 == 0 else False

    var count_true = buffer.count(True)
    # Indices: 0, 3, 6, 9, ..., 99 = 34 occurrences
    var expected = 0
    for i in range(100):
        if i % 3 == 0:
            expected += 1
    assert_true(count_true == expected)


fn test_count_bool_false_cnt() raises:
    """Test counting False in boolean buffer."""
    print("test_count_bool_false_cnt")
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = i % 2 == 0

    var count_false = buffer.count(False)
    assert_true(count_false == 50)


fn test_count_bool_all_true_cnt() raises:
    """Test counting in all-True boolean buffer."""
    print("test_count_bool_all_true_cnt")
    var buffer = Buffer[DType.bool].full(True, 100)
    var count = buffer.count(True)
    assert_true(count == 100)

    var count_false = buffer.count(False)
    assert_true(count_false == 0)


fn test_count_bool_all_false_cnt() raises:
    """Test counting in all-False boolean buffer."""
    print("test_count_bool_all_false_cnt")
    var buffer = Buffer[DType.bool].full(False, 100)
    var count = buffer.count(False)
    assert_true(count == 100)

    var count_true = buffer.count(True)
    assert_true(count_true == 0)


fn test_count_bool_range_cnt() raises:
    """Test counting with range on boolean buffer."""
    print("test_count_bool_range_cnt")
    var buffer = Buffer[DType.bool](50)
    for i in range(50):
        buffer[i] = i < 25

    var count_all = buffer.count(True)
    assert_true(count_all == 25)

    var count_range = buffer.count(True, 10, 30)
    assert_true(count_range == 15)  # Indices 10-24


fn test_count_large_buffer_cnt_777_with_default() raises:
    """Test counting in large buffer."""
    print("test_count_large_buffer_cnt")
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = 777 if i % 100 == 0 else -1  # Use -1 as default

    var count = buffer.count(777)
    assert_true(count == 100)  # Exactly 100 multiples of 100


fn test_count_large_buffer_cnt_777() raises:
    """Test counting in large buffer."""
    print("test_count_large_buffer_cnt")
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = 777 if i % 100 == 0 else i

    var count = buffer.count(777)
    assert_true(count == 101)


fn test_count_large_buffer_cnt() raises:
    """Test counting in large buffer."""
    print("test_count_large_buffer_cnt")
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = 99999 if i % 100 == 0 else i  # Use value outside range

    var count = buffer.count(99999)
    assert_true(count == 100)  # Exactly 100 multiples of 100


fn test_count_negative_values_cnt() raises:
    """Test counting negative values."""
    print("test_count_negative_values_cnt")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = -5 if i % 10 == 0 else i

    var count = buffer.count(-5)
    assert_true(count == 10)


fn test_count_zero_cnt() raises:
    """Test counting zeros."""
    print("test_count_zero_cnt")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = 0 if i % 7 == 0 else 1

    var count_zero = buffer.count(0)
    # Indices: 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98 = 15
    var expected = 0
    for i in range(100):
        if i % 7 == 0:
            expected += 1
    assert_true(count_zero == expected)


fn test_count_start_equals_end_cnt() raises:
    """Test counting with start_index == end_index."""
    print("test_count_start_equals_end_cnt")
    var buffer = Buffer[DType.int32].full(42, 100)
    var count = buffer.count(42, 50, 50)
    assert_true(count == 0)


fn test_count_invalid_range_cnt() raises:
    """Test counting with start > end."""
    print("test_count_invalid_range_cnt")
    var buffer = Buffer[DType.int32].full(42, 100)
    var count = buffer.count(42, 60, 50)
    assert_true(count == 0)


fn test_count_uint8_cnt() raises:
    """Test counting with uint8 dtype."""
    print("test_count_uint8_cnt")
    var buffer = Buffer[DType.uint8](256)
    for i in range(256):
        buffer[i] = UInt8(i)

    var count_128 = buffer.count(UInt8(128))
    assert_true(count_128 == 1)

    var count_255 = buffer.count(UInt8(255))
    assert_true(count_255 == 1)


# ============================================================================
# Buffer Inplace Scalar Operations Tests
# ============================================================================


fn test_inplace_multiply_scalar_ips() raises:
    """Test inplace multiplication by scalar."""
    print("test_inplace_multiply_scalar_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](2)

    for i in range(100):
        assert_true(buffer[i] == i * 2)


fn test_inplace_add_scalar_ips() raises:
    """Test inplace addition of scalar."""
    print("test_inplace_add_scalar_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Add](10)

    for i in range(100):
        assert_true(buffer[i] == i + 10)


fn test_inplace_subtract_scalar_ips() raises:
    """Test inplace subtraction of scalar."""
    print("test_inplace_subtract_scalar_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i + 50

    buffer.inplace_ops_scalar[Subtract](50)

    for i in range(100):
        assert_true(buffer[i] == i)


fn test_inplace_divide_scalar_ips() raises:
    """Test inplace division by scalar."""
    print("test_inplace_divide_scalar_ips")
    var buffer = Buffer[DType.float32](100)
    for i in range(100):
        buffer[i] = Float32(i * 4)

    buffer.inplace_ops_scalar[Divide](2.0)

    for i in range(100):
        assert_true(abs(buffer[i] - Float32(i * 2)) < 0.0001)


fn test_inplace_multiply_with_range_ips() raises:
    """Test inplace multiplication with range."""
    print("test_inplace_multiply_with_range_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](3, 20, 40)

    for i in range(100):
        if i >= 20 and i < 40:
            assert_true(buffer[i] == i * 3)
        else:
            assert_true(buffer[i] == i)


fn test_inplace_add_with_range_ips() raises:
    """Test inplace addition with range."""
    print("test_inplace_add_with_range_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Add](100, 50, 60)

    for i in range(100):
        if i >= 50 and i < 60:
            assert_true(buffer[i] == i + 100)
        else:
            assert_true(buffer[i] == i)


fn test_inplace_empty_buffer_ips() raises:
    """Test inplace operations on empty buffer."""
    print("test_inplace_empty_buffer_ips")
    var buffer = Buffer[DType.int32](0)
    buffer.inplace_ops_scalar[Multiply](5)
    # Should not crash
    assert_true(buffer.size == 0)


fn test_inplace_multiply_by_zero_ips() raises:
    """Test inplace multiplication by zero."""
    print("test_inplace_multiply_by_zero_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i + 1

    buffer.inplace_ops_scalar[Multiply](0)

    for i in range(50):
        assert_true(buffer[i] == 0)


fn test_inplace_multiply_by_one_ips() raises:
    """Test inplace multiplication by one (identity)."""
    print("test_inplace_multiply_by_one_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](1)

    for i in range(50):
        assert_true(buffer[i] == i)


fn test_inplace_add_zero_ips() raises:
    """Test inplace addition of zero (identity)."""
    print("test_inplace_add_zero_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    buffer.inplace_ops_scalar[Add](0)

    for i in range(50):
        assert_true(buffer[i] == i)


fn test_inplace_subtract_zero_ips() raises:
    """Test inplace subtraction of zero (identity)."""
    print("test_inplace_subtract_zero_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    buffer.inplace_ops_scalar[Subtract](0)

    for i in range(50):
        assert_true(buffer[i] == i)


fn test_inplace_divide_by_one_ips() raises:
    """Test inplace division by one (identity)."""
    print("test_inplace_divide_by_one_ips")
    var buffer = Buffer[DType.float32](50)
    for i in range(50):
        buffer[i] = Float32(i)

    buffer.inplace_ops_scalar[Divide](1.0)

    for i in range(50):
        assert_true(abs(buffer[i] - Float32(i)) < 0.0001)


fn test_inplace_multiply_negative_ips() raises:
    """Test inplace multiplication by negative scalar."""
    print("test_inplace_multiply_negative_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](-1)

    for i in range(50):
        assert_true(buffer[i] == -i)


fn test_inplace_add_negative_ips() raises:
    """Test inplace addition of negative scalar."""
    print("test_inplace_add_negative_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i + 100

    buffer.inplace_ops_scalar[Add](-50)

    for i in range(50):
        assert_true(buffer[i] == i + 50)


fn test_inplace_float32_precision_ips() raises:
    """Test inplace operations on float32 with precision check."""
    print("test_inplace_float32_precision_ips")
    var buffer = Buffer[DType.float32](100)
    for i in range(100):
        buffer[i] = Float32(i) * 0.1

    buffer.inplace_ops_scalar[Multiply](10.0)

    for i in range(100):
        assert_true(abs(buffer[i] - Float32(i)) < 0.01)


fn test_inplace_float64_precision_ips() raises:
    """Test inplace operations on float64 with precision check."""
    print("test_inplace_float64_precision_ips")
    var buffer = Buffer[DType.float64](100)
    for i in range(100):
        buffer[i] = Float64(i) * 0.1

    buffer.inplace_ops_scalar[Multiply](10.0)

    for i in range(100):
        assert_true(abs(buffer[i] - Float64(i)) < 0.0001)


fn test_inplace_simd_boundary_ips() raises:
    """Test inplace operations across SIMD boundaries."""
    print("test_inplace_simd_boundary_ips")
    alias simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 5 + 7  # Not aligned to SIMD width
    var buffer = Buffer[DType.int32](size)

    for i in range(size):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](2)

    for i in range(size):
        assert_true(buffer[i] == i * 2)


fn test_inplace_range_partial_simd_ips() raises:
    """Test inplace operations with range not aligned to SIMD."""
    print("test_inplace_range_partial_simd_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Add](1000, 13, 47)

    for i in range(100):
        if i >= 13 and i < 47:
            assert_true(buffer[i] == i + 1000)
        else:
            assert_true(buffer[i] == i)


fn test_inplace_single_element_ips() raises:
    """Test inplace operations on single element."""
    print("test_inplace_single_element_ips")
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 42

    buffer.inplace_ops_scalar[Multiply](3)
    assert_true(buffer[0] == 126)


fn test_inplace_bool_multiply_ips() raises:
    """Test inplace multiplication on boolean buffer."""
    print("test_inplace_bool_multiply_ips")
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = True if i % 2 == 0 else False

    buffer.inplace_ops_scalar[Multiply](False)

    # All should be False after multiplying by False
    for i in range(100):
        assert_true(buffer[i] == False)


fn test_inplace_int8_ips() raises:
    """Test inplace operations on int8."""
    print("test_inplace_int8_ips")
    var buffer = Buffer[DType.int8](128)
    for i in range(128):
        buffer[i] = Int8(i)

    buffer.inplace_ops_scalar[Add](Int8(10))

    for i in range(128):
        var expected = Int8((i + 10) % 256)
        assert_true(buffer[i] == expected)


fn test_inplace_int64_ips() raises:
    """Test inplace operations on int64."""
    print("test_inplace_int64_ips")
    var buffer = Buffer[DType.int64](100)
    for i in range(100):
        buffer[i] = Int64(i) * 1000000

    buffer.inplace_ops_scalar[Divide](Int64(1000000))

    for i in range(100):
        assert_true(buffer[i] == Int64(i))


fn test_inplace_uint8_ips() raises:
    """Test inplace operations on uint8."""
    print("test_inplace_uint8_ips")
    var buffer = Buffer[DType.uint8](100)
    for i in range(100):
        buffer[i] = UInt8(i)

    buffer.inplace_ops_scalar[Multiply](UInt8(2))

    for i in range(100):
        var expected = UInt8((i * 2) % 256)
        assert_true(buffer[i] == expected)


fn test_inplace_chained_operations_ips() raises:
    """Test chaining multiple inplace operations."""
    print("test_inplace_chained_operations_ips")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](2)
    buffer.inplace_ops_scalar[Add](10)
    buffer.inplace_ops_scalar[Subtract](5)

    for i in range(50):
        assert_true(buffer[i] == i * 2 + 10 - 5)


fn test_inplace_large_buffer_ips() raises:
    """Test inplace operations on large buffer."""
    print("test_inplace_large_buffer_ips")
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](3)

    # Check some samples
    assert_true(buffer[0] == 0)
    assert_true(buffer[100] == 300)
    assert_true(buffer[5000] == 15000)
    assert_true(buffer[9999] == 29997)


fn test_inplace_start_equals_end_ips() raises:
    """Test inplace operation with start == end (no-op)."""
    print("test_inplace_start_equals_end_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](999, 50, 50)

    # Nothing should change
    for i in range(100):
        assert_true(buffer[i] == i)


fn test_inplace_invalid_range_ips() raises:
    """Test inplace operation with start > end (no-op)."""
    print("test_inplace_invalid_range_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Multiply](999, 70, 50)

    # Nothing should change
    for i in range(100):
        assert_true(buffer[i] == i)


fn test_inplace_full_range_explicit_ips() raises:
    """Test inplace operation with explicit full range."""
    print("test_inplace_full_range_explicit_ips")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    buffer.inplace_ops_scalar[Add](100, 0, 100)

    for i in range(100):
        assert_true(buffer[i] == i + 100)


# ============================================================================
# Consolidated Test Runner
# ============================================================================


fn run_all_inplace_scalar_tests() raises:
    """Run all buffer inplace scalar operation tests."""
    print("\n=== Running Buffer Inplace Scalar Operations Test Suite ===\n")

    # Basic operation tests
    test_inplace_multiply_scalar_ips()
    test_inplace_add_scalar_ips()
    test_inplace_subtract_scalar_ips()
    test_inplace_divide_scalar_ips()

    # Range tests
    test_inplace_multiply_with_range_ips()
    test_inplace_add_with_range_ips()
    test_inplace_range_partial_simd_ips()

    # Edge case tests
    test_inplace_empty_buffer_ips()
    test_inplace_single_element_ips()
    test_inplace_start_equals_end_ips()
    test_inplace_invalid_range_ips()
    test_inplace_full_range_explicit_ips()

    # Identity operation tests
    test_inplace_multiply_by_zero_ips()
    test_inplace_multiply_by_one_ips()
    test_inplace_add_zero_ips()
    test_inplace_subtract_zero_ips()
    test_inplace_divide_by_one_ips()

    # Negative values tests
    test_inplace_multiply_negative_ips()
    test_inplace_add_negative_ips()

    # Precision tests
    test_inplace_float32_precision_ips()
    test_inplace_float64_precision_ips()

    # SIMD boundary tests
    test_inplace_simd_boundary_ips()

    # Dtype-specific tests
    test_inplace_bool_multiply_ips()
    test_inplace_int8_ips()
    test_inplace_int64_ips()
    test_inplace_uint8_ips()

    # Complex scenarios
    test_inplace_chained_operations_ips()
    test_inplace_large_buffer_ips()

    print("\n=== All Buffer Inplace Scalar Operations Tests Passed! ===\n")


# ============================================================================
# Buffer Arithmetic Operations (Buffer-Buffer) Tests
# ============================================================================


fn test_arithmetic_multiply_buffers_arith() raises:
    """Test element-wise multiplication of two buffers."""
    print("test_arithmetic_multiply_buffers_arith")
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = i
        buf2[i] = 2

    var result = buf1.arithmetic_ops[Multiply](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == i * 2)


fn test_arithmetic_add_buffers_arith() raises:
    """Test element-wise addition of two buffers."""
    print("test_arithmetic_add_buffers_arith")
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = i
        buf2[i] = 10

    var result = buf1.arithmetic_ops[Add](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == i + 10)


fn test_arithmetic_subtract_buffers_arith() raises:
    """Test element-wise subtraction of two buffers."""
    print("test_arithmetic_subtract_buffers_arith")
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = i + 50
        buf2[i] = 50

    var result = buf1.arithmetic_ops[Subtract](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == i)


fn test_arithmetic_divide_buffers_arith() raises:
    """Test element-wise division of two buffers."""
    print("test_arithmetic_divide_buffers_arith")
    var buf1 = Buffer[DType.float32](100)
    var buf2 = Buffer[DType.float32](100)

    for i in range(100):
        buf1[i] = Float32(i * 4)
        buf2[i] = 2.0

    var result = buf1.arithmetic_ops[Divide](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(abs(result[i] - Float32(i * 2)) < 0.001)


fn test_arithmetic_with_ranges_arith() raises:
    """Test arithmetic operations with specified ranges."""
    print("test_arithmetic_with_ranges_arith")
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = i
        buf2[i] = 1

    var result = buf1.arithmetic_ops[Multiply](buf2, 20, 40, 30, 50)

    assert_true(result.size == 20)
    for i in range(20):
        assert_true(
            result[i] == (20 + i)
        )  # buf1[20+i] * buf2[30+i] = (20+i) * 1


fn test_arithmetic_float32_precision_arith() raises:
    """Test arithmetic on float32 with precision."""
    print("test_arithmetic_float32_precision_arith")
    var buf1 = Buffer[DType.float32](100)
    var buf2 = Buffer[DType.float32](100)

    for i in range(100):
        buf1[i] = Float32(i) * 0.5
        buf2[i] = Float32(i) * 0.3

    var result = buf1.arithmetic_ops[Add](buf2)

    for i in range(100):
        var expected = Float32(i) * 0.8
        assert_true(abs(result[i] - expected) < 0.01)


fn test_arithmetic_simd_boundary_arith() raises:
    """Test arithmetic across SIMD boundaries."""
    print("test_arithmetic_simd_boundary_arith")
    alias simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 4 + 7

    var buf1 = Buffer[DType.int32](size)
    var buf2 = Buffer[DType.int32](size)

    for i in range(size):
        buf1[i] = i
        buf2[i] = 3

    var result = buf1.arithmetic_ops[Multiply](buf2)

    for i in range(size):
        assert_true(result[i] == i * 3)


fn test_arithmetic_bool_buffers_arith() raises:
    """Test arithmetic on boolean buffers."""
    print("test_arithmetic_bool_buffers_arith")
    var buf1 = Buffer[DType.bool](100)
    var buf2 = Buffer[DType.bool](100)

    for i in range(100):
        buf1[i] = i % 2 == 0
        buf2[i] = i % 3 == 0

    var result = buf1.arithmetic_ops[Multiply](buf2)  # Logical AND

    for i in range(100):
        var expected = (i % 2 == 0) and (i % 3 == 0)
        assert_true(result[i] == expected)


fn test_arithmetic_int64_arith() raises:
    """Test arithmetic on int64 buffers."""
    print("test_arithmetic_int64_arith")
    var buf1 = Buffer[DType.int64](100)
    var buf2 = Buffer[DType.int64](100)

    for i in range(100):
        buf1[i] = Int64(i) * 1000000
        buf2[i] = Int64(2)

    var result = buf1.arithmetic_ops[Divide](buf2)

    for i in range(100):
        assert_true(result[i] == Int64(i) * 500000)


# ============================================================================
# Buffer Arithmetic Operations (Buffer-Scalar) Tests
# ============================================================================


fn test_arithmetic_scalar_multiply_arith() raises:
    """Test multiplication by scalar."""
    print("test_arithmetic_scalar_multiply_arith")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    var result = buffer.arithmetic_ops_scalar[Multiply](3)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == i * 3)


fn test_arithmetic_scalar_add_arith() raises:
    """Test addition with scalar."""
    print("test_arithmetic_scalar_add_arith")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    var result = buffer.arithmetic_ops_scalar[Add](100)

    for i in range(100):
        assert_true(result[i] == i + 100)


fn test_arithmetic_scalar_subtract_arith() raises:
    """Test subtraction with scalar."""
    print("test_arithmetic_scalar_subtract_arith")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i + 50

    var result = buffer.arithmetic_ops_scalar[Subtract](50)

    for i in range(100):
        assert_true(result[i] == i)


fn test_arithmetic_scalar_reverse_subtract_arith() raises:
    """Test reverse subtraction (scalar - buffer)."""
    print("test_arithmetic_scalar_reverse_subtract_arith")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    var result = buffer.arithmetic_ops_scalar[ReverseSubtract](100)

    for i in range(100):
        assert_true(result[i] == 100 - i)


fn test_arithmetic_scalar_divide_arith() raises:
    """Test division by scalar."""
    print("test_arithmetic_scalar_divide_arith")
    var buffer = Buffer[DType.float32](100)
    for i in range(100):
        buffer[i] = Float32(i * 4)

    var result = buffer.arithmetic_ops_scalar[Divide](2.0)

    for i in range(100):
        assert_true(abs(result[i] - Float32(i * 2)) < 0.001)


fn test_arithmetic_scalar_reverse_divide_arith() raises:
    """Test reverse division (scalar / buffer)."""
    print("test_arithmetic_scalar_reverse_divide_arith")
    var buffer = Buffer[DType.float32](100)
    for i in range(1, 101):  # Start from 1 to avoid division by zero
        buffer[i - 1] = Float32(i)

    var result = buffer.arithmetic_ops_scalar[ReverseDivide](100.0)

    for i in range(100):
        var expected = 100.0 / Float32(i + 1)
        assert_true(abs(result[i] - expected) < 0.01)


fn test_arithmetic_scalar_with_range_arith() raises:
    """Test scalar arithmetic with range."""
    print("test_arithmetic_scalar_with_range_arith")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    var result = buffer.arithmetic_ops_scalar[Multiply](10, 20, 50)

    assert_true(result.size == 30)
    for i in range(30):
        assert_true(result[i] == (20 + i) * 10)


fn test_arithmetic_scalar_float64_arith() raises:
    """Test scalar arithmetic on float64."""
    print("test_arithmetic_scalar_float64_arith")
    var buffer = Buffer[DType.float64](100)
    for i in range(100):
        buffer[i] = Float64(i) * 0.1

    var result = buffer.arithmetic_ops_scalar[Multiply](10.0)

    for i in range(100):
        assert_true(abs(result[i] - Float64(i)) < 0.0001)


fn test_arithmetic_scalar_simd_boundary_arith() raises:
    """Test scalar arithmetic across SIMD boundaries."""
    print("test_arithmetic_scalar_simd_boundary_arith")
    alias simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 3 + 5

    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = i

    var result = buffer.arithmetic_ops_scalar[Add](1000)

    for i in range(size):
        assert_true(result[i] == i + 1000)


fn test_arithmetic_scalar_bool_arith() raises:
    """Test scalar arithmetic on boolean buffer."""
    print("test_arithmetic_scalar_bool_arith")
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = i % 2 == 0

    var result = buffer.arithmetic_ops_scalar[Multiply](False)

    for i in range(100):
        assert_true(result[i] == False)


fn test_arithmetic_scalar_negative_arith() raises:
    """Test scalar arithmetic with negative values."""
    print("test_arithmetic_scalar_negative_arith")
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = i

    var result = buffer.arithmetic_ops_scalar[Multiply](-1)

    for i in range(100):
        assert_true(result[i] == -i)


fn test_arithmetic_scalar_int8_arith() raises:
    """Test scalar arithmetic on int8."""
    print("test_arithmetic_scalar_int8_arith")
    var buffer = Buffer[DType.int8](128)
    for i in range(128):
        buffer[i] = Int8(i)

    var result = buffer.arithmetic_ops_scalar[Add](Int8(10))

    for i in range(128):
        var expected = Int8((i + 10) % 256)
        assert_true(result[i] == expected)


fn test_arithmetic_ops_preserve_original_arith() raises:
    """Test that arithmetic ops don't modify original buffers."""
    print("test_arithmetic_ops_preserve_original_arith")
    var buf1 = Buffer[DType.int32](50)
    var buf2 = Buffer[DType.int32](50)

    for i in range(50):
        buf1[i] = i
        buf2[i] = 10

    _result = buf1.arithmetic_ops[Add](buf2)

    # Check original buffers unchanged
    for i in range(50):
        assert_true(buf1[i] == i)
        assert_true(buf2[i] == 10)


fn test_arithmetic_scalar_preserve_original_arith() raises:
    """Test that scalar ops don't modify original buffer."""
    print("test_arithmetic_scalar_preserve_original_arith")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    _result = buffer.arithmetic_ops_scalar[Multiply](5)

    # Check original buffer unchanged
    for i in range(50):
        assert_true(buffer[i] == i)


fn test_arithmetic_chained_operations_arith() raises:
    """Test chaining multiple arithmetic operations."""
    print("test_arithmetic_chained_operations_arith")
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = i

    var result1 = buffer.arithmetic_ops_scalar[Multiply](2)
    var result2 = result1.arithmetic_ops_scalar[Add](10)
    var result3 = result2.arithmetic_ops_scalar[Subtract](5)

    for i in range(50):
        assert_true(result3[i] == i * 2 + 10 - 5)


fn test_arithmetic_large_buffers_arith() raises:
    """Test arithmetic on large buffers."""
    print("test_arithmetic_large_buffers_arith")
    var buf1 = Buffer[DType.int32](10000)
    var buf2 = Buffer[DType.int32](10000)

    for i in range(10000):
        buf1[i] = i
        buf2[i] = 2

    var result = buf1.arithmetic_ops[Multiply](buf2)

    # Check some samples
    assert_true(result[0] == 0)
    assert_true(result[100] == 200)
    assert_true(result[5000] == 10000)
    assert_true(result[9999] == 19998)


fn test_arithmetic_uint8_arith() raises:
    """Test arithmetic on uint8 buffers."""
    print("test_arithmetic_uint8_arith")
    var buf1 = Buffer[DType.uint8](100)
    var buf2 = Buffer[DType.uint8](100)

    for i in range(100):
        buf1[i] = UInt8(i)
        buf2[i] = UInt8(2)

    var result = buf1.arithmetic_ops[Multiply](buf2)

    for i in range(100):
        var expected = UInt8((i * 2) % 256)
        assert_true(result[i] == expected)


# ============================================================================
# Consolidated Test Runner
# ============================================================================


fn run_all_arithmetic_ops_tests() raises:
    """Run all buffer arithmetic operation tests."""
    print("\n=== Running Buffer Arithmetic Operations Test Suite ===\n")

    print("--- Buffer-Buffer Operations ---")
    test_arithmetic_multiply_buffers_arith()
    test_arithmetic_add_buffers_arith()
    test_arithmetic_subtract_buffers_arith()
    test_arithmetic_divide_buffers_arith()
    test_arithmetic_with_ranges_arith()
    test_arithmetic_float32_precision_arith()
    test_arithmetic_simd_boundary_arith()
    # test_arithmetic_bool_buffers_arith()
    test_arithmetic_int64_arith()
    test_arithmetic_ops_preserve_original_arith()
    test_arithmetic_large_buffers_arith()
    test_arithmetic_uint8_arith()

    print("\n--- Buffer-Scalar Operations ---")
    test_arithmetic_scalar_multiply_arith()
    test_arithmetic_scalar_add_arith()
    test_arithmetic_scalar_subtract_arith()
    test_arithmetic_scalar_reverse_subtract_arith()
    test_arithmetic_scalar_divide_arith()
    test_arithmetic_scalar_reverse_divide_arith()
    test_arithmetic_scalar_with_range_arith()
    test_arithmetic_scalar_float64_arith()
    test_arithmetic_scalar_simd_boundary_arith()
    test_arithmetic_scalar_bool_arith()
    test_arithmetic_scalar_negative_arith()
    test_arithmetic_scalar_int8_arith()
    test_arithmetic_scalar_preserve_original_arith()
    test_arithmetic_chained_operations_arith()

    print("\n=== All Buffer Arithmetic Operations Tests Passed! ===\n")


# ========== Add Operation Tests ==========


fn test_mv_inplace_add_basic() raises:
    """Test basic in-place addition."""
    print("test_mv_inplace_add_basic")
    var buffer1 = Buffer[DType.float32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.float32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = Float32(i)
        buffer2[i] = Float32(i * 2)

    buffer1.inplace_ops[Add](buffer2)

    for i in range(SMALL_SIZE_NEW):
        var expected = Float32(i + i * 2)
        assert_true(
            buffer1[i] == expected,
            "Add failed at index "
            + String(i)
            + ": got "
            + String(buffer1[i])
            + ", expected "
            + String(expected),
        )

    print("✓ test_mv_inplace_add_basic passed")


fn test_mv_inplace_add_range() raises:
    """Test in-place addition with range."""
    print("test_mv_inplace_add_range")
    var buffer1 = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var buffer2 = Buffer[DType.float32].full(5.0, MEDIUM_SIZE)

    # Add only middle portion [20:50)
    buffer1.inplace_ops[Add](buffer2, 20, 50, 20, 50)

    # Check unchanged regions
    for i in range(20):
        assert_true(buffer1[i] == 10.0, "Before range changed")

    # Check modified region
    for i in range(20, 50):
        assert_true(buffer1[i] == 15.0, "Inside range incorrect")

    # Check unchanged tail
    for i in range(50, MEDIUM_SIZE):
        assert_true(buffer1[i] == 10.0, "After range changed")

    print("✓ test_mv_inplace_add_range passed")


# ========== Multiply Operation Tests ==========


fn test_mv_inplace_multiply_basic() raises:
    """Test basic in-place multiplication."""
    print("test_mv_inplace_multiply_basic")
    var buffer1 = Buffer[DType.int32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.int32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = i + 1
        buffer2[i] = 2

    buffer1.inplace_ops[Multiply](buffer2)

    for i in range(SMALL_SIZE_NEW):
        var expected = (i + 1) * 2
        assert_true(buffer1[i] == expected, "Multiply failed at " + String(i))

    print("✓ test_mv_inplace_multiply_basic passed")


fn test_mv_inplace_multiply_large() raises:
    """Test vectorization efficiency on large buffer."""
    print("test_mv_inplace_multiply_large")
    var buffer1 = Buffer[DType.float64](LARGE_SIZE)
    var buffer2 = Buffer[DType.float64](LARGE_SIZE)

    for i in range(LARGE_SIZE):
        buffer1[i] = Float64(i)
        buffer2[i] = 1.5

    buffer1.inplace_ops[Multiply](buffer2)

    for i in range(LARGE_SIZE):
        var expected = Float64(i) * 1.5
        assert_true(abs(buffer1[i] - expected) < 1e-10, "Large multiply failed")

    print("✓ test_mv_inplace_multiply_large passed")


# ========== Subtract Operation Tests ==========


fn test_mv_inplace_subtract_basic() raises:
    """Test basic in-place subtraction."""
    print("test_mv_inplace_subtract_basic")
    var buffer1 = Buffer[DType.int32](MEDIUM_SIZE)
    var buffer2 = Buffer[DType.int32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buffer1[i] = 100
        buffer2[i] = i

    buffer1.inplace_ops[Subtract](buffer2)

    for i in range(MEDIUM_SIZE):
        var expected = 100 - i
        assert_true(buffer1[i] == expected, "Subtract failed")

    print("✓ test_mv_inplace_subtract_basic passed")


fn test_mv_inplace_subtract_negative() raises:
    """Test subtraction resulting in negatives."""
    print("test_mv_inplace_subtract_negative")
    var buffer1 = Buffer[DType.float32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.float32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = Float32(i)
        buffer2[i] = Float32(i + 10)

    buffer1.inplace_ops[Subtract](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == -10.0, "Negative result incorrect")

    print("✓ test_mv_inplace_subtract_negative passed")


# ========== Divide Operation Tests ==========


fn test_mv_inplace_divide_basic() raises:
    """Test basic in-place division."""
    print("test_mv_inplace_divide_basic")
    var buffer1 = Buffer[DType.float32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.float32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = Float32(100.0)
        buffer2[i] = Float32(4.0)

    buffer1.inplace_ops[Divide](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == 25.0, "Divide failed")

    print("✓ test_mv_inplace_divide_basic passed")


fn test_mv_inplace_divide_fractional() raises:
    """Test division with fractional results."""
    print("test_mv_inplace_divide_fractional")
    var buffer1 = Buffer[DType.float64](MEDIUM_SIZE)
    var buffer2 = Buffer[DType.float64](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buffer1[i] = Float64(i + 1)
        buffer2[i] = 3.0

    buffer1.inplace_ops[Divide](buffer2)

    for i in range(MEDIUM_SIZE):
        var expected = Float64(i + 1) / 3.0
        assert_true(
            abs(buffer1[i] - expected) < 1e-10, "Fractional divide failed"
        )

    print("✓ test_mv_inplace_divide_fractional passed")


# ========== Overwrite Operation Tests ==========


fn test_mv_inplace_overwrite_basic() raises:
    """Test basic overwrite operation."""
    print("test_mv_inplace_overwrite_basic")
    var buffer1 = Buffer[DType.int32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.int32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = 999
        buffer2[i] = i * 10

    buffer1.inplace_ops[Overwrite](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == i * 10, "Overwrite failed at " + String(i))

    print("✓ test_mv_inplace_overwrite_basic passed")


fn test_mv_inplace_overwrite_partial() raises:
    """Test partial overwrite with range."""
    print("test_mv_inplace_overwrite_partial")
    var buffer1 = Buffer[DType.float32].full(1.0, MEDIUM_SIZE)
    var buffer2 = Buffer[DType.float32].full(5.0, MEDIUM_SIZE)

    # Overwrite middle section [10:30)
    buffer1.inplace_ops[Overwrite](buffer2, 10, 30, 10, 30)

    # Check preserved head
    for i in range(10):
        assert_true(buffer1[i] == 1.0, "Head should be unchanged")

    # Check overwritten middle
    for i in range(10, 30):
        assert_true(buffer1[i] == 5.0, "Middle should be overwritten")

    # Check preserved tail
    for i in range(30, MEDIUM_SIZE):
        assert_true(buffer1[i] == 1.0, "Tail should be unchanged")

    print("✓ test_mv_inplace_overwrite_partial passed")


fn test_mv_inplace_bool_multiply() raises:
    """Test boolean multiplication (AND operation)."""
    print("test_mv_inplace_bool_multiply")
    var buffer1 = Buffer[DType.bool](MEDIUM_SIZE)
    var buffer2 = Buffer[DType.bool](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buffer1[i] = i < 40
        buffer2[i] = i > 20

    buffer1.inplace_ops[Multiply](buffer2)

    # Check results (bool multiply is AND)
    for i in range(MEDIUM_SIZE):
        var expected = (i < 40) and (i > 20)  # True for [21, 39]
        assert_true(buffer1[i] == expected, "Bool multiply failed")

    print("✓ test_mv_inplace_bool_multiply passed")


fn test_mv_inplace_bool_overwrite() raises:
    """Test boolean overwrite."""
    print("test_mv_inplace_bool_overwrite")
    var buffer1 = Buffer[DType.bool](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.bool](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = True
        buffer2[i] = (i % 2) == 0

    buffer1.inplace_ops[Overwrite](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == ((i % 2) == 0), "Bool overwrite failed")

    print("✓ test_mv_inplace_bool_overwrite passed")


# ========== Edge Cases ==========


fn test_mv_inplace_single_element() raises:
    """Test with single element buffer."""
    print("test_mv_inplace_single_element")
    var buffer1 = Buffer[DType.int32](1)
    var buffer2 = Buffer[DType.int32](1)

    buffer1[0] = 10
    buffer2[0] = 5

    buffer1.inplace_ops[Add](buffer2)
    assert_true(buffer1[0] == 15, "Single element add failed")

    buffer1.inplace_ops[Multiply](buffer2)
    assert_true(buffer1[0] == 75, "Single element multiply failed")

    print("✓ test_mv_inplace_single_element passed")


fn test_mv_inplace_tail_alignment() raises:
    """Test SIMD tail loop handling."""
    print("test_mv_inplace_tail_alignment")

    # Size that's not SIMD-aligned (e.g., 13 for 16-byte SIMD)
    alias SIZE = 13
    var buffer1 = Buffer[DType.float32](SIZE)
    var buffer2 = Buffer[DType.float32](SIZE)

    for i in range(SIZE):
        buffer1[i] = Float32(i)
        buffer2[i] = 1.0

    buffer1.inplace_ops[Add](buffer2)

    # Verify all elements, especially tail
    for i in range(SIZE):
        var expected = Float32(i + 1)
        assert_true(buffer1[i] == expected, "Tail element failed")

    print("✓ test_mv_inplace_tail_alignment passed")


fn test_mv_inplace_mismatched_ranges() raises:
    """Test validation catches mismatched ranges."""
    print("test_mv_inplace_mismatched_ranges")
    var buffer1 = Buffer[DType.int32](100)
    var buffer2 = Buffer[DType.int32](100)

    for i in range(100):
        buffer1[i] = i
        buffer2[i] = 0

    # This should fail validation (range 10 vs 20)
    buffer1.inplace_ops[Add, validate=True](buffer2, 0, 10, 0, 20)

    # Buffer1 should be unchanged (operation skipped)
    for i in range(100):
        assert_true(buffer1[i] == i, "Validation bypass failed")

    print("✓ test_mv_inplace_mismatched_ranges passed")


# ========== Performance Comparison Test ==========


fn test_mv_performance_comparison() raises:
    """Compare performance: small vs large buffers."""
    print("test_mv_performance_comparison")

    alias PERF_SIZE = 10000
    var buffer1 = Buffer[DType.float64](PERF_SIZE)
    var buffer2 = Buffer[DType.float64](PERF_SIZE)

    for i in range(PERF_SIZE):
        buffer1[i] = Float64(i)
        buffer2[i] = 2.0

    var t0 = perf_counter_ns()
    buffer1.inplace_ops[Multiply](buffer2)
    var t1 = perf_counter_ns()

    var time_ms = Float64(t1 - t0) / 1e6
    print(
        "  Multiply "
        + String(PERF_SIZE)
        + " elements: "
        + String(time_ms)
        + " ms"
    )

    # Verify correctness
    for i in range(PERF_SIZE):
        var expected = Float64(i * 2)
        assert_true(
            abs(buffer1[i] - expected) < 1e-10,
            "Performance test: result incorrect",
        )

    print("✓ test_mv_performance_comparison passed")


# ========== Test Runner ==========


fn run_all_manual_vectorization_tests() raises:
    """Run all manual vectorization tests."""
    print("\n" + "=" * 60)
    print("Manual Vectorization In-place Ops Tests")
    print("=" * 60 + "\n")

    # Add tests
    test_mv_inplace_add_basic()
    test_mv_inplace_add_range()

    # Multiply tests
    test_mv_inplace_multiply_basic()
    test_mv_inplace_multiply_large()

    # Subtract tests
    test_mv_inplace_subtract_basic()
    test_mv_inplace_subtract_negative()

    # Divide tests
    test_mv_inplace_divide_basic()
    test_mv_inplace_divide_fractional()

    # Overwrite tests
    test_mv_inplace_overwrite_basic()
    test_mv_inplace_overwrite_partial()

    # Bool tests (bit-packed)
    # test_mv_inplace_bool_multiply()
    test_mv_inplace_bool_overwrite()

    # Edge cases
    test_mv_inplace_single_element()
    test_mv_inplace_tail_alignment()
    test_mv_inplace_mismatched_ranges()

    # Performance
    test_mv_performance_comparison()

    print("\n" + "=" * 60)
    print("✅ All Manual Vectorization Tests Passed!")
    print("=" * 60)


alias TEST_SMALL = 17
alias TEST_MEDIUM = 68
alias TEST_LARGE = 1024

# ========== SUM TESTS ==========


fn test_bufops_sum_basic() raises:
    """Test basic sum operation."""
    print("test_bufops_sum_basic")
    var buffer = Buffer[DType.int32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = i + 1  # 1, 2, 3, ..., 17

    var result = buffer.sum()
    var expected = (TEST_SMALL * (TEST_SMALL + 1)) // 2  # Sum formula

    assert_true(
        result == expected,
        "Sum failed: got " + String(result) + ", expected " + String(expected),
    )

    print("✓ test_bufops_sum_basic passed")


fn test_bufops_sum_range() raises:
    """Test sum with start/end indices."""
    print("test_bufops_sum_range")
    var buffer = Buffer[DType.float32](TEST_MEDIUM)

    for i in range(TEST_MEDIUM):
        buffer[i] = 1.0

    # Sum only middle portion [10:30)
    var result = buffer.sum(10, 30)
    assert_true(result == 20.0, "Sum range failed")

    print("✓ test_bufops_sum_range passed")


fn test_bufops_sum_negative() raises:
    """Test sum with negative numbers."""
    print("test_bufops_sum_negative")
    var buffer = Buffer[DType.int32](10)

    for i in range(10):
        buffer[i] = i - 5  # -5, -4, ..., 3, 4

    var result = buffer.sum()
    var expected = -5  # Sum of -5 to 4

    assert_true(result == expected, "Sum with negatives failed")

    print("✓ test_bufops_sum_negative passed")


fn test_bufops_sum_large() raises:
    """Test sum on large buffer (vectorization efficiency)."""
    print("test_bufops_sum_large")
    var buffer = Buffer[DType.float64](TEST_LARGE)

    for i in range(TEST_LARGE):
        buffer[i] = 1.0

    var result = buffer.sum()
    assert_true(result == Float64(TEST_LARGE), "Large sum failed")

    print("✓ test_bufops_sum_large passed")


fn test_bufops_sum_empty_range() raises:
    """Test sum with empty range."""
    print("test_bufops_sum_empty_range")
    var buffer = Buffer[DType.int32](100)

    var result = buffer.sum(50, 50)  # Empty range
    assert_true(result == 0, "Empty sum should be 0")

    print("✓ test_bufops_sum_empty_range passed")


# ========== PRODUCT TESTS ==========


fn test_bufops_product_basic() raises:
    """Test basic product operation."""
    print("test_bufops_product_basic")
    var buffer = Buffer[DType.int32](5)

    for i in range(5):
        buffer[i] = i + 1  # 1, 2, 3, 4, 5

    var result = buffer.product()
    assert_true(result == 120, "Product failed: expected 120")

    print("✓ test_bufops_product_basic passed")


fn test_bufops_product_range() raises:
    """Test product with range."""
    print("test_bufops_product_range")
    var buffer = Buffer[DType.float32](20)

    for i in range(20):
        buffer[i] = 2.0

    # Product of indices [5:10) = 2^5 = 32
    var result = buffer.product(5, 10)
    assert_true(result == 32.0, "Product range failed")

    print("✓ test_bufops_product_range passed")


fn test_bufops_product_with_zero() raises:
    """Test product with zero element."""
    print("test_bufops_product_with_zero")
    var buffer = Buffer[DType.int32](10)

    for i in range(10):
        buffer[i] = i  # 0, 1, 2, ..., 9

    var result = buffer.product()
    assert_true(result == 0, "Product with zero should be 0")

    print("✓ test_bufops_product_with_zero passed")


fn test_bufops_product_fractional() raises:
    """Test product with fractional values."""
    print("test_bufops_product_fractional")
    var buffer = Buffer[DType.float64](4)

    buffer[0] = 0.5
    buffer[1] = 2.0
    buffer[2] = 4.0
    buffer[3] = 0.25

    var result = buffer.product()
    assert_true(result == 1.0, "Fractional product failed")

    print("✓ test_bufops_product_fractional passed")


fn test_bufops_product_empty() raises:
    """Test product of empty range (should be 1)."""
    print("test_bufops_product_empty")
    var buffer = Buffer[DType.int32](100)

    var result = buffer.product(10, 10)  # Empty range
    assert_true(result == 1, "Empty product should be 1")

    print("✓ test_bufops_product_empty passed")


# ========== POWER TESTS ==========


fn test_bufops_pow_basic() raises:
    """Test basic power operation."""
    print("test_bufops_pow_basic")
    var buffer = Buffer[DType.float32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = Float32(i)

    var result = buffer.__pow__(2.0)

    for i in range(TEST_SMALL):
        var expected = Float32(i * i)
        assert_true(result[i] == expected, "Pow failed at " + String(i))

    print("✓ test_bufops_pow_basic passed")


fn test_bufops_pow_fractional() raises:
    """Test power with fractional exponent."""
    print("test_bufops_pow_fractional")
    var buffer = Buffer[DType.float64](10)

    for i in range(10):
        buffer[i] = Float64((i + 1) * (i + 1))  # 1, 4, 9, 16, ...

    var result = buffer.__pow__(0.5)  # Square root

    for i in range(10):
        var expected = Float64(i + 1)
        assert_true(abs(result[i] - expected) < 1e-8, "Fractional pow failed")

    print("✓ test_bufops_pow_fractional passed")


fn test_bufops_pow_zero_exponent() raises:
    """Test power with exponent 0 (should be all 1s)."""
    print("test_bufops_pow_zero_exponent")
    var buffer = Buffer[DType.int32](TEST_MEDIUM)

    for i in range(TEST_MEDIUM):
        buffer[i] = i * 10

    var result = buffer.__pow__(0)

    for i in range(TEST_MEDIUM):
        assert_true(result[i] == 1, "Pow 0 should be 1")

    print("✓ test_bufops_pow_zero_exponent passed")


fn test_bufops_pow_negative_base() raises:
    """Test power with negative base values."""
    print("test_bufops_pow_negative_base")
    var buffer = Buffer[DType.float32](5)

    buffer[0] = -2.0
    buffer[1] = -1.0
    buffer[2] = 0.0
    buffer[3] = 1.0
    buffer[4] = 2.0

    var result = buffer.__pow__(3.0)

    assert_true(result[0] == -8.0, "Negative pow failed")
    assert_true(result[1] == -1.0, "Negative pow failed")
    assert_true(result[2] == 0.0, "Zero pow failed")
    assert_true(result[3] == 1.0, "Positive pow failed")
    assert_true(result[4] == 8.0, "Positive pow failed")

    print("✓ test_bufops_pow_negative_base passed")


# ========== ABSOLUTE VALUE TESTS ==========


fn test_bufops_abs_basic() raises:
    """Test basic absolute value."""
    print("test_bufops_abs_basic")
    var buffer = Buffer[DType.int32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = i - 8  # -8, -7, ..., 7, 8

    var result = buffer.__abs__()

    for i in range(TEST_SMALL):
        var expected = abs(i - 8)
        assert_true(result[i] == expected, "Abs failed")

    print("✓ test_bufops_abs_basic passed")


fn test_bufops_abs_all_negative() raises:
    """Test abs with all negative values."""
    print("test_bufops_abs_all_negative")
    var buffer = Buffer[DType.float32](TEST_MEDIUM)

    for i in range(TEST_MEDIUM):
        buffer[i] = Float32(-(i + 1))

    var result = buffer.__abs__()

    for i in range(TEST_MEDIUM):
        assert_true(result[i] == Float32(i + 1), "Abs negative failed")

    print("✓ test_bufops_abs_all_negative passed")


fn test_bufops_abs_all_positive() raises:
    """Test abs with all positive values (should be unchanged)."""
    print("test_bufops_abs_all_positive")
    var buffer = Buffer[DType.int32](20)

    for i in range(20):
        buffer[i] = i * 5

    var result = buffer.__abs__()

    for i in range(20):
        assert_true(result[i] == i * 5, "Abs positive changed value")

    print("✓ test_bufops_abs_all_positive passed")


fn test_bufops_abs_mixed() raises:
    """Test abs with mixed positive/negative."""
    print("test_bufops_abs_mixed")
    var buffer = Buffer[DType.float64](10)

    buffer[0] = -100.0
    buffer[1] = -1.5
    buffer[2] = -0.001
    buffer[3] = 0.0
    buffer[4] = 0.001
    buffer[5] = 1.5
    buffer[6] = 100.0
    buffer[7] = -50.0
    buffer[8] = 50.0
    buffer[9] = -0.5

    var result = buffer.__abs__()

    assert_true(result[0] == 100.0, "Abs mixed failed")
    assert_true(result[3] == 0.0, "Abs zero failed")
    assert_true(result[7] == 50.0, "Abs mixed failed")

    print("✓ test_bufops_abs_mixed passed")


# ========== FILL TESTS ==========


fn test_bufops_fill_basic() raises:
    """Test basic fill operation."""
    print("test_bufops_fill_basic")
    var buffer = Buffer[DType.int32](TEST_SMALL)

    buffer.fill(42)

    for i in range(TEST_SMALL):
        assert_true(buffer[i] == 42, "Fill failed")

    print("✓ test_bufops_fill_basic passed")


fn test_bufops_fill_range() raises:
    """Test fill with range."""
    print("test_bufops_fill_range")
    var buffer = Buffer[DType.float32](TEST_MEDIUM)

    # Initialize with 1.0
    buffer.fill(1.0)

    # Fill middle section with 5.0
    buffer.fill(5.0, 20, 40)

    # Check head
    for i in range(20):
        assert_true(buffer[i] == 1.0, "Fill head changed")

    # Check middle
    for i in range(20, 40):
        assert_true(buffer[i] == 5.0, "Fill range failed")

    # Check tail
    for i in range(40, TEST_MEDIUM):
        assert_true(buffer[i] == 1.0, "Fill tail changed")

    print("✓ test_bufops_fill_range passed")


fn test_bufops_fill_bool() raises:
    """Test fill with boolean type."""
    print("test_bufops_fill_bool")
    var buffer = Buffer[DType.bool](TEST_SMALL)

    buffer.fill(True)

    for i in range(TEST_SMALL):
        assert_true(buffer[i] == True, "Bool fill failed")

    buffer.fill(False, 5, 10)

    for i in range(5, 10):
        assert_true(buffer[i] == False, "Bool fill range failed")

    print("✓ test_bufops_fill_bool passed")


fn test_bufops_fill_zero() raises:
    """Test fill with zero."""
    print("test_bufops_fill_zero")
    var buffer = Buffer[DType.float64](TEST_MEDIUM)

    # Set to non-zero
    for i in range(TEST_MEDIUM):
        buffer[i] = Float64(i + 100)

    # Fill with zero
    buffer.fill(0.0)

    for i in range(TEST_MEDIUM):
        assert_true(buffer[i] == 0.0, "Fill zero failed")

    print("✓ test_bufops_fill_zero passed")


fn test_bufops_fill_negative() raises:
    """Test fill with negative value."""
    print("test_bufops_fill_negative")
    var buffer = Buffer[DType.int32](30)

    buffer.fill(-999)

    for i in range(30):
        assert_true(buffer[i] == -999, "Fill negative failed")

    print("✓ test_bufops_fill_negative passed")


# ========== NEGATION TESTS ==========


fn test_bufops_neg_basic() raises:
    """Test basic negation."""
    print("test_bufops_neg_basic")
    var buffer = Buffer[DType.int32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = i + 1

    var result = buffer.__neg__()

    for i in range(TEST_SMALL):
        assert_true(result[i] == -(i + 1), "Neg failed")

    print("✓ test_bufops_neg_basic passed")


fn test_bufops_neg_mixed() raises:
    """Test negation with mixed signs."""
    print("test_bufops_neg_mixed")
    var buffer = Buffer[DType.float32](10)

    for i in range(10):
        buffer[i] = Float32(i - 5)  # -5, -4, ..., 3, 4

    var result = buffer.__neg__()

    for i in range(10):
        var expected = Float32(5 - i)
        assert_true(result[i] == expected, "Neg mixed failed")

    print("✓ test_bufops_neg_mixed passed")


fn test_bufops_neg_zero() raises:
    """Test negation with zeros."""
    print("test_bufops_neg_zero")
    var buffer = Buffer[DType.float64](TEST_MEDIUM)

    buffer.fill(0.0)

    var result = buffer.__neg__()

    for i in range(TEST_MEDIUM):
        assert_true(result[i] == 0.0, "Neg zero failed")

    print("✓ test_bufops_neg_zero passed")


fn test_bufops_neg_double_negation() raises:
    """Test double negation (should restore original)."""
    print("test_bufops_neg_double_negation")
    var buffer = Buffer[DType.int32](20)

    for i in range(20):
        buffer[i] = i * 10

    var neg1 = buffer.__neg__()
    var neg2 = neg1.__neg__()

    for i in range(20):
        assert_true(neg2[i] == buffer[i], "Double neg failed")

    print("✓ test_bufops_neg_double_negation passed")


# ========== EDGE CASE TESTS ==========


fn test_bufops_single_element() raises:
    """Test all operations on single element buffer."""
    print("test_bufops_single_element")

    var buffer = Buffer[DType.float32](1)
    buffer[0] = 5.0

    assert_true(buffer.sum() == 5.0, "Single sum failed")
    assert_true(buffer.product() == 5.0, "Single product failed")

    var pow_result = buffer.__pow__(2.0)
    assert_true(pow_result[0] == 25.0, "Single pow failed")

    var abs_result = buffer.__abs__()
    assert_true(abs_result[0] == 5.0, "Single abs failed")

    var neg_result = buffer.__neg__()
    assert_true(neg_result[0] == -5.0, "Single neg failed")

    print("✓ test_bufops_single_element passed")


fn test_bufops_tail_alignment() raises:
    """Test tail loop handling for non-SIMD-aligned sizes."""
    print("test_bufops_tail_alignment")

    alias SIZE = 13  # Likely not SIMD-aligned
    var buffer = Buffer[DType.float64](SIZE)

    for i in range(SIZE):
        buffer[i] = Float64(i + 1)

    # Test sum (verify tail elements)
    var sum_result = buffer.sum()
    var expected_sum = Float64((SIZE * (SIZE + 1)) // 2)
    assert_true(sum_result == expected_sum, "Tail sum failed")

    # Test fill (verify tail elements)
    buffer.fill(7.0)
    for i in range(SIZE):
        assert_true(buffer[i] == 7.0, "Tail fill failed at " + String(i))

    print("✓ test_bufops_tail_alignment passed")


fn test_bufops_performance_comparison() raises:
    """Compare performance across operations."""
    print("test_bufops_performance_comparison")

    alias PERF_SIZE = 10000
    var buffer = Buffer[DType.float64](PERF_SIZE)

    for i in range(PERF_SIZE):
        buffer[i] = Float64(i + 1)

    var t0 = perf_counter_ns()
    var _sum_result = buffer.sum()
    var t1 = perf_counter_ns()

    var t2 = perf_counter_ns()
    var _abs_result = buffer.__abs__()
    var t3 = perf_counter_ns()

    var t4 = perf_counter_ns()
    buffer.fill(1.0)
    var t5 = perf_counter_ns()

    print(
        "  Sum "
        + String(PERF_SIZE)
        + " elements: "
        + String(Float64(t1 - t0) / 1e6)
        + " ms"
    )
    print(
        "  Abs "
        + String(PERF_SIZE)
        + " elements: "
        + String(Float64(t3 - t2) / 1e6)
        + " ms"
    )
    print(
        "  Fill "
        + String(PERF_SIZE)
        + " elements: "
        + String(Float64(t5 - t4) / 1e6)
        + " ms"
    )

    print("✓ test_bufops_performance_comparison passed")


# ========== TEST RUNNER ==========


fn run_all_buffer_ops_tests() raises:
    """Run all buffer operations tests."""
    print("\n" + "=" * 70)
    print("Manual Vectorization Buffer Operations Tests")
    print("=" * 70 + "\n")

    # Sum tests (5)
    test_bufops_sum_basic()
    test_bufops_sum_range()
    test_bufops_sum_negative()
    test_bufops_sum_large()
    test_bufops_sum_empty_range()

    # Product tests (5)
    test_bufops_product_basic()
    test_bufops_product_range()
    test_bufops_product_with_zero()
    test_bufops_product_fractional()
    test_bufops_product_empty()

    # Power tests (4)
    test_bufops_pow_basic()
    test_bufops_pow_fractional()
    test_bufops_pow_zero_exponent()
    test_bufops_pow_negative_base()

    # Absolute value tests (4)
    test_bufops_abs_basic()
    test_bufops_abs_all_negative()
    test_bufops_abs_all_positive()
    test_bufops_abs_mixed()

    # Fill tests (5)
    test_bufops_fill_basic()
    test_bufops_fill_range()
    test_bufops_fill_bool()
    test_bufops_fill_zero()
    test_bufops_fill_negative()

    # Negation tests (4)
    test_bufops_neg_basic()
    test_bufops_neg_mixed()
    test_bufops_neg_zero()
    test_bufops_neg_double_negation()

    # Edge cases (3)
    test_bufops_single_element()
    test_bufops_tail_alignment()
    test_bufops_performance_comparison()

    print("\n" + "=" * 70)
    print("✅ All 30 Buffer Operations Tests Passed!")
    print("=" * 70)


fn test_compare_buffer_manual() raises:
    print("test_compare_buffer_manual")
    var buf1 = Buffer[DType.int32](MEDIUM_SIZE)
    var buf2 = Buffer[DType.int32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buf1[i] = i
        buf2[i] = i if i < 30 else i + 1

    var result = buf1.compare_buffer_full[Equal](buf2)

    # Count equal elements (first 30 should be equal)
    var equal_count = 0
    for i in range(MEDIUM_SIZE):
        if result[i]:
            equal_count += 1

    assert_true(
        equal_count == 30,
        "compare_buffer_manual: expected 30 equal, got "
        + equal_count.__str__(),
    )
    print("test_compare_buffer_manual passed")


fn test_compare_scalar_manual() raises:
    print("test_compare_scalar_manual")
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = i % 10

    var result = buffer.compare_scalar_full[Equal](5)

    # Count elements equal to 5
    var count = 0
    for i in range(MEDIUM_SIZE):
        if result[i]:
            count += 1

    # Values 0-67, with i % 10 == 5 at indices: 5, 15, 25, 35, 45, 55, 65 = 7 occurrences
    assert_true(
        count == 7,
        "compare_scalar_manual: expected 7 fives, got " + count.__str__(),
    )
    print("test_compare_scalar_manual passed")


fn test_unary_ops_manual_relu() raises:
    print("test_unary_ops_manual_relu")
    var buffer = Buffer[DType.float32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Float32(i - 30)  # Values from -30 to 37

    var result = buffer.unary_ops[ReLUForwardOp]()

    # Check that negative values are zeroed
    var all_correct = True
    for i in range(MEDIUM_SIZE):
        var expected = Float32(0) if i < 30 else Float32(i - 30)
        if abs(result[i] - expected) > 1e-6:
            all_correct = False
            break

    assert_true(all_correct, "unary_ops_manual_relu: ReLU forward failed")
    print("test_unary_ops_manual_relu passed")


fn test_select_manual_relu_backward() raises:
    print("test_select_manual_relu_backward")
    var input_buf = Buffer[DType.float32](MEDIUM_SIZE)
    var grad_buf = Buffer[DType.float32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        input_buf[i] = Float32(i - 30)  # Values from -30 to 37
        grad_buf[i] = 1.0  # All gradients are 1.0

    var result = input_buf.select[ReLUBackwardOp](grad_buf)

    # Check that gradients are zeroed where input <= 0
    var all_correct = True
    for i in range(MEDIUM_SIZE):
        var expected = Float32(0) if i < 31 else Float32(1.0)
        if abs(result[i] - expected) > 1e-6:
            all_correct = False
            break

    assert_true(all_correct, "select_manual_relu_backward: failed")
    print("test_select_manual_relu_backward passed")


fn test_compare_buffer_manual_gt() raises:
    print("test_compare_buffer_manual_gt")
    var buf1 = Buffer[DType.int32](MEDIUM_SIZE)
    var buf2 = Buffer[DType.int32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buf1[i] = i
        buf2[i] = 30

    var result = buf1.compare_buffer_full[GreaterThan](buf2)

    # Count elements > 30
    var count = 0
    for i in range(MEDIUM_SIZE):
        if result[i]:
            count += 1

    # Elements 31-67 are > 30, so 37 elements
    assert_true(
        count == 37,
        "compare_buffer_manual_gt: expected 37, got " + count.__str__(),
    )
    print("test_compare_buffer_manual_gt passed")


fn test_unary_ops_manual_sigmoid() raises:
    print("test_unary_ops_manual_sigmoid")
    var buffer = Buffer[DType.float32](10)
    for i in range(10):
        buffer[i] = Float32(i - 5)  # Values from -5 to 4

    var result = buffer.unary_ops[SigmoidOp]()

    # Check that all values are in (0, 1) range
    var all_in_range = True
    for i in range(10):
        if result[i] <= 0.0 or result[i] >= 1.0:
            all_in_range = False
            break

    assert_true(all_in_range, "unary_ops_manual_sigmoid: values out of range")
    print("test_unary_ops_manual_sigmoid passed")


# Consolidated test runner
fn run_all_manual_vectorization_tests_2() raises:
    print("\n========== Running Manual Vectorization Tests ==========\n")
    test_compare_buffer_manual()
    test_compare_scalar_manual()
    test_unary_ops_manual_relu()
    test_select_manual_relu_backward()
    test_compare_buffer_manual_gt()
    test_unary_ops_manual_sigmoid()
    print("\n========== All Manual Vectorization Tests Passed ==========\n")


fn test_relu_forward_with_mask() raises:
    print("test_relu_forward_with_mask")
    var buffer = Buffer[DType.float32](10)
    for i in range(10):
        buffer[i] = Float32(i - 5)  # Values from -5 to 4

    var result = buffer.unary_ops_with_mask[ReLUForwardOp]()
    var output = result[0]
    var mask = result[1]

    # Check output: negative values should be 0
    var output_correct = True
    for i in range(10):
        var expected = Float32(0) if i < 6 else Float32(i - 5)
        if abs(output[i] - expected) > 1e-6:
            output_correct = False
            break

    # Check mask: should be 0.0 for negative inputs, 1.0 for positive
    var mask_correct = True
    for i in range(10):
        var expected = Float32(0) if i < 6 else Float32(1)
        if abs(mask[i] - expected) > 1e-6:
            mask_correct = False
            break

    assert_true(output_correct, "relu_forward_with_mask: output incorrect")
    assert_true(mask_correct, "relu_forward_with_mask: mask incorrect")
    print("test_relu_forward_with_mask passed")


fn test_buffer_multiply() raises:
    print("test_buffer_multiply")
    var buf1 = Buffer[DType.float32](MEDIUM_SIZE)
    var buf2 = Buffer[DType.float32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buf1[i] = Float32(i)
        buf2[i] = Float32(2) if i < 30 else Float32(0)  # Mask: 2.0 or 0.0

    var result = buf1 * buf2  # Using Buffer's multiply operator

    # Check: first 30 should be doubled, rest should be 0
    var all_correct = True
    for i in range(MEDIUM_SIZE):
        var expected = Float32(i * 2) if i < 30 else Float32(0)
        if abs(result[i] - expected) > 1e-6:
            all_correct = False
            break

    assert_true(all_correct, "buffer_multiply: incorrect results")
    print("test_buffer_multiply passed")


fn test_relu_backward_multiplication() raises:
    print("test_relu_backward_multiplication")

    # Simulate: input had values [-2, -1, 0, 1, 2]
    # Mask should be [0, 0, 0, 1, 1]
    var mask = Buffer[DType.float32](5)
    mask[0] = 0.0
    mask[1] = 0.0
    mask[2] = 0.0
    mask[3] = 1.0
    mask[4] = 1.0

    # Incoming gradients [1, 1, 1, 1, 1]
    var grad = Buffer[DType.float32](5)
    for i in range(5):
        grad[i] = 1.0

    # Backward: grad * mask (using Buffer's operator)
    var result = grad * mask

    # Expected: [0, 0, 0, 1, 1]
    var correct = (
        abs(result[0] - 0.0) < 1e-6
        and abs(result[1] - 0.0) < 1e-6
        and abs(result[2] - 0.0) < 1e-6
        and abs(result[3] - 1.0) < 1e-6
        and abs(result[4] - 1.0) < 1e-6
    )

    assert_true(
        correct, "relu_backward_multiplication: incorrect gradient masking"
    )
    print("test_relu_backward_multiplication passed")


fn run_relu_optimization_tests() raises:
    print("\n========== Running ReLU Optimization Tests ==========\n")
    test_relu_forward_with_mask()
    test_buffer_multiply()
    test_relu_backward_multiplication()
    print("\n========== All ReLU Optimization Tests Passed ==========\n")
