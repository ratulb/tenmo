from tenmo.buffers import Buffer
from tenmo.mnemonics import *
from tenmo.numpy_interop import ndarray_ptr
from std.sys import simd_width_of
from std.time import perf_counter_ns
from std.testing import assert_almost_equal, TestSuite
from std.testing import assert_true, assert_false
from std.python import Python


# ============================================
# Test Constants
# ============================================
comptime SMALL_SIZE = 7  # Less than typical SIMD width
comptime MEDIUM_SIZE = 68  # Multiple SIMD blocks + remainder
comptime LARGE_SIZE = 1000  # Stress test
comptime SMALL_SIZE_NEW = 17  # Prime number to test tail loop


# ============================================
# Constructor Tests
# ============================================
def test_constructor_empty() raises:
    var buffer = Buffer[DType.int32]()
    assert_true(buffer.size == 0, "Empty buffer size should be 0")
    assert_true(len(buffer) == 0, "Empty buffer len should be 0")


def test_constructor_size() raises:
    var buffer = Buffer[DType.float32](MEDIUM_SIZE)
    assert_true(buffer.size == MEDIUM_SIZE, "Buffer size mismatch")
    assert_true(len(buffer) == MEDIUM_SIZE, "Buffer len mismatch")


def test_constructor_from_list() raises:
    var ll = List[Scalar[DType.int32]](capacity=5)
    for i in range(5):
        ll.append(Scalar[DType.int32](i * 10))

    var buffer = Buffer[DType.int32](ll)
    assert_true(buffer.size == 5, "Buffer size from list mismatch")
    for i in range(5):
        assert_true(
            buffer[i] == Int32(i * 10),
            "Buffer value from list mismatch at " + String(i),
        )


def test_constructor_external_ptr() raises:
    """Buffer wrapping an external pointer (copy=False — the old 'rebind' path).
    """
    var n = 5
    var ptr = alloc[Scalar[DType.float32]](n)
    for i in range(n):
        ptr[i] = Float32(Float32(i) * 1.5)
    var buffer = Buffer[DType.float32](n, ptr, copy=False)
    assert_true(buffer.size == n)
    assert_true(buffer.external)
    for i in range(n):
        assert_true(buffer[i] == Float32(Float32(i) * 1.5))
    ptr.free()


def test_constructor_external_ptr_copy() raises:
    """Buffer deep-copying from an external pointer (copy=True)."""
    var n = 4
    var ptr = alloc[Scalar[DType.int64]](n)
    for i in range(n):
        ptr[i] = Int64(i * 100)
    var buffer = Buffer[DType.int64](n, ptr, copy=True)
    assert_true(buffer.size == n)
    assert_false(buffer.external)
    for i in range(n):
        assert_true(buffer[i] == Int64(i * 100))
    # Mutate original — buffer must be unaffected
    ptr[0] = Int64(-999)
    assert_true(
        buffer[0] == Int64(0), "Buffer must not share memory with external ptr"
    )
    ptr.free()


# ============================================
# Static Factory Tests
# ============================================
def test_full() raises:
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)
    assert_true(buffer.size == MEDIUM_SIZE, "full: size mismatch")
    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 42, "full: value mismatch at " + String(i))


def test_zeros() raises:
    var buffer = Buffer[DType.float32].zeros(MEDIUM_SIZE)
    assert_true(buffer.size == MEDIUM_SIZE, "zeros: size mismatch")
    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 0.0, "zeros: value mismatch at " + String(i))


def test_arange_single_arg() raises:
    var buffer = Buffer[DType.int32].arange(10)
    assert_true(buffer.size == 10, "arange: size mismatch")
    for i in range(10):
        assert_true(
            buffer[i] == Int32(i), "arange: value mismatch at " + String(i)
        )


def test_arange_two_args() raises:
    var buffer = Buffer[DType.int32].arange(5, 10)
    assert_true(buffer.size == 5, "arange: size mismatch")
    for i in range(5):
        assert_true(buffer[i] == Int32(i + 5), "arange: value mismatch")


def test_arange_three_args() raises:
    var buffer = Buffer[DType.int32].arange(0, 10, 2)
    assert_true(buffer.size == 5, "arange step 2: size mismatch")
    for i in range(5):
        assert_true(buffer[i] == Int32(i * 2), "arange step 2: value mismatch")


def test_linspace() raises:
    var buffer = Buffer[DType.float32].linspace(0.0, 10.0, 11)
    assert_true(buffer.size == 11, "linspace: size mismatch")
    for i in range(11):
        assert_true(
            abs(buffer[i] - Scalar[DType.float32](i)) < 0.001,
            "linspace: value mismatch at " + String(i),
        )


# ============================================
# Indexing and Slicing Tests
# ============================================
def test_getitem_setitem() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i) * 5

    for i in range(10):
        assert_true(buffer[i] == Int32(i * 5), "getitem/setitem mismatch")


def test_slice_contiguous() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i)

    var sliced = buffer[2:7]
    assert_true(sliced.size == 5, "slice: size mismatch")
    for i in range(5):
        assert_true(sliced[i] == Int32(i) + 2, "slice: value mismatch")


def test_slice_with_step() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i)

    var sliced = buffer[0:10:2]
    assert_true(sliced.size == 5, "slice step: size mismatch")
    for i in range(5):
        assert_true(sliced[i] == Int32(i * 2), "slice step: value mismatch")


def test_slice_empty() raises:
    var buffer = Buffer[DType.int32](10)
    var sliced = buffer[5:5]
    assert_true(sliced.size == 0, "empty slice: size should be 0")


# ============================================
# Arithmetic Operations (Buffer + Buffer)
# ============================================
def test_add_buffers() raises:
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    var result = a + b

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 15, "add buffers: value mismatch")


def test_sub_buffers() raises:
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    var result = a - b

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 7, "sub buffers: value mismatch")


def test_mul_buffers() raises:
    var a = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    var result = a * b

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 12, "mul buffers: value mismatch")


def test_div_buffers() raises:
    var a = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var b = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    var result = a / b

    for i in range(MEDIUM_SIZE):
        assert_true(abs(result[i] - 5.0) < 0.001, "div buffers: value mismatch")


# ============================================
# In-place Arithmetic (Buffer += Buffer)
# ============================================
def test_iadd_buffers() raises:
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    a += b

    for i in range(MEDIUM_SIZE):
        assert_true(a[i] == 15, "iadd buffers: value mismatch")


def test_isub_buffers() raises:
    var a = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    a -= b

    for i in range(MEDIUM_SIZE):
        assert_true(a[i] == 7, "isub buffers: value mismatch")


def test_imul_buffers() raises:
    var a = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    a *= b

    for i in range(MEDIUM_SIZE):
        assert_true(a[i] == 12, "imul buffers: value mismatch")


def test_itruediv_buffers() raises:
    var a = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var b = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    a /= b

    for i in range(MEDIUM_SIZE):
        assert_true(abs(a[i] - 5.0) < 0.001, "itruediv buffers: value mismatch")


# ============================================
# Arithmetic Operations (Buffer + Scalar)
# ============================================
def test_add_scalar() raises:
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var result = buffer + 5

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 15, "add scalar: value mismatch")


def test_radd_scalar() raises:
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var result = 5 + buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 15, "radd scalar: value mismatch")


def test_sub_scalar() raises:
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    var result = buffer - 3

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 7, "sub scalar: value mismatch")


def test_rsub_scalar() raises:
    var buffer = Buffer[DType.int32].full(3, MEDIUM_SIZE)
    var result = 10 - buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 7, "rsub scalar: value mismatch")


def test_mul_scalar() raises:
    var buffer = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var result = buffer * 3

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 12, "mul scalar: value mismatch")


def test_rmul_scalar() raises:
    var buffer = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    var result = 3 * buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == 12, "rmul scalar: value mismatch")


def test_truediv_scalar() raises:
    var buffer = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    var result = buffer / 2.0

    for i in range(MEDIUM_SIZE):
        assert_true(
            abs(result[i] - 5.0) < 0.001, "truediv scalar: value mismatch"
        )


def test_rtruediv_scalar() raises:
    var buffer = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    var result = 10.0 / buffer

    for i in range(MEDIUM_SIZE):
        assert_true(
            abs(result[i] - 5.0) < 0.001, "rtruediv scalar: value mismatch"
        )


# ============================================
# In-place Scalar Operations
# ============================================
def test_iadd_scalar() raises:
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    buffer += 5

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 15, "iadd scalar: value mismatch")


def test_isub_scalar() raises:
    var buffer = Buffer[DType.int32].full(10, MEDIUM_SIZE)
    buffer -= 3

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 7, "isub scalar: value mismatch")


def test_imul_scalar() raises:
    var buffer = Buffer[DType.int32].full(4, MEDIUM_SIZE)
    buffer *= 3

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 12, "imul scalar: value mismatch")


def test_itruediv_scalar() raises:
    var buffer = Buffer[DType.float32].full(10.0, MEDIUM_SIZE)
    buffer /= 2.0

    for i in range(MEDIUM_SIZE):
        assert_true(
            abs(buffer[i] - 5.0) < 0.001, "itruediv scalar: value mismatch"
        )


# ============================================
# Unary Operations
# ============================================
def test_neg() raises:
    var buffer = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    var result = -buffer

    for i in range(MEDIUM_SIZE):
        assert_true(result[i] == -5, "neg: value mismatch")


def test_abs() raises:
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Int32(i - 30)  # Mix of negative and positive

    var result = abs(buffer)
    for i in range(MEDIUM_SIZE):
        assert_true(result[i] >= 0, "abs: should be non-negative")
        assert_true(result[i] == abs(Int32(i) - 30), "abs: value mismatch")


def test_pow() raises:
    var buffer = Buffer[DType.float32].full(2.0, MEDIUM_SIZE)
    var result = buffer**3.0

    for i in range(MEDIUM_SIZE):
        assert_true(abs(result[i] - 8.0) < 0.001, "pow: value mismatch")


def test_exp() raises:
    var buffer = Buffer[DType.float32].full(0.0, 10)
    var result = buffer.exp()

    for i in range(10):
        assert_true(abs(result[i] - 1.0) < 0.001, "exp(0) should be 1")


def test_log() raises:
    var buffer = Buffer[DType.float32].full(1.0, 10)
    var result = buffer.log()

    for i in range(10):
        assert_true(abs(result[i] - 0.0) < 0.001, "log(1) should be 0")


def test_log_buffer() raises:
    comptime dtype = DType.float32
    var a = Buffer[dtype].arange(3, MEDIUM_SIZE)
    var result = a.log(10, 50)

    assert_almost_equal(result[17], 3.40119738, "ln(30) should be 3.4011974")


def test_invert_bool() raises:
    var buffer = Buffer[DType.bool](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Scalar[DType.bool](i % 2 == 0)  # Alternating True/False

    var result = ~buffer
    for i in range(MEDIUM_SIZE):
        expected = i % 2 != 0
        assert_true(
            result[i] == expected, "invert: value mismatch at " + String(i)
        )


# ============================================
# Reduction Operations
# ============================================
def test_sum() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i + 1)  # 1 to 10

    var result = buffer.sum()
    assert_true(result == 55, "sum: expected 55, got " + String(result))


def test_sum_with_range() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i + 1)  # 1 to 10

    var result = buffer.sum(2, 5)  # Sum indices 2,3,4 -> 3+4+5 = 12
    assert_true(result == 12, "sum range: expected 12, got " + String(result))


def test_product() raises:
    var buffer = Buffer[DType.int32](5)
    for i in range(5):
        buffer[i] = Int32(i + 1)  # 1 to 5

    var result = buffer.product()
    assert_true(
        result == 120, "product: expected 120 (5!), got " + String(result)
    )


def test_product_with_range() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i + 1)

    var result = buffer.product(0, 3)  # 1*2*3 = 6
    assert_true(result == 6, "product range: expected 6, got " + String(result))


def test_dot() raises:
    var a = Buffer[DType.int32].full(2, 5)
    var b = Buffer[DType.int32].full(3, 5)

    var result = a.dot(b)
    assert_true(result == 30, "dot: expected 30 (2*3*5), got " + String(result))
    comptime dtype = DType.float32
    var A = Buffer[dtype].full(2, 42)
    var B = Buffer[dtype].full(2, 42)
    var scalar = A.dot(B)
    assert_true(
        scalar == 168, "dot: expected 168 (2*2*42), got " + String(scalar)
    )


# ============================================
# Comparison Operations (Scalar)
# ============================================
def test_eq_scalar_all_equal() raises:
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer == 42, "eq scalar: all equal should be True")
    assert_true(not (buffer == 0), "eq scalar: none equal should be False")


def test_ne_scalar() raises:
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer != 0, "ne scalar: all != should be True")
    assert_true(not (buffer != 42), "ne scalar: none != should be False")


def test_gt_scalar() raises:
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer > 99, "gt scalar: all > 99")
    assert_true(not (buffer > 100), "gt scalar: none > 100")


def test_ge_scalar() raises:
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer >= 100, "ge scalar: all >= 100")
    assert_true(not (buffer >= 101), "ge scalar: none >= 101")


def test_lt_scalar() raises:
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer < 51, "lt scalar: all < 51")
    assert_true(not (buffer < 50), "lt scalar: none < 50")


def test_le_scalar() raises:
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer <= 50, "le scalar: all <= 50")
    assert_true(not (buffer <= 49), "le scalar: none <= 49")


# ============================================
# Element-wise Comparison (Scalar) -> Buffer[bool]
# ============================================
def test_eq_full_scalar() raises:
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Int32(i % 10)

    var result = buffer.eq(5)
    for i in range(MEDIUM_SIZE):
        expected = (i % 10) == 5
        assert_true(result[i] == expected, "eq full: mismatch at " + String(i))


def test_ne_full_scalar() raises:
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Int32(i % 10)

    var result = buffer.ne(5)
    for i in range(MEDIUM_SIZE):
        expected = (i % 10) != 5
        assert_true(result[i] == expected, "ne full: mismatch at " + String(i))


def test_lt_full_scalar() raises:
    var buffer = Buffer[DType.int32](20)
    for i in range(20):
        buffer[i] = Int32(i)

    var result = buffer.lt(10)
    for i in range(20):
        expected = i < 10
        assert_true(result[i] == expected, "lt full: mismatch at " + String(i))


def test_gt_full_scalar() raises:
    var buffer = Buffer[DType.int32](20)
    for i in range(20):
        buffer[i] = Int32(i)

    var result = buffer.gt(10)
    for i in range(20):
        expected = i > 10
        assert_true(result[i] == expected, "gt full: mismatch at " + String(i))


# ============================================
# Comparison Operations (Buffer)
# ============================================
def test_eq_buffers() raises:
    var a = Buffer[DType.int32].full(42, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(a == b, "eq buffers: identical should be True")

    b[0] = 0
    assert_true(not (a == b), "eq buffers: one different should be False")


def test_ne_buffers() raises:
    var a = Buffer[DType.int32](MEDIUM_SIZE)
    var b = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = Int32(i)
        b[i] = Int32(i + 1)  # All different

    assert_true(a != b, "ne buffers: all different should be True")

    a[0] = b[0]
    assert_true(not (a != b), "ne buffers: one equal should be False")


def test_lt_buffers() raises:
    var a = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    var b = Buffer[DType.int32].full(10, MEDIUM_SIZE)

    assert_true(a < b, "lt buffers: all a < b")

    a[0] = 10
    assert_true(not (a < b), "lt buffers: one not < should be False")


# ============================================
# Element-wise Comparison (Buffer) -> Buffer[bool]
# ============================================
def test_eq_full_buffers() raises:
    var a = Buffer[DType.int32](MEDIUM_SIZE)
    var b = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = Int32(i)
        b[i] = Int32(i % 10)

    var result = a.eq(b)
    for i in range(MEDIUM_SIZE):
        expected = a[i] == b[i]
        assert_true(
            result[i] == expected, "eq full buffers: mismatch at " + String(i)
        )


def test_lt_full_buffers() raises:
    var a = Buffer[DType.int32](MEDIUM_SIZE)
    var b = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        a[i] = Int32(i)
        b[i] = 30

    var result = a.lt(b)
    for i in range(MEDIUM_SIZE):
        expected = a[i] < b[i]
        assert_true(
            result[i] == expected, "lt full buffers: mismatch at " + String(i)
        )


# ============================================
# Utility Methods
# ============================================
def test_fill() raises:
    var buffer = Buffer[DType.int32].zeros(MEDIUM_SIZE)
    buffer.fill(99, 10, 20)

    for i in range(MEDIUM_SIZE):
        if i >= 10 and i < 20:
            assert_true(buffer[i] == 99, "fill: should be 99 in range")
        else:
            assert_true(buffer[i] == 0, "fill: should be 0 outside range")


def test_zero() raises:
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)
    buffer.zero()

    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == 0, "zero: all should be 0")


def test_overwrite() raises:
    var buffer = Buffer[DType.int32].zeros(MEDIUM_SIZE)
    var source = Buffer[DType.int32].full(99, 10)
    buffer.overwrite(source, 10, 20)

    for i in range(MEDIUM_SIZE):
        if i >= 10 and i < 20:
            assert_true(buffer[i] == 99, "overwrite: should be 99 in range")
        else:
            assert_true(buffer[i] == 0, "overwrite: should be 0 outside range")


def test_count() raises:
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Int32(i % 10)

    var count_5 = buffer.count(5)
    # Values 0-67, with i % 10 == 5 at indices: 5, 15, 25, 35, 45, 55, 65 = 7 occurrences
    assert_true(count_5 == 7, "count: expected 7 fives, got " + String(count_5))

    var count_0 = buffer.count(0)
    # i % 10 == 0 at indices: 0, 10, 20, 30, 40, 50, 60 = 7 occurrences
    assert_true(count_0 == 7, "count: expected 7 zeros, got " + String(count_0))


def test_count_with_range() raises:
    var buffer = Buffer[DType.int32].full(5, MEDIUM_SIZE)
    buffer[10] = 99
    buffer[20] = 99
    buffer[30] = 99

    var count_all = buffer.count(99)
    assert_true(count_all == 3, "count all: expected 3")

    var count_range = buffer.count(99, 15, 35)
    assert_true(count_range == 2, "count range: expected 2 (at 20 and 30)")


# ============================================
# Type Conversion Tests
# ============================================
def test_to_dtype_int_to_float() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i * 10)

    var result = buffer.to_dtype[DType.float32]()
    for i in range(10):
        assert_true(
            abs(result[i] - Scalar[DType.float32](i * 10)) < 0.001,
            "to_dtype int->float: mismatch at " + String(i),
        )


def test_to_dtype_float_to_int() raises:
    var buffer = Buffer[DType.float32](10)
    for i in range(10):
        buffer[i] = Scalar[DType.float32](i) + 0.7

    var result = buffer.to_dtype[DType.int32]()
    for i in range(10):
        # Float to int truncates
        assert_true(
            result[i] == Int32(i),
            "to_dtype float->int: mismatch at " + String(i),
        )


def test_to_dtype_to_bool() raises:
    var buffer = Buffer[DType.int32](10)
    for i in range(10):
        buffer[i] = Int32(i)  # 0, 1, 2, ...

    var result = buffer.to_dtype[DType.bool]()
    assert_true(result[0] == False, "to_dtype to bool: 0 should be False")
    for i in range(1, 10):
        assert_true(
            result[i] == True, "to_dtype to bool: non-zero should be True"
        )


def test_to_dtype_from_bool() raises:
    var buffer = Buffer[DType.bool](10)
    for i in range(10):
        buffer[i] = i % 2 == 0  # Alternating

    var result = buffer.to_dtype[DType.int32]()
    for i in range(10):
        expected = 1 if i % 2 == 0 else 0
        assert_true(
            result[i] == Int32(expected),
            "to_dtype from bool: mismatch at " + String(i),
        )


def test_float_convenience() raises:
    var buffer = Buffer[DType.int32].full(42, 10)
    var result = buffer.float()

    assert_true(result[0] == 42.0, "float() convenience: value mismatch")


def test_float64_convenience() raises:
    var buffer = Buffer[DType.int32].full(42, 10)
    var result = buffer.float64()

    assert_true(result[0] == 42.0, "float64() convenience: value mismatch")


# ============================================
# Boolean Buffer Operations
# ============================================


def test_imul_bool_scalar() raises:
    var buffer = Buffer[DType.bool](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = True

    buffer *= False
    for i in range(MEDIUM_SIZE):
        assert_true(buffer[i] == False, "imul bool scalar: all should be False")


# ============================================
# Edge Cases and SIMD Boundary Tests
# ============================================
def test_simd_boundary_sizes() raises:
    comptime sizes: List[Int] = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]

    comptime for idx in range(len(sizes)):
        comptime size = sizes[idx]
        var a = Buffer[DType.int32].full(10, size)
        var b = Buffer[DType.int32].full(5, size)

        var sum_result = a + b
        var mul_result = a * b

        for i in range(size):
            assert_true(
                sum_result[i] == 15, "SIMD boundary add: size " + String(size)
            )
            assert_true(
                mul_result[i] == 50, "SIMD boundary mul: size " + String(size)
            )


def test_arithmetic_ops_with_range() raises:
    var a = Buffer[DType.int32].full(10, 20)
    var b = Buffer[DType.int32].full(3, 10)

    # Test arithmetic_ops with ranges
    var result = a.arithmetic_ops[Add, True](b, 5, 15, 0, 10)
    assert_true(result.size == 10, "arithmetic_ops range: size mismatch")
    for i in range(10):
        assert_true(result[i] == 13, "arithmetic_ops range: value mismatch")


def test_inplace_ops_with_range() raises:
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


def test_inplace_ops_scalar_with_range() raises:
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


# ============================================
# Different DType Tests
# ============================================
def test_operations_uint8() raises:
    var a = Buffer[DType.uint8].full(100, MEDIUM_SIZE)
    var b = Buffer[DType.uint8].full(50, MEDIUM_SIZE)

    var sum_result = a + b
    # var mul_result = a * b

    assert_true(sum_result[0] == 150, "uint8 add")
    # Note: 100 * 50 = 5000, but uint8 max is 255, so overflow


def test_operations_int16() raises:
    var a = Buffer[DType.int16].full(1000, MEDIUM_SIZE)
    var b = Buffer[DType.int16].full(500, MEDIUM_SIZE)

    var sum_result = a + b
    assert_true(sum_result[0] == 1500, "int16 add")


def test_operations_float64() raises:
    var a = Buffer[DType.float64].full(1.5, MEDIUM_SIZE)
    var b = Buffer[DType.float64].full(2.5, MEDIUM_SIZE)

    var sum_result = a + b
    var mul_result = a * b

    assert_true(abs(sum_result[0] - 4.0) < 0.001, "float64 add")
    assert_true(abs(mul_result[0] - 3.75) < 0.001, "float64 mul")


# ============================================
# __eq__ Tests - new tests above
# ============================================
def test_dunder_eq_all_equal_int32() raises:
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer == 42, "All elements are 42, should return True")
    assert_true(not (buffer == 0), "No elements are 0, should return False")


def test_dunder_eq_one_different() raises:
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


def test_dunder_eq_small_buffer() raises:
    var buffer = Buffer[DType.int32].full(5, SMALL_SIZE)
    assert_true(buffer == 5, "Small buffer: all equal")

    buffer[SMALL_SIZE - 1] = 6
    assert_true(not (buffer == 5), "Small buffer: one different")


# ============================================
# __ne__ Tests
# ============================================
def test_dunder_ne_all_different() raises:
    var buffer = Buffer[DType.int32].full(42, MEDIUM_SIZE)

    assert_true(buffer != 0, "All != 0, should return True")
    assert_true(not (buffer != 42), "None != 42, should return False")


def test_dunder_ne_one_equal() raises:
    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[MEDIUM_SIZE // 2] = 50

    assert_true(
        not (buffer != 50), "One element equals 50, should return False"
    )
    assert_true(
        not (buffer != 100), "Most elements equal 100, should return False"
    )


# ============================================
# __gt__ Tests
# ============================================
def test_dunder_gt_all_greater() raises:
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer > 99, "All > 99")
    assert_true(not (buffer > 100), "None > 100 (equal)")
    assert_true(not (buffer > 101), "None > 101")


def test_dunder_gt_one_not_greater() raises:
    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[0] = 50

    assert_true(not (buffer > 99), "First element 50 not > 99")


def test_dunder_gt_negative() raises:
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Int32(i) - 30

    assert_true(buffer > -31, "All > -31")
    assert_true(not (buffer > -30), "First element is -30, not > -30")


# ============================================
# __ge__ Tests
# ============================================
def test_dunder_ge_all_greater_equal() raises:
    var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)

    assert_true(buffer >= 100, "All >= 100 (equal)")
    assert_true(buffer >= 50, "All >= 50")
    assert_true(not (buffer >= 101), "None >= 101")


def test_dunder_ge_one_less() raises:
    var buffer = Buffer[DType.int64].full(100, MEDIUM_SIZE)
    buffer[MEDIUM_SIZE - 1] = 99

    assert_true(not (buffer >= 100), "Last element 99 not >= 100")


# ============================================
# __lt__ Tests
# ============================================
def test_dunder_lt_all_less() raises:
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer < 51, "All < 51")
    assert_true(not (buffer < 50), "None < 50 (equal)")
    assert_true(not (buffer < 49), "None < 49")


def test_dunder_lt_one_not_less() raises:
    var buffer = Buffer[DType.int64].full(50, MEDIUM_SIZE)
    buffer[0] = 100

    assert_true(not (buffer < 51), "First element 100 not < 51")


def test_dunder_lt_float32() raises:
    var buffer = Buffer[DType.float32](20)
    for i in range(20):
        buffer[i] = Scalar[DType.float32](i)

    assert_true(buffer < 20.0, "All < 20.0")
    assert_true(not (buffer < 19.0), "Last element 19.0 not < 19.0")


# ============================================
# __le__ Tests
# ============================================
def test_dunder_le_all_less_equal() raises:
    var buffer = Buffer[DType.int32].full(50, MEDIUM_SIZE)

    assert_true(buffer <= 50, "All <= 50 (equal)")
    assert_true(buffer <= 100, "All <= 100")
    assert_true(not (buffer <= 49), "None <= 49")


def test_dunder_le_one_greater() raises:
    var buffer = Buffer[DType.int64].full(50, MEDIUM_SIZE)
    buffer[0] = 51

    assert_true(not (buffer <= 50), "First element 51 not <= 50")


# ============================================
# Edge Cases
# ============================================
def test_dunder_compare_single_element() raises:
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 42

    assert_true(buffer == 42, "Single: == 42")
    assert_true(buffer != 0, "Single: != 0")
    assert_true(buffer > 41, "Single: > 41")
    assert_true(buffer >= 42, "Single: >= 42")
    assert_true(buffer < 43, "Single: < 43")
    assert_true(buffer <= 42, "Single: <= 42")


def test_dunder_compare_empty() raises:
    var buffer = Buffer[DType.int32]()

    assert_true(not (buffer == 0), "Empty: == returns False")
    assert_true(not (buffer != 0), "Empty: != returns False")
    assert_true(not (buffer > 0), "Empty: > returns False")
    assert_true(not (buffer >= 0), "Empty: >= returns False")
    assert_true(not (buffer < 0), "Empty: < returns False")
    assert_true(not (buffer <= 0), "Empty: <= returns False")


def test_dunder_compare_simd_boundary() raises:
    # Test sizes around SIMD width (8 for int32)
    comptime sizes: List[Int] = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33]

    comptime for idx in range(len(sizes)):
        comptime size = sizes[idx]
        var buffer = Buffer[DType.int32].full(10, size)

        assert_true(buffer == 10, "Size " + String(size) + ": ==")
        assert_true(buffer >= 10, "Size " + String(size) + ": >=")
        assert_true(buffer <= 10, "Size " + String(size) + ": <=")

        buffer[size - 1] = 11
        assert_true(
            not (buffer == 10), "Size " + String(size) + ": != after change"
        )
        assert_true(
            not (buffer <= 10),
            "Size " + String(size) + ": not <= after change",
        )


def test_dunder_compare_position_sensitivity() raises:
    comptime positions: List[Int] = [0, 7, 8, 9, 16, 33, 64, 67]

    comptime for idx in range(len(positions)):
        comptime pos = positions[idx]
        if pos < MEDIUM_SIZE:
            var buffer = Buffer[DType.int32].full(100, MEDIUM_SIZE)
            buffer[pos] = 99

            assert_true(
                not (buffer >= 100), "Position " + String(pos) + ": >= fails"
            )
            assert_true(
                not (buffer == 100), "Position " + String(pos) + ": == fails"
            )


def test_dunder_compare_uint8() raises:
    var buffer = Buffer[DType.uint8].full(100, MEDIUM_SIZE)

    assert_true(buffer == 100, "uint8: ==")
    assert_true(buffer > 99, "uint8: >")
    assert_true(buffer < 101, "uint8: <")


def test_dunder_compare_float64() raises:
    var buffer = Buffer[DType.float64].full(3.14159, MEDIUM_SIZE)

    assert_true(buffer == 3.14159, "float64: ==")
    assert_true(buffer > 3.14, "float64: >")
    assert_true(buffer < 3.15, "float64: <")


# ============================================
# EQ Tests
# ============================================
def test_eq_int32() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i) % 10

    var result = buffer.eq(5)

    # Check indices where i % 10 == 5: 5, 15, 25, 35, 45, 55, 65
    for i in range(size):
        expected = (i % 10) == 5
        assert_true(
            result[i] == expected,
            "test_eq_int32: index "
            + String(i)
            + " expected "
            + String(expected),
        )


def test_eq_float64() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float64](i % 5)

    var result = buffer.eq(3.0)

    for i in range(size):
        expected = (i % 5) == 3
        assert_true(
            result[i] == expected,
            "test_eq_float64: index " + String(i) + " failed",
        )


def test_eq_all_same() raises:
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


def test_eq_small_buffer() raises:
    size = SMALL_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i)

    var result = buffer.eq(3)

    for i in range(size):
        expected = i == 3
        assert_true(
            result[i] == expected,
            "test_eq_small_buffer: index " + String(i) + " failed",
        )


# ============================================
# NE Tests
# ============================================
def test_ne_int32() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i) % 10

    var result = buffer.ne(5)

    for i in range(size):
        expected = (i % 10) != 5
        assert_true(
            result[i] == expected,
            "test_ne_int32: index " + String(i) + " failed",
        )


def test_ne_float32() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float32](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float32](i)

    var result = buffer.ne(10.0)

    for i in range(size):
        expected = i != 10
        assert_true(
            result[i] == expected,
            "test_ne_float32: index " + String(i) + " failed",
        )


def test_ne_all_different() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(100, size)

    var result = buffer.ne(0)

    for i in range(size):
        assert_true(
            result[i] == True, "test_ne_all_different: all should be True"
        )


# ============================================
# GT Tests
# ============================================
def test_gt_int64() raises:
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
            "test_gt_int64: index " + String(i) + " failed",
        )


def test_gt_float32() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float32](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float32](i) * 0.5

    var result = buffer.gt(15.0)

    for i in range(size):
        expected = (Scalar[DType.float32](i) * 0.5) > 15.0
        assert_true(
            result[i] == expected,
            "test_gt_float32: index " + String(i) + " failed",
        )


def test_gt_negative_values() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i) - 30  # Values from -30 to 37

    var result = buffer.gt(0)

    for i in range(size):
        expected = (i - 30) > 0
        assert_true(
            result[i] == expected,
            "test_gt_negative_values: index " + String(i) + " failed",
        )


def test_gt_boundary() raises:
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


# ============================================
# GE Tests
# ============================================
def test_ge_int32() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i)

    var result = buffer.ge(50)

    for i in range(size):
        expected = i >= 50
        assert_true(
            result[i] == expected,
            "test_ge_int32: index " + String(i) + " failed",
        )


def test_ge_float64() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float64](i)

    var result = buffer.ge(33.0)

    for i in range(size):
        expected = i >= 33
        assert_true(
            result[i] == expected,
            "test_ge_float64: index " + String(i) + " failed",
        )


def test_ge_equal_value() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(100, size)

    var result = buffer.ge(100)

    for i in range(size):
        assert_true(
            result[i] == True, "test_ge_equal_value: all should be >= 100"
        )


# ============================================
# LT Tests
# ============================================
def test_lt_int64() raises:
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


def test_lt_float32() raises:
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
            "test_lt_float32: index " + String(i) + " failed",
        )


def test_lt_all_less() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32].full(0, size)

    var result = buffer.lt(100)

    for i in range(size):
        assert_true(result[i] == True, "test_lt_all_less: all should be < 100")


def test_lt_none_less() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32].full(100, size)

    var result = buffer.lt(50)

    for i in range(size):
        assert_true(
            result[i] == False, "test_lt_none_less: none should be < 50"
        )


# ============================================
# LE Tests
# ============================================
def test_le_int32() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i)

    var result = buffer.le(50)

    for i in range(size):
        expected = i <= 50
        assert_true(
            result[i] == expected,
            "test_le_int32: index " + String(i) + " failed",
        )


def test_le_float64() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float64](i)

    var result = buffer.le(33.0)

    for i in range(size):
        expected = i <= 33
        assert_true(
            result[i] == expected,
            "test_le_float64: index " + String(i) + " failed",
        )


def test_le_equal_value() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int64].full(100, size)

    var result = buffer.le(100)

    for i in range(size):
        assert_true(
            result[i] == True, "test_le_equal_value: all should be <= 100"
        )


# ============================================
# Edge Case Tests
# ============================================
def test_compare_single_element() raises:
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 5

    assert_true(buffer.eq(5)[0] == True, "Single element eq failed")
    assert_true(buffer.ne(5)[0] == False, "Single element ne failed")
    assert_true(buffer.gt(4)[0] == True, "Single element gt failed")
    assert_true(buffer.ge(5)[0] == True, "Single element ge failed")
    assert_true(buffer.lt(6)[0] == True, "Single element lt failed")
    assert_true(buffer.le(5)[0] == True, "Single element le failed")


def test_compare_simd_boundary() raises:
    # Test sizes around typical SIMD widths
    comptime sizes: List[Int] = [
        1,
        2,
        4,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
    ]

    comptime for s_idx in range(len(sizes)):
        comptime size = sizes[s_idx]
        var buffer = Buffer[DType.int32](size)
        for i in range(size):
            buffer[i] = Int32(i)

        var result = buffer.lt(Int32(size) // 2)

        for i in range(size):
            expected = i < (size // 2)
            assert_true(
                result[i] == expected,
                "test_compare_simd_boundary: size "
                + String(size)
                + " index "
                + String(i)
                + " failed",
            )


def test_compare_large_buffer() raises:
    size = LARGE_SIZE
    var buffer = Buffer[DType.float32](size)
    for i in range(size):
        buffer[i] = Scalar[DType.float32](i)

    var result = buffer.ge(500.0)

    for i in range(size):
        expected = i >= 500
        assert_true(
            result[i] == expected,
            "test_compare_large_buffer: index " + String(i) + " failed",
        )


def test_compare_uint8() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.uint8](size)
    for i in range(size):
        buffer[i] = Scalar[DType.uint8](i % 256)

    var result = buffer.gt(100)

    for i in range(size):
        expected = (i % 256) > 100
        assert_true(
            result[i] == expected,
            "test_compare_uint8: index " + String(i) + " failed",
        )


def test_compare_int8() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int8](size)
    for i in range(size):
        buffer[i] = Scalar[DType.int8](i - 30)  # -30 to 37

    var result = buffer.ge(0)

    for i in range(size):
        expected = (i - 30) >= 0
        assert_true(
            result[i] == expected,
            "test_compare_int8: index " + String(i) + " failed",
        )


def test_compare_int16() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.int16](size)
    for i in range(size):
        buffer[i] = Scalar[DType.int16](i * 100)

    var result = buffer.lt(3000)

    for i in range(size):
        expected = (i * 100) < 3000
        assert_true(
            result[i] == expected,
            "test_compare_int16: index " + String(i) + " failed",
        )


def test_compare_uint64() raises:
    size = MEDIUM_SIZE
    var buffer = Buffer[DType.uint64](size)
    for i in range(size):
        buffer[i] = Scalar[DType.uint64](i * 1000)

    var result = buffer.le(50000)

    for i in range(size):
        expected = (i * 1000) <= 50000
        assert_true(
            result[i] == expected,
            "test_compare_uint64: index " + String(i) + " failed",
        )


# ====================================="
def test_lt() raises:
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


def test_lt_breaks_original() raises:
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


def test_simd_indexing() raises:
    comptime simd_width = 8

    # Create a SIMD vector with distinct values
    var vec = SIMD[DType.int32, simd_width](0, 1, 2, 3, 4, 5, 6, 7)

    for i in range(simd_width):
        assert_true(vec[i] == Int32(i), "SIMD vector indexing failed")

    # This will segfault - commenting out for safety
    # var should_crash = vec[8]


def test_what_values() raises:
    comptime simd_width = 8

    var buffer = Buffer[DType.float32](16)
    for i in range(16):
        buffer[i] = Float32(i)

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
                    + String(k)
                    + "] (value "
                    + String((8 + k))
                    + ") should be >= 10",
                )


def test_to_dtype_same_type() raises:
    var buffer = Buffer[DType.int32](5)
    for i in range(5):
        buffer[i] = Int32(i * 10)

    var result = buffer.to_dtype[DType.int32]()

    for i in range(5):
        assert_true(
            result[i] == buffer[i],
            "Same type conversion should preserve values",
        )


def test_to_dtype_non_bool() raises:
    var buffer = Buffer[DType.int32](5)
    for i in range(5):
        buffer[i] = Int32(i * 10)

    var result = buffer.to_dtype[DType.float32]()

    for i in range(5):
        assert_true(
            result[i] == Scalar[DType.float32](i * 10),
            "int32 to float32 conversion failed",
        )


def test_to_dtype_to_bool_orig() raises:
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


def test_to_dtype_from_bool_orig() raises:
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


def test_to_dtype_large_buffer() raises:
    size = 100
    var buffer = Buffer[DType.float64](size)
    for i in range(size):
        buffer[i] = Float64(i) * 0.5

    var result = buffer.to_dtype[DType.int32]()

    for i in range(size):
        expected = Int32(Float64(i) * 0.5)
        assert_true(
            result[i] == expected,
            "Large buffer conversion failed at index " + String(i),
        )


def test_count_orig() raises:
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


def test_log_orig() raises:
    ll = List[Scalar[DType.float32]](capacity=100)
    for i in range(1, 100):
        ll.insert(0, Scalar[DType.float32](i))
    buf = Buffer[DType.float32](ll)
    logs = buf.log()

    assert_true(logs[len(logs) - 1] == 0, "Buffer log zero assertion failed")
    assert_true(
        logs[0] == 4.59512, "Buffer log assertion failed for value at index 0"
    )


def test_buffer_iter() raises:
    comptime dtype = DType.float32
    buff = Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sliced = buff[4:1:-1]
    var expect = 5
    for elem in sliced:
        assert_true(elem == Float32(expect), "Buffer iter assertion failed")
        expect -= 1


def test_buffer_slice() raises:
    comptime dtype = DType.float32
    buff = Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sliced = buff[4:1:-2]
    assert_true(
        (sliced == Buffer[dtype]([5, 3])),
        "Buffer slicing assertion failed",
    )


def _buffer_buffer_mul() raises:
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


def test_buffer_buffer_add() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(43.0)
    b = Buffer[dtype](72)
    b.fill(43.0)
    expected = Buffer[dtype](72)
    expected.fill(86)
    added = a + b
    result = added == expected
    assert_true(
        result,
        "Buffer buffer add assertion failed",
    )


def test_buffer_scalar_float_greater_than_eq() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(43.0)
    result1 = a >= 42
    a.fill(42.0)
    result2 = a >= 42
    assert_true(
        result1 and result2,
        "Buffer scalar float greater than eq assertion failed",
    )


def test_buffer_scalar_float_less_than_eq() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    result1 = a <= 43
    result2 = a <= 42
    assert_true(
        result1 and result2,
        "Buffer scalar float less than eq assertion failed",
    )


def test_buffer_scalar_float_greater_than() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    result = a > 41
    assert_true(result, "Buffer scalar float greater than assertion failed")


def test_buffer_scalar_float_less_than() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    result = a < 43
    assert_true(result, "Buffer scalar float less than assertion failed")


def test_buffer_scalar_float_inequality() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    result = a != 43
    assert_true(result, "Buffer scalar float inequality assertion failed")


def test_buffer_scalar_float_equality() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    result = a == 42
    assert_true(result, "Buffer scalar float equality assertion failed")


def test_buffer_dot() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](33)
    a.fill(42.0)
    b = Buffer[dtype](33)
    b.fill(2.0)
    assert_true(
        a.dot(b) == b.dot(a) and a.dot(b) == 2772, "dot assertion failed"
    )


def test_buffer_prod() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](2)
    a.fill(42.0)
    result = a.product()
    assert_true(result == 1764, "prod assertion failed")


def test_buffer_sum() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    result = a.sum()
    assert_true(result == 3024, "Sum assertion failed")


def test_buffer_float_greater_than_eq() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    b = Buffer[dtype](72)
    b.fill(420)
    result = b >= a
    assert_true(result, "72 float greater than eq assertion failed")

    a = Buffer[dtype](31)
    a.fill(42.0)
    b = Buffer[dtype](31)
    b.fill(42)
    result = b >= a
    assert_true(result, "31 float greater than eq assertion failed")


def test_buffer_float_greater_than() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    b = Buffer[dtype](72)
    b.fill(420)
    result = b > a
    assert_true(result, "72 float greater than assertion failed")


def test_buffer_float_less_eq_than() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    b = Buffer[dtype](72)
    b.fill(420)
    result = a <= b
    assert_true(result, "72 float less than eq assertion failed")

    a = Buffer[dtype](65)
    a.fill(42.0)
    b = Buffer[dtype](65)
    b.fill(42)
    result = a <= b
    assert_true(result, "65 float less than eq assertion failed")


def test_buffer_float_less_than() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    b = Buffer[dtype](72)
    b.fill(420)
    result = a < b
    assert_true(result, "72 float less than assertion failed")


def test_buffer_float_equality() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    b = Buffer[dtype](72)
    b.fill(42)
    result = a == b
    assert_true(result, "72 float equality assertion failed")

    a = Buffer[dtype](1)
    a.fill(42.0)
    b = Buffer[dtype](1)
    b.fill(42)
    result = a == b
    assert_true(result, "1 float equality assertion failed")

    a = Buffer[dtype](1024)
    a.fill(42.0)
    b = Buffer[dtype](1024)
    b.fill(42)
    result = a == b
    assert_true(result, "1024 float equality assertion failed")


def test_buffer_float_inequality() raises:
    comptime dtype = DType.float32
    a = Buffer[dtype](72)
    a.fill(42.0)
    b = Buffer[dtype](72)
    b.fill(420)
    result = a != b
    assert_true(result, "72 float inequality assertion failed")

    a = Buffer[dtype](1)
    a.fill(42.0)
    b = Buffer[dtype](1)
    b.fill(420)
    result = a != b
    assert_true(result, "1 float inequality assertion failed")

    a = Buffer[dtype](1024)
    a.fill(42.0)
    b = Buffer[dtype](1024)
    b.fill(420)
    result = a != b
    assert_true(result, "1024 float inequality assertion failed")


def test_fill_segment() raises:
    comptime dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=Int(size))
    for i in range(size):
        l.append(Int32(i))

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


def test_overwrite_orig() raises:
    comptime dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=Int(size))
    for i in range(size):
        l.append(Int32(i))

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


# ============================================================================
# Buffer Count Tests
# ============================================================================


def test_count_basic_cnt() raises:
    """Test basic counting functionality."""
    var buffer = Buffer[DType.int32](68)
    for i in range(68):
        buffer[i] = Int32(i % 10)

    var count_5 = buffer.count(5)
    # Values 0-67, with i % 10 == 5 at indices: 5, 15, 25, 35, 45, 55, 65 = 7 occurrences
    assert_true(count_5 == 7)

    var count_0 = buffer.count(0)
    # i % 10 == 0 at indices: 0, 10, 20, 30, 40, 50, 60 = 7 occurrences
    assert_true(count_0 == 7)

    var count_9 = buffer.count(9)
    # i % 10 == 9 at indices: 9, 19, 29, 39, 49, 59 = 6 occurrences
    assert_true(count_9 == 6)


def test_count_with_range_cnt() raises:
    """Test counting within a specified range."""
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


def test_count_empty_buffer_cnt() raises:
    """Test counting on empty buffer."""
    var buffer = Buffer[DType.int32](0)
    var count = buffer.count(42)
    assert_true(count == 0)


def test_count_single_element_cnt() raises:
    """Test counting in single-element buffer."""
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 42

    var count_match = buffer.count(42)
    assert_true(count_match == 1)

    var count_no_match = buffer.count(99)
    assert_true(count_no_match == 0)


def test_count_all_match_cnt() raises:
    """Test counting when all elements match."""
    var buffer = Buffer[DType.int32].full(7, 100)
    var count = buffer.count(7)
    assert_true(count == 100)


def test_count_none_match_cnt() raises:
    """Test counting when no elements match."""
    var buffer = Buffer[DType.int32].full(5, 100)
    var count = buffer.count(999)
    assert_true(count == 0)


def test_count_alternating_pattern_cnt() raises:
    """Test counting with alternating pattern."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(1 if i % 2 == 0 else 0)

    var count_ones = buffer.count(1)
    assert_true(count_ones == 50)

    var count_zeros = buffer.count(0)
    assert_true(count_zeros == 50)


def test_count_simd_boundary_cnt() raises:
    """Test counting across SIMD boundaries."""
    comptime simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 3 + 5  # Not aligned to SIMD width
    var buffer = Buffer[DType.int32](size)

    # Fill with pattern
    for i in range(size):
        buffer[i] = Int32(42 if i % 7 == 0 else 0)

    var count = buffer.count(42)
    # Count manually
    var expected = 0
    for i in range(size):
        if i % 7 == 0:
            expected += 1
    assert_true(count == expected)


def test_count_range_partial_simd_cnt() raises:
    """Test counting with range that doesn't align to SIMD width."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    # Set specific values
    buffer[10] = 999
    buffer[15] = 999
    buffer[20] = 999
    buffer[25] = 999

    # Count in range [12, 23) - should find at 15 and 20
    var count = buffer.count(999, 12, 23)
    assert_true(count == 2)


def test_count_float32_cnt() raises:
    """Test counting with float32 dtype."""
    var buffer = Buffer[DType.float64](50)
    for i in range(50):
        buffer[i] = 3.14 if i % 5 == 0 else 2.71

    var count_pi = buffer.count(3.14)
    assert_true(count_pi == 10)  # Indices: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45

    var count_e = buffer.count(2.71)
    assert_true(count_e == 40)


def test_count_float64_cnt() raises:
    """Test counting with float64 dtype."""
    var buffer = Buffer[DType.float64](64)
    for i in range(64):
        buffer[i] = 1.5 if i < 32 else 2.5

    var count_1_5 = buffer.count(1.5)
    assert_true(count_1_5 == 32)

    var count_2_5 = buffer.count(2.5)
    assert_true(count_2_5 == 32)


def test_count_int8_cnt() raises:
    """Test counting with int8 dtype."""
    var buffer = Buffer[DType.int8](100)
    for i in range(100):
        buffer[i] = Int8(i % 128)

    var count_42 = buffer.count(Int8(42))
    assert_true(count_42 == 1)  # Only at index 42

    var count_0 = buffer.count(Int8(0))
    assert_true(count_0 == 1)  # Only at index 0


def test_count_int64_cnt() raises:
    """Test counting with int64 dtype."""
    var buffer = Buffer[DType.int64](80)
    for i in range(80):
        buffer[i] = Int64(1000000 if i % 10 == 0 else i)

    var count = buffer.count(1000000)
    assert_true(count == 8)  # At indices: 0, 10, 20, 30, 40, 50, 60, 70


def test_count_bool_true_cnt() raises:
    """Test counting True in boolean buffer."""
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


def test_count_bool_false_cnt() raises:
    """Test counting False in boolean buffer."""
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = i % 2 == 0

    var count_false = buffer.count(False)
    assert_true(count_false == 50)


def test_count_bool_all_true_cnt() raises:
    """Test counting in all-True boolean buffer."""
    var buffer = Buffer[DType.bool].full(True, 100)
    var count = buffer.count(True)
    assert_true(count == 100)

    var count_false = buffer.count(False)
    assert_true(count_false == 0)


def test_count_bool_all_false_cnt() raises:
    """Test counting in all-False boolean buffer."""
    var buffer = Buffer[DType.bool].full(False, 100)
    var count = buffer.count(False)
    assert_true(count == 100)

    var count_true = buffer.count(True)
    assert_true(count_true == 0)


def test_count_bool_range_cnt() raises:
    """Test counting with range on boolean buffer."""
    var buffer = Buffer[DType.bool](50)
    for i in range(50):
        buffer[i] = i < 25

    var count_all = buffer.count(True)
    assert_true(count_all == 25)

    var count_range = buffer.count(True, 10, 30)
    assert_true(count_range == 15)  # Indices 10-24


def test_count_large_buffer_cnt_777_with_default() raises:
    """Test counting in large buffer."""
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = Int32(777 if i % 100 == 0 else -1)  # Use -1 as default

    var count = buffer.count(777)
    assert_true(count == 100)  # Exactly 100 multiples of 100


def test_count_large_buffer_cnt_777() raises:
    """Test counting in large buffer."""
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = Int32(777 if i % 100 == 0 else i)

    var count = buffer.count(777)
    assert_true(count == 101)


def test_count_large_buffer_cnt() raises:
    """Test counting in large buffer."""
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = 99999 if i % 100 == 0 else Int32(
            i
        )  # Use value outside range

    var count = buffer.count(99999)
    assert_true(count == 100)  # Exactly 100 multiples of 100


def test_count_negative_values_cnt() raises:
    """Test counting negative values."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = -5 if i % 10 == 0 else Int32(i)

    var count = buffer.count(-5)
    assert_true(count == 10)


def test_count_zero_cnt() raises:
    """Test counting zeros."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(0 if i % 7 == 0 else 1)

    var count_zero = buffer.count(0)
    # Indices: 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98 = 15
    var expected = 0
    for i in range(100):
        if i % 7 == 0:
            expected += 1
    assert_true(count_zero == expected)


def test_count_start_equals_end_cnt() raises:
    """Test counting with start_index == end_index."""
    var buffer = Buffer[DType.int32].full(42, 100)
    var count = buffer.count(42, 50, 50)
    assert_true(count == 0)


def test_count_invalid_range_cnt() raises:
    """Test counting with start > end."""
    var buffer = Buffer[DType.int32].full(42, 100)
    var count = buffer.count(42, 60, 50)
    assert_true(count == 0)


def test_count_uint8_cnt() raises:
    """Test counting with uint8 dtype."""
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


def test_inplace_multiply_scalar_ips() raises:
    """Test inplace multiplication by scalar."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](2)

    for i in range(100):
        assert_true(buffer[i] == Int32(i) * 2)


def test_inplace_add_scalar_ips() raises:
    """Test inplace addition of scalar."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Add](10)

    for i in range(100):
        assert_true(buffer[i] == Int32(i) + 10)


def test_inplace_subtract_scalar_ips() raises:
    """Test inplace subtraction of scalar."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i + 50)

    buffer.inplace_ops_scalar[Subtract](50)

    for i in range(100):
        assert_true(buffer[i] == Int32(i))


def test_inplace_divide_scalar_ips() raises:
    """Test inplace division by scalar."""
    var buffer = Buffer[DType.float32](100)
    for i in range(100):
        buffer[i] = Float32(i * 4)

    buffer.inplace_ops_scalar[Divide](2.0)

    for i in range(100):
        assert_true(abs(buffer[i] - Float32(i * 2)) < 0.0001)


def test_inplace_multiply_with_range_ips() raises:
    """Test inplace multiplication with range."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](3, 20, 40)

    for i in range(100):
        if i >= 20 and i < 40:
            assert_true(buffer[i] == Int32(i) * 3)
        else:
            assert_true(buffer[i] == Int32(i))


def test_inplace_add_with_range_ips() raises:
    """Test inplace addition with range."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Add](100, 50, 60)

    for i in range(100):
        if i >= 50 and i < 60:
            assert_true(buffer[i] == Int32(i + 100))
        else:
            assert_true(buffer[i] == Int32(i))


def test_inplace_empty_buffer_ips() raises:
    """Test inplace operations on empty buffer."""
    var buffer = Buffer[DType.int32](0)
    buffer.inplace_ops_scalar[Multiply](5)
    # Should not crash
    assert_true(buffer.size == 0)


def test_inplace_multiply_by_zero_ips() raises:
    """Test inplace multiplication by zero."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i + 1)

    buffer.inplace_ops_scalar[Multiply](0)

    for i in range(50):
        assert_true(buffer[i] == 0)


def test_inplace_multiply_by_one_ips() raises:
    """Test inplace multiplication by one (identity)."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](1)

    for i in range(50):
        assert_true(buffer[i] == Int32(i))


def test_inplace_add_zero_ips() raises:
    """Test inplace addition of zero (identity)."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Add](0)

    for i in range(50):
        assert_true(buffer[i] == Int32(i))


def test_inplace_subtract_zero_ips() raises:
    """Test inplace subtraction of zero (identity)."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Subtract](0)

    for i in range(50):
        assert_true(buffer[i] == Int32(i))


def test_inplace_divide_by_one_ips() raises:
    """Test inplace division by one (identity)."""
    var buffer = Buffer[DType.float32](50)
    for i in range(50):
        buffer[i] = Float32(i)

    buffer.inplace_ops_scalar[Divide](1.0)

    for i in range(50):
        assert_true(abs(buffer[i] - Float32(i)) < 0.0001)


def test_inplace_multiply_negative_ips() raises:
    """Test inplace multiplication by negative scalar."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](-1)

    for i in range(50):
        assert_true(buffer[i] == Int32(-i))


def test_inplace_add_negative_ips() raises:
    """Test inplace addition of negative scalar."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i + 100)

    buffer.inplace_ops_scalar[Add](-50)

    for i in range(50):
        assert_true(buffer[i] == Int32(i + 50))


def test_inplace_float32_precision_ips() raises:
    """Test inplace operations on float32 with precision check."""
    var buffer = Buffer[DType.float32](100)
    for i in range(100):
        buffer[i] = Float32(i) * 0.1

    buffer.inplace_ops_scalar[Multiply](10.0)

    for i in range(100):
        assert_true(abs(buffer[i] - Float32(i)) < 0.01)


def test_inplace_float64_precision_ips() raises:
    """Test inplace operations on float64 with precision check."""
    var buffer = Buffer[DType.float64](100)
    for i in range(100):
        buffer[i] = Float64(i) * 0.1

    buffer.inplace_ops_scalar[Multiply](10.0)

    for i in range(100):
        assert_true(abs(buffer[i] - Float64(i)) < 0.0001)


def test_inplace_simd_boundary_ips() raises:
    """Test inplace operations across SIMD boundaries."""
    comptime simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 5 + 7  # Not aligned to SIMD width
    var buffer = Buffer[DType.int32](size)

    for i in range(size):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](2)

    for i in range(size):
        assert_true(buffer[i] == Int32(i) * 2)


def test_inplace_range_partial_simd_ips() raises:
    """Test inplace operations with range not aligned to SIMD."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Add](1000, 13, 47)

    for i in range(100):
        if i >= 13 and i < 47:
            assert_true(buffer[i] == Int32(i) + 1000)
        else:
            assert_true(buffer[i] == Int32(i))


def test_inplace_single_element_ips() raises:
    """Test inplace operations on single element."""
    var buffer = Buffer[DType.int32](1)
    buffer[0] = 42

    buffer.inplace_ops_scalar[Multiply](3)
    assert_true(buffer[0] == 126)


def test_inplace_bool_multiply_ips() raises:
    """Test inplace multiplication on boolean buffer."""
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = True if i % 2 == 0 else False

    buffer.inplace_ops_scalar[Multiply](False)

    # All should be False after multiplying by False
    for i in range(100):
        assert_true(buffer[i] == False)


def test_inplace_int8_ips() raises:
    """Test inplace operations on int8."""
    var buffer = Buffer[DType.int8](128)
    for i in range(128):
        buffer[i] = Int8(i)

    buffer.inplace_ops_scalar[Add](Int8(10))

    for i in range(128):
        var expected = Int8((i + 10) % 256)
        assert_true(buffer[i] == expected)


def test_inplace_int64_ips() raises:
    """Test inplace operations on int64."""
    var buffer = Buffer[DType.int64](100)
    for i in range(100):
        buffer[i] = Int64(i) * 1000000

    buffer.inplace_ops_scalar[Divide](Int64(1000000))

    for i in range(100):
        assert_true(buffer[i] == Int64(i))


def test_inplace_uint8_ips() raises:
    """Test inplace operations on uint8."""
    var buffer = Buffer[DType.uint8](100)
    for i in range(100):
        buffer[i] = UInt8(i)

    buffer.inplace_ops_scalar[Multiply](UInt8(2))

    for i in range(100):
        var expected = UInt8((i * 2) % 256)
        assert_true(buffer[i] == expected)


def test_inplace_chained_operations_ips() raises:
    """Test chaining multiple inplace operations."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](2)
    buffer.inplace_ops_scalar[Add](10)
    buffer.inplace_ops_scalar[Subtract](5)

    for i in range(50):
        assert_true(buffer[i] == Int32(i * 2 + 10 - 5))


def test_inplace_large_buffer_ips() raises:
    """Test inplace operations on large buffer."""
    var buffer = Buffer[DType.int32](10000)
    for i in range(10000):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](3)

    # Check some samples
    assert_true(buffer[0] == 0)
    assert_true(buffer[100] == 300)
    assert_true(buffer[5000] == 15000)
    assert_true(buffer[9999] == 29997)


def test_inplace_start_equals_end_ips() raises:
    """Test inplace operation with start == end (no-op)."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](999, 50, 50)

    # Nothing should change
    for i in range(100):
        assert_true(buffer[i] == Int32(i))


def test_inplace_invalid_range_ips() raises:
    """Test inplace operation with start > end (no-op)."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Multiply](999, 70, 50)

    # Nothing should change
    for i in range(100):
        assert_true(buffer[i] == Int32(i))


def test_inplace_full_range_explicit_ips() raises:
    """Test inplace operation with explicit full range."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    buffer.inplace_ops_scalar[Add](100, 0, 100)

    for i in range(100):
        assert_true(buffer[i] == Int32(i + 100))


# ============================================================================
# Buffer Arithmetic Operations (Buffer-Buffer) Tests
# ============================================================================


def test_arithmetic_multiply_buffers_arith() raises:
    """Test element-wise multiplication of two buffers."""
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = Int32(i)
        buf2[i] = 2

    var result = buf1.arithmetic_ops[Multiply](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == Int32(i) * 2)


def test_arithmetic_add_buffers_arith() raises:
    """Test element-wise addition of two buffers."""
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = Int32(i)
        buf2[i] = 10

    var result = buf1.arithmetic_ops[Add](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == Int32(i) + 10)


def test_arithmetic_subtract_buffers_arith() raises:
    """Test element-wise subtraction of two buffers."""
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = Int32(i + 50)
        buf2[i] = 50

    var result = buf1.arithmetic_ops[Subtract](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == Int32(i))


def test_arithmetic_divide_buffers_arith() raises:
    """Test element-wise division of two buffers."""
    var buf1 = Buffer[DType.float32](100)
    var buf2 = Buffer[DType.float32](100)

    for i in range(100):
        buf1[i] = Float32(i * 4)
        buf2[i] = 2.0

    var result = buf1.arithmetic_ops[Divide](buf2)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(abs(result[i] - Float32(i * 2)) < 0.001)


def test_arithmetic_with_ranges_arith() raises:
    """Test arithmetic operations with specified ranges."""
    var buf1 = Buffer[DType.int32](100)
    var buf2 = Buffer[DType.int32](100)

    for i in range(100):
        buf1[i] = Int32(i)
        buf2[i] = 1

    var result = buf1.arithmetic_ops[Multiply](buf2, 20, 40, 30, 50)

    assert_true(result.size == 20)
    for i in range(20):
        assert_true(
            result[i] == (20 + Int32(i))
        )  # buf1[20+i] * buf2[30+i] = (20+i) * 1


def test_arithmetic_float32_precision_arith() raises:
    """Test arithmetic on float32 with precision."""
    var buf1 = Buffer[DType.float32](100)
    var buf2 = Buffer[DType.float32](100)

    for i in range(100):
        buf1[i] = Float32(i) * 0.5
        buf2[i] = Float32(i) * 0.3

    var result = buf1.arithmetic_ops[Add](buf2)

    for i in range(100):
        var expected = Float32(i) * 0.8
        assert_true(abs(result[i] - expected) < 0.01)


def test_arithmetic_simd_boundary_arith() raises:
    """Test arithmetic across SIMD boundaries."""
    comptime simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 4 + 7

    var buf1 = Buffer[DType.int32](size)
    var buf2 = Buffer[DType.int32](size)

    for i in range(size):
        buf1[i] = Int32(i)
        buf2[i] = 3

    var result = buf1.arithmetic_ops[Multiply](buf2)

    for i in range(size):
        assert_true(result[i] == Int32(i) * 3)


def test_arithmetic_int64_arith() raises:
    """Test arithmetic on int64 buffers."""
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


def test_arithmetic_scalar_multiply_arith() raises:
    """Test multiplication by scalar."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    var result = buffer.arithmetic_ops_scalar[Multiply](3)

    assert_true(result.size == 100)
    for i in range(100):
        assert_true(result[i] == Int32(i) * 3)


def test_arithmetic_scalar_add_arith() raises:
    """Test addition with scalar."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    var result = buffer.arithmetic_ops_scalar[Add](100)

    for i in range(100):
        assert_true(result[i] == Int32(i) + 100)


def test_arithmetic_scalar_subtract_arith() raises:
    """Test subtraction with scalar."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i + 50)

    var result = buffer.arithmetic_ops_scalar[Subtract](50)

    for i in range(100):
        assert_true(result[i] == Int32(i))


def test_arithmetic_scalar_reverse_subtract_arith() raises:
    """Test reverse subtraction (scalar - buffer)."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    var result = buffer.arithmetic_ops_scalar[ReverseSubtract](100)

    for i in range(100):
        assert_true(result[i] == 100 - Int32(i))


def test_arithmetic_scalar_divide_arith() raises:
    """Test division by scalar."""
    var buffer = Buffer[DType.float32](100)
    for i in range(100):
        buffer[i] = Float32(i * 4)

    var result = buffer.arithmetic_ops_scalar[Divide](2.0)

    for i in range(100):
        assert_true(abs(result[i] - Float32(i * 2)) < 0.001)


def test_arithmetic_scalar_reverse_divide_arith() raises:
    """Test reverse division (scalar / buffer)."""
    var buffer = Buffer[DType.float32](100)
    for i in range(1, 101):  # Start from 1 to avoid division by zero
        buffer[i - 1] = Float32(i)

    var result = buffer.arithmetic_ops_scalar[ReverseDivide](100.0)

    for i in range(100):
        var expected = 100.0 / Float32(i + 1)
        assert_true(abs(result[i] - expected) < 0.01)


def test_arithmetic_scalar_with_range_arith() raises:
    """Test scalar arithmetic with range."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    var result = buffer.arithmetic_ops_scalar[Multiply](10, 20, 50)

    assert_true(result.size == 30)
    for i in range(30):
        assert_true(result[i] == (20 + Int32(i)) * 10)


def test_arithmetic_scalar_float64_arith() raises:
    """Test scalar arithmetic on float64."""
    var buffer = Buffer[DType.float64](100)
    for i in range(100):
        buffer[i] = Float64(i) * 0.1

    var result = buffer.arithmetic_ops_scalar[Multiply](10.0)

    for i in range(100):
        assert_true(abs(result[i] - Float64(i)) < 0.0001)


def test_arithmetic_scalar_simd_boundary_arith() raises:
    """Test scalar arithmetic across SIMD boundaries."""
    comptime simd_w = simd_width_of[DType.int32]()
    var size = simd_w * 3 + 5

    var buffer = Buffer[DType.int32](size)
    for i in range(size):
        buffer[i] = Int32(i)

    var result = buffer.arithmetic_ops_scalar[Add](1000)

    for i in range(size):
        assert_true(result[i] == Int32(i) + 1000)


def test_arithmetic_scalar_bool_arith() raises:
    """Test scalar arithmetic on boolean buffer."""
    var buffer = Buffer[DType.bool](100)
    for i in range(100):
        buffer[i] = i % 2 == 0

    var result = buffer.arithmetic_ops_scalar[Multiply](False)

    for i in range(100):
        assert_true(result[i] == False)


def test_arithmetic_scalar_negative_arith() raises:
    """Test scalar arithmetic with negative values."""
    var buffer = Buffer[DType.int32](100)
    for i in range(100):
        buffer[i] = Int32(i)

    var result = buffer.arithmetic_ops_scalar[Multiply](-1)

    for i in range(100):
        assert_true(result[i] == -Int32(i))


def test_arithmetic_scalar_int8_arith() raises:
    """Test scalar arithmetic on int8."""
    var buffer = Buffer[DType.int8](128)
    for i in range(128):
        buffer[i] = Int8(i)

    var result = buffer.arithmetic_ops_scalar[Add](Int8(10))

    for i in range(128):
        var expected = Int8((i + 10) % 256)
        assert_true(result[i] == expected)


def test_arithmetic_ops_preserve_original_arith() raises:
    """Test that arithmetic ops don't modify original buffers."""
    var buf1 = Buffer[DType.int32](50)
    var buf2 = Buffer[DType.int32](50)

    for i in range(50):
        buf1[i] = Int32(i)
        buf2[i] = 10

    _result = buf1.arithmetic_ops[Add](buf2)

    # Check original buffers unchanged
    for i in range(50):
        assert_true(buf1[i] == Int32(i))
        assert_true(buf2[i] == 10)


def test_arithmetic_scalar_preserve_original_arith() raises:
    """Test that scalar ops don't modify original buffer."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    _result = buffer.arithmetic_ops_scalar[Multiply](5)

    # Check original buffer unchanged
    for i in range(50):
        assert_true(buffer[i] == Int32(i))


def test_arithmetic_chained_operations_arith() raises:
    """Test chaining multiple arithmetic operations."""
    var buffer = Buffer[DType.int32](50)
    for i in range(50):
        buffer[i] = Int32(i)

    var result1 = buffer.arithmetic_ops_scalar[Multiply](2)
    var result2 = result1.arithmetic_ops_scalar[Add](10)
    var result3 = result2.arithmetic_ops_scalar[Subtract](5)

    for i in range(50):
        assert_true(result3[i] == Int32(i) * 2 + 10 - 5)


def test_arithmetic_large_buffers_arith() raises:
    """Test arithmetic on large buffers."""
    var buf1 = Buffer[DType.int32](10000)
    var buf2 = Buffer[DType.int32](10000)

    for i in range(10000):
        buf1[i] = Int32(i)
        buf2[i] = 2

    var result = buf1.arithmetic_ops[Multiply](buf2)

    # Check some samples
    assert_true(result[0] == 0)
    assert_true(result[100] == 200)
    assert_true(result[5000] == 10000)
    assert_true(result[9999] == 19998)


def test_arithmetic_uint8_arith() raises:
    """Test arithmetic on uint8 buffers."""
    var buf1 = Buffer[DType.uint8](100)
    var buf2 = Buffer[DType.uint8](100)

    for i in range(100):
        buf1[i] = UInt8(i)
        buf2[i] = UInt8(2)

    var result = buf1.arithmetic_ops[Multiply](buf2)

    for i in range(100):
        var expected = UInt8((i * 2) % 256)
        assert_true(result[i] == expected)


# ========== Add Operation Tests ==========


def test_mv_inplace_add_basic() raises:
    """Test basic in-place addition."""
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


def test_mv_inplace_add_range() raises:
    """Test in-place addition with range."""
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


# ========== Multiply Operation Tests ==========


def test_mv_inplace_multiply_basic() raises:
    """Test basic in-place multiplication."""
    var buffer1 = Buffer[DType.int32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.int32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = Int32(i + 1)
        buffer2[i] = 2

    buffer1.inplace_ops[Multiply](buffer2)

    for i in range(SMALL_SIZE_NEW):
        var expected = (i + 1) * 2
        assert_true(
            buffer1[i] == Int32(expected), "Multiply failed at " + String(i)
        )


def test_mv_inplace_multiply_large() raises:
    """Test vectorization efficiency on large buffer."""
    var buffer1 = Buffer[DType.float64](LARGE_SIZE)
    var buffer2 = Buffer[DType.float64](LARGE_SIZE)

    for i in range(LARGE_SIZE):
        buffer1[i] = Float64(i)
        buffer2[i] = 1.5

    buffer1.inplace_ops[Multiply](buffer2)

    for i in range(LARGE_SIZE):
        var expected = Float64(i) * 1.5
        assert_true(abs(buffer1[i] - expected) < 1e-10, "Large multiply failed")


# ========== Subtract Operation Tests ==========


def test_mv_inplace_subtract_basic() raises:
    """Test basic in-place subtraction."""
    var buffer1 = Buffer[DType.int32](MEDIUM_SIZE)
    var buffer2 = Buffer[DType.int32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buffer1[i] = 100
        buffer2[i] = Int32(i)

    buffer1.inplace_ops[Subtract](buffer2)

    for i in range(MEDIUM_SIZE):
        var expected = Int32(100 - i)
        assert_true(buffer1[i] == expected, "Subtract failed")


def test_mv_inplace_subtract_negative() raises:
    """Test subtraction resulting in negatives."""
    var buffer1 = Buffer[DType.float32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.float32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = Float32(i)
        buffer2[i] = Float32(i + 10)

    buffer1.inplace_ops[Subtract](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == -10.0, "Negative result incorrect")


# ========== Divide Operation Tests ==========


def test_mv_inplace_divide_basic() raises:
    """Test basic in-place division."""
    var buffer1 = Buffer[DType.float32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.float32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = Float32(100.0)
        buffer2[i] = Float32(4.0)

    buffer1.inplace_ops[Divide](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == 25.0, "Divide failed")


def test_mv_inplace_divide_fractional() raises:
    """Test division with fractional results."""
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


# ========== Overwrite Operation Tests ==========


def test_mv_inplace_overwrite_basic() raises:
    """Test basic overwrite operation."""
    var buffer1 = Buffer[DType.int32](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.int32](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = 999
        buffer2[i] = Int32(i * 10)

    buffer1.inplace_ops[Overwrite](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(
            buffer1[i] == Int32(i * 10), "Overwrite failed at " + String(i)
        )


def test_mv_inplace_overwrite_partial() raises:
    """Test partial overwrite with range."""
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


def test_mv_inplace_bool_overwrite() raises:
    """Test boolean overwrite."""
    var buffer1 = Buffer[DType.bool](SMALL_SIZE_NEW)
    var buffer2 = Buffer[DType.bool](SMALL_SIZE_NEW)

    for i in range(SMALL_SIZE_NEW):
        buffer1[i] = True
        buffer2[i] = (i % 2) == 0

    buffer1.inplace_ops[Overwrite](buffer2)

    for i in range(SMALL_SIZE_NEW):
        assert_true(buffer1[i] == ((i % 2) == 0), "Bool overwrite failed")


# ========== Edge Cases ==========


def test_mv_inplace_single_element() raises:
    """Test with single element buffer."""
    var buffer1 = Buffer[DType.int32](1)
    var buffer2 = Buffer[DType.int32](1)

    buffer1[0] = 10
    buffer2[0] = 5

    buffer1.inplace_ops[Add](buffer2)
    assert_true(buffer1[0] == 15, "Single element add failed")

    buffer1.inplace_ops[Multiply](buffer2)
    assert_true(buffer1[0] == 75, "Single element multiply failed")


def test_mv_inplace_tail_alignment() raises:
    """Test SIMD tail loop handling."""

    # Size that's not SIMD-aligned (e.g., 13 for 16-byte SIMD)
    comptime SIZE = 13
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


def test_mv_inplace_mismatched_ranges() raises:
    """Test validation catches mismatched ranges."""
    var buffer1 = Buffer[DType.int32](100)
    var buffer2 = Buffer[DType.int32](100)

    for i in range(100):
        buffer1[i] = Int32(i)
        buffer2[i] = 0

    # This should fail validation (range 10 vs 20)
    buffer1.inplace_ops[Add, validate=True](buffer2, 0, 10, 0, 20)

    # Buffer1 should be unchanged (operation skipped)
    for i in range(100):
        assert_true(buffer1[i] == Int32(i), "Validation bypass failed")


# ========== Performance Comparison Test ==========


def test_mv_performance_comparison() raises:
    """Compare performance: small vs large buffers."""

    comptime PERF_SIZE = 10000
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


comptime TEST_SMALL = 17
comptime TEST_MEDIUM = 68
comptime TEST_LARGE = 1024

# ========== SUM TESTS ==========


def test_bufops_sum_basic() raises:
    """Test basic sum operation."""
    var buffer = Buffer[DType.int32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = Int32(i + 1)  # 1, 2, 3, ..., 17

    var result = buffer.sum()
    var expected = (TEST_SMALL * (TEST_SMALL + 1)) // 2  # Sum formula

    assert_true(
        result == Int32(expected),
        "Sum failed: got " + String(result) + ", expected " + String(expected),
    )


def test_bufops_sum_range() raises:
    """Test sum with start/end indices."""
    var buffer = Buffer[DType.float32](TEST_MEDIUM)

    for i in range(TEST_MEDIUM):
        buffer[i] = 1.0

    # Sum only middle portion [10:30)
    var result = buffer.sum(10, 30)
    assert_true(result == 20.0, "Sum range failed")


def test_bufops_sum_negative() raises:
    """Test sum with negative numbers."""
    var buffer = Buffer[DType.int32](10)

    for i in range(10):
        buffer[i] = Int32(i - 5)  # -5, -4, ..., 3, 4

    var result = buffer.sum()
    var expected = -5  # Sum of -5 to 4

    assert_true(result == Int32(expected), "Sum with negatives failed")


def test_bufops_sum_large() raises:
    """Test sum on large buffer (vectorization efficiency)."""
    var buffer = Buffer[DType.float64](TEST_LARGE)

    for i in range(TEST_LARGE):
        buffer[i] = 1.0

    var result = buffer.sum()
    assert_true(result == Float64(TEST_LARGE), "Large sum failed")


def test_bufops_sum_empty_range() raises:
    """Test sum with empty range."""
    var buffer = Buffer[DType.int32](100)

    var result = buffer.sum(50, 50)  # Empty range
    assert_true(result == 0, "Empty sum should be 0")


# ========== PRODUCT TESTS ==========


def test_bufops_product_basic() raises:
    """Test basic product operation."""
    var buffer = Buffer[DType.int32](5)

    for i in range(5):
        buffer[i] = Int32(i + 1)  # 1, 2, 3, 4, 5

    var result = buffer.product()
    assert_true(result == 120, "Product failed: expected 120")


def test_bufops_product_range() raises:
    """Test product with range."""
    var buffer = Buffer[DType.float32](20)

    for i in range(20):
        buffer[i] = 2.0

    # Product of indices [5:10) = 2^5 = 32
    var result = buffer.product(5, 10)
    assert_true(result == 32.0, "Product range failed")


def test_bufops_product_with_zero() raises:
    """Test product with zero element."""
    var buffer = Buffer[DType.int32](10)

    for i in range(10):
        buffer[i] = Int32(i)  # 0, 1, 2, ..., 9

    var result = buffer.product()
    assert_true(result == 0, "Product with zero should be 0")


def test_bufops_product_fractional() raises:
    """Test product with fractional values."""
    var buffer = Buffer[DType.float64](4)

    buffer[0] = 0.5
    buffer[1] = 2.0
    buffer[2] = 4.0
    buffer[3] = 0.25

    var result = buffer.product()
    assert_true(result == 1.0, "Fractional product failed")


def test_bufops_product_empty() raises:
    """Test product of empty range (should be 1)."""
    var buffer = Buffer[DType.int32](100)

    var result = buffer.product(10, 10)  # Empty range
    assert_true(result == 1, "Empty product should be 1")


# ========== POWER TESTS ==========


def test_bufops_pow_basic() raises:
    """Test basic power operation."""
    var buffer = Buffer[DType.float32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = Float32(i)

    var result = buffer.__pow__(2.0)

    for i in range(TEST_SMALL):
        var expected = Float32(i * i)
        assert_true(result[i] == expected, "Pow failed at " + String(i))


def test_bufops_pow_fractional() raises:
    """Test power with fractional exponent."""
    var buffer = Buffer[DType.float64](10)

    for i in range(10):
        buffer[i] = Float64((i + 1) * (i + 1))  # 1, 4, 9, 16, ...

    var result = buffer.__pow__(0.5)  # Square root

    for i in range(10):
        var expected = Float64(i + 1)
        assert_true(abs(result[i] - expected) < 1e-8, "Fractional pow failed")


def test_bufops_pow_zero_exponent() raises:
    """Test power with exponent 0 (should be all 1s)."""
    var buffer = Buffer[DType.int32](TEST_MEDIUM)

    for i in range(TEST_MEDIUM):
        buffer[i] = Int32(i * 10)

    var result = buffer.__pow__(0)

    for i in range(TEST_MEDIUM):
        assert_true(result[i] == 1, "Pow 0 should be 1")


def test_bufops_pow_negative_base() raises:
    """Test power with negative base values."""
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


# ========== ABSOLUTE VALUE TESTS ==========


def test_bufops_abs_basic() raises:
    """Test basic absolute value."""
    var buffer = Buffer[DType.int32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = Int32(i - 8)  # -8, -7, ..., 7, 8

    var result = buffer.__abs__()

    for i in range(TEST_SMALL):
        var expected = abs(Int32(i) - 8)
        assert_true(result[i] == expected, "Abs failed")


def test_bufops_abs_all_negative() raises:
    """Test abs with all negative values."""
    var buffer = Buffer[DType.float32](TEST_MEDIUM)

    for i in range(TEST_MEDIUM):
        buffer[i] = Float32(-(i + 1))

    var result = buffer.__abs__()

    for i in range(TEST_MEDIUM):
        assert_true(result[i] == Float32(i + 1), "Abs negative failed")


def test_bufops_abs_all_positive() raises:
    """Test abs with all positive values (should be unchanged)."""
    var buffer = Buffer[DType.int32](20)

    for i in range(20):
        buffer[i] = Int32(i * 5)

    var result = buffer.__abs__()

    for i in range(20):
        assert_true(result[i] == Int32(i) * 5, "Abs positive changed value")


def test_bufops_abs_mixed() raises:
    """Test abs with mixed positive/negative."""
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


# ========== FILL TESTS ==========


def test_bufops_fill_basic() raises:
    """Test basic fill operation."""
    var buffer = Buffer[DType.int32](TEST_SMALL)

    buffer.fill(42)

    for i in range(TEST_SMALL):
        assert_true(buffer[i] == 42, "Fill failed")


def test_bufops_fill_range() raises:
    """Test fill with range."""
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


def test_bufops_fill_bool() raises:
    """Test fill with boolean type."""
    var buffer = Buffer[DType.bool](TEST_SMALL)

    buffer.fill(True)

    for i in range(TEST_SMALL):
        assert_true(buffer[i] == True, "Bool fill failed")

    buffer.fill(False, 5, 10)

    for i in range(5, 10):
        assert_true(buffer[i] == False, "Bool fill range failed")


def test_bufops_fill_zero() raises:
    """Test fill with zero."""
    var buffer = Buffer[DType.float64](TEST_MEDIUM)

    # Set to non-zero
    for i in range(TEST_MEDIUM):
        buffer[i] = Float64(i + 100)

    # Fill with zero
    buffer.fill(0.0)

    for i in range(TEST_MEDIUM):
        assert_true(buffer[i] == 0.0, "Fill zero failed")


def test_bufops_fill_negative() raises:
    """Test fill with negative value."""
    var buffer = Buffer[DType.int32](30)

    buffer.fill(-999)

    for i in range(30):
        assert_true(buffer[i] == -999, "Fill negative failed")


# ========== NEGATION TESTS ==========


def test_bufops_neg_basic() raises:
    """Test basic negation."""
    var buffer = Buffer[DType.int32](TEST_SMALL)

    for i in range(TEST_SMALL):
        buffer[i] = Int32(i + 1)

    var result = buffer.__neg__()

    for i in range(TEST_SMALL):
        assert_true(result[i] == -Int32(i + 1), "Neg failed")


def test_bufops_neg_mixed() raises:
    """Test negation with mixed signs."""
    var buffer = Buffer[DType.float32](10)

    for i in range(10):
        buffer[i] = Float32(i - 5)  # -5, -4, ..., 3, 4

    var result = buffer.__neg__()

    for i in range(10):
        var expected = Float32(5 - i)
        assert_true(result[i] == expected, "Neg mixed failed")


def test_bufops_neg_zero() raises:
    """Test negation with zeros."""
    var buffer = Buffer[DType.float64](TEST_MEDIUM)

    buffer.fill(0.0)

    var result = buffer.__neg__()

    for i in range(TEST_MEDIUM):
        assert_true(result[i] == 0.0, "Neg zero failed")


def test_bufops_neg_double_negation() raises:
    """Test double negation (should restore original)."""
    var buffer = Buffer[DType.int32](20)

    for i in range(20):
        buffer[i] = Int32(i * 10)

    var neg1 = buffer.__neg__()
    var neg2 = neg1.__neg__()

    for i in range(20):
        assert_true(neg2[i] == buffer[i], "Double neg failed")


# ========== EDGE CASE TESTS ==========


def test_bufops_single_element() raises:
    """Test all operations on single element buffer."""

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


def test_bufops_tail_alignment() raises:
    """Test tail loop handling for non-SIMD-aligned sizes."""

    comptime SIZE = 13  # Likely not SIMD-aligned
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


def test_bufops_performance_comparison() raises:
    """Compare performance across operations."""

    comptime PERF_SIZE = 10000
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


def test_compare_buffer_manual() raises:
    var buf1 = Buffer[DType.int32](MEDIUM_SIZE)
    var buf2 = Buffer[DType.int32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buf1[i] = Int32(i)
        buf2[i] = Int32(i) if i < 30 else Int32(i + 1)

    var result = buf1.compare_buffer_full[Equal](buf2)

    # Count equal elements (first 30 should be equal)
    var equal_count = 0
    for i in range(MEDIUM_SIZE):
        if result[i]:
            equal_count += 1

    assert_true(
        equal_count == 30,
        "compare_buffer_manual: expected 30 equal, got " + String(equal_count),
    )


def test_compare_scalar_manual() raises:
    var buffer = Buffer[DType.int32](MEDIUM_SIZE)
    for i in range(MEDIUM_SIZE):
        buffer[i] = Int32(i % 10)

    var result = buffer.compare_scalar_full[Equal](5)

    # Count elements equal to 5
    var count = 0
    for i in range(MEDIUM_SIZE):
        if result[i]:
            count += 1

    # Values 0-67, with i % 10 == 5 at indices: 5, 15, 25, 35, 45, 55, 65 = 7 occurrences
    assert_true(
        count == 7,
        "compare_scalar_manual: expected 7 fives, got " + String(count),
    )



def test_compare_buffer_manual_gt() raises:
    var buf1 = Buffer[DType.int32](MEDIUM_SIZE)
    var buf2 = Buffer[DType.int32](MEDIUM_SIZE)

    for i in range(MEDIUM_SIZE):
        buf1[i] = Int32(i)
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
        "compare_buffer_manual_gt: expected 37, got " + String(count),
    )


def test_unary_ops_manual_sigmoid() raises:
    var buffer = Buffer[DType.float32](10)
    for i in range(10):
        buffer[i] = Float32(i - 5)  # Values from -5 to 4

    var result = buffer.sigmoid()

    # Check that all values are in (0, 1) range
    var all_in_range = True
    for i in range(10):
        if result[i] <= 0.0 or result[i] >= 1.0:
            all_in_range = False
            break

    assert_true(all_in_range, "unary_ops_manual_sigmoid: values out of range")


def test_buffer_multiply() raises:
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


def test_relu_backward_multiplication() raises:
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


# ============================================
# Buffer.copied Tests
# ============================================


def test_copied_empty() raises:
    var buf = Buffer[DType.float32]()
    var copy = buf.copied()
    assert_true(copy.size == 0, "copied: empty buffer size mismatch")


def test_copied_full() raises:
    comptime dtype = DType.float32
    var buf = Buffer[dtype](10)
    for i in range(10):
        buf[i] = Float32(i)
    var copy = buf.copied()
    assert_true(copy.size == 10, "copied: full copy size mismatch")
    for i in range(10):
        assert_true(
            copy[i] == Float32(i),
            "copied: full copy value mismatch at " + String(i),
        )


def test_copied_independent() raises:
    comptime dtype = DType.float32
    var buf = Buffer[dtype](5)
    for i in range(5):
        buf[i] = Float32(i)
    var copy = buf.copied()
    buf[0] = Float32(99)
    copy[1] = Float32(88)
    assert_true(
        buf[0] == Float32(99),
        "copied: modifying original should not affect copy",
    )
    assert_true(buf[1] == Float32(1), "copied: original unchanged at index 1")
    assert_true(copy[0] == Float32(0), "copied: copy unchanged at index 0")
    assert_true(
        copy[1] == Float32(88),
        "copied: modifying copy should not affect original",
    )


def test_copied_start_index() raises:
    comptime dtype = DType.float32
    var buf = Buffer[dtype](10)
    for i in range(10):
        buf[i] = Float32(i)
    var copy = buf.copied(start_index=3)
    assert_true(copy.size == 7, "copied: start_index size mismatch")
    for i in range(7):
        assert_true(
            copy[i] == Float32(i + 3),
            "copied: start_index value mismatch at " + String(i),
        )


def test_copied_start_end() raises:
    comptime dtype = DType.float32
    var buf = Buffer[dtype](10)
    for i in range(10):
        buf[i] = Float32(i)
    var copy = buf.copied(start_index=3, end_index=7)
    assert_true(copy.size == 4, "copied: start_end size mismatch")
    for i in range(4):
        assert_true(
            copy[i] == Float32(i + 3),
            "copied: start_end value mismatch at " + String(i),
        )


def test_copied_shared_independent() raises:
    comptime dtype = DType.float32
    var buf = Buffer[dtype](5)
    for i in range(5):
        buf[i] = Float32(i)
    buf.shared()  # convert to shared in-place
    var copy = buf.copied()
    buf[0] = Float32(99)
    copy[0] = Float32(88)
    assert_true(
        buf[0] == Float32(99),
        "copied: modifying original should not affect copy",
    )
    assert_true(buf[1] == Float32(1), "copied: original unchanged at index 1")
    assert_true(
        copy[0] == Float32(88),
        "copied: modifying copy should not affect original",
    )
    assert_true(copy[1] == Float32(1), "copied: copy unchanged at index 1")


def test_buffer_numpy_noncopy_roundtrip() raises:
    """Buffer wrapping a numpy ndarray without copying (the non-copy / 'rebind' path).
    """
    var np = Python.import_module("numpy")
    var py_list = Python.list(
        Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)
    )
    var arr = np.array(py_list, dtype=np.float32)
    var ptr = ndarray_ptr[DType.float32](arr)
    var buf = Buffer[DType.float32](4, ptr, copy=False)
    assert_true(buf.size == 4)
    assert_true(buf.external, "non-copy Buffer should be marked external")
    assert_true(buf[0] == 1.0)
    assert_true(buf[1] == 2.0)
    assert_true(buf[2] == 3.0)
    assert_true(buf[3] == 4.0)
    # Modify numpy array in-place — Buffer must reflect changes (shared memory)
    arr[0] = 99.0
    arr[2] = -77.0
    assert_true(buf[0] == 99.0, "Buffer must see numpy modifications (no copy)")
    assert_true(
        buf[2] == -77.0, "Buffer must see numpy modifications (no copy)"
    )
    # Buffer destructor will NOT free numpy's memory (external=True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
