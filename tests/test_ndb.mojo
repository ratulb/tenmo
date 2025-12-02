from testing import assert_true, assert_false
from buffers import Buffer
from intarray import IntArray
from shapes import Shape
from strides import Strides
from ndbuffer import NDBuffer
from operators import *

# ============================================
# Test Constants
# ============================================
alias SMALL_SIZE = 3
alias MEDIUM_SIZE = 4

# ============================================
# arithmetic_ops Tests
# ============================================


fn ndb_ops_test_arithmetic_add_same_shape_both_contiguous() raises:
    print("ndb_ops_test_arithmetic_add_same_shape_both_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 5)

    var result = a.arithmetic_ops[Add](b)

    assert_true(result.shape == Shape(3, 4), "Shape should be preserved")
    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 15, "10 + 5 = 15")
    print("ndb_ops_test_arithmetic_add_same_shape_both_contiguous passed")


fn ndb_ops_test_arithmetic_sub_same_shape_both_contiguous() raises:
    print("ndb_ops_test_arithmetic_sub_same_shape_both_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    var result = a.arithmetic_ops[Subtract](b)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 7, "10 - 3 = 7")
    print("ndb_ops_test_arithmetic_sub_same_shape_both_contiguous passed")


fn ndb_ops_test_arithmetic_mul_same_shape_both_contiguous() raises:
    print("ndb_ops_test_arithmetic_mul_same_shape_both_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    var result = a.arithmetic_ops[Multiply](b)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 12, "4 * 3 = 12")
    print("ndb_ops_test_arithmetic_mul_same_shape_both_contiguous passed")


fn ndb_ops_test_arithmetic_div_same_shape_both_contiguous() raises:
    print("ndb_ops_test_arithmetic_div_same_shape_both_contiguous")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 10.0)
    var b = NDBuffer[DType.float32].full(Shape(3, 4), 2.0)

    var result = a.arithmetic_ops[Divide](b)

    for i in range(3):
        for j in range(4):
            assert_true(abs(result[IntArray(i, j)] - 5.0) < 0.001, "10 / 2 = 5")
    print("ndb_ops_test_arithmetic_div_same_shape_both_contiguous passed")


fn ndb_ops_test_arithmetic_self_contiguous_other_noncontiguous() raises:
    print("ndb_ops_test_arithmetic_self_contiguous_other_noncontiguous")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 10)

    # Create non-contiguous other
    var other_buffer = Buffer[DType.int32](12)
    for i in range(12):
        other_buffer[i] = i
    var b = NDBuffer[DType.int32](
        other_buffer^, Shape(2, 3), Strides(1, 2), offset=0  # Non-contiguous
    )

    var result = a.arithmetic_ops[Add](b)

    for i in range(2):
        for j in range(3):
            var expected = 10 + b[IntArray(i, j)]
            assert_true(
                result[IntArray(i, j)] == expected,
                "Mismatch at " + i.__str__() + "," + j.__str__(),
            )
    print("ndb_ops_test_arithmetic_self_contiguous_other_noncontiguous passed")


fn ndb_ops_test_arithmetic_self_noncontiguous_other_contiguous() raises:
    print("ndb_ops_test_arithmetic_self_noncontiguous_other_contiguous")

    # Create non-contiguous self
    var self_buffer = Buffer[DType.int32](12)
    for i in range(12):
        self_buffer[i] = i * 10
    var a = NDBuffer[DType.int32](
        self_buffer^, Shape(2, 3), Strides(1, 2), offset=0
    )

    var b = NDBuffer[DType.int32].full(Shape(2, 3), 5)

    var result = a.arithmetic_ops[Add](b)

    for i in range(2):
        for j in range(3):
            var expected = a[IntArray(i, j)] + 5
            assert_true(result[IntArray(i, j)] == expected, "Mismatch")
    print("ndb_ops_test_arithmetic_self_noncontiguous_other_contiguous passed")


fn ndb_ops_test_arithmetic_both_noncontiguous() raises:
    print("ndb_ops_test_arithmetic_both_noncontiguous")

    var a_buffer = Buffer[DType.int32](12)
    var b_buffer = Buffer[DType.int32](12)
    for i in range(12):
        a_buffer[i] = i
        b_buffer[i] = i * 2

    var a = NDBuffer[DType.int32](a_buffer^, Shape(2, 3), Strides(1, 2), 0)
    var b = NDBuffer[DType.int32](b_buffer^, Shape(2, 3), Strides(1, 2), 0)

    var result = a.arithmetic_ops[Multiply](b)

    for i in range(2):
        for j in range(3):
            var expected = a[IntArray(i, j)] * b[IntArray(i, j)]
            assert_true(result[IntArray(i, j)] == expected, "Mismatch")
    print("ndb_ops_test_arithmetic_both_noncontiguous passed")


fn ndb_ops_test_arithmetic_with_offset() raises:
    print("ndb_ops_test_arithmetic_with_offset")

    # Create buffers with offsets
    var a_buffer = Buffer[DType.int32](20)
    var b_buffer = Buffer[DType.int32](20)
    for i in range(20):
        a_buffer[i] = i
        b_buffer[i] = 100 + i

    var a = NDBuffer[DType.int32](
        a_buffer^, Shape(2, 3), Strides.default(Shape(2, 3)), offset=5
    )
    var b = NDBuffer[DType.int32](
        b_buffer^, Shape(2, 3), Strides.default(Shape(2, 3)), offset=8
    )

    var result = a.arithmetic_ops[Add](b)

    for i in range(2):
        for j in range(3):
            var expected = a[IntArray(i, j)] + b[IntArray(i, j)]
            assert_true(result[IntArray(i, j)] == expected, "Offset mismatch")
    print("ndb_ops_test_arithmetic_with_offset passed")


fn ndb_ops_test_arithmetic_broadcast_row() raises:
    print("ndb_ops_test_arithmetic_broadcast_row")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32](Shape(1, 4))
    for j in range(4):
        b[IntArray(0, j)] = j

    var result = a.arithmetic_ops[Add](b)

    assert_true(result.shape == Shape(3, 4), "Broadcast shape")
    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 10 + j, "Broadcast row add")
    print("ndb_ops_test_arithmetic_broadcast_row passed")


fn ndb_ops_test_arithmetic_broadcast_col() raises:
    print("ndb_ops_test_arithmetic_broadcast_col")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32](Shape(3, 1))
    for i in range(3):
        b[IntArray(i, 0)] = i * 100

    var result = a.arithmetic_ops[Add](b)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 10 + i * 100, "Broadcast col")
    print("ndb_ops_test_arithmetic_broadcast_col passed")


fn ndb_ops_test_arithmetic_broadcast_scalar() raises:
    print("ndb_ops_test_arithmetic_broadcast_scalar")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32](Shape())
    b[IntArray()] = 5

    var result = a.arithmetic_ops[Multiply](b)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 50, "Broadcast scalar mul")
    print("ndb_ops_test_arithmetic_broadcast_scalar passed")


fn ndb_ops_test_arithmetic_1d() raises:
    print("ndb_ops_test_arithmetic_1d")
    var a = NDBuffer[DType.float32].full(Shape(10), 2.5)
    var b = NDBuffer[DType.float32].full(Shape(10), 1.5)

    var result = a.arithmetic_ops[Add](b)

    assert_true(result.shape == Shape(10), "1D shape")
    for i in range(10):
        assert_true(abs(result[IntArray(i)] - 4.0) < 0.001, "1D add")
    print("ndb_ops_test_arithmetic_1d passed")


fn ndb_ops_test_arithmetic_3d() raises:
    print("ndb_ops_test_arithmetic_3d")
    var a = NDBuffer[DType.int32].full(Shape(2, 3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(2, 3, 4), 5)

    var result = a.arithmetic_ops[Subtract](b)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_true(result[IntArray(i, j, k)] == 5, "3D subtract")
    print("ndb_ops_test_arithmetic_3d passed")


# ============================================
# inplace_ops Tests
# ============================================


fn ndb_ops_test_inplace_add_same_shape_both_contiguous() raises:
    print("ndb_ops_test_inplace_add_same_shape_both_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 5)

    a.inplace_ops[Add](b)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 15, "10 + 5 = 15")
    print("ndb_ops_test_inplace_add_same_shape_both_contiguous passed")


fn ndb_ops_test_inplace_sub_same_shape() raises:
    print("ndb_ops_test_inplace_sub_same_shape")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    a.inplace_ops[Subtract](b)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 7, "10 - 3 = 7")
    print("ndb_ops_test_inplace_sub_same_shape passed")


fn ndb_ops_test_inplace_mul_same_shape() raises:
    print("ndb_ops_test_inplace_mul_same_shape")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    a.inplace_ops[Multiply](b)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 12, "4 * 3 = 12")
    print("ndb_ops_test_inplace_mul_same_shape passed")


fn ndb_ops_test_inplace_div_same_shape() raises:
    print("ndb_ops_test_inplace_div_same_shape")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 10.0)
    var b = NDBuffer[DType.float32].full(Shape(3, 4), 2.0)

    a.inplace_ops[Divide](b)

    for i in range(3):
        for j in range(4):
            assert_true(abs(a[IntArray(i, j)] - 5.0) < 0.001, "10 / 2 = 5")
    print("ndb_ops_test_inplace_div_same_shape passed")


fn ndb_ops_test_inplace_self_contiguous_other_noncontiguous() raises:
    print("ndb_ops_test_inplace_self_contiguous_other_noncontiguous")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 10)

    var other_buffer = Buffer[DType.int32](12)
    for i in range(12):
        other_buffer[i] = i
    var b = NDBuffer[DType.int32](other_buffer^, Shape(2, 3), Strides(1, 2), 0)

    # Store expected values before inplace op
    var expected = NDBuffer[DType.int32](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            expected[IntArray(i, j)] = 10 + b[IntArray(i, j)]

    a.inplace_ops[Add](b)

    for i in range(2):
        for j in range(3):
            assert_true(
                a[IntArray(i, j)] == expected[IntArray(i, j)], "Mismatch"
            )
    print("ndb_ops_test_inplace_self_contiguous_other_noncontiguous passed")


fn ndb_ops_test_inplace_self_noncontiguous_other_contiguous() raises:
    print("ndb_ops_test_inplace_self_noncontiguous_other_contiguous")

    var self_buffer = Buffer[DType.int32](12)
    for i in range(12):
        self_buffer[i] = i * 10
    var a = NDBuffer[DType.int32](self_buffer^, Shape(2, 3), Strides(1, 2), 0)

    var b = NDBuffer[DType.int32].full(Shape(2, 3), 5)

    # Store expected
    var expected = NDBuffer[DType.int32](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            expected[IntArray(i, j)] = a[IntArray(i, j)] + 5

    a.inplace_ops[Add](b)

    for i in range(2):
        for j in range(3):
            assert_true(
                a[IntArray(i, j)] == expected[IntArray(i, j)], "Mismatch"
            )
    print("ndb_ops_test_inplace_self_noncontiguous_other_contiguous passed")


fn ndb_ops_test_inplace_both_noncontiguous() raises:
    print("ndb_ops_test_inplace_both_noncontiguous")

    var a_buffer = Buffer[DType.int32](12)
    var b_buffer = Buffer[DType.int32](12)
    for i in range(12):
        a_buffer[i] = 10
        b_buffer[i] = i

    var a = NDBuffer[DType.int32](a_buffer^, Shape(2, 3), Strides(1, 2), 0)
    var b = NDBuffer[DType.int32](b_buffer^, Shape(2, 3), Strides(1, 2), 0)

    # Store expected
    var expected = NDBuffer[DType.int32](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            expected[IntArray(i, j)] = a[IntArray(i, j)] * b[IntArray(i, j)]

    a.inplace_ops[Multiply](b)

    for i in range(2):
        for j in range(3):
            assert_true(
                a[IntArray(i, j)] == expected[IntArray(i, j)], "Mismatch"
            )
    print("ndb_ops_test_inplace_both_noncontiguous passed")


fn ndb_ops_test_inplace_broadcast_row() raises:
    print("ndb_ops_test_inplace_broadcast_row")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32](Shape(1, 4))
    for j in range(4):
        b[IntArray(0, j)] = j

    a.inplace_ops[Add](b)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 10 + j, "Broadcast row inplace")
    print("ndb_ops_test_inplace_broadcast_row passed")


fn ndb_ops_test_inplace_broadcast_col() raises:
    print("ndb_ops_test_inplace_broadcast_col")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32](Shape(3, 1))
    for i in range(3):
        b[IntArray(i, 0)] = i * 10

    a.inplace_ops[Multiply](b)

    for i in range(3):
        for j in range(4):
            assert_true(
                a[IntArray(i, j)] == 10 * i * 10, "Broadcast col inplace"
            )
    print("ndb_ops_test_inplace_broadcast_col passed")


fn ndb_ops_test_inplace_with_offset() raises:
    print("ndb_ops_test_inplace_with_offset")

    var a_buffer = Buffer[DType.int32](20)
    var b_buffer = Buffer[DType.int32](20)
    for i in range(20):
        a_buffer[i] = 10
        b_buffer[i] = i

    var a = NDBuffer[DType.int32](
        a_buffer^, Shape(2, 3), Strides.default(Shape(2, 3)), offset=5
    )
    var b = NDBuffer[DType.int32](
        b_buffer^, Shape(2, 3), Strides.default(Shape(2, 3)), offset=8
    )

    # Store expected
    var expected = NDBuffer[DType.int32](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            expected[IntArray(i, j)] = a[IntArray(i, j)] + b[IntArray(i, j)]

    a.inplace_ops[Add](b)

    for i in range(2):
        for j in range(3):
            assert_true(a[IntArray(i, j)] == expected[IntArray(i, j)], "Offset")
    print("ndb_ops_test_inplace_with_offset passed")


# ============================================
# inplace_scalar_ops Tests
# ============================================


fn ndb_ops_test_inplace_scalar_add_contiguous() raises:
    print("ndb_ops_test_inplace_scalar_add_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)

    a.inplace_scalar_ops[Add](5)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 15, "10 + 5 = 15")
    print("ndb_ops_test_inplace_scalar_add_contiguous passed")


fn ndb_ops_test_inplace_scalar_sub_contiguous() raises:
    print("ndb_ops_test_inplace_scalar_sub_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)

    a.inplace_scalar_ops[Subtract](3)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 7, "10 - 3 = 7")
    print("ndb_ops_test_inplace_scalar_sub_contiguous passed")


fn ndb_ops_test_inplace_scalar_mul_contiguous() raises:
    print("ndb_ops_test_inplace_scalar_mul_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)

    a.inplace_scalar_ops[Multiply](3)

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 12, "4 * 3 = 12")
    print("ndb_ops_test_inplace_scalar_mul_contiguous passed")


fn ndb_ops_test_inplace_scalar_div_contiguous() raises:
    print("ndb_ops_test_inplace_scalar_div_contiguous")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 10.0)

    a.inplace_scalar_ops[Divide](2.0)

    for i in range(3):
        for j in range(4):
            assert_true(abs(a[IntArray(i, j)] - 5.0) < 0.001, "10 / 2 = 5")
    print("ndb_ops_test_inplace_scalar_div_contiguous passed")


fn ndb_ops_test_inplace_scalar_noncontiguous() raises:
    print("ndb_ops_test_inplace_scalar_noncontiguous")

    var a_buffer = Buffer[DType.int32](12)
    for i in range(12):
        a_buffer[i] = 10
    var a = NDBuffer[DType.int32](a_buffer^, Shape(2, 3), Strides(1, 2), 0)

    a.inplace_scalar_ops[Add](5)

    for i in range(2):
        for j in range(3):
            assert_true(a[IntArray(i, j)] == 15, "Non-contiguous scalar add")
    print("ndb_ops_test_inplace_scalar_noncontiguous passed")


fn ndb_ops_test_inplace_scalar_with_offset() raises:
    print("ndb_ops_test_inplace_scalar_with_offset")

    var a_buffer = Buffer[DType.int32](20)
    for i in range(20):
        a_buffer[i] = 10
    var a = NDBuffer[DType.int32](
        a_buffer^, Shape(2, 3), Strides.default(Shape(2, 3)), offset=5
    )

    a.inplace_scalar_ops[Multiply](3)

    for i in range(2):
        for j in range(3):
            assert_true(a[IntArray(i, j)] == 30, "Offset scalar mul")
    print("ndb_ops_test_inplace_scalar_with_offset passed")


fn ndb_ops_test_inplace_scalar_1d() raises:
    print("ndb_ops_test_inplace_scalar_1d")
    var a = NDBuffer[DType.float32].full(Shape(10), 2.5)

    a.inplace_scalar_ops[Add](1.5)

    for i in range(10):
        assert_true(abs(a[IntArray(i)] - 4.0) < 0.001, "1D scalar add")
    print("ndb_ops_test_inplace_scalar_1d passed")


fn ndb_ops_test_inplace_scalar_3d() raises:
    print("ndb_ops_test_inplace_scalar_3d")
    var a = NDBuffer[DType.int32].full(Shape(2, 3, 4), 10)

    a.inplace_scalar_ops[Subtract](3)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_true(a[IntArray(i, j, k)] == 7, "3D scalar sub")
    print("ndb_ops_test_inplace_scalar_3d passed")


# ============================================
# scalar_ops Tests
# ============================================


fn ndb_ops_test_scalar_add_contiguous() raises:
    print("ndb_ops_test_scalar_add_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)

    var result = a.scalar_ops[Add](5)

    assert_true(result.shape == Shape(3, 4), "Shape preserved")
    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 15, "10 + 5 = 15")
            assert_true(a[IntArray(i, j)] == 10, "Original unchanged")
    print("ndb_ops_test_scalar_add_contiguous passed")


fn ndb_ops_test_scalar_sub_contiguous() raises:
    print("ndb_ops_test_scalar_sub_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)

    var result = a.scalar_ops[Subtract](3)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 7, "10 - 3 = 7")
    print("ndb_ops_test_scalar_sub_contiguous passed")


fn ndb_ops_test_scalar_mul_contiguous() raises:
    print("ndb_ops_test_scalar_mul_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)

    var result = a.scalar_ops[Multiply](3)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 12, "4 * 3 = 12")
    print("ndb_ops_test_scalar_mul_contiguous passed")


fn ndb_ops_test_scalar_div_contiguous() raises:
    print("ndb_ops_test_scalar_div_contiguous")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 10.0)

    var result = a.scalar_ops[Divide](2.0)

    for i in range(3):
        for j in range(4):
            assert_true(abs(result[IntArray(i, j)] - 5.0) < 0.001, "10 / 2 = 5")
    print("ndb_ops_test_scalar_div_contiguous passed")


fn ndb_ops_test_scalar_reverse_sub_contiguous() raises:
    print("ndb_ops_test_scalar_reverse_sub_contiguous")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    var result = a.scalar_ops[ReverseSubtract](10)

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 7, "10 - 3 = 7")
    print("ndb_ops_test_scalar_reverse_sub_contiguous passed")


fn ndb_ops_test_scalar_reverse_div_contiguous() raises:
    print("ndb_ops_test_scalar_reverse_div_contiguous")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 2.0)

    var result = a.scalar_ops[ReverseDivide](10.0)

    for i in range(3):
        for j in range(4):
            assert_true(abs(result[IntArray(i, j)] - 5.0) < 0.001, "10 / 2 = 5")


fn ndb_ops_test_scalar_noncontiguous() raises:
    print("ndb_ops_test_scalar_noncontiguous")

    var a_buffer = Buffer[DType.int32](12)
    for i in range(12):
        a_buffer[i] = 10
    var a = NDBuffer[DType.int32](a_buffer^, Shape(2, 3), Strides(1, 2), 0)

    var result = a.scalar_ops[Add](5)

    for i in range(2):
        for j in range(3):
            assert_true(
                result[IntArray(i, j)] == 15, "Non-contiguous scalar add"
            )
    print("ndb_ops_test_scalar_noncontiguous passed")


fn ndb_ops_test_scalar_with_offset() raises:
    print("ndb_ops_test_scalar_with_offset")

    var a_buffer = Buffer[DType.int32](20)
    for i in range(20):
        a_buffer[i] = 10
    var a = NDBuffer[DType.int32](
        a_buffer^, Shape(2, 3), Strides.default(Shape(2, 3)), offset=5
    )

    var result = a.scalar_ops[Multiply](3)

    for i in range(2):
        for j in range(3):
            assert_true(result[IntArray(i, j)] == 30, "Offset scalar mul")
    print("ndb_ops_test_scalar_with_offset passed")


fn ndb_ops_test_scalar_1d() raises:
    print("ndb_ops_test_scalar_1d")
    var a = NDBuffer[DType.float32].full(Shape(10), 2.5)

    var result = a.scalar_ops[Add](1.5)

    for i in range(10):
        assert_true(abs(result[IntArray(i)] - 4.0) < 0.001, "1D scalar add")
    print("ndb_ops_test_scalar_1d passed")


fn ndb_ops_test_scalar_3d() raises:
    print("ndb_ops_test_scalar_3d")
    var a = NDBuffer[DType.int32].full(Shape(2, 3, 4), 10)

    var result = a.scalar_ops[Subtract](3)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_true(result[IntArray(i, j, k)] == 7, "3D scalar sub")
    print("ndb_ops_test_scalar_3d passed")


# ============================================
# Operator Overload Tests (__add__, __mul__, etc.)
# ============================================


fn ndb_ops_test_dunder_add() raises:
    print("ndb_ops_test_dunder_add")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 5)

    var result = a + b

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 15, "__add__")
    print("ndb_ops_test_dunder_add passed")


fn ndb_ops_test_dunder_sub() raises:
    print("ndb_ops_test_dunder_sub")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    var result = a - b

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 7, "__sub__")
    print("ndb_ops_test_dunder_sub passed")


fn ndb_ops_test_dunder_mul() raises:
    print("ndb_ops_test_dunder_mul")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    var result = a * b

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 12, "__mul__")
    print("ndb_ops_test_dunder_mul passed")


fn ndb_ops_test_dunder_truediv() raises:
    print("ndb_ops_test_dunder_truediv")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 10.0)
    var b = NDBuffer[DType.float32].full(Shape(3, 4), 2.0)

    var result = a / b

    for i in range(3):
        for j in range(4):
            assert_true(
                abs(result[IntArray(i, j)] - 5.0) < 0.001, "__truediv__"
            )
    print("ndb_ops_test_dunder_truediv passed")


fn ndb_ops_test_dunder_mul_scalar() raises:
    print("ndb_ops_test_dunder_mul_scalar")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)

    var result = a * 3

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 12, "__mul__ scalar")
    print("ndb_ops_test_dunder_mul_scalar passed")


fn ndb_ops_test_dunder_rmul_scalar() raises:
    print("ndb_ops_test_dunder_rmul_scalar")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)

    var result = 3 * a

    for i in range(3):
        for j in range(4):
            assert_true(result[IntArray(i, j)] == 12, "__rmul__ scalar")
    print("ndb_ops_test_dunder_rmul_scalar passed")


fn ndb_ops_test_dunder_iadd() raises:
    print("ndb_ops_test_dunder_iadd")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 5)

    a += b

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 15, "__iadd__")
    print("ndb_ops_test_dunder_iadd passed")


fn ndb_ops_test_dunder_isub() raises:
    print("ndb_ops_test_dunder_isub")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    a -= b

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 7, "__isub__")
    print("ndb_ops_test_dunder_isub passed")


fn ndb_ops_test_dunder_imul() raises:
    print("ndb_ops_test_dunder_imul")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 4)
    var b = NDBuffer[DType.int32].full(Shape(3, 4), 3)

    a *= b

    for i in range(3):
        for j in range(4):
            assert_true(a[IntArray(i, j)] == 12, "__imul__")
    print("ndb_ops_test_dunder_imul passed")


fn ndb_ops_test_dunder_itruediv() raises:
    print("ndb_ops_test_dunder_itruediv")
    var a = NDBuffer[DType.float32].full(Shape(3, 4), 10.0)
    var b = NDBuffer[DType.float32].full(Shape(3, 4), 2.0)

    a /= b

    for i in range(3):
        for j in range(4):
            assert_true(abs(a[IntArray(i, j)] - 5.0) < 0.001, "__itruediv__")
    print("ndb_ops_test_dunder_itruediv passed")


# ============================================
# Edge Cases
# ============================================


fn ndb_ops_test_scalar_ndbuffer_ops() raises:
    print("ndb_ops_test_scalar_ndbuffer_ops")
    var a = NDBuffer[DType.int32](Shape())
    a[IntArray()] = 10

    var b = NDBuffer[DType.int32](Shape())
    b[IntArray()] = 5

    var result = a.arithmetic_ops[Add](b)
    assert_true(result.item() == 15, "Scalar NDBuffer add")

    a.inplace_ops[Multiply](b)
    assert_true(a.item() == 50, "Scalar NDBuffer inplace mul")
    print("ndb_ops_test_scalar_ndbuffer_ops passed")


fn ndb_ops_test_different_dtypes() raises:
    print("ndb_ops_test_different_dtypes")

    # Float64
    var f64 = NDBuffer[DType.float64].full(Shape(2, 2), 3.14159)
    var f64_result = f64.scalar_ops[Multiply](2.0)
    assert_true(
        abs(f64_result[IntArray(0, 0)] - 6.28318) < 0.0001, "float64 scalar mul"
    )

    # Int64
    var i64 = NDBuffer[DType.int64].full(Shape(2, 2), 1000000)
    var i64_result = i64.scalar_ops[Add](999999)
    assert_true(i64_result[IntArray(0, 0)] == 1999999, "int64 scalar add")

    # Int8
    var i8 = NDBuffer[DType.int8].full(Shape(2, 2), 10)
    var i8_result = i8.scalar_ops[Multiply](2)
    assert_true(i8_result[IntArray(0, 0)] == 20, "int8 scalar mul")

    print("ndb_ops_test_different_dtypes passed")


fn ndb_ops_test_large_buffer() raises:
    print("ndb_ops_test_large_buffer")
    var a = NDBuffer[DType.int32].full(Shape(100, 100), 1)
    var b = NDBuffer[DType.int32].full(Shape(100, 100), 2)

    var result = a.arithmetic_ops[Add](b)

    # Spot check
    assert_true(result[IntArray(0, 0)] == 3, "Large buffer [0,0]")
    assert_true(result[IntArray(50, 50)] == 3, "Large buffer [50,50]")
    assert_true(result[IntArray(99, 99)] == 3, "Large buffer [99,99]")
    print("ndb_ops_test_large_buffer passed")


# ============================================
# Test Runner
# ============================================


fn run_all_ndb_ops_tests() raises:
    print("=" * 60)
    print("Running all NDBuffer ops tests (ndb_ops_*)")
    print("=" * 60)

    # arithmetic_ops tests
    ndb_ops_test_arithmetic_add_same_shape_both_contiguous()
    ndb_ops_test_arithmetic_sub_same_shape_both_contiguous()
    ndb_ops_test_arithmetic_mul_same_shape_both_contiguous()
    ndb_ops_test_arithmetic_div_same_shape_both_contiguous()
    ndb_ops_test_arithmetic_self_contiguous_other_noncontiguous()
    ndb_ops_test_arithmetic_self_noncontiguous_other_contiguous()
    ndb_ops_test_arithmetic_both_noncontiguous()
    ndb_ops_test_arithmetic_with_offset()
    ndb_ops_test_arithmetic_broadcast_row()
    ndb_ops_test_arithmetic_broadcast_col()
    ndb_ops_test_arithmetic_broadcast_scalar()
    ndb_ops_test_arithmetic_1d()
    ndb_ops_test_arithmetic_3d()

    # inplace_ops tests
    ndb_ops_test_inplace_add_same_shape_both_contiguous()
    ndb_ops_test_inplace_sub_same_shape()
    ndb_ops_test_inplace_mul_same_shape()
    ndb_ops_test_inplace_div_same_shape()
    ndb_ops_test_inplace_self_contiguous_other_noncontiguous()
    ndb_ops_test_inplace_self_noncontiguous_other_contiguous()
    ndb_ops_test_inplace_both_noncontiguous()
    ndb_ops_test_inplace_broadcast_row()
    ndb_ops_test_inplace_broadcast_col()
    ndb_ops_test_inplace_with_offset()

    # inplace_scalar_ops tests
    ndb_ops_test_inplace_scalar_add_contiguous()
    ndb_ops_test_inplace_scalar_sub_contiguous()
    ndb_ops_test_inplace_scalar_mul_contiguous()
    ndb_ops_test_inplace_scalar_div_contiguous()
    ndb_ops_test_inplace_scalar_noncontiguous()
    ndb_ops_test_inplace_scalar_with_offset()
    ndb_ops_test_inplace_scalar_1d()
    ndb_ops_test_inplace_scalar_3d()

    # scalar_ops tests
    ndb_ops_test_scalar_add_contiguous()
    ndb_ops_test_scalar_sub_contiguous()
    ndb_ops_test_scalar_mul_contiguous()
    ndb_ops_test_scalar_div_contiguous()
    ndb_ops_test_scalar_reverse_sub_contiguous()
    ndb_ops_test_scalar_reverse_div_contiguous()
    ndb_ops_test_scalar_noncontiguous()
    ndb_ops_test_scalar_with_offset()
    ndb_ops_test_scalar_1d()
    ndb_ops_test_scalar_3d()

    # Operator overloads
    ndb_ops_test_dunder_add()
    ndb_ops_test_dunder_sub()
    ndb_ops_test_dunder_mul()
    ndb_ops_test_dunder_truediv()
    ndb_ops_test_dunder_mul_scalar()
    ndb_ops_test_dunder_rmul_scalar()
    ndb_ops_test_dunder_iadd()
    ndb_ops_test_dunder_isub()
    ndb_ops_test_dunder_imul()
    ndb_ops_test_dunder_itruediv()

    # Edge cases
    ndb_ops_test_scalar_ndbuffer_ops()
    ndb_ops_test_different_dtypes()
    ndb_ops_test_large_buffer()

    print("=" * 60)
    print("All NDBuffer ops tests passed!")
    print("=" * 60)


# ============================================
# copy_from_alike Tests
# ============================================


fn test_copy_from_equal_shaped_both_contiguous_overwrite() raises:
    print("test_copy_from_equal_shaped_both_contiguous_overwrite")
    var dest = NDBuffer[DType.int32].full(Shape(3, 4), 0)
    var src = NDBuffer[DType.int32].full(Shape(3, 4), 42)

    dest.copy_from_alike[overwrite=True](src)

    for i in range(3):
        for j in range(4):
            assert_true(dest[IntArray(i, j)] == 42, "Should be 42")
    print("test_copy_from_equal_shaped_both_contiguous_overwrite passed")


fn test_copy_from_equal_shaped_both_contiguous_add() raises:
    print("test_copy_from_equal_shaped_both_contiguous_add")
    var dest = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var src = NDBuffer[DType.int32].full(Shape(3, 4), 5)

    dest.copy_from_alike[overwrite=False](src)

    for i in range(3):
        for j in range(4):
            assert_true(dest[IntArray(i, j)] == 15, "Should be 10 + 5 = 15")
    print("test_copy_from_equal_shaped_both_contiguous_add passed")


fn test_copy_from_equal_shaped_dest_contiguous_src_noncontiguous_overwrite() raises:
    print(
        "test_copy_from_equal_shaped_dest_contiguous_src_noncontiguous_overwrite"
    )
    var dest = NDBuffer[DType.int32].full(Shape(2, 3), 0)

    # Create non-contiguous source (e.g., a transposed view or strided view)
    # For testing, we'll create a larger buffer and use custom strides
    var src_buffer = Buffer[DType.int32](12)
    for i in range(12):
        src_buffer[i] = i * 10

    # Create NDBuffer with non-default strides (column-major for example)
    # Shape (2, 3) with strides (1, 2) instead of default (3, 1)
    var src = NDBuffer[DType.int32](
        src_buffer^, Shape(2, 3), Strides(1, 2), offset=0  # Non-contiguous
    )

    dest.copy_from_alike[overwrite=True](src)

    # Verify values were copied correctly
    for i in range(2):
        for j in range(3):
            var expected = src[IntArray(i, j)]
            var actual = dest[IntArray(i, j)]
            assert_true(
                actual == expected,
                "Mismatch at " + i.__str__() + "," + j.__str__(),
            )
    print(
        "test_copy_from_equal_shaped_dest_contiguous_src_noncontiguous_overwrite"
        " passed"
    )


fn test_copy_from_equal_shaped_dest_noncontiguous_src_contiguous_overwrite() raises:
    print(
        "test_copy_from_equal_shaped_dest_noncontiguous_src_contiguous_overwrite"
    )

    # Create non-contiguous dest
    var dest_buffer = Buffer[DType.int32](12)
    dest_buffer.fill(0)
    var dest = NDBuffer[DType.int32](
        dest_buffer^, Shape(2, 3), Strides(1, 2), offset=0  # Non-contiguous
    )

    var src = NDBuffer[DType.int32].full(Shape(2, 3), 99)

    dest.copy_from_alike[overwrite=True](src)

    for i in range(2):
        for j in range(3):
            assert_true(
                dest[IntArray(i, j)] == 99,
                "Should be 99 at " + i.__str__() + "," + j.__str__(),
            )
    print(
        "test_copy_from_equal_shaped_dest_noncontiguous_src_contiguous_overwrite"
        " passed"
    )


fn test_copy_from_equal_shaped_both_noncontiguous_overwrite() raises:
    print("test_copy_from_equal_shaped_both_noncontiguous_overwrite")

    # Non-contiguous dest
    var dest_buffer = Buffer[DType.int32](12)
    dest_buffer.fill(0)
    var dest = NDBuffer[DType.int32](
        dest_buffer^, Shape(2, 3), Strides(1, 2), offset=0
    )

    # Non-contiguous src
    var src_buffer = Buffer[DType.int32](12)
    for i in range(12):
        src_buffer[i] = i + 100
    var src = NDBuffer[DType.int32](
        src_buffer^, Shape(2, 3), Strides(1, 2), offset=0
    )

    dest.copy_from_alike[overwrite=True](src)

    for i in range(2):
        for j in range(3):
            var expected = src[IntArray(i, j)]
            var actual = dest[IntArray(i, j)]
            assert_true(
                actual == expected,
                "Mismatch at " + i.__str__() + "," + j.__str__(),
            )
    print("test_copy_from_equal_shaped_both_noncontiguous_overwrite passed")


fn test_copy_from_equal_shaped_both_noncontiguous_add() raises:
    print("test_copy_from_equal_shaped_both_noncontiguous_add")

    # Non-contiguous dest with initial values
    var dest_buffer = Buffer[DType.int32](12)
    dest_buffer.fill(10)
    var dest = NDBuffer[DType.int32](
        dest_buffer^, Shape(2, 3), Strides(1, 2), offset=0
    )

    # Non-contiguous src
    var src_buffer = Buffer[DType.int32](12)
    src_buffer.fill(5)
    var src = NDBuffer[DType.int32](
        src_buffer^, Shape(2, 3), Strides(1, 2), offset=0
    )

    dest.copy_from_alike[overwrite=False](src)

    for i in range(2):
        for j in range(3):
            assert_true(dest[IntArray(i, j)] == 15, "Should be 10 + 5 = 15")
    print("test_copy_from_equal_shaped_both_noncontiguous_add passed")


fn test_copy_from_equal_shaped_with_offset() raises:
    print("test_copy_from_equal_shaped_with_offset")

    # Dest with offset (simulating a view into larger buffer)
    var dest_buffer = Buffer[DType.int32](20)
    dest_buffer.fill(0)
    var dest = NDBuffer[DType.int32](
        dest_buffer^,
        Shape(2, 3),
        Strides.default(Shape(2, 3)),
        offset=5,  # Start at index 5
    )

    # Src with offset
    var src_buffer = Buffer[DType.int32](20)
    for i in range(20):
        src_buffer[i] = i * 10
    var src = NDBuffer[DType.int32](
        src_buffer^,
        Shape(2, 3),
        Strides.default(Shape(2, 3)),
        offset=8,  # Start at index 8
    )

    dest.copy_from_alike[overwrite=True](src)

    # Verify correct values were copied
    for i in range(2):
        for j in range(3):
            var expected = src[IntArray(i, j)]
            var actual = dest[IntArray(i, j)]
            assert_true(
                actual == expected,
                "Mismatch at " + i.__str__() + "," + j.__str__(),
            )
    print("test_copy_from_equal_shaped_with_offset passed")


fn test_copy_from_equal_shaped_1d() raises:
    print("test_copy_from_equal_shaped_1d")
    var dest = NDBuffer[DType.float32].zeros(Shape(10))
    var src = NDBuffer[DType.float32].full(Shape(10), 3.14)

    dest.copy_from_alike[overwrite=True](src)

    for i in range(10):
        assert_true(abs(dest[IntArray(i)] - 3.14) < 0.001, "Should be 3.14")
    print("test_copy_from_equal_shaped_1d passed")


fn test_copy_from_equal_shaped_3d() raises:
    print("test_copy_from_equal_shaped_3d")
    var dest = NDBuffer[DType.int32].zeros(Shape(2, 3, 4))
    var src = NDBuffer[DType.int32](Shape(2, 3, 4))

    # Fill src with pattern
    for i in range(2):
        for j in range(3):
            for k in range(4):
                src[IntArray(i, j, k)] = i * 100 + j * 10 + k

    dest.copy_from_alike[overwrite=True](src)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                var expected = i * 100 + j * 10 + k
                assert_true(
                    dest[IntArray(i, j, k)] == expected, "3D copy mismatch"
                )
    print("test_copy_from_equal_shaped_3d passed")


fn test_copy_from_equal_shaped_scalar() raises:
    print("test_copy_from_equal_shaped_scalar")
    var dest = NDBuffer[DType.int32](Shape())
    dest[IntArray()] = 0

    var src = NDBuffer[DType.int32](Shape())
    src[IntArray()] = 42

    dest.copy_from_alike[overwrite=True](src)

    assert_true(dest.item() == 42, "Scalar copy should work")
    print("test_copy_from_equal_shaped_scalar passed")


# ============================================
# fill(other: NDBuffer) Tests
# ============================================


fn test_fill_same_shape() raises:
    print("test_fill_same_shape")
    var dest = NDBuffer[DType.int32].zeros(Shape(3, 4))
    var src = NDBuffer[DType.int32].full(Shape(3, 4), 77)

    dest.fill(src)

    for i in range(3):
        for j in range(4):
            assert_true(dest[IntArray(i, j)] == 77, "Should be 77")
    print("test_fill_same_shape passed")


fn test_fill_from_scalar_ndbuffer() raises:
    print("test_fill_from_scalar_ndbuffer")
    var dest = NDBuffer[DType.int32].zeros(Shape(3, 4))
    var scalar_src = NDBuffer[DType.int32](Shape())
    scalar_src[IntArray()] = 99

    dest.fill(scalar_src)

    for i in range(3):
        for j in range(4):
            assert_true(dest[IntArray(i, j)] == 99, "Should be 99")
    print("test_fill_from_scalar_ndbuffer passed")


fn test_fill_from_singleton_ndbuffer() raises:
    print("test_fill_from_singleton_ndbuffer")
    var dest = NDBuffer[DType.int32].zeros(Shape(3, 4))
    var singleton_src = NDBuffer[DType.int32].full(Shape(1), 88)

    dest.fill(singleton_src)

    for i in range(3):
        for j in range(4):
            assert_true(dest[IntArray(i, j)] == 88, "Should be 88")
    print("test_fill_from_singleton_ndbuffer passed")


fn test_fill_broadcast_row() raises:
    print("test_fill_broadcast_row")
    var dest = NDBuffer[DType.int32].zeros(Shape(3, 4))
    var src = NDBuffer[DType.int32](Shape(1, 4))
    for j in range(4):
        src[IntArray(0, j)] = j * 10

    dest.fill(src)

    for i in range(3):
        for j in range(4):
            assert_true(
                dest[IntArray(i, j)] == j * 10, "Broadcast row fill mismatch"
            )
    print("test_fill_broadcast_row passed")


fn test_fill_broadcast_col() raises:
    print("test_fill_broadcast_col")
    var dest = NDBuffer[DType.int32].zeros(Shape(3, 4))
    var src = NDBuffer[DType.int32](Shape(3, 1))
    for i in range(3):
        src[IntArray(i, 0)] = i * 100

    dest.fill(src)

    for i in range(3):
        for j in range(4):
            assert_true(
                dest[IntArray(i, j)] == i * 100, "Broadcast col fill mismatch"
            )
    print("test_fill_broadcast_col passed")


fn test_fill_broadcast_3d() raises:
    print("test_fill_broadcast_3d")
    var dest = NDBuffer[DType.int32].zeros(Shape(2, 3, 4))
    var src = NDBuffer[DType.int32](Shape(1, 3, 1))
    for j in range(3):
        src[IntArray(0, j, 0)] = j + 1

    dest.fill(src)

    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_true(
                    dest[IntArray(i, j, k)] == j + 1,
                    "3D broadcast fill mismatch",
                )
    print("test_fill_broadcast_3d passed")


fn test_fill_self_panics() raises:
    print("test_fill_self_panics")
    var _ndb = NDBuffer[DType.int32].full(Shape(3, 4), 42)

    # This should panic - we can't easily test panics, so just document behavior
    # ndb.fill(ndb)  # Would panic with "cannot fill with self"

    print("test_fill_self_panics passed (manual verification needed)")


fn test_fill_preserves_dest_structure() raises:
    print("test_fill_preserves_dest_structure")

    # Non-contiguous dest
    var dest_buffer = Buffer[DType.int32](12)
    dest_buffer.fill(0)
    var dest = NDBuffer[DType.int32](
        dest_buffer^, Shape(2, 3), Strides(1, 2), offset=0
    )

    var src = NDBuffer[DType.int32].full(Shape(2, 3), 55)

    dest.fill(src)

    # Verify fill worked with non-contiguous dest
    for i in range(2):
        for j in range(3):
            assert_true(
                dest[IntArray(i, j)] == 55, "Non-contiguous dest fill mismatch"
            )
    print("test_fill_preserves_dest_structure passed")


fn test_fill_different_dtypes() raises:
    print("test_fill_different_dtypes")

    # Float32
    var dest_f32 = NDBuffer[DType.float32].zeros(Shape(2, 2))
    var src_f32 = NDBuffer[DType.float32].full(Shape(2, 2), 3.14)
    dest_f32.fill(src_f32)
    assert_true(abs(dest_f32[IntArray(0, 0)] - 3.14) < 0.001, "float32 fill")

    # Float64
    var dest_f64 = NDBuffer[DType.float64].zeros(Shape(2, 2))
    var src_f64 = NDBuffer[DType.float64].full(Shape(2, 2), 2.718)
    dest_f64.fill(src_f64)
    assert_true(abs(dest_f64[IntArray(0, 0)] - 2.718) < 0.001, "float64 fill")

    # Int64
    var dest_i64 = NDBuffer[DType.int64].zeros(Shape(2, 2))
    var src_i64 = NDBuffer[DType.int64].full(Shape(2, 2), 9999)
    dest_i64.fill(src_i64)
    assert_true(dest_i64[IntArray(0, 0)] == 9999, "int64 fill")

    print("test_fill_different_dtypes passed")


# ============================================
# Test Runner
# ============================================


fn run_all_copy_fill_tests() raises:
    print("=" * 60)
    print("Running copy_from_alike and fill tests")
    print("=" * 60)

    # copy_from_alike tests
    test_copy_from_equal_shaped_both_contiguous_overwrite()
    test_copy_from_equal_shaped_both_contiguous_add()
    test_copy_from_equal_shaped_dest_contiguous_src_noncontiguous_overwrite()
    test_copy_from_equal_shaped_dest_noncontiguous_src_contiguous_overwrite()
    test_copy_from_equal_shaped_both_noncontiguous_overwrite()
    test_copy_from_equal_shaped_both_noncontiguous_add()
    test_copy_from_equal_shaped_with_offset()
    test_copy_from_equal_shaped_1d()
    test_copy_from_equal_shaped_3d()
    test_copy_from_equal_shaped_scalar()

    # fill(other: NDBuffer) tests
    test_fill_same_shape()
    test_fill_from_scalar_ndbuffer()
    test_fill_from_singleton_ndbuffer()
    test_fill_broadcast_row()
    test_fill_broadcast_col()
    test_fill_broadcast_3d()
    test_fill_self_panics()
    test_fill_preserves_dest_structure()
    test_fill_different_dtypes()

    print("=" * 60)
    print("All copy/fill tests passed!")
    print("=" * 60)


# ============================================
# Constructor Tests
# ============================================
fn test_ndbuffer_from_varargs() raises:
    print("test_ndbuffer_from_varargs")
    var ndb = NDBuffer[DType.float32](1.0, 2.0, 3.0, 4.0)
    assert_true(ndb.shape == Shape(4), "Shape should be (4,)")
    assert_true(ndb[IntArray(0)] == 1.0, "First element")
    assert_true(ndb[IntArray(3)] == 4.0, "Last element")
    print("test_ndbuffer_from_varargs passed")


fn test_ndbuffer_from_shape() raises:
    print("test_ndbuffer_from_shape")
    var ndb = NDBuffer[DType.int32](Shape(3, 4))
    assert_true(ndb.shape == Shape(3, 4), "Shape mismatch")
    assert_true(ndb.numels() == 12, "Numels mismatch")
    assert_true(ndb.rank() == 2, "Rank mismatch")
    print("test_ndbuffer_from_shape passed")


fn test_ndbuffer_zeros() raises:
    print("test_ndbuffer_zeros")
    var ndb = NDBuffer[DType.float32].zeros(Shape(2, 3))
    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(ndb[idx] == 0.0, "Should be zero")
    print("test_ndbuffer_zeros passed")


fn test_ndbuffer_full() raises:
    print("test_ndbuffer_full")
    var ndb = NDBuffer[DType.int32].full(Shape(2, 3), 42)
    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(ndb[idx] == 42, "Should be 42")
    print("test_ndbuffer_full passed")


# ============================================
# Indexing Tests
# ============================================
fn test_ndbuffer_getitem_setitem() raises:
    print("test_ndbuffer_getitem_setitem")
    var ndb = NDBuffer[DType.int32](Shape(3, 4))

    for i in range(3):
        for j in range(4):
            var idx = IntArray(i, j)
            ndb[idx] = i * 10 + j

    for i in range(3):
        for j in range(4):
            var idx = IntArray(i, j)
            assert_true(
                ndb[idx] == i * 10 + j,
                "Value mismatch at " + i.__str__() + "," + j.__str__(),
            )
    print("test_ndbuffer_getitem_setitem passed")


fn test_ndbuffer_item_scalar() raises:
    print("test_ndbuffer_item_scalar")
    var ndb = NDBuffer[DType.float32](Shape(1))
    ndb[IntArray(0)] = 42.5
    assert_true(ndb.item() == 42.5, "item() should return scalar value")
    print("test_ndbuffer_item_scalar passed")


# ============================================
# Arithmetic Tests
# ============================================
fn test_ndbuffer_add() raises:
    print("test_ndbuffer_add")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 10)
    var b = NDBuffer[DType.int32].full(Shape(2, 3), 5)
    var result = a + b

    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(result[idx] == 15, "Add result should be 15")
    print("test_ndbuffer_add passed")


fn test_ndbuffer_mul() raises:
    print("test_ndbuffer_mul")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 4)
    var b = NDBuffer[DType.int32].full(Shape(2, 3), 3)
    var result = a * b

    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(result[idx] == 12, "Mul result should be 12")
    print("test_ndbuffer_mul passed")


fn test_ndbuffer_scalar_mul() raises:
    print("test_ndbuffer_scalar_mul")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 5)
    var result = a * 3

    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(result[idx] == 15, "Scalar mul result should be 15")
    print("test_ndbuffer_scalar_mul passed")


fn test_ndbuffer_iadd() raises:
    print("test_ndbuffer_iadd")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 10)
    var b = NDBuffer[DType.int32].full(Shape(2, 3), 5)
    a += b

    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(a[idx] == 15, "iadd result should be 15")
    print("test_ndbuffer_iadd passed")


# ============================================
# Broadcasting Tests
# ============================================
fn test_ndbuffer_broadcast_scalar() raises:
    print("test_ndbuffer_broadcast_scalar")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 10)
    var b = NDBuffer[DType.int32].full(Shape(), 5)  # Scalar
    var result = a + b

    assert_true(result.shape == Shape(2, 3), "Broadcast shape")
    for i in range(2):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(result[idx] == 15, "Broadcast add result should be 15")
    print("test_ndbuffer_broadcast_scalar passed")


fn test_ndbuffer_broadcast_row() raises:
    print("test_ndbuffer_broadcast_row")
    var a = NDBuffer[DType.int32].full(Shape(3, 4), 10)
    var b = NDBuffer[DType.int32](Shape(1, 4))
    for j in range(4):
        b[IntArray(0, j)] = j

    var result = a + b
    assert_true(result.shape == Shape(3, 4), "Broadcast shape")

    for i in range(3):
        for j in range(4):
            var idx = IntArray(i, j)
            assert_true(result[idx] == 10 + j, "Broadcast row add")
    print("test_ndbuffer_broadcast_row passed")


fn test_ndbuffer_broadcast_to() raises:
    print("test_ndbuffer_broadcast_to")
    var a = NDBuffer[DType.int32].full(Shape(1, 3), 5)
    var result = a.broadcast_to(Shape(4, 3))

    assert_true(result.shape == Shape(4, 3), "broadcast_to shape")
    for i in range(4):
        for j in range(3):
            var idx = IntArray(i, j)
            assert_true(result[idx] == 5, "broadcast_to value")
    print("test_ndbuffer_broadcast_to passed")


# ============================================
# Reduction Tests
# ============================================
fn test_ndbuffer_sum_all() raises:
    print("test_ndbuffer_sum_all")
    var ndb = NDBuffer[DType.int32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = val
            val += 1
    # Values: 1,2,3,4,5,6 -> sum = 21

    var result = ndb.sum_all()
    assert_true(result == 21, "sum_all should be 21")
    print("test_ndbuffer_sum_all passed")


fn test_ndbuffer_sum_axis() raises:
    print("test_ndbuffer_sum_axis")
    var ndb = NDBuffer[DType.int32](Shape(2, 3))
    # Row 0: 1, 2, 3
    # Row 1: 4, 5, 6
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = val
            val += 1

    # Sum over axis 0 (rows) -> shape (3,)
    var result = ndb.sum(IntArray(0), keepdims=False)
    assert_true(result.shape == Shape(3), "Sum axis 0 shape")
    assert_true(result[IntArray(0)] == 5, "Sum col 0: 1+4=5")
    assert_true(result[IntArray(1)] == 7, "Sum col 1: 2+5=7")
    assert_true(result[IntArray(2)] == 9, "Sum col 2: 3+6=9")
    print("test_ndbuffer_sum_axis passed")


fn test_ndbuffer_sum_axis_keepdims() raises:
    print("test_ndbuffer_sum_axis_keepdims")
    var ndb = NDBuffer[DType.int32](Shape(2, 3))
    var val = 1
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = val
            val += 1

    var result = ndb.sum(IntArray(0), keepdims=True)
    assert_true(result.shape == Shape(1, 3), "Sum keepdims shape")
    print("test_ndbuffer_sum_axis_keepdims passed")


# ============================================
# Contiguity Tests
# ============================================
fn test_ndbuffer_contiguous() raises:
    print("test_ndbuffer_contiguous")
    var ndb = NDBuffer[DType.int32](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = i * 10 + j

    assert_true(ndb.is_contiguous(), "Should be contiguous")

    var cont = ndb.contiguous()
    assert_true(
        cont.is_contiguous(), "contiguous() result should be contiguous"
    )
    assert_true(cont.shape == ndb.shape, "Shape should match")
    print("test_ndbuffer_contiguous passed")


fn test_ndbuffer_contiguous_buffer() raises:
    print("test_ndbuffer_contiguous_buffer")
    var ndb = NDBuffer[DType.int32](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            ndb[IntArray(i, j)] = i * 10 + j

    var buf = ndb.contiguous_buffer()
    assert_true(buf.size == 6, "Contiguous buffer size")
    assert_true(buf[0] == 0, "First element")
    assert_true(buf[5] == 12, "Last element (1*10+2)")
    print("test_ndbuffer_contiguous_buffer passed")


# ============================================
# Comparison Tests
# ============================================
fn test_ndbuffer_eq() raises:
    print("test_ndbuffer_eq")
    var a = NDBuffer[DType.int32].full(Shape(2, 3), 42)
    var b = NDBuffer[DType.int32].full(Shape(2, 3), 42)
    var c = NDBuffer[DType.int32].full(Shape(2, 3), 0)

    assert_true(a == b, "Equal buffers should be ==")
    assert_true(not (a == c), "Different buffers should not be ==")
    print("test_ndbuffer_eq passed")


fn test_ndbuffer_compare_elementwise() raises:
    print("test_ndbuffer_compare_elementwise")
    var a = NDBuffer[DType.int32](Shape(4))
    for i in range(4):
        a[IntArray(i)] = i  # 0, 1, 2, 3

    var result = a.compare_scalar[LessThan](2)
    assert_true(result[IntArray(0)] == True, "0 < 2")
    assert_true(result[IntArray(1)] == True, "1 < 2")
    assert_true(result[IntArray(2)] == False, "2 < 2")
    assert_true(result[IntArray(3)] == False, "3 < 2")
    print("test_ndbuffer_compare_elementwise passed")


fn test_ndbuffer_all_close() raises:
    print("test_ndbuffer_all_close")
    var a = NDBuffer[DType.float32].full(Shape(2, 3), 1.0)
    var b = NDBuffer[DType.float32].full(Shape(2, 3), 1.0 + 1e-9)
    var c = NDBuffer[DType.float32].full(Shape(2, 3), 2.0)

    assert_true(a.all_close(b), "Should be close")
    assert_true(not a.all_close(c), "Should not be close")
    print("test_ndbuffer_all_close passed")


# ============================================
# Utility Tests
# ============================================
fn test_ndbuffer_fill() raises:
    print("test_ndbuffer_fill")
    var ndb = NDBuffer[DType.int32].zeros(Shape(2, 3))
    ndb.fill(99)

    for i in range(2):
        for j in range(3):
            assert_true(ndb[IntArray(i, j)] == 99, "Fill value")
    print("test_ndbuffer_fill passed")


fn test_ndbuffer_count() raises:
    print("test_ndbuffer_count")
    var ndb = NDBuffer[DType.int32](Shape(3, 4))
    for i in range(3):
        for j in range(4):
            ndb[IntArray(i, j)] = (i + j) % 3

    var count_0 = ndb.count(0)
    var count_1 = ndb.count(1)
    var count_2 = ndb.count(2)

    assert_true(count_0 + count_1 + count_2 == 12, "Total count should be 12")
    print("test_ndbuffer_count passed")


fn test_ndbuffer_flatten() raises:
    print("test_ndbuffer_flatten")
    var ndb = NDBuffer[DType.int32](Shape(2, 3, 4))
    for i in range(2):
        for j in range(3):
            for k in range(4):
                ndb[IntArray(i, j, k)] = i * 100 + j * 10 + k

    var flat = ndb.flatten()
    assert_true(flat.shape == Shape(24), "Flatten shape")
    assert_true(flat.numels() == 24, "Flatten numels")
    print("test_ndbuffer_flatten passed")


fn test_ndbuffer_flatten_partial() raises:
    print("test_ndbuffer_flatten_partial")
    var ndb = NDBuffer[DType.int32](Shape(2, 3, 4))

    var flat = ndb.flatten(1, 2)  # Flatten dims 1 and 2
    assert_true(flat.shape == Shape(2, 12), "Partial flatten shape")
    print("test_ndbuffer_flatten_partial passed")


fn test_ndbuffer_to_dtype() raises:
    print("test_ndbuffer_to_dtype")
    var ndb = NDBuffer[DType.int32].full(Shape(2, 3), 42)
    var result = ndb.to_dtype[DType.float32]()

    assert_true(result.shape == Shape(2, 3), "Shape preserved")
    for i in range(2):
        for j in range(3):
            assert_true(result[IntArray(i, j)] == 42.0, "Value converted")
    print("test_ndbuffer_to_dtype passed")


# ============================================
# Sharing Tests
# ============================================
fn test_ndbuffer_share() raises:
    print("test_ndbuffer_share")
    var ndb = NDBuffer[DType.int32].full(Shape(2, 3), 42)
    assert_true(not ndb.shared(), "Initially not shared")

    var view = ndb.share()
    assert_true(ndb.shared(), "Now shared")
    assert_true(view.shared(), "View is shared")

    # Modify original
    ndb[IntArray(0, 0)] = 99
    assert_true(view[IntArray(0, 0)] == 99, "View sees modification")
    print("test_ndbuffer_share passed")


# ============================================
# SIMD Load/Store Tests
# ============================================
fn test_ndbuffer_simd_load_store() raises:
    print("test_ndbuffer_simd_load_store")
    var ndb = NDBuffer[DType.float32](Shape(4, 8))

    # Fill with pattern
    for i in range(4):
        for j in range(8):
            ndb[IntArray(i, j)] = Scalar[DType.float32](i * 10 + j)

    # SIMD load row 1, starting at col 0, width 4
    var vec = ndb.load[4](1, 0)
    assert_true(vec[0] == 10.0, "SIMD load [0]")
    assert_true(vec[1] == 11.0, "SIMD load [1]")
    assert_true(vec[2] == 12.0, "SIMD load [2]")
    assert_true(vec[3] == 13.0, "SIMD load [3]")

    # SIMD store
    var new_vec = SIMD[DType.float32, 4](100.0, 101.0, 102.0, 103.0)
    ndb.store[4](2, 0, new_vec)
    assert_true(ndb[IntArray(2, 0)] == 100.0, "SIMD store [0]")
    assert_true(ndb[IntArray(2, 3)] == 103.0, "SIMD store [3]")
    print("test_ndbuffer_simd_load_store passed")


# ============================================
# Test Runner
# ============================================
fn run_all_ndbuffer_tests() raises:
    print("=" * 60)
    print("Running all NDBuffer tests")
    print("=" * 60)

    # Constructors
    test_ndbuffer_from_varargs()
    test_ndbuffer_from_shape()
    test_ndbuffer_zeros()
    test_ndbuffer_full()

    # Indexing
    test_ndbuffer_getitem_setitem()
    test_ndbuffer_item_scalar()

    # Arithmetic
    test_ndbuffer_add()
    test_ndbuffer_mul()
    test_ndbuffer_scalar_mul()
    test_ndbuffer_iadd()

    # Broadcasting
    test_ndbuffer_broadcast_scalar()
    test_ndbuffer_broadcast_row()
    test_ndbuffer_broadcast_to()

    # Reductions
    test_ndbuffer_sum_all()
    test_ndbuffer_sum_axis()
    test_ndbuffer_sum_axis_keepdims()

    # Contiguity
    test_ndbuffer_contiguous()
    test_ndbuffer_contiguous_buffer()

    # Comparisons
    test_ndbuffer_eq()
    test_ndbuffer_compare_elementwise()
    test_ndbuffer_all_close()

    # Utilities
    test_ndbuffer_fill()
    test_ndbuffer_count()
    test_ndbuffer_flatten()
    test_ndbuffer_flatten_partial()
    test_ndbuffer_to_dtype()

    # Sharing
    test_ndbuffer_share()

    # SIMD
    test_ndbuffer_simd_load_store()

    print("=" * 60)
    print("All NDBuffer tests passed!")
    print("=" * 60)


fn main() raises:
    var runs = 1
    alias _dtype = DType.float32

    for _ in range(runs):
        test_ndbuffer_set_get()
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
        test_ndbuffer_fill_orig()
        test_buffer_sum_all()
        test_buffer_sum()
        # Consolidated
        run_all_ndbuffer_tests()
        run_all_copy_fill_tests()
        run_all_ndb_ops_tests()


fn test_buffer_sum() raises:
    print("test_buffer_sum")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))
    result = ndb.sum(IntArray(0), True)
    assert_true(
        result.data_buffer() == Buffer[dtype]([21, 24, 27, 30, 33, 36, 39])
    )

    result = ndb.sum(IntArray(0), False)
    assert_true(
        result.data_buffer() == Buffer[dtype]([21, 24, 27, 30, 33, 36, 39])
    )

    result = ndb.sum(IntArray(0, 1), True)
    assert_true(result.data_buffer() == Buffer[dtype]([210]))

    result = ndb.sum(IntArray(1), True)
    assert_true(result.data_buffer() == Buffer[dtype]([21, 70, 119]))


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
    assert_true(ndb.sum_all() == 39 and ndb[IntArray(0)] == 39)
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
    ndb.data_buffer().overwrite(result, 3, 6)


fn test_compare_buffer() raises:
    print("test_compare_buffer")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb2 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 3, 4, 6]), Shape(2, 3))
    result = ndb.compare[GreaterThan](ndb2)
    assert_true(
        result.data_buffer()
        == Buffer[DType.bool]([False, False, False, True, True, False])
    )


fn test_compare_scalar() raises:
    print("test_compare_scalar")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    result = ndb.compare_scalar[GreaterThan](3)
    assert_true(
        result.data_buffer()
        == Buffer[DType.bool]([False, False, False, True, True, True])
    )

    shared = ndb.share(Shape(1, 3), strides=Strides(1, 2), offset=1)
    result = shared.compare_scalar[Equal](4)
    assert_true(
        result.data_buffer() == Buffer[DType.bool]([False, True, False])
    )


fn test_inplace_broadcast_operations() raises:
    print("test_inplace_broadcast_operations")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb2 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3]), Shape(3))
    ndb += ndb2
    assert_true(ndb.data_buffer() == Buffer[dtype]([2, 4, 6, 5, 7, 9]))

    ndb -= ndb2
    assert_true(ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6]))

    ndb_shared = ndb.share()
    ndb2_shared = ndb2.share()

    ndb_shared += ndb2_shared
    assert_true(ndb.data_buffer() == Buffer[dtype]([2, 4, 6, 5, 7, 9]))

    ndb_shared -= ndb2_shared
    assert_true(ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6]))


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
    assert_true(
        ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2
    )
    ndb -= ndb2
    assert_true(ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    ndb *= ndb2
    assert_true(
        ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2
    )
    ndb /= ndb2
    assert_true(ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared = ndb.share()

    ndb += ndb2
    assert_true(
        shared.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2
    )
    ndb -= ndb2
    assert_true(
        shared.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    ndb *= ndb2
    assert_true(
        shared.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2
    )
    ndb /= ndb2
    assert_true(
        shared.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9])
    )

    shared2 = ndb.share(Shape(2, 3), offset=3)
    ndb2_shared = ndb2.share(Shape(2, 3))

    shared2 += ndb2_shared

    assert_true(
        ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 5, 7, 9, 11, 13, 15])
    )
    shared2 -= ndb2_shared

    assert_true(ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared3 = ndb.share(Shape(1, 3), offset=3, strides=Strides(1, 2))
    shared4 = ndb2.share(Shape(1, 3), strides=Strides(1, 3))

    shared3 += shared4

    assert_true(
        ndb.data_buffer() == Buffer[dtype]([1, 2, 3, 5, 5, 10, 7, 15, 9])
    )


fn test_unique() raises:
    print("test_unique")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([2, 2, 3, 4, 2, 6]), Shape(2, 3))
    assert_true(ndb.unique().data_buffer() == Buffer[dtype]([2, 3, 4, 6]))


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
    assert_true(
        ndb.data_buffer() == Buffer[dtype]([100, 101, 102, 103, 104, 105])
    )
    shared = ndb.share(Shape(3, 1), offset=3)
    shared.inplace_scalar_ops[Add](10)
    assert_true(
        ndb.data_buffer() == Buffer[dtype]([100, 101, 102, 113, 114, 115])
    )

    shared2 = ndb.share(Shape(1, 3), offset=0, strides=Strides(1, 2))
    shared2.inplace_scalar_ops[Add](100)
    assert_true(
        ndb.data_buffer() == Buffer[dtype]([200, 101, 202, 113, 214, 115])
    )


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
    assert_true(result.data_buffer() == Buffer[dtype]([46, 47, 48]))


fn test_dtype_conversion() raises:
    print("test_dtype_conversion")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb_shared = ndb.share(Shape(1, 3), offset=3)
    converted = ndb_shared.to_dtype[DType.float64]()

    assert_true(
        converted.data_buffer() == Buffer[DType.float64]([4, 5, 6])
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
    assert_true(result.data_buffer() == Buffer[DType.bool]([True, False, True]))


fn test_add() raises:
    print("test_add")
    alias dtype = DType.float32
    ndb1 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb1_shared = ndb1.share(Shape(1, 3), offset=3)
    ndb2 = NDBuffer[dtype](Buffer[dtype]([10, 20, 30]), Shape(1, 3))

    result = ndb1_shared + ndb2
    assert_true(
        result.data_buffer() == Buffer[dtype]([14, 25, 36])
        and result.shared() == False
    )


fn test_zero() raises:
    print("test_zero")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    ndb.fill(42)
    shared = ndb.share(Shape(3), offset=3)
    shared.zero()
    assert_true(ndb.data_buffer() == Buffer[dtype]([42, 42, 42, 0, 0, 0]))


fn test_broadcast_fill() raises:
    print("test_broadcast_fill")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    filler = NDBuffer[dtype](Shape(2, 1))
    filler.fill(42)
    ndb.fill(filler)
    assert_true(ndb.data_buffer() == Buffer[dtype]([42, 42, 42, 42, 42, 42]))

    filler.fill(89)
    shared = filler.share()
    ndb.fill(shared)
    assert_true(ndb.data_buffer() == Buffer[dtype]([89, 89, 89, 89, 89, 89]))


fn test_fill_2() raises:
    print("test_fill_2")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    filler = NDBuffer[dtype](Shape(2, 3))
    filler.fill(91)
    ndb.fill(filler)
    assert_true(ndb.data_buffer() == Buffer[dtype].full(91, 6))

    shared1 = ndb.share(Shape(3), offset=3)
    filler = NDBuffer[dtype](Shape(3))
    filler.fill(92)
    shared1.fill(filler)

    assert_true(
        shared1.data_buffer() == Buffer[dtype]([91, 91, 91, 92, 92, 92])
    )
    assert_true(ndb.data_buffer() == Buffer[dtype]([91, 91, 91, 92, 92, 92]))

    # Left contiguous, right non-contiguous
    ndb = NDBuffer[dtype](Shape(2, 2))
    filler = NDBuffer[dtype](Shape(2, 1, 4))
    filler.fill(102)
    filler_shared = filler.share(Shape(2, 2), offset=4)
    ndb.fill(filler_shared)

    assert_true(ndb.data_buffer() == Buffer[dtype]([102, 102, 102, 102]))
    # Both shared
    ndb = NDBuffer[dtype](Shape(2, 2))
    filler_shared.fill(31)
    ndb_shared = ndb.share()
    ndb_shared.fill(filler_shared)

    assert_true(ndb.data_buffer() == Buffer[dtype]([31, 31, 31, 31]))

    filler = NDBuffer[dtype](Shape(2, 1, 4))
    filler.fill(1919)

    filler_shared = filler.share(Shape(2, 2), strides=Strides(1, 2))

    ndb_shared.fill(filler_shared)

    assert_true(
        ndb.data_buffer() == Buffer[dtype]([1919, 1919, 1919, 1919])
        and not filler_shared._contiguous,
    )
    # Left non-contiguous and right contiguous
    filler1 = NDBuffer[dtype](Shape(2, 2))
    filler1.fill(47)

    ndb1 = NDBuffer[dtype](Shape(2, 1, 4))
    ndb1.fill(1)
    ndb_shared1 = ndb1.share(Shape(2, 2), strides=Strides(1, 2), offset=1)
    ndb_shared1.fill(filler1)

    assert_true(
        ndb1.data_buffer() == Buffer[dtype]([1, 47, 47, 47, 47, 1, 1, 1])
    )

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
        ndb1.data_buffer()
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
    assert_true(
        ndb_shared.data_buffer() == Buffer[dtype]([1, 2, 42, 42, 42, 6, 7, 8])
    )


fn test_ndbuffer_fill_orig() raises:
    print("test_ndbuffer_fill")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(8))
    ndb.fill(42)
    expected = Buffer[dtype].full(42, 8)
    assert_true(
        ndb.data_buffer() == expected, "NDBuffer fill assertion 1 failed"
    )
    assert_false(ndb.shared(), "NDBuffer not shared assertion failed")
    shared = ndb.share()
    assert_true(ndb.shared(), "NDBuffer shared assertion failed - post sharing")
    shared.fill(91)
    expected = Buffer[dtype].full(91, 8)
    assert_true(
        ndb.data_buffer() == expected, "NDBuffer fill assertion 2 failed"
    )
    share2 = ndb.share(Shape(3), Strides(2), offset=2)
    share2.fill(81)
    var l: List[Scalar[dtype]] = [91, 91, 81, 91, 81, 91, 81, 91]

    expected = Buffer[dtype](l)
    assert_true(
        share2.data_buffer() == expected
        and ndb.data_buffer() == expected
        and shared.data_buffer() == expected,
        "Fill via shape, strides and offset failed",
    )
    ndb = NDBuffer[dtype](Shape(1))
    filler = NDBuffer[dtype](Shape(1))
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
        ndb_bool.data_buffer()
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
    assert_true(
        result.data_buffer() == (Buffer[dtype]([42, 42, 42, 42, 42, 42]) + 3)
    )

    result = result.arithmetic_ops[Subtract](ndbuffer2)
    assert_true(result.data_buffer() == Buffer[dtype]([42, 42, 42, 42, 42, 42]))


fn test_ndbuffer_inplace_ops() raises:
    print("test_ndbuffer_inplace_ops")

    alias dtype = DType.float32
    buffer1 = Buffer[dtype](30)
    buffer1.fill(42)
    shape = Shape(5, 6)
    ndbuffer1 = NDBuffer[dtype](buffer1^, shape, None)
    index1 = IntArray(4, 5)
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
        ndbuffer1.data_buffer() == expected, "In place add failed for NDBuffer"
    )

    shared_buffer = ndbuffer1.share(shape1)
    assert_true(
        shared_buffer.data_buffer() == expected, "NDBuffer sharing failed"
    )
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
