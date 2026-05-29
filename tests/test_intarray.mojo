from tenmo.intarray import IntArray
from std.testing import assert_equal, assert_true, assert_false, TestSuite

# ========== Construction Tests ==========


def test_ia2_empty_construction() raises:
    """Test empty IntArray construction."""
    var ia = IntArray()
    assert_equal(len(ia), 0)
    # assert_equal(ia.capacity(), IntArray.SMALL_SIZE)
    assert_equal(ia.capacity(), 0)
    assert_true(ia.is_empty())


def test_ia2_variadic_construction() raises:
    """Test variadic constructor."""
    var ia1 = IntArray(5)
    assert_equal(len(ia1), 1)
    assert_equal(ia1[0], 5)

    var ia2 = IntArray(1, 2, 3, 4, 5)
    assert_equal(len(ia2), 5)
    assert_equal(ia2[0], 1)
    assert_equal(ia2[4], 5)

    var ia3 = IntArray(-1, -2, -3)
    assert_equal(len(ia3), 3)
    assert_equal(ia3[0], -1)


def test_ia2_list_construction() raises:
    """Test List[Int] constructor."""
    var lst: List[Int] = [1, 2, 3, 4]
    var ia = IntArray(lst)
    assert_equal(len(ia), 4)
    assert_equal(ia[0], 1)
    assert_equal(ia[3], 4)


def test_ia2_filled_construction() raises:
    """Test filled() static constructor."""
    var ia = IntArray.filled(5, 42)
    assert_equal(len(ia), 5)
    for i in range(5):
        assert_equal(ia[i], 42)

    var ia_empty = IntArray.filled(0, 99)
    assert_equal(len(ia_empty), 0)


def test_ia2_range_construction() raises:
    """Test range() static constructor."""
    # Positive step
    var ia1 = IntArray.range(0, 5)
    assert_equal(len(ia1), 5)
    assert_equal(ia1[0], 0)
    assert_equal(ia1[4], 4)

    # Custom step
    var ia2 = IntArray.range(0, 10, 2)
    assert_equal(len(ia2), 5)
    assert_equal(ia2[0], 0)
    assert_equal(ia2[4], 8)

    # Negative step
    var ia3 = IntArray.range(10, 0, -2)
    assert_equal(len(ia3), 5)
    assert_equal(ia3[0], 10)
    assert_equal(ia3[4], 2)

    # Empty range
    var ia4 = IntArray.range(5, 2)
    assert_equal(len(ia4), 0)

    # Negative to positive
    var ia5 = IntArray.range(-3, 3)
    assert_equal(len(ia5), 6)
    assert_equal(ia5[0], -3)
    assert_equal(ia5[5], 2)


def test_ia2_with_capacity() raises:
    """Test with_capacity() constructor."""
    var ia = IntArray.with_capacity(10)
    assert_equal(len(ia), 0)
    assert_equal(ia.capacity(), 10)
    assert_true(ia.is_empty())


def test_ia2_copy_construction() raises:
    """Test copy constructor (deep copy)."""
    var ia1 = IntArray(1, 2, 3)
    var ia2 = ia1
    assert_equal(len(ia2), 3)
    assert_equal(ia2[0], 1)

    # Modify ia2, ia1 should be unchanged (deep copy)
    ia2[0] = 99
    assert_equal(ia1[0], 1)  # Original unchanged
    assert_equal(ia2[0], 99)  # Copy modified


# ========== Access Tests ==========


def test_ia2_positive_indexing() raises:
    """Test positive index access."""
    var ia = IntArray(10, 20, 30, 40, 50)
    assert_equal(ia[0], 10)
    assert_equal(ia[2], 30)
    assert_equal(ia[4], 50)


def test_ia2_negative_indexing() raises:
    """Test negative index access."""
    var ia = IntArray(10, 20, 30, 40, 50)
    assert_equal(ia[-1], 50)
    assert_equal(ia[-2], 40)
    assert_equal(ia[-5], 10)


def test_ia2_setitem() raises:
    """Test element assignment."""
    var ia = IntArray(1, 2, 3, 4, 5)
    ia[0] = 100
    ia[4] = 500
    ia[-2] = 400
    assert_equal(ia[0], 100)
    assert_equal(ia[4], 500)
    assert_equal(ia[3], 400)


def test_ia2_slicing_basic() raises:
    """Test basic slicing."""
    var ia = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # Forward slice
    var s1 = ia[2:5]
    assert_equal(len(s1), 3)
    assert_equal(s1[0], 2)
    assert_equal(s1[2], 4)

    # Slice to end
    var s2 = ia[7:]
    assert_equal(len(s2), 3)
    assert_equal(s2[0], 7)

    # Slice from start
    var s3 = ia[:3]
    assert_equal(len(s3), 3)
    assert_equal(s3[2], 2)

    # Full slice
    var s4 = ia[:]
    assert_equal(len(s4), 10)


def test_ia2_slicing_step() raises:
    """Test slicing with step."""
    var ia = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # Every 2nd element
    var s1 = ia[::2]
    assert_equal(len(s1), 5)
    assert_equal(s1[0], 0)
    assert_equal(s1[4], 8)

    # Reverse
    var s2 = ia[::-1]
    assert_equal(len(s2), 10)
    assert_equal(s2[0], 9)
    assert_equal(s2[9], 0)

    # Reverse with step
    var s3 = ia[8:2:-2]
    assert_equal(len(s3), 3)
    assert_equal(s3[0], 8)
    assert_equal(s3[2], 4)


def test_ia2_slicing_negative_indices() raises:
    """Test slicing with negative indices."""
    var ia = IntArray(0, 1, 2, 3, 4, 5)

    var s1 = ia[-3:-1]
    assert_equal(len(s1), 2)
    assert_equal(s1[0], 3)
    assert_equal(s1[1], 4)

    var s2 = ia[-5:]
    assert_equal(len(s2), 5)
    assert_equal(s2[0], 1)


# ========== Growth Operations Tests ==========


def test_ia2_append() raises:
    """Test append operation."""
    var ia = IntArray()
    assert_equal(len(ia), 0)

    ia.append(1)
    assert_equal(len(ia), 1)
    assert_equal(ia[0], 1)

    ia.append(2)
    ia.append(3)
    assert_equal(len(ia), 3)
    assert_equal(ia[2], 3)

    # Test capacity growth
    for i in range(100):
        ia.append(i)
    assert_equal(len(ia), 103)


def test_ia2_prepend() raises:
    """Test prepend operation."""
    var ia = IntArray()
    ia.prepend(3)
    assert_equal(len(ia), 1)
    assert_equal(ia[0], 3)

    ia.prepend(2)
    ia.prepend(1)
    assert_equal(len(ia), 3)
    assert_equal(ia[0], 1)
    assert_equal(ia[1], 2)
    assert_equal(ia[2], 3)


def test_ia2__add__() raises:
    """Test __add__ (immutable append)."""
    var ia1 = IntArray(1, 2, 3)
    var ia2 = ia1 + 4

    assert_equal(len(ia1), 3)  # Original unchanged
    assert_equal(len(ia2), 4)  # New array extended
    assert_equal(ia2[3], 4)


def test_ia2_pop() raises:
    """Test pop operation."""
    var ia = IntArray(1, 2, 3, 4, 5)

    # Pop last
    var val1 = ia.pop()
    assert_equal(val1, 5)
    assert_equal(len(ia), 4)

    # Pop specific index
    var val2 = ia.pop(1)
    assert_equal(val2, 2)
    assert_equal(len(ia), 3)
    assert_equal(ia[1], 3)  # 3 shifted down

    # Pop with negative index
    var val3 = ia.pop(-1)
    assert_equal(val3, 4)
    assert_equal(len(ia), 2)


def test_ia2_clear() raises:
    """Test clear operation."""
    var ia = IntArray(1, 2, 3, 4, 5)
    assert_equal(len(ia), 5)

    ia.clear()
    assert_equal(len(ia), 0)
    assert_true(ia.is_empty())

    # Can still append after clear
    ia.append(10)
    assert_equal(len(ia), 1)
    assert_equal(ia[0], 10)


# ========== Insert/Replace Tests ==========


def test_ia2_insert_single() raises:
    """Test single element insertion."""
    var ia1 = IntArray(1, 3, 4)
    var ia2 = ia1.insert(1, 2)

    assert_equal(len(ia1), 3)  # Original unchanged
    assert_equal(len(ia2), 4)
    assert_equal(ia2[0], 1)
    assert_equal(ia2[1], 2)
    assert_equal(ia2[2], 3)
    assert_equal(ia2[3], 4)

    # Insert at start
    var ia3 = ia1.insert(0, 0)
    assert_equal(ia3[0], 0)
    assert_equal(len(ia3), 4)

    # Insert at end
    var ia4 = ia1.insert(3, 5)
    assert_equal(ia4[3], 5)
    assert_equal(len(ia4), 4)


def test_ia2_insert_multiple() raises:
    """Test multiple element insertion."""
    var ia = IntArray(0, 1, 5, 6)
    var indices = IntArray(2, 3)
    var values = IntArray(2, 3)
    var result = ia.insert(indices, values)

    assert_equal(len(result), 6)
    assert_equal(result[0], 0)
    assert_equal(result[1], 1)
    assert_equal(result[2], 2)
    assert_equal(result[3], 3)
    assert_equal(result[4], 5)
    assert_equal(result[5], 6)


def test_ia2_insert_empty_array() raises:
    """Test insertion into empty array."""
    var ia = IntArray()
    var indices = IntArray(0, 1, 2)
    var values = IntArray(10, 20, 30)
    var result = ia.insert(indices, values)

    assert_equal(len(result), 3)
    assert_equal(result[0], 10)
    assert_equal(result[1], 20)
    assert_equal(result[2], 30)


def test_ia2_replace_single() raises:
    """Test single element replacement."""
    var ia1 = IntArray(1, 2, 3, 4, 5)
    var ia2 = ia1.replace(2, 99)

    assert_equal(len(ia1), 5)  # Original unchanged
    assert_equal(ia1[2], 3)
    assert_equal(len(ia2), 5)
    assert_equal(ia2[2], 99)

    # Test negative index
    var ia3 = ia1.replace(-1, 50)
    assert_equal(ia3[4], 50)


def test_ia2_replace_multiple() raises:
    """Test multiple element replacement."""
    var ia = IntArray(0, 1, 2, 3, 4, 5)
    var indices = IntArray(1, 3, 5)
    var values = IntArray(10, 30, 50)
    var result = ia.replace(indices, values)

    assert_equal(len(result), 6)
    assert_equal(result[0], 0)
    assert_equal(result[1], 10)
    assert_equal(result[2], 2)
    assert_equal(result[3], 30)
    assert_equal(result[4], 4)
    assert_equal(result[5], 50)


# ========== Operation Tests ==========


def test_ia2_fill() raises:
    """Test fill operation."""
    var ia = IntArray(1, 2, 3, 4, 5)
    ia.fill(0)

    assert_equal(len(ia), 5)
    for i in range(5):
        assert_equal(ia[i], 0)


def test_ia2_contains() raises:
    """Test contains operation."""
    var ia = IntArray(10, 20, 30, 40, 50)

    assert_true(10 in ia)
    assert_true(30 in ia)
    assert_true(50 in ia)
    assert_false(5 in ia)
    assert_false(100 in ia)

    var empty = IntArray()
    assert_false(1 in empty)


def test_ia2_equality() raises:
    """Test equality comparison."""
    var ia1 = IntArray(1, 2, 3)
    var ia2 = IntArray(1, 2, 3)
    var ia3 = IntArray(1, 2, 4)
    var ia4 = IntArray(1, 2)

    assert_true(ia1 == ia2)
    assert_false(ia1 == ia3)
    assert_false(ia1 == ia4)

    # Test with List
    var lst: List[Int] = [1, 2, 3]
    assert_true(ia1 == lst)


def test_ia2_reverse() raises:
    """Test reverse operation."""
    var ia = IntArray(1, 2, 3, 4, 5)
    ia.reverse()

    assert_equal(len(ia), 5)
    assert_equal(ia[0], 5)
    assert_equal(ia[4], 1)

    # Test single element
    var ia2 = IntArray(42)
    ia2.reverse()
    assert_equal(ia2[0], 42)

    # Test empty
    var ia3 = IntArray()
    ia3.reverse()
    assert_equal(len(ia3), 0)


def test_ia2_reversed() raises:
    """Test reversed (immutable) operation."""
    var ia1 = IntArray(1, 2, 3, 4, 5)
    var ia2 = ia1.reversed()

    # Original unchanged
    assert_equal(ia1[0], 1)
    assert_equal(ia1[4], 5)

    # Reversed copy
    assert_equal(ia2[0], 5)
    assert_equal(ia2[4], 1)


# ========== Math Operations Tests ==========


def test_ia2_product() raises:
    """Test product operation."""
    var ia1 = IntArray(2, 3, 4)
    assert_equal(ia1.product(), 24)

    var ia2 = IntArray(5)
    assert_equal(ia2.product(), 5)

    var ia3 = IntArray()
    assert_equal(ia3.product(), 1)  # Empty product = 1

    var ia4 = IntArray(1, 2, 0, 4)
    assert_equal(ia4.product(), 0)


def test_ia2_sum() raises:
    """Test sum operation."""
    var ia1 = IntArray(1, 2, 3, 4, 5)
    assert_equal(ia1.sum(), 15)

    var ia2 = IntArray(10)
    assert_equal(ia2.sum(), 10)

    var ia3 = IntArray()
    assert_equal(ia3.sum(), 0)  # Empty sum = 0

    var ia4 = IntArray(-5, 5, -3, 3)
    assert_equal(ia4.sum(), 0)


# ========== Conversion Tests ==========


def test_ia2_tolist() raises:
    """Test conversion to List[Int]."""
    var ia = IntArray(1, 2, 3, 4, 5)
    var lst = ia.tolist()

    assert_equal(len(lst), 5)
    assert_equal(lst[0], 1)
    assert_equal(lst[4], 5)

    var empty = IntArray()
    var empty_list = empty.tolist()
    assert_equal(len(empty_list), 0)


def test_ia2_string_representation() raises:
    """Test string conversion."""
    var ia1 = IntArray(1, 2, 3)
    var s1 = ia1.__str__()
    assert_equal(s1, "[1, 2, 3]")

    var ia2 = IntArray()
    var s2 = ia2.__str__()
    assert_equal(s2, "[]")

    var ia3 = IntArray(42)
    var s3 = ia3.__str__()
    assert_equal(s3, "[42]")


# ========== Edge Case Tests ==========


def test_ia2_large_array() raises:
    """Test with large array."""
    var ia = IntArray.with_capacity(1000)
    for i in range(1000):
        ia.append(i)

    assert_equal(len(ia), 1000)
    assert_equal(ia[0], 0)
    assert_equal(ia[999], 999)
    assert_equal(ia.sum(), 499500)


def test_ia2_capacity_growth() raises:
    """Test capacity growth strategy."""
    var ia = IntArray()
    var prev_cap = 0

    for i in range(100):
        ia.append(i)
        var curr_cap = ia.capacity()
        # Capacity should only grow, never shrink
        assert_true(curr_cap >= prev_cap)
        prev_cap = curr_cap

    assert_equal(len(ia), 100)


def test_ia2_negative_values() raises:
    """Test with negative values."""
    var ia = IntArray(-5, -3, -1, 0, 1, 3, 5)

    assert_equal(len(ia), 7)
    assert_equal(ia[0], -5)
    assert_equal(ia.sum(), 0)

    var ia2 = ia.reversed()
    assert_equal(ia2[0], 5)
    assert_equal(ia2[6], -5)


def test_ia2_mixed_operations() raises:
    """Test combination of operations."""
    var ia = IntArray(1, 2, 3)
    ia.append(4)
    ia.prepend(0)
    assert_equal(len(ia), 5)

    var ia2 = ia.insert(3, 99)
    assert_equal(len(ia2), 6)

    var ia3 = ia2.replace(3, 50)
    assert_equal(ia3[3], 50)

    ia3.reverse()
    assert_equal(ia3[0], 4)


# ============================================
# INTARRAY INSERT/REPLACE TESTS
# ============================================


def test_intarray_replace_single_v2() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    var result = arr.replace(2, 999)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[2] == 999, "element at index 2 should be 999")
    assert_true(result[0] == 10, "other elements unchanged")
    assert_true(result[4] == 50, "other elements unchanged")
    assert_true(arr[2] == 30, "original array unchanged")


def test_intarray_replace_multiple_v2() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    var indices = IntArray(1, 3)
    var values = IntArray(200, 400)
    var result = arr.replace(indices, values)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[1] == 200, "element at index 1 should be 200")
    assert_true(result[3] == 400, "element at index 3 should be 400")
    assert_true(result[0] == 10, "other elements unchanged")
    assert_true(result[2] == 30, "other elements unchanged")
    assert_true(result[4] == 50, "other elements unchanged")
    assert_true(arr[1] == 20, "original array unchanged")


def test_intarray_replace_first_and_last() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    var indices = IntArray(0, 4)
    var values = IntArray(100, 500)
    var result = arr.replace(indices, values)
    assert_true(result[0] == 100, "first element should be 100")
    assert_true(result[4] == 500, "last element should be 500")
    assert_true(result[2] == 3, "middle unchanged")


def test_intarray_insert_single_beginning() raises:
    var arr = IntArray(10, 20, 30)
    var result = arr.insert(0, 5)
    assert_true(len(result) == 4, "result should have 4 elements")
    assert_true(result[0] == 5, "inserted element should be at index 0")
    assert_true(result[1] == 10, "original first element shifted")
    assert_true(result[3] == 30, "last element preserved")


def test_intarray_insert_single_middle() raises:
    var arr = IntArray(10, 20, 40, 50)
    var result = arr.insert(2, 30)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[2] == 30, "inserted element at index 2")
    assert_true(result[1] == 20, "element before unchanged")
    assert_true(result[3] == 40, "elements after shifted")


def test_intarray_insert_single_end() raises:
    var arr = IntArray(10, 20, 30)
    var result = arr.insert(3, 40)
    assert_true(len(result) == 4, "result should have 4 elements")
    assert_true(result[3] == 40, "inserted element at end")
    assert_true(result[2] == 30, "previous elements unchanged")


def test_intarray_insert_multiple_v2() raises:
    var arr = IntArray(10, 30, 50)
    var indices = IntArray(1, 3)
    var values = IntArray(20, 40)
    var result = arr.insert(indices, values)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[0] == 10, "first original element")
    assert_true(result[1] == 20, "first inserted element")
    assert_true(result[2] == 30, "second original element")
    assert_true(result[3] == 40, "second inserted element")
    assert_true(result[4] == 50, "third original element")


def test_intarray_insert_multiple_consecutive() raises:
    var arr = IntArray(10, 40)
    var indices = IntArray(1, 2, 3)
    var values = IntArray(20, 30, 35)
    var result = arr.insert(indices, values)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[0] == 10, "original first")
    assert_true(result[1] == 20, "inserted at 1")
    assert_true(result[2] == 30, "inserted at 2")
    assert_true(result[3] == 35, "inserted at 3")
    assert_true(result[4] == 40, "original second")


def test_intarray_insert_into_empty() raises:
    var arr = IntArray()
    var indices = IntArray(0, 1, 2)
    var values = IntArray(10, 20, 30)
    var result = arr.insert(indices, values)
    assert_true(len(result) == 3, "result should have 3 elements")
    assert_true(result[0] == 10, "first value")
    assert_true(result[1] == 20, "second value")
    assert_true(result[2] == 30, "third value")


def test_intarray_insert_empty_indices() raises:
    var arr = IntArray(1, 2, 3)
    var indices = IntArray()
    var values = IntArray()
    var result = arr.insert(indices, values)
    assert_true(len(result) == 3, "result should be unchanged")
    assert_true(result[0] == 1, "elements preserved")


def test_intarray_replace_single_negative_index() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    var result = arr.replace(-1, 999)
    assert_true(result[-1] == 999, "last element should be 999")
    assert_true(result[0] == 10, "other elements unchanged")


def test_intarray_default_constructor() raises:
    var arr = IntArray()
    assert_true(len(arr) == 0, "default constructor should create empty array")
    assert_true(arr.is_empty(), "is_empty should return True")


def test_intarray_with_capacity() raises:
    var arr = IntArray.with_capacity(10)
    assert_true(
        len(arr) == 0, "with_capacity should create empty array (size 0)"
    )
    assert_true(arr.capacity() >= 10, "capacity should be at least 10")


def test_intarray_size_constructor() raises:
    var arr = IntArray.with_capacity(10)
    assert_true(len(arr) == 0, "size constructor should create array of size 0")
    assert_true(arr.capacity() >= 10, "capacity should be at least 10")


def test_intarray_variadic_constructor() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    assert_true(
        len(arr) == 5, "variadic constructor should create array of size 5"
    )
    assert_true(arr[0] == 1, "first element should be 1")
    assert_true(arr[4] == 5, "last element should be 5")


def test_intarray_list_constructor() raises:
    var lst = List[Int]()
    lst.append(10)
    lst.append(20)
    lst.append(30)
    var arr = IntArray(lst)
    assert_true(len(arr) == 3, "list constructor should create array of size 3")
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[1] == 20, "second element should be 20")
    assert_true(arr[2] == 30, "third element should be 30")


def test_intarray_filled() raises:
    var arr = IntArray.filled(7, 42)
    assert_true(len(arr) == 7, "filled should create array of size 7")
    for i in range(7):
        assert_true(arr[i] == 42, "all elements should be 42")


def test_intarray_range() raises:
    var arr = IntArray.range(0, 10)
    assert_true(len(arr) == 10, "range should create array of size 10")
    for i in range(10):
        assert_true(
            arr[i] == i, "element " + String(i) + " should equal " + String(i)
        )


def test_intarray_range_with_step() raises:
    var arr = IntArray.range(0, 20, 2)
    assert_true(
        len(arr) == 10, "range with step 2 should create array of size 10"
    )
    for i in range(10):
        assert_true(arr[i] == i * 2, "element should be " + String(i * 2))


def test_intarray_getitem() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[2] == 30, "third element should be 30")
    assert_true(arr[4] == 50, "last element should be 50")


def test_intarray_getitem_negative() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    assert_true(arr[-1] == 50, "arr[-1] should be 50")
    assert_true(arr[-2] == 40, "arr[-2] should be 40")
    assert_true(arr[-5] == 10, "arr[-5] should be 10")


def test_intarray_setitem() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    arr[0] = 100
    arr[2] = 200
    arr[4] = 300
    assert_true(arr[0] == 100, "arr[0] should be 100")
    assert_true(arr[2] == 200, "arr[2] should be 200")
    assert_true(arr[4] == 300, "arr[4] should be 300")


def test_intarray_setitem_negative() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    arr[-1] = 999
    arr[-3] = 777
    assert_true(arr[-1] == 999, "arr[-1] should be 999")
    assert_true(arr[-3] == 777, "arr[-3] should be 777")


def test_intarray_slice() raises:
    var arr = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var slice1 = arr[2:5]
    assert_true(len(slice1) == 3, "slice [2:5] should have 3 elements")
    assert_true(slice1[0] == 2, "first element should be 2")
    assert_true(slice1[2] == 4, "last element should be 4")


def test_intarray_slice_with_step() raises:
    var arr = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var slice1 = arr[0:10:2]
    assert_true(len(slice1) == 5, "slice [0:10:2] should have 5 elements")
    assert_true(slice1[0] == 0, "first element should be 0")
    assert_true(slice1[4] == 8, "last element should be 8")


def test_intarray_append() raises:
    var arr = IntArray()
    arr.append(10)
    arr.append(20)
    arr.append(30)
    assert_true(len(arr) == 3, "array should have 3 elements")
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[1] == 20, "second element should be 20")
    assert_true(arr[2] == 30, "third element should be 30")


def test_intarray_append_growth() raises:
    var arr = IntArray.with_capacity(2)
    for i in range(10):
        arr.append(i)
    assert_true(len(arr) == 10, "array should have 10 elements after growth")
    assert_true(arr.capacity() >= 10, "capacity should have grown")
    for i in range(10):
        assert_true(arr[i] == i, "element " + String(i) + " should be correct")


def test_intarray_prepend() raises:
    var arr = IntArray()
    arr.prepend(30)
    arr.prepend(20)
    arr.prepend(10)
    assert_true(len(arr) == 3, "array should have 3 elements")
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[1] == 20, "second element should be 20")
    assert_true(arr[2] == 30, "third element should be 30")


def test_intarray__add__() raises:
    var arr = IntArray(1, 2, 3)
    var extended = arr + 4
    assert_true(len(extended) == 4, "added array should have 4 elements")
    assert_true(extended[3] == 4, "last element should be 4")
    assert_true(len(arr) == 3, "original array unchanged")
    extended = 0 + arr
    assert_true(
        len(extended) == 4,
        "added array should have 4 elements after addition at front",
    )
    assert_true(extended == IntArray(0, 1, 2, 3), "front addition should match")


def test_intarray__add__self() raises:
    var arr = IntArray(1, 2, 3)
    var extended = arr + IntArray(4)
    assert_true(len(extended) == 4, "added array should have 4 elements")
    assert_true(extended[3] == 4, "last element should be 4")
    assert_true(len(arr) == 3, "original array unchanged")

    extended = IntArray(1, 2, 3) + IntArray(4) + IntArray(5, 6, 7)
    assert_true(len(extended) == 7, "added array should have 7 elements")
    assert_true(
        extended == IntArray(1, 2, 3, 4, 5, 6, 7), "Added result should match"
    )


def test_intarray_pop() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    var val = arr.pop()
    assert_true(val == 50, "popped value should be 50")
    assert_true(len(arr) == 4, "array should have 4 elements")
    assert_true(arr[3] == 40, "last element should now be 40")


def test_intarray_pop_index() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    var val = arr.pop(2)
    assert_true(val == 30, "popped value should be 30")
    assert_true(len(arr) == 4, "array should have 4 elements")
    assert_true(arr[2] == 40, "element at index 2 should now be 40")


def test_intarray_clear() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    arr.clear()
    assert_true(len(arr) == 0, "array should be empty after clear")
    assert_true(arr.is_empty(), "is_empty should return True")


def test_intarray_fill() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    arr.fill(99)
    for i in range(5):
        assert_true(arr[i] == 99, "all elements should be 99")


def test_intarray_contains() raises:
    var arr = IntArray(10, 20, 30, 40, 50)
    assert_true(30 in arr, "30 should be in array")
    assert_true(50 in arr, "50 should be in array")
    assert_true(not (99 in arr), "99 should not be in array")


def test_intarray_eq() raises:
    var arr1 = IntArray(1, 2, 3, 4, 5)
    var arr2 = IntArray(1, 2, 3, 4, 5)
    var arr3 = IntArray(1, 2, 3, 4, 6)
    assert_true(arr1 == arr2, "arr1 should equal arr2")
    assert_true(not (arr1 == arr3), "arr1 should not equal arr3")


def test_intarray_eq_list() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    var lst = List[Int]()
    for i in range(1, 6):
        lst.append(i)
    assert_true(arr == lst, "array should equal list")


def test_intarray_tolist() raises:
    var arr = IntArray(10, 20, 30)
    var lst = arr.tolist()
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[0] == 10, "first element should be 10")
    assert_true(lst[2] == 30, "last element should be 30")


def test_intarray_str() raises:
    var arr = IntArray(1, 2, 3)
    var s = arr.__str__()
    assert_true(s == "[1, 2, 3]", "string representation should be '[1, 2, 3]'")


def test_intarray_product() raises:
    var arr = IntArray(2, 3, 4)
    assert_true(arr.product() == 24, "product should be 24")
    var empty = IntArray()
    assert_true(empty.product() == 1, "empty product should be 1")


def test_intarray_sum() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    assert_true(arr.sum() == 15, "sum should be 15")
    var empty = IntArray()
    assert_true(empty.sum() == 0, "empty sum should be 0")


def test_intarray_reverse() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    arr.reverse()
    assert_true(arr[0] == 5, "first element should be 5")
    assert_true(arr[4] == 1, "last element should be 1")


def test_intarray_reversed() raises:
    var arr = IntArray(1, 2, 3, 4, 5)
    var rev = arr.reversed()
    assert_true(rev[0] == 5, "first element should be 5")
    assert_true(rev[4] == 1, "last element should be 1")
    assert_true(arr[0] == 1, "original array unchanged")


def test_intarray_copyinit() raises:
    var arr1 = IntArray(1, 2, 3, 4, 5)
    var arr2 = arr1
    arr2[0] = 999
    assert_true(arr1[0] == 1, "original array should be unchanged (deep copy)")
    assert_true(arr2[0] == 999, "copy should be modified")


# ========== Optional / Null-Safety Tests ==========


def test_empty_operations_no_crash() raises:
    """All operations on empty IntArray must not segfault."""
    var a = IntArray()
    a.fill(0)
    assert_true(a.is_empty())
    assert_false(42 in a)
    assert_equal(a.tolist().__len__(), 0)
    assert_equal(a.__str__(), "[]")
    assert_true(a == IntArray())
    a.reverse()
    assert_true(a.is_empty())
    assert_equal(a.sum(), 0)
    assert_equal(a.product(), 1)


def test_empty_append_after_clear() raises:
    """Append after clear — exercises reserve() Optional reinit path."""
    var a = IntArray()
    for i in range(5):
        a.append(i)
    assert_equal(len(a), 5)
    a.clear()
    assert_true(a.is_empty())
    a.append(99)
    assert_equal(len(a), 1)
    assert_equal(a[0], 99)


def test_repeated_growth() raises:
    """Repeated reserve growth — exercises Optional reassign in reserve()."""
    var a = IntArray()
    for i in range(1000):
        a.append(i)
    assert_equal(len(a), 1000)
    for i in range(1000):
        assert_equal(a[i], i)


def test_reserve_preserves_data() raises:
    """Reserve() must preserve existing data through multiple growth phases."""
    var a = IntArray.with_capacity(2)
    a.append(10)
    a.append(20)
    a.reserve(100)
    assert_equal(a[0], 10)
    assert_equal(a[1], 20)
    assert_equal(len(a), 2)
    for i in range(200):
        a.append(i)
    assert_equal(a[0], 10)
    assert_equal(a[1], 20)
    assert_equal(a[201], 199)


def test_copy_constructor_empty() raises:
    """Copy of empty IntArray must create valid empty array."""
    var a = IntArray()
    var b = a
    assert_true(b.is_empty())
    assert_equal(b.capacity(), 0)
    b.append(7)
    assert_equal(b[0], 7)


def test_slice_of_empty() raises:
    """Slicing empty array must return empty array."""
    var a = IntArray()
    var s = a[:]
    assert_true(s.is_empty())
    s = a[0:0]
    assert_true(s.is_empty())


def test_concat_empty() raises:
    """Concatenation with empty arrays must not dereference null."""
    var a = IntArray()
    var b = IntArray(1, 2)
    var c = a + b
    assert_equal(len(c), 2)
    assert_equal(c[0], 1)

    var d = b + a
    assert_equal(len(d), 2)
    assert_equal(d[0], 1)

    var e = a + a
    assert_true(e.is_empty())


def test_pop_until_empty() raises:
    """Pop all elements — exercises left-shift through Optional pointer."""
    var a = IntArray(10, 20, 30)
    assert_equal(a.pop(), 30)
    assert_equal(a.pop(), 20)
    assert_equal(a.pop(), 10)
    assert_true(a.is_empty())


def test_setitem_on_constructed() raises:
    """Write through Optional pointer after capacity growth."""
    var a = IntArray(0, 0, 0)
    a[0] = 100
    a[-1] = 300
    assert_equal(a[0], 100)
    assert_equal(a[2], 300)


def test_data_integrity_after_multiple_ops() raises:
    """Sequence of operations through Optional pointer must preserve data."""
    var a = IntArray()
    for i in range(50):
        a.append(i * 2)
    a.reverse()
    for i in range(50):
        assert_equal(a[i], (49 - i) * 2)
    a.sort()
    for i in range(50):
        assert_equal(a[i], i * 2)


# ========== Memmove / Bulk Tests ==========


def test_prepend_large() raises:
    """Prepend 10000 elements — triggers SIMD memmove."""
    var a = IntArray()
    for i in range(10000):
        a.prepend(i)
    assert_equal(len(a), 10000)
    assert_equal(a[0], 9999)
    assert_equal(a[9999], 0)


def test_prepend_to_existing() raises:
    """Prepend to non-empty array — memmove with partial shift."""
    var a = IntArray(10, 20, 30)
    a.prepend(0)
    assert_equal(a[0], 0)
    assert_equal(a[1], 10)
    assert_equal(a[2], 20)
    assert_equal(a[3], 30)

    a.prepend(-10)
    assert_equal(a[0], -10)
    assert_equal(a[1], 0)
    assert_equal(a[2], 10)
    assert_equal(len(a), 5)


def test_prepend_after_append() raises:
    """Mix of append and prepend — exercises both growth paths."""
    var a = IntArray()
    for i in range(100):
        a.append(i)
    for i in range(100):
        a.prepend(-i - 1)
    assert_equal(len(a), 200)
    assert_equal(a[0], -100)
    assert_equal(a[99], -1)
    assert_equal(a[100], 0)
    assert_equal(a[199], 99)


def test_pop_front() raises:
    """Pop from front — triggers full memmove of remaining elements."""
    var a = IntArray(10, 20, 30, 40, 50)
    var v = a.pop(0)
    assert_equal(v, 10)
    assert_equal(len(a), 4)
    assert_equal(a[0], 20)
    assert_equal(a[3], 50)
    var v2 = a.pop(0)
    assert_equal(v2, 20)
    assert_equal(a[0], 30)


def test_pop_back() raises:
    """Pop from back — no memmove, pure size decrement."""
    var a = IntArray(1, 2, 3, 4, 5)
    for i in range(5):
        var v = a.pop()
        assert_equal(v, 5 - i)
    assert_true(a.is_empty())


def test_pop_middle_large() raises:
    """Pop from middle of large array — triggers SIMD memmove."""
    var a = IntArray()
    for i in range(5000):
        a.append(i)
    var v = a.pop(2500)
    assert_equal(v, 2500)
    assert_equal(len(a), 4999)
    assert_equal(a[2500], 2501)
    for i in range(2500):
        assert_equal(a[i], i)
    for i in range(2500, 4999):
        assert_equal(a[i], i + 1)


def test_insert_single_memcpy() raises:
    """Insert single element — exercises memcpy-based insert."""
    var a = IntArray(10, 20, 40, 50)
    var r = a.insert(2, 30)
    assert_equal(len(r), 5)
    assert_equal(r[0], 10)
    assert_equal(r[1], 20)
    assert_equal(r[2], 30)
    assert_equal(r[3], 40)
    assert_equal(r[4], 50)


def test_insert_beginning() raises:
    """Insert at position 0 — full right-shift memcpy."""
    var a = IntArray(1, 2, 3)
    var r = a.insert(0, 0)
    assert_equal(len(r), 4)
    assert_equal(r[0], 0)
    assert_equal(r[1], 1)
    assert_equal(r[3], 3)


def test_insert_end() raises:
    """Insert at last position — single element memcpy after."""
    var a = IntArray(1, 2, 3)
    var r = a.insert(3, 4)
    assert_equal(len(r), 4)
    assert_equal(r[0], 1)
    assert_equal(r[3], 4)


def test_insert_large_middle() raises:
    """Insert in middle of large array — triggers SIMD memcpy."""
    var a = IntArray()
    for i in range(10000):
        a.append(i)
    var r = a.insert(5000, 9999)
    assert_equal(len(r), 10001)
    assert_equal(r[0], 0)
    assert_equal(r[5000], 9999)
    assert_equal(r[5001], 5000)
    assert_equal(r[10000], 9999)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
