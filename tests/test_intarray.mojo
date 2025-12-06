from intarray import IntArray
from testing import assert_equal, assert_true, assert_false

# ========== Construction Tests ==========


fn test_ia2_empty_construction() raises:
    """Test empty IntArray construction."""
    var ia = IntArray()
    assert_equal(len(ia), 0)
    # assert_equal(ia.capacity(), IntArray.SMALL_SIZE)
    assert_equal(ia.capacity(), 0)
    assert_true(ia.is_empty())
    print("✓ test_ia2_empty_construction")


fn test_ia2_variadic_construction() raises:
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
    print("✓ test_ia2_variadic_construction")


fn test_ia2_list_construction() raises:
    """Test List[Int] constructor."""
    var lst = List[Int](1, 2, 3, 4)
    var ia = IntArray(lst)
    assert_equal(len(ia), 4)
    assert_equal(ia[0], 1)
    assert_equal(ia[3], 4)
    print("✓ test_ia2_list_construction")


fn test_ia2_filled_construction() raises:
    """Test filled() static constructor."""
    var ia = IntArray.filled(5, 42)
    assert_equal(len(ia), 5)
    for i in range(5):
        assert_equal(ia[i], 42)

    var ia_empty = IntArray.filled(0, 99)
    assert_equal(len(ia_empty), 0)
    print("✓ test_ia2_filled_construction")


fn test_ia2_range_construction() raises:
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
    print("✓ test_ia2_range_construction")


fn test_ia2_with_capacity() raises:
    """Test with_capacity() constructor."""
    var ia = IntArray.with_capacity(10)
    assert_equal(len(ia), 0)
    assert_equal(ia.capacity(), 10)
    assert_true(ia.is_empty())
    print("✓ test_ia2_with_capacity")


fn test_ia2_copy_construction() raises:
    """Test copy constructor (deep copy)."""
    var ia1 = IntArray(1, 2, 3)
    var ia2 = ia1
    assert_equal(len(ia2), 3)
    assert_equal(ia2[0], 1)

    # Modify ia2, ia1 should be unchanged (deep copy)
    ia2[0] = 99
    assert_equal(ia1[0], 1)  # Original unchanged
    assert_equal(ia2[0], 99)  # Copy modified
    print("✓ test_ia2_copy_construction")


# ========== Access Tests ==========


fn test_ia2_positive_indexing() raises:
    """Test positive index access."""
    var ia = IntArray(10, 20, 30, 40, 50)
    assert_equal(ia[0], 10)
    assert_equal(ia[2], 30)
    assert_equal(ia[4], 50)
    print("✓ test_ia2_positive_indexing")


fn test_ia2_negative_indexing() raises:
    """Test negative index access."""
    var ia = IntArray(10, 20, 30, 40, 50)
    assert_equal(ia[-1], 50)
    assert_equal(ia[-2], 40)
    assert_equal(ia[-5], 10)
    print("✓ test_ia2_negative_indexing")


fn test_ia2_setitem() raises:
    """Test element assignment."""
    var ia = IntArray(1, 2, 3, 4, 5)
    ia[0] = 100
    ia[4] = 500
    ia[-2] = 400
    assert_equal(ia[0], 100)
    assert_equal(ia[4], 500)
    assert_equal(ia[3], 400)
    print("✓ test_ia2_setitem")


fn test_ia2_slicing_basic() raises:
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
    print("✓ test_ia2_slicing_basic")


fn test_ia2_slicing_step() raises:
    """Test slicing with step."""
    var ia = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # Every 2nd element
    var s1 = ia[::2]
    assert_equal(len(s1), 5)
    assert_equal(s1[0], 0)
    assert_equal(s1[4], 8)

    # Reverse
    var s2 = ia[::-1]
    print("s2: ", s2)
    assert_equal(len(s2), 10)
    assert_equal(s2[0], 9)
    assert_equal(s2[9], 0)

    # Reverse with step
    var s3 = ia[8:2:-2]
    assert_equal(len(s3), 3)
    assert_equal(s3[0], 8)
    assert_equal(s3[2], 4)
    print("✓ test_ia2_slicing_step")


fn test_ia2_slicing_negative_indices() raises:
    """Test slicing with negative indices."""
    var ia = IntArray(0, 1, 2, 3, 4, 5)

    var s1 = ia[-3:-1]
    assert_equal(len(s1), 2)
    assert_equal(s1[0], 3)
    assert_equal(s1[1], 4)

    var s2 = ia[-5:]
    assert_equal(len(s2), 5)
    assert_equal(s2[0], 1)
    print("✓ test_ia2_slicing_negative_indices")


# ========== Growth Operations Tests ==========


fn test_ia2_append() raises:
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
    print("✓ test_ia2_append")


fn test_ia2_prepend() raises:
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
    print("✓ test_ia2_prepend")


fn test_ia2__add__() raises:
    """Test __add__ (immutable append)."""
    var ia1 = IntArray(1, 2, 3)
    var ia2 = ia1 + 4

    assert_equal(len(ia1), 3)  # Original unchanged
    assert_equal(len(ia2), 4)  # New array extended
    assert_equal(ia2[3], 4)
    print("✓ test_ia2__add__")


fn test_ia2_pop() raises:
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
    print("✓ test_ia2_pop")


fn test_ia2_clear() raises:
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
    print("✓ test_ia2_clear")


# ========== Insert/Replace Tests ==========


fn test_ia2_insert_single() raises:
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
    print("✓ test_ia2_insert_single")


fn test_ia2_insert_multiple() raises:
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
    print("✓ test_ia2_insert_multiple")


fn test_ia2_insert_empty_array() raises:
    """Test insertion into empty array."""
    var ia = IntArray()
    var indices = IntArray(0, 1, 2)
    var values = IntArray(10, 20, 30)
    var result = ia.insert(indices, values)

    assert_equal(len(result), 3)
    assert_equal(result[0], 10)
    assert_equal(result[1], 20)
    assert_equal(result[2], 30)
    print("✓ test_ia2_insert_empty_array")


fn test_ia2_replace_single() raises:
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
    print("✓ test_ia2_replace_single")


fn test_ia2_replace_multiple() raises:
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
    print("✓ test_ia2_replace_multiple")


# ========== Operation Tests ==========


fn test_ia2_fill() raises:
    """Test fill operation."""
    var ia = IntArray(1, 2, 3, 4, 5)
    ia.fill(0)

    assert_equal(len(ia), 5)
    for i in range(5):
        assert_equal(ia[i], 0)
    print("✓ test_ia2_fill")


fn test_ia2_contains() raises:
    """Test contains operation."""
    var ia = IntArray(10, 20, 30, 40, 50)

    assert_true(10 in ia)
    assert_true(30 in ia)
    assert_true(50 in ia)
    assert_false(5 in ia)
    assert_false(100 in ia)

    var empty = IntArray()
    assert_false(1 in empty)
    print("✓ test_ia2_contains")


fn test_ia2_equality() raises:
    """Test equality comparison."""
    var ia1 = IntArray(1, 2, 3)
    var ia2 = IntArray(1, 2, 3)
    var ia3 = IntArray(1, 2, 4)
    var ia4 = IntArray(1, 2)

    assert_true(ia1 == ia2)
    assert_false(ia1 == ia3)
    assert_false(ia1 == ia4)

    # Test with List
    var lst = List[Int](1, 2, 3)
    assert_true(ia1 == lst)
    print("✓ test_ia2_equality")


fn test_ia2_reverse() raises:
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
    print("✓ test_ia2_reverse")


fn test_ia2_reversed() raises:
    """Test reversed (immutable) operation."""
    var ia1 = IntArray(1, 2, 3, 4, 5)
    var ia2 = ia1.reversed()

    # Original unchanged
    assert_equal(ia1[0], 1)
    assert_equal(ia1[4], 5)

    # Reversed copy
    assert_equal(ia2[0], 5)
    assert_equal(ia2[4], 1)
    print("✓ test_ia2_reversed")


# ========== Math Operations Tests ==========


fn test_ia2_product() raises:
    """Test product operation."""
    var ia1 = IntArray(2, 3, 4)
    assert_equal(ia1.product(), 24)

    var ia2 = IntArray(5)
    assert_equal(ia2.product(), 5)

    var ia3 = IntArray()
    assert_equal(ia3.product(), 1)  # Empty product = 1

    var ia4 = IntArray(1, 2, 0, 4)
    assert_equal(ia4.product(), 0)
    print("✓ test_ia2_product")


fn test_ia2_sum() raises:
    """Test sum operation."""
    var ia1 = IntArray(1, 2, 3, 4, 5)
    assert_equal(ia1.sum(), 15)

    var ia2 = IntArray(10)
    assert_equal(ia2.sum(), 10)

    var ia3 = IntArray()
    assert_equal(ia3.sum(), 0)  # Empty sum = 0

    var ia4 = IntArray(-5, 5, -3, 3)
    assert_equal(ia4.sum(), 0)
    print("✓ test_ia2_sum")


# ========== Conversion Tests ==========


fn test_ia2_tolist() raises:
    """Test conversion to List[Int]."""
    var ia = IntArray(1, 2, 3, 4, 5)
    var lst = ia.tolist()

    assert_equal(len(lst), 5)
    assert_equal(lst[0], 1)
    assert_equal(lst[4], 5)

    var empty = IntArray()
    var empty_list = empty.tolist()
    assert_equal(len(empty_list), 0)
    print("✓ test_ia2_tolist")


fn test_ia2_string_representation() raises:
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
    print("✓ test_ia2_string_representation")


# ========== Edge Case Tests ==========


fn test_ia2_large_array() raises:
    """Test with large array."""
    var ia = IntArray.with_capacity(1000)
    for i in range(1000):
        ia.append(i)

    assert_equal(len(ia), 1000)
    assert_equal(ia[0], 0)
    assert_equal(ia[999], 999)
    assert_equal(ia.sum(), 499500)
    print("✓ test_ia2_large_array")


fn test_ia2_capacity_growth() raises:
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
    print("✓ test_ia2_capacity_growth")


fn test_ia2_negative_values() raises:
    """Test with negative values."""
    var ia = IntArray(-5, -3, -1, 0, 1, 3, 5)

    assert_equal(len(ia), 7)
    assert_equal(ia[0], -5)
    assert_equal(ia.sum(), 0)

    var ia2 = ia.reversed()
    assert_equal(ia2[0], 5)
    assert_equal(ia2[6], -5)
    print("✓ test_ia2_negative_values")


fn test_ia2_mixed_operations() raises:
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
    print("✓ test_ia2_mixed_operations")


# ========== Main Test Runner ==========


fn run_all_intarray_v2_tests() raises:
    """Run all IntArray v2 tests."""
    print("\n=== IntArray v2 Comprehensive Tests ===\n")

    # Construction
    test_ia2_empty_construction()
    test_ia2_variadic_construction()
    test_ia2_list_construction()
    test_ia2_filled_construction()
    test_ia2_range_construction()
    test_ia2_with_capacity()
    test_ia2_copy_construction()

    # Access
    test_ia2_positive_indexing()
    test_ia2_negative_indexing()
    test_ia2_setitem()
    test_ia2_slicing_basic()
    test_ia2_slicing_step()
    test_ia2_slicing_negative_indices()

    # Growth
    test_ia2_append()
    test_ia2_prepend()
    test_ia2__add__()
    test_ia2_pop()
    test_ia2_clear()

    # Insert/Replace
    test_ia2_insert_single()
    test_ia2_insert_multiple()
    test_ia2_insert_empty_array()
    test_ia2_replace_single()
    test_ia2_replace_multiple()

    # Operations
    test_ia2_fill()
    test_ia2_contains()
    test_ia2_equality()
    test_ia2_reverse()
    test_ia2_reversed()

    # Math
    test_ia2_product()
    test_ia2_sum()

    # Conversions
    test_ia2_tolist()
    test_ia2_string_representation()

    # Edge cases
    test_ia2_large_array()
    test_ia2_capacity_growth()
    test_ia2_negative_values()
    test_ia2_mixed_operations()

    print("\n=== All IntArray v2 Tests Passed! ===")


# ============================================
# INTARRAY INSERT/REPLACE TESTS
# ============================================


fn test_intarray_replace_single_v2() raises:
    print("test_intarray_replace_single_v2")
    var arr = IntArray(10, 20, 30, 40, 50)
    var result = arr.replace(2, 999)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[2] == 999, "element at index 2 should be 999")
    assert_true(result[0] == 10, "other elements unchanged")
    assert_true(result[4] == 50, "other elements unchanged")
    assert_true(arr[2] == 30, "original array unchanged")
    print("test_intarray_replace_single_v2 passed")


fn test_intarray_replace_multiple_v2() raises:
    print("test_intarray_replace_multiple_v2")
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
    print("test_intarray_replace_multiple_v2 passed")


fn test_intarray_replace_first_and_last() raises:
    print("test_intarray_replace_first_and_last")
    var arr = IntArray(1, 2, 3, 4, 5)
    var indices = IntArray(0, 4)
    var values = IntArray(100, 500)
    var result = arr.replace(indices, values)
    assert_true(result[0] == 100, "first element should be 100")
    assert_true(result[4] == 500, "last element should be 500")
    assert_true(result[2] == 3, "middle unchanged")
    print("test_intarray_replace_first_and_last passed")


fn test_intarray_insert_single_beginning() raises:
    print("test_intarray_insert_single_beginning")
    var arr = IntArray(10, 20, 30)
    var result = arr.insert(0, 5)
    assert_true(len(result) == 4, "result should have 4 elements")
    assert_true(result[0] == 5, "inserted element should be at index 0")
    assert_true(result[1] == 10, "original first element shifted")
    assert_true(result[3] == 30, "last element preserved")
    print("test_intarray_insert_single_beginning passed")


fn test_intarray_insert_single_middle() raises:
    print("test_intarray_insert_single_middle")
    var arr = IntArray(10, 20, 40, 50)
    var result = arr.insert(2, 30)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[2] == 30, "inserted element at index 2")
    assert_true(result[1] == 20, "element before unchanged")
    assert_true(result[3] == 40, "elements after shifted")
    print("test_intarray_insert_single_middle passed")


fn test_intarray_insert_single_end() raises:
    print("test_intarray_insert_single_end")
    var arr = IntArray(10, 20, 30)
    var result = arr.insert(3, 40)
    assert_true(len(result) == 4, "result should have 4 elements")
    assert_true(result[3] == 40, "inserted element at end")
    assert_true(result[2] == 30, "previous elements unchanged")
    print("test_intarray_insert_single_end passed")


fn test_intarray_insert_multiple_v2() raises:
    print("test_intarray_insert_multiple_v2")
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
    print("test_intarray_insert_multiple_v2 passed")


fn test_intarray_insert_multiple_consecutive() raises:
    print("test_intarray_insert_multiple_consecutive")
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
    print("test_intarray_insert_multiple_consecutive passed")


fn test_intarray_insert_into_empty() raises:
    print("test_intarray_insert_into_empty")
    var arr = IntArray()
    var indices = IntArray(0, 1, 2)
    var values = IntArray(10, 20, 30)
    var result = arr.insert(indices, values)
    assert_true(len(result) == 3, "result should have 3 elements")
    assert_true(result[0] == 10, "first value")
    assert_true(result[1] == 20, "second value")
    assert_true(result[2] == 30, "third value")
    print("test_intarray_insert_into_empty passed")


fn test_intarray_insert_empty_indices() raises:
    print("test_intarray_insert_empty_indices")
    var arr = IntArray(1, 2, 3)
    var indices = IntArray()
    var values = IntArray()
    var result = arr.insert(indices, values)
    assert_true(len(result) == 3, "result should be unchanged")
    assert_true(result[0] == 1, "elements preserved")
    print("test_intarray_insert_empty_indices passed")


fn test_intarray_replace_single_negative_index() raises:
    print("test_intarray_replace_single_negative_index")
    var arr = IntArray(10, 20, 30, 40, 50)
    var result = arr.replace(-1, 999)
    assert_true(result[-1] == 999, "last element should be 999")
    assert_true(result[0] == 10, "other elements unchanged")
    print("test_intarray_replace_single_negative_index passed")


# ============================================
# CONSOLIDATED TEST RUNNER FOR INSERT/REPLACE
# ============================================


fn run_intarray_insert_replace_tests() raises:
    print("\n" + "=" * 60)
    print("RUNNING INTARRAY INSERT/REPLACE TESTS")
    print("=" * 60)

    test_intarray_replace_single_v2()
    test_intarray_replace_multiple_v2()
    test_intarray_replace_first_and_last()
    test_intarray_insert_single_beginning()
    test_intarray_insert_single_middle()
    test_intarray_insert_single_end()
    test_intarray_insert_multiple_v2()
    test_intarray_insert_multiple_consecutive()
    test_intarray_insert_into_empty()
    test_intarray_insert_empty_indices()
    test_intarray_replace_single_negative_index()

    print("\n" + "=" * 60)
    print("ALL INTARRAY INSERT/REPLACE TESTS PASSED ✓")
    print("=" * 60)


# ============================================
# INTARRAY TESTS
# ============================================


fn test_intarray_default_constructor() raises:
    print("test_intarray_default_constructor")
    var arr = IntArray()
    assert_true(len(arr) == 0, "default constructor should create empty array")
    assert_true(arr.is_empty(), "is_empty should return True")
    print("test_intarray_default_constructor passed")


fn test_intarray_with_capacity() raises:
    print("test_intarray_with_capacity")
    var arr = IntArray.with_capacity(10)
    assert_true(
        len(arr) == 0, "with_capacity should create empty array (size 0)"
    )
    assert_true(arr.capacity() >= 10, "capacity should be at least 10")
    print("test_intarray_with_capacity passed")


fn test_intarray_size_constructor() raises:
    print("test_intarray_size_constructor")
    var arr = IntArray.with_capacity(10)
    assert_true(len(arr) == 0, "size constructor should create array of size 0")
    assert_true(arr.capacity() >= 10, "capacity should be at least 10")
    print("test_intarray_size_constructor passed")


fn test_intarray_variadic_constructor() raises:
    print("test_intarray_variadic_constructor")
    var arr = IntArray(1, 2, 3, 4, 5)
    assert_true(
        len(arr) == 5, "variadic constructor should create array of size 5"
    )
    assert_true(arr[0] == 1, "first element should be 1")
    assert_true(arr[4] == 5, "last element should be 5")
    print("test_intarray_variadic_constructor passed")


fn test_intarray_list_constructor() raises:
    print("test_intarray_list_constructor")
    var lst = List[Int]()
    lst.append(10)
    lst.append(20)
    lst.append(30)
    var arr = IntArray(lst)
    assert_true(len(arr) == 3, "list constructor should create array of size 3")
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[1] == 20, "second element should be 20")
    assert_true(arr[2] == 30, "third element should be 30")
    print("test_intarray_list_constructor passed")


fn test_intarray_filled() raises:
    print("test_intarray_filled")
    var arr = IntArray.filled(7, 42)
    assert_true(len(arr) == 7, "filled should create array of size 7")
    for i in range(7):
        assert_true(arr[i] == 42, "all elements should be 42")
    print("test_intarray_filled passed")


fn test_intarray_range() raises:
    print("test_intarray_range")
    var arr = IntArray.range(0, 10)
    assert_true(len(arr) == 10, "range should create array of size 10")
    for i in range(10):
        assert_true(
            arr[i] == i, "element " + String(i) + " should equal " + String(i)
        )
    print("test_intarray_range passed")


fn test_intarray_range_with_step() raises:
    print("test_intarray_range_with_step")
    var arr = IntArray.range(0, 20, 2)
    assert_true(
        len(arr) == 10, "range with step 2 should create array of size 10"
    )
    for i in range(10):
        assert_true(arr[i] == i * 2, "element should be " + String(i * 2))
    print("test_intarray_range_with_step passed")


fn test_intarray_getitem() raises:
    print("test_intarray_getitem")
    var arr = IntArray(10, 20, 30, 40, 50)
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[2] == 30, "third element should be 30")
    assert_true(arr[4] == 50, "last element should be 50")
    print("test_intarray_getitem passed")


fn test_intarray_getitem_negative() raises:
    print("test_intarray_getitem_negative")
    var arr = IntArray(10, 20, 30, 40, 50)
    assert_true(arr[-1] == 50, "arr[-1] should be 50")
    assert_true(arr[-2] == 40, "arr[-2] should be 40")
    assert_true(arr[-5] == 10, "arr[-5] should be 10")
    print("test_intarray_getitem_negative passed")


fn test_intarray_setitem() raises:
    print("test_intarray_setitem")
    var arr = IntArray(1, 2, 3, 4, 5)
    arr[0] = 100
    arr[2] = 200
    arr[4] = 300
    assert_true(arr[0] == 100, "arr[0] should be 100")
    assert_true(arr[2] == 200, "arr[2] should be 200")
    assert_true(arr[4] == 300, "arr[4] should be 300")
    print("test_intarray_setitem passed")


fn test_intarray_setitem_negative() raises:
    print("test_intarray_setitem_negative")
    var arr = IntArray(1, 2, 3, 4, 5)
    arr[-1] = 999
    arr[-3] = 777
    assert_true(arr[-1] == 999, "arr[-1] should be 999")
    assert_true(arr[-3] == 777, "arr[-3] should be 777")
    print("test_intarray_setitem_negative passed")


fn test_intarray_slice() raises:
    print("test_intarray_slice")
    var arr = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var slice1 = arr[2:5]
    assert_true(len(slice1) == 3, "slice [2:5] should have 3 elements")
    assert_true(slice1[0] == 2, "first element should be 2")
    assert_true(slice1[2] == 4, "last element should be 4")
    print("test_intarray_slice passed")


fn test_intarray_slice_with_step() raises:
    print("test_intarray_slice_with_step")
    var arr = IntArray(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var slice1 = arr[0:10:2]
    assert_true(len(slice1) == 5, "slice [0:10:2] should have 5 elements")
    assert_true(slice1[0] == 0, "first element should be 0")
    assert_true(slice1[4] == 8, "last element should be 8")
    print("test_intarray_slice_with_step passed")


fn test_intarray_append() raises:
    print("test_intarray_append")
    var arr = IntArray()
    arr.append(10)
    arr.append(20)
    arr.append(30)
    assert_true(len(arr) == 3, "array should have 3 elements")
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[1] == 20, "second element should be 20")
    assert_true(arr[2] == 30, "third element should be 30")
    print("test_intarray_append passed")


fn test_intarray_append_growth() raises:
    print("test_intarray_append_growth")
    var arr = IntArray.with_capacity(2)
    for i in range(10):
        arr.append(i)
    assert_true(len(arr) == 10, "array should have 10 elements after growth")
    assert_true(arr.capacity() >= 10, "capacity should have grown")
    for i in range(10):
        assert_true(arr[i] == i, "element " + String(i) + " should be correct")
    print("test_intarray_append_growth passed")


fn test_intarray_prepend() raises:
    print("test_intarray_prepend")
    var arr = IntArray()
    arr.prepend(30)
    arr.prepend(20)
    arr.prepend(10)
    assert_true(len(arr) == 3, "array should have 3 elements")
    assert_true(arr[0] == 10, "first element should be 10")
    assert_true(arr[1] == 20, "second element should be 20")
    assert_true(arr[2] == 30, "third element should be 30")
    print("test_intarray_prepend passed")


fn test_intarray__add__() raises:
    print("test_intarray__add__")
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
    print("test_intarray__add__ passed")


fn test_intarray__add__self() raises:
    print("test_intarray__add__self")
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
    print("test_intarray__add__self passed")


fn test_intarray_pop() raises:
    print("test_intarray_pop")
    var arr = IntArray(10, 20, 30, 40, 50)
    var val = arr.pop()
    assert_true(val == 50, "popped value should be 50")
    assert_true(len(arr) == 4, "array should have 4 elements")
    assert_true(arr[3] == 40, "last element should now be 40")
    print("test_intarray_pop passed")


fn test_intarray_pop_index() raises:
    print("test_intarray_pop_index")
    var arr = IntArray(10, 20, 30, 40, 50)
    var val = arr.pop(2)
    assert_true(val == 30, "popped value should be 30")
    assert_true(len(arr) == 4, "array should have 4 elements")
    assert_true(arr[2] == 40, "element at index 2 should now be 40")
    print("test_intarray_pop_index passed")


fn test_intarray_clear() raises:
    print("test_intarray_clear")
    var arr = IntArray(1, 2, 3, 4, 5)
    arr.clear()
    assert_true(len(arr) == 0, "array should be empty after clear")
    assert_true(arr.is_empty(), "is_empty should return True")
    print("test_intarray_clear passed")


fn test_intarray_fill() raises:
    print("test_intarray_fill")
    var arr = IntArray(1, 2, 3, 4, 5)
    arr.fill(99)
    for i in range(5):
        assert_true(arr[i] == 99, "all elements should be 99")
    print("test_intarray_fill passed")


fn test_intarray_contains() raises:
    print("test_intarray_contains")
    var arr = IntArray(10, 20, 30, 40, 50)
    assert_true(30 in arr, "30 should be in array")
    assert_true(50 in arr, "50 should be in array")
    assert_true(not (99 in arr), "99 should not be in array")
    print("test_intarray_contains passed")


fn test_intarray_eq() raises:
    print("test_intarray_eq")
    var arr1 = IntArray(1, 2, 3, 4, 5)
    var arr2 = IntArray(1, 2, 3, 4, 5)
    var arr3 = IntArray(1, 2, 3, 4, 6)
    assert_true(arr1 == arr2, "arr1 should equal arr2")
    assert_true(not (arr1 == arr3), "arr1 should not equal arr3")
    print("test_intarray_eq passed")


fn test_intarray_eq_list() raises:
    print("test_intarray_eq_list")
    var arr = IntArray(1, 2, 3, 4, 5)
    var lst = List[Int]()
    for i in range(1, 6):
        lst.append(i)
    assert_true(arr == lst, "array should equal list")
    print("test_intarray_eq_list passed")


fn test_intarray_tolist() raises:
    print("test_intarray_tolist")
    var arr = IntArray(10, 20, 30)
    var lst = arr.tolist()
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[0] == 10, "first element should be 10")
    assert_true(lst[2] == 30, "last element should be 30")
    print("test_intarray_tolist passed")


fn test_intarray_str() raises:
    print("test_intarray_str")
    var arr = IntArray(1, 2, 3)
    var s = arr.__str__()
    assert_true(s == "[1, 2, 3]", "string representation should be '[1, 2, 3]'")
    print("test_intarray_str passed")


fn test_intarray_product() raises:
    print("test_intarray_product")
    var arr = IntArray(2, 3, 4)
    assert_true(arr.product() == 24, "product should be 24")
    var empty = IntArray()
    assert_true(empty.product() == 1, "empty product should be 1")
    print("test_intarray_product passed")


fn test_intarray_sum() raises:
    print("test_intarray_sum")
    var arr = IntArray(1, 2, 3, 4, 5)
    assert_true(arr.sum() == 15, "sum should be 15")
    var empty = IntArray()
    assert_true(empty.sum() == 0, "empty sum should be 0")
    print("test_intarray_sum passed")


fn test_intarray_reverse() raises:
    print("test_intarray_reverse")
    var arr = IntArray(1, 2, 3, 4, 5)
    arr.reverse()
    assert_true(arr[0] == 5, "first element should be 5")
    assert_true(arr[4] == 1, "last element should be 1")
    print("test_intarray_reverse passed")


fn test_intarray_reversed() raises:
    print("test_intarray_reversed")
    var arr = IntArray(1, 2, 3, 4, 5)
    var rev = arr.reversed()
    assert_true(rev[0] == 5, "first element should be 5")
    assert_true(rev[4] == 1, "last element should be 1")
    assert_true(arr[0] == 1, "original array unchanged")
    print("test_intarray_reversed passed")


fn test_intarray_copyinit() raises:
    print("test_intarray_copyinit")
    var arr1 = IntArray(1, 2, 3, 4, 5)
    var arr2 = arr1
    arr2[0] = 999
    assert_true(arr1[0] == 1, "original array should be unchanged (deep copy)")
    assert_true(arr2[0] == 999, "copy should be modified")
    print("test_intarray_copyinit passed")


fn run_all_intarray_tests() raises:
    print("\n" + "=" * 60)
    print("RUNNING INTARRAY TESTS")
    print("=" * 60)

    test_intarray_default_constructor()
    test_intarray_size_constructor()
    test_intarray_variadic_constructor()
    test_intarray_list_constructor()
    test_intarray_filled()
    test_intarray_range()
    test_intarray_range_with_step()
    test_intarray_with_capacity()
    test_intarray_getitem()
    test_intarray_getitem_negative()
    test_intarray_setitem()
    test_intarray_setitem_negative()
    test_intarray_slice()
    test_intarray_slice_with_step()
    test_intarray_append()
    test_intarray_append_growth()
    test_intarray_prepend()
    test_intarray__add__()
    test_intarray__add__self()
    test_intarray_pop()
    test_intarray_pop_index()
    test_intarray_clear()
    test_intarray_fill()
    test_intarray_contains()
    test_intarray_eq()
    test_intarray_eq_list()
    test_intarray_tolist()
    test_intarray_str()
    test_intarray_product()
    test_intarray_sum()
    test_intarray_reverse()
    test_intarray_reversed()
    test_intarray_copyinit()
    run_intarray_insert_replace_tests()
    print("\n" + "=" * 60)
    print("ALL INTARRAY TESTS PASSED ✓")
    print("=" * 60)


fn main() raises:
    run_all_intarray_tests()
    run_all_intarray_v2_tests()
