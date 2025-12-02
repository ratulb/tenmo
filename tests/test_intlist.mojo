from intlist import IntList
from testing import assert_true, assert_false

fn test_intlist_filled() raises:
    print("test_intlist_filled")
    var lst = IntList.filled(5, 42)
    assert_true(len(lst) == 5, "filled should create list of size 5")
    for i in range(5):
        assert_true(lst[i] == 42, "all elements should be 42")
    print("test_intlist_filled passed")

fn test_intlist_range_list() raises:
    print("test_intlist_range_list")
    var lst = IntList.range_list(10)
    assert_true(len(lst) == 10, "range_list should create list of size 10")
    for i in range(10):
        assert_true(lst[i] == i, "element " + String(i) + " should be " + String(i))
    print("test_intlist_range_list passed")

fn test_intlist_with_capacity() raises:
    print("test_intlist_with_capacity")
    var lst = IntList.with_capacity(20)
    assert_true(len(lst) == 0, "with_capacity should start with size 0")
    assert_true(lst.capacity() >= 20, "capacity should be at least 20")
    print("test_intlist_with_capacity passed")

fn test_intlist_append() raises:
    print("test_intlist_append")
    var lst = IntList()
    lst.append(1)
    lst.append(2)
    lst.append(3)
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[2] == 3, "last element should be 3")
    print("test_intlist_append passed")

fn test_intlist_prepend() raises:
    print("test_intlist_prepend")
    var lst = IntList()
    lst.prepend(3)
    lst.prepend(2)
    lst.prepend(1)
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[0] == 1, "first element should be 1")
    print("test_intlist_prepend passed")

fn test_intlist_pop() raises:
    print("test_intlist_pop")
    var lst = IntList(10, 20, 30, 40, 50)
    var val = lst.pop()
    assert_true(val == 50, "popped value should be 50")
    assert_true(len(lst) == 4, "list should have 4 elements")
    print("test_intlist_pop passed")

fn test_intlist_clear() raises:
    print("test_intlist_clear")
    var lst = IntList(1, 2, 3, 4, 5)
    lst.clear()
    assert_true(len(lst) == 0, "list should be empty")
    print("test_intlist_clear passed")

fn test_intlist_add() raises:
    print("test_intlist_add")
    var lst1 = IntList(1, 2, 3)
    var lst2 = IntList(4, 5, 6)
    var result = lst1 + lst2
    assert_true(len(result) == 6, "concatenated list should have 6 elements")
    assert_true(result[0] == 1, "first element should be 1")
    assert_true(result[5] == 6, "last element should be 6")
    print("test_intlist_add passed")

fn test_intlist_mul_scalar() raises:
    print("test_intlist_mul_scalar")
    var lst = IntList(1, 2, 3)
    var result = lst * 3
    assert_true(len(result) == 9, "repeated list should have 9 elements")
    assert_true(result[0] == 1, "first element should be 1")
    assert_true(result[6] == 1, "element 6 should be 1")
    print("test_intlist_mul_scalar passed")

fn test_intlist_mul_elementwise() raises:
    print("test_intlist_mul_elementwise")
    var lst1 = IntList(2, 3, 4)
    var lst2 = IntList(5, 6, 7)
    var result = lst1 * lst2
    assert_true(len(result) == 3, "result should have 3 elements")
    assert_true(result[0] == 10, "2 * 5 = 10")
    assert_true(result[1] == 18, "3 * 6 = 18")
    assert_true(result[2] == 28, "4 * 7 = 28")
    print("test_intlist_mul_elementwise passed")

fn test_intlist_product() raises:
    print("test_intlist_product")
    var lst = IntList(2, 3, 4)
    assert_true(lst.product() == 24, "product should be 24")
    print("test_intlist_product passed")

fn test_intlist_sum() raises:
    print("test_intlist_sum")
    var lst = IntList(1, 2, 3, 4, 5)
    assert_true(lst.sum() == 15, "sum should be 15")
    print("test_intlist_sum passed")

fn test_intlist_reverse() raises:
    print("test_intlist_reverse")
    var lst = IntList(1, 2, 3, 4, 5)
    lst.reverse()
    assert_true(lst[0] == 5, "first element should be 5")
    assert_true(lst[4] == 1, "last element should be 1")
    print("test_intlist_reverse passed")

fn test_intlist_reversed() raises:
    print("test_intlist_reversed")
    var lst = IntList(1, 2, 3, 4, 5)
    var rev = lst.reversed()
    assert_true(rev[0] == 5, "first element should be 5")
    assert_true(lst[0] == 1, "original unchanged")
    print("test_intlist_reversed passed")

fn test_intlist_count() raises:
    print("test_intlist_count")
    var lst = IntList(1, 2, 3, 2, 4, 2, 5)
    assert_true(lst.count(2) == 3, "count of 2 should be 3")
    assert_true(lst.count(99) == 0, "count of 99 should be 0")
    print("test_intlist_count passed")

fn test_intlist_has_duplicates() raises:
    print("test_intlist_has_duplicates")
    var lst1 = IntList(1, 2, 3, 4, 5)
    var lst2 = IntList(1, 2, 3, 2, 4)
    assert_true(not lst1.has_duplicates(), "lst1 should have no duplicates")
    assert_true(lst2.has_duplicates(), "lst2 should have duplicates")
    print("test_intlist_has_duplicates passed")

fn test_intlist_sort() raises:
    print("test_intlist_sort")
    var lst = IntList(5, 2, 8, 1, 9)
    lst.sort()
    assert_true(lst[0] == 1, "first element should be 1")
    assert_true(lst[4] == 9, "last element should be 9")
    print("test_intlist_sort passed")

fn test_intlist_sorted() raises:
    print("test_intlist_sorted")
    var lst = IntList(5, 2, 8, 1, 9)
    var sorted_lst = lst.sorted()
    assert_true(sorted_lst[0] == 1, "sorted first element should be 1")
    assert_true(lst[0] == 5, "original unchanged")
    print("test_intlist_sorted passed")

fn test_intlist_swap() raises:
    print("test_intlist_swap")
    var lst = IntList(10, 20, 30, 40, 50)
    lst.swap(1, 3)
    assert_true(lst[1] == 40, "element at index 1 should be 40")
    assert_true(lst[3] == 20, "element at index 3 should be 20")
    print("test_intlist_swap passed")

fn test_intlist_replace_single() raises:
    print("test_intlist_replace_single")
    var lst = IntList(1, 2, 3, 4, 5)
    var result = lst.replace(2, 99)
    assert_true(result[2] == 99, "element at index 2 should be 99")
    assert_true(lst[2] == 3, "original unchanged")
    print("test_intlist_replace_single passed")

fn test_intlist_replace_multiple() raises:
    print("test_intlist_replace_multiple")
    var lst = IntList(1, 2, 3, 4, 5)
    var indices = IntList(1, 3)
    var values = IntList(20, 40)
    var result = lst.replace(indices, values)
    assert_true(result[1] == 20, "element at index 1 should be 20")
    assert_true(result[3] == 40, "element at index 3 should be 40")
    print("test_intlist_replace_multiple passed")

fn test_intlist_permute() raises:
    print("test_intlist_permute")
    var lst = IntList(10, 20, 30, 40)
    var axes = IntList(2, 0, 3, 1)
    var result = lst.permute(axes)
    assert_true(result[0] == 30, "first element should be 30")
    assert_true(result[1] == 10, "second element should be 10")
    assert_true(result[2] == 40, "third element should be 40")
    assert_true(result[3] == 20, "fourth element should be 20")
    print("test_intlist_permute passed")

fn test_intlist_select() raises:
    print("test_intlist_select")
    var lst = IntList(10, 20, 30, 40, 50)
    var indices = IntList(0, 2, 4)
    var result = lst.select(indices)
    assert_true(len(result) == 3, "result should have 3 elements")
    assert_true(result[0] == 10, "first element should be 10")
    assert_true(result[1] == 30, "second element should be 30")
    assert_true(result[2] == 50, "third element should be 50")
    print("test_intlist_select passed")

fn test_intlist_indices_of() raises:
    print("test_intlist_indices_of")
    var lst = IntList(1, 2, 3, 2, 4, 2, 5)
    var indices = lst.indices_of(2)
    assert_true(len(indices) == 3, "should find 3 occurrences")
    assert_true(indices[0] == 1, "first occurrence at index 1")
    assert_true(indices[1] == 3, "second occurrence at index 3")
    assert_true(indices[2] == 5, "third occurrence at index 5")
    print("test_intlist_indices_of passed")

fn test_intlist_insert_single() raises:
    print("test_intlist_insert_single")
    var lst = IntList(1, 2, 4, 5)
    var result = lst.insert(2, 3)
    assert_true(len(result) == 5, "result should have 5 elements")
    assert_true(result[2] == 3, "inserted element should be at index 2")
    print("test_intlist_insert_single passed")

fn test_intlist_invert_permutation() raises:
    print("test_intlist_invert_permutation")
    var perm = IntList(2, 0, 1)
    var inv = IntList.invert_permutation(perm)
    assert_true(inv[0] == 1, "inv[0] should be 1")
    assert_true(inv[1] == 2, "inv[1] should be 2")
    assert_true(inv[2] == 0, "inv[2] should be 0")
    print("test_intlist_invert_permutation passed")

fn test_intlist_tolist() raises:
    print("test_intlist_tolist")
    var lst = IntList(10, 20, 30)
    var l = lst.tolist()
    assert_true(len(l) == 3, "list should have 3 elements")
    assert_true(l[0] == 10, "first element should be 10")
    print("test_intlist_tolist passed")

fn test_intlist_intarray() raises:
    print("test_intlist_intarray")
    var lst = IntList(5, 10, 15)
    var arr = lst.intarray()
    assert_true(len(arr) == 3, "array should have 3 elements")
    assert_true(arr[1] == 10, "second element should be 10")
    print("test_intlist_intarray passed")


fn intlist_test_product_empty() raises:
    print("intlist_test_product_empty")
    var il = IntList()
    assert_true(il.product() == 1, "Empty product should be 1")
    print("intlist_test_product_empty passed")


fn intlist_test_product() raises:
    print("intlist_test_product")
    var il = IntList(1, 2, 3, 4, 5)
    assert_true(il.product() == 120, "1*2*3*4*5 = 120")
    print("intlist_test_product passed")


fn intlist_test_sum() raises:
    print("intlist_test_sum")
    var il = IntList(1, 2, 3, 4, 5)
    assert_true(il.sum() == 15, "1+2+3+4+5 = 15")
    print("intlist_test_sum passed")


fn intlist_test_append_grow() raises:
    print("intlist_test_append_grow")
    var il = IntList()
    for i in range(100):
        il.append(i)
    assert_true(len(il) == 100, "Length should be 100")
    for i in range(100):
        assert_true(il[i] == i, "Value mismatch")
    print("intlist_test_append_grow passed")


fn intlist_test_negative_indexing() raises:
    print("intlist_test_negative_indexing")
    var il = IntList(10, 20, 30, 40, 50)
    assert_true(il[-1] == 50, "il[-1] should be 50")
    assert_true(il[-2] == 40, "il[-2] should be 40")
    assert_true(il[-5] == 10, "il[-5] should be 10")
    print("intlist_test_negative_indexing passed")


fn intlist_test_slice() raises:
    print("intlist_test_slice")
    var il = IntList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    var slice1 = il[2:5]
    assert_true(len(slice1) == 3, "Slice length")
    assert_true(slice1[0] == 2, "Slice[0]")
    assert_true(slice1[2] == 4, "Slice[2]")

    var slice2 = il[::2]  # Every other element
    assert_true(len(slice2) == 5, "Step slice length")
    assert_true(slice2[0] == 0, "Step slice[0]")
    assert_true(slice2[2] == 4, "Step slice[2]")
    print("intlist_test_slice passed")


fn intlist_test_sort() raises:
    print("intlist_test_sort")
    var il = IntList(5, 2, 8, 1, 9, 3)
    il.sort()
    assert_true(il[0] == 1, "Sorted[0]")
    assert_true(il[5] == 9, "Sorted[5]")

    il.sort(asc=False)
    assert_true(il[0] == 9, "Desc sorted[0]")
    assert_true(il[5] == 1, "Desc sorted[5]")
    print("intlist_test_sort passed")


fn intlist_test_sorted() raises:
    print("intlist_test_sorted")
    var il = IntList(5, 2, 8, 1, 9, 3)
    var sorted_il = il.sorted()

    # Original unchanged
    assert_true(il[0] == 5, "Original unchanged")
    # Sorted copy
    assert_true(sorted_il[0] == 1, "Sorted copy")
    print("intlist_test_sorted passed")


fn intlist_test_sort_and_deduplicate() raises:
    print("intlist_test_sort_and_deduplicate")
    var il = IntList(3, 1, 2, 1, 3, 2, 1)
    il.sort_and_deduplicate()
    assert_true(len(il) == 3, "Deduplicated length")
    assert_true(il[0] == 1, "Dedup[0]")
    assert_true(il[1] == 2, "Dedup[1]")
    assert_true(il[2] == 3, "Dedup[2]")
    print("intlist_test_sort_and_deduplicate passed")


fn intlist_test_has_duplicates() raises:
    print("intlist_test_has_duplicates")
    var il1 = IntList(1, 2, 3, 4, 5)
    var il2 = IntList(1, 2, 3, 2, 5)
    assert_true(not il1.has_duplicates(), "No duplicates")
    assert_true(il2.has_duplicates(), "Has duplicates")
    print("intlist_test_has_duplicates passed")


fn intlist_test_select() raises:
    print("intlist_test_select")
    var il = IntList(10, 20, 30, 40, 50)
    var indices = IntList(0, 2, 4)
    var selected = il.select(indices)
    assert_true(len(selected) == 3, "Select length")
    assert_true(selected[0] == 10, "Select[0]")
    assert_true(selected[1] == 30, "Select[1]")
    assert_true(selected[2] == 50, "Select[2]")
    print("intlist_test_select passed")


fn intlist_test_permute() raises:
    print("intlist_test_permute")
    var il = IntList(10, 20, 30, 40)
    var axes = IntList(3, 1, 2, 0)
    var permuted = il.permute(axes)
    assert_true(permuted[0] == 40, "Permute[0]")
    assert_true(permuted[1] == 20, "Permute[1]")
    assert_true(permuted[2] == 30, "Permute[2]")
    assert_true(permuted[3] == 10, "Permute[3]")
    print("intlist_test_permute passed")


fn intlist_test_insert_single() raises:
    print("intlist_test_insert_single")
    var il = IntList(1, 2, 4, 5)
    var result = il.insert(2, 3)
    assert_true(len(result) == 5, "Insert length")
    assert_true(result[2] == 3, "Inserted value")
    assert_true(result == IntList(1, 2, 3, 4, 5), "Full comparison")
    print("intlist_test_insert_single passed")


fn intlist_test_insert_at_start() raises:
    print("intlist_test_insert_at_start")
    var il = IntList(2, 3, 4)
    var result = il.insert(0, 1)
    assert_true(result == IntList(1, 2, 3, 4), "Insert at start")
    print("intlist_test_insert_at_start passed")


fn intlist_test_insert_at_end() raises:
    print("intlist_test_insert_at_end")
    var il = IntList(1, 2, 3)
    var result = il.insert(3, 4)
    assert_true(result == IntList(1, 2, 3, 4), "Insert at end")
    print("intlist_test_insert_at_end passed")


fn intlist_test_pop() raises:
    print("intlist_test_pop")
    var il = IntList(1, 2, 3, 4, 5)

    var last = il.pop()
    assert_true(last == 5, "Pop last")
    assert_true(len(il) == 4, "Length after pop")

    var first = il.pop(0)
    assert_true(first == 1, "Pop first")
    assert_true(len(il) == 3, "Length after pop(0)")

    var middle = il.pop(1)
    assert_true(middle == 3, "Pop middle")
    assert_true(il == IntList(2, 4), "Remaining")
    print("intlist_test_pop passed")


fn intlist_test_replace_single() raises:
    print("intlist_test_replace_single")
    var il = IntList(1, 2, 3, 4, 5)
    var result = il.replace(2, 99)
    assert_true(result[2] == 99, "Replace single")
    assert_true(il[2] == 3, "Original unchanged")
    print("intlist_test_replace_single passed")


fn intlist_test_replace_multiple() raises:
    print("intlist_test_replace_multiple")
    var il = IntList(1, 2, 3, 4, 5)
    var indices = IntList(0, 2, 4)
    var values = IntList(10, 30, 50)
    var result = il.replace(indices, values)
    assert_true(result == IntList(10, 2, 30, 4, 50), "Replace multiple")
    print("intlist_test_replace_multiple passed")


fn intlist_test_swap() raises:
    print("intlist_test_swap")
    var il = IntList(1, 2, 3, 4, 5)
    il.swap(0, 4)
    assert_true(il[0] == 5, "Swap[0]")
    assert_true(il[4] == 1, "Swap[4]")
    print("intlist_test_swap passed")


fn intlist_test_reverse() raises:
    print("intlist_test_reverse")
    var il = IntList(1, 2, 3, 4, 5)
    il.reverse()
    assert_true(il == IntList(5, 4, 3, 2, 1), "Reversed")
    print("intlist_test_reverse passed")


fn intlist_test_reversed() raises:
    print("intlist_test_reversed")
    var il = IntList(1, 2, 3, 4, 5)
    var rev = il.reversed()
    assert_true(rev == IntList(5, 4, 3, 2, 1), "Reversed copy")
    assert_true(il == IntList(1, 2, 3, 4, 5), "Original unchanged")
    print("intlist_test_reversed passed")


fn intlist_test_prepend() raises:
    print("intlist_test_prepend")
    var il = IntList(2, 3, 4)
    il.prepend(1)
    assert_true(il == IntList(1, 2, 3, 4), "Prepend")
    print("intlist_test_prepend passed")


fn intlist_test_clear() raises:
    print("intlist_test_clear")
    var il = IntList(1, 2, 3, 4, 5)
    il.clear()
    assert_true(len(il) == 0, "Cleared length")
    assert_true(il.is_empty(), "Is empty")
    print("intlist_test_clear passed")


fn intlist_test_contains() raises:
    print("intlist_test_contains")
    var il = IntList(1, 2, 3, 4, 5)
    assert_true(3 in il, "Contains 3")
    assert_true(not (99 in il), "Not contains 99")
    print("intlist_test_contains passed")


fn intlist_test_count() raises:
    print("intlist_test_count")
    var il = IntList(1, 2, 2, 3, 2, 4)
    assert_true(il.count(2) == 3, "Count 2s")
    assert_true(il.count(99) == 0, "Count 99s")
    print("intlist_test_count passed")


fn intlist_test_indices_of() raises:
    print("intlist_test_indices_of")
    var il = IntList(1, 2, 3, 2, 4, 2)
    var indices = il.indices_of(2)
    assert_true(indices == IntList(1, 3, 5), "Indices of 2")
    print("intlist_test_indices_of passed")


fn intlist_test_add_lists() raises:
    print("intlist_test_add_lists")
    var il1 = IntList(1, 2, 3)
    var il2 = IntList(4, 5, 6)
    var result = il1 + il2
    assert_true(result == IntList(1, 2, 3, 4, 5, 6), "Concatenate")
    print("intlist_test_add_lists passed")


fn intlist_test_mul_factor() raises:
    print("intlist_test_mul_factor")
    var il = IntList(1, 2, 3)
    var result = il * 3
    assert_true(result == IntList(1, 2, 3, 1, 2, 3, 1, 2, 3), "Repeat")
    print("intlist_test_mul_factor passed")


fn intlist_test_mul_elementwise() raises:
    print("intlist_test_mul_elementwise")
    var il1 = IntList(1, 2, 3)
    var il2 = IntList(4, 5, 6)
    var result = il1 * il2
    assert_true(result == IntList(4, 10, 18), "Elementwise mul")
    print("intlist_test_mul_elementwise passed")


fn intlist_test_iterator() raises:
    print("intlist_test_iterator")
    var il = IntList(1, 2, 3, 4, 5)
    var sum = 0
    for val in il:
        sum += val
    assert_true(sum == 15, "Iterator sum")
    print("intlist_test_iterator passed")


fn intlist_test_reversed_iterator() raises:
    print("intlist_test_reversed_iterator")
    var il = IntList(1, 2, 3, 4, 5)
    var result = IntList()
    for val in il.__reversed__():
        result.append(val)
    assert_true(result == IntList(5, 4, 3, 2, 1), "Reversed iterator")
    print("intlist_test_reversed_iterator passed")


fn intlist_test_range_list() raises:
    print("intlist_test_range_list")
    var il = IntList.range_list(5)
    assert_true(il == IntList(0, 1, 2, 3, 4), "Range list")
    print("intlist_test_range_list passed")


fn intlist_test_filled() raises:
    print("intlist_test_filled")
    var il = IntList.filled(5, 42)
    assert_true(len(il) == 5, "Filled length")
    for i in range(5):
        assert_true(il[i] == 42, "Filled value")
    print("intlist_test_filled passed")


fn intlist_test_invert_permutation() raises:
    print("intlist_test_invert_permutation")
    var perm = IntList(2, 0, 3, 1)
    var inv = IntList.invert_permutation(perm)
    # Verify: inv[perm[i]] == i
    for i in range(4):
        assert_true(inv[perm[i]] == i, "Invert permutation")
    print("intlist_test_invert_permutation passed")


fn intlist_test_any() raises:
    print("intlist_test_any")
    var il = IntList(1, 2, 3, 4, 5)

    fn gt_3(x: Int) -> Bool:
        return x > 3

    fn gt_10(x: Int) -> Bool:
        return x > 10

    assert_true(il.any(gt_3), "Any > 3")
    assert_true(not il.any(gt_10), "None > 10")
    print("intlist_test_any passed")


fn run_all_intlist_tests() raises:
    print("=" * 60)
    print("Running all IntList tests")
    print("=" * 60)

    intlist_test_product_empty()
    intlist_test_product()
    intlist_test_sum()
    intlist_test_append_grow()
    intlist_test_negative_indexing()
    intlist_test_slice()
    intlist_test_sort()
    intlist_test_sorted()
    intlist_test_sort_and_deduplicate()
    intlist_test_has_duplicates()
    intlist_test_select()
    intlist_test_permute()
    intlist_test_insert_single()
    intlist_test_insert_at_start()
    intlist_test_insert_at_end()
    intlist_test_pop()
    intlist_test_replace_single()
    intlist_test_replace_multiple()
    intlist_test_swap()
    intlist_test_reverse()
    intlist_test_reversed()
    intlist_test_prepend()
    intlist_test_clear()
    intlist_test_contains()
    intlist_test_count()
    intlist_test_indices_of()
    intlist_test_add_lists()
    intlist_test_mul_factor()
    intlist_test_mul_elementwise()
    intlist_test_iterator()
    intlist_test_reversed_iterator()
    intlist_test_range_list()
    intlist_test_filled()
    intlist_test_invert_permutation()
    intlist_test_any()

    print("=" * 60)
    print("All IntList tests passed!")
    print("=" * 60)


#######################
fn test_large_list() raises:
    print("test_large_list")
    var big = IntList()
    for i in range(100, 0, -1):  # 100 to 1
        big.append(i)
    big.sort()

    for i in range(1, 101):
        assert_true(i in big)
    assert_true(0 not in big)
    assert_true(101 not in big)


fn test_contains_unsorted() raises:
    print("test_contains_unsorted")
    var raw = IntList()
    raw.append(10)
    raw.append(3)
    raw.append(99)

    assert_true(10 in raw)
    assert_true(3 in raw)
    assert_true(99 in raw)
    assert_true(1 not in raw)


fn test_negative_and_duplicates() raises:
    print("test_negative_and_duplicates")
    var nums = IntList()
    nums.append(-10)
    nums.append(0)
    nums.append(-5)
    nums.append(-10)
    nums.append(5)
    nums.sort()

    assert_true(-10 in nums)
    assert_true(-5 in nums)
    assert_true(0 in nums)
    assert_true(5 in nums)
    assert_true(1 not in nums)


fn test_edge_cases() raises:
    print("test_edge_cases")
    var empty = IntList()
    assert_true(42 not in empty)

    var single = IntList()
    single.append(7)
    assert_true(7 in single)
    assert_true(8 not in single)

    single.sort()
    assert_true(single[0] == 7)


fn test_sorted_contains() raises:
    print("test_sorted_contains")
    var list = IntList()
    list.append(5)
    list.append(1)
    list.append(3)
    list.sort(asc=True)

    assert_true(list[0] == 1)
    assert_true(list[1] == 3)
    assert_true(list[2] == 5)

    assert_true(1 in list)
    assert_true(3 in list)
    assert_true(5 in list)
    assert_true(2 not in list)


fn test_prepend() raises:
    print("test_prepend")
    il = IntList()
    il.prepend(2)
    assert_true(il == IntList(2) and len(il) == 1, "append assertion 1 failed")
    il.prepend(1)
    assert_true(
        il == IntList(1, 2) and len(il) == 2, "append assertion 2 failed"
    )
    il = IntList(2, 3)
    il.prepend(1)
    assert_true(
        il == IntList(1, 2, 3) and len(il) == 3, "append assertion 3 failed"
    )


fn test_slice() raises:
    print("test_slice")
    il = IntList.range_list(15)
    sliced = il[2::3]
    assert_true(sliced == IntList(2, 5, 8, 11, 14), "slice assertion failed")


fn test_deduplicate() raises:
    print("test_deduplicate")
    il = IntList(9, 2, 9, 1, 4, 3, 1, 5, 7, 2, 1, 4, 7)
    il.sort_and_deduplicate()
    assert_true(
        il
        == IntList(
            1,
            2,
            3,
            4,
            5,
            7,
            9,
        ),
        "deduplicate assertion failed",
    )


fn test_new() raises:
    print("test_new")
    l = List(1, 2, 3)
    il = IntList.new(l)
    assert_true(il == IntList(1, 2, 3), "new assertion 1 failed")
    l = List[Int]()
    il = IntList.new(l)
    assert_true(il == IntList(), "new assertion 2 failed")


fn test_range_list() raises:
    print("test_range_list")
    il = IntList.range_list(3)
    assert_true(il == IntList(0, 1, 2), "range_list assertion 1 failed")
    il = IntList.range_list(0)
    assert_true(il == IntList(), "range_list assertion 2 failed")


fn test_has_duplicates() raises:
    print("test_has_duplicates")
    il = IntList(1, 0, 1, 2, 1)
    assert_true(il.has_duplicates(), "has_duplicates True assertion failed")
    il = IntList(1)
    assert_false(il.has_duplicates(), "has_duplicates False assertion failed")
    il = IntList(1, 2, 3)
    assert_false(
        il.has_duplicates(), "has_duplicates False assertion 2  failed"
    )


fn test_indices_of() raises:
    print("test_indices_of")
    il = IntList(1, 0, 1, 2, 1)
    indices = il.indices_of(1)
    assert_true(indices == IntList(0, 2, 4), "indices_of assertion 1 failed")
    indices = il.indices_of(0)
    assert_true(indices == IntList(1), "indices_of assertion 2 failed")
    indices = il.indices_of(2)
    assert_true(indices == IntList(3), "indices_of assertion 3 failed")
    indices = il.indices_of(5)
    assert_true(indices == IntList(), "indices_of assertion 4 failed")


fn test_count() raises:
    print("test_count")
    il = IntList(0, 3, 0, 5, 0)
    assert_true(il.count(0) == 3, "count assertion failed")


fn test_bulk_replace() raises:
    print("test_bulk_replace")
    il = IntList(0, 1, 0, 1, 0)
    result = il.replace(IntList(1, 3), IntList(3, 5))
    assert_true(
        result == IntList(0, 3, 0, 5, 0), "bulk replace assertion failed"
    )


fn test_bulk_insert() raises:
    print("test_bulk_insert")
    il = IntList(0, 0, 0)
    result = il.insert(IntList(1, 3), IntList(3, 5))
    assert_true(
        result == IntList(0, 3, 0, 5, 0), "bulk insert assertion failed"
    )


fn test_with_capacity_fill() raises:
    print("test_with_capacity_fill")
    il = IntList.with_capacity(3, -10)
    assert_true(
        il == IntList(-10, -10, -10) and len(il) == 3,
        "with_capacity with fill assertion failed",
    )


fn test_select() raises:
    print("test_select")
    il = IntList(9, 2, 3, 4, 5, 6)
    assert_true(
        il.select(IntList(2, 5)) == IntList(3, 6),
        "select assertion 1 failed",
    )
    assert_true(
        il.select(IntList(0, 4, 1)) == IntList(9, 5, 2),
        "select assertion 2 failed",
    )


fn test_sorted() raises:
    print("test_sorted")
    il = IntList(9, 2, 3, 4, 5, 6)
    assert_true(
        il.sorted() == IntList(2, 3, 4, 5, 6, 9),
        "Ascending sorted assertion failed",
    )
    assert_true(
        il.sorted(False) == IntList(9, 6, 5, 4, 3, 2),
        "Descending sorted assertion failed",
    )


fn test_of() raises:
    print("test_of")
    l = List(9, 2, 3, 4, 5, 6)
    il = IntList.new(l)
    il.sort()
    assert_true(
        il == IntList(2, 3, 4, 5, 6, 9), "IntList of and sort assertion failed"
    )
    l2 = [9, 2, 3, 4, 5, 6]
    il2 = IntList.new(l2)
    il2.sort(False)
    assert_true(
        il2 == IntList(9, 6, 5, 4, 3, 2),
        "IntList of and sort assertion 2 failed",
    )


fn test_sort() raises:
    print("test_sort")
    il = IntList(9, 2, 3, 4, 5, 6)
    il.sort()
    assert_true(
        il == IntList(2, 3, 4, 5, 6, 9), "Ascending sort assertion failed"
    )
    il.sort(False)
    assert_true(
        il == IntList(9, 6, 5, 4, 3, 2), "Descending sort assertion failed"
    )


fn test_insert() raises:
    print("test_insert")
    il = IntList(2, 3, 4, 5, 6)
    inserted = il.insert(0, 9)
    assert_true(
        inserted == IntList(9, 2, 3, 4, 5, 6),
        "IntList -> insert at 0 assertion failed",
    )
    inserted = il.insert(1, 9)
    assert_true(
        inserted == IntList(2, 9, 3, 4, 5, 6),
        "IntList -> insert at 1 assertion failed",
    )

    inserted = il.insert(2, 9)
    assert_true(
        inserted == IntList(2, 3, 9, 4, 5, 6),
        "IntList -> insert at 2 assertion failed",
    )

    inserted = il.insert(3, 9)
    assert_true(
        inserted == IntList(2, 3, 4, 9, 5, 6),
        "IntList -> insert at 3 assertion failed",
    )

    inserted = il.insert(4, 9)
    assert_true(
        inserted == IntList(2, 3, 4, 5, 9, 6),
        "IntList -> insert at 4 assertion failed",
    )

    inserted = il.insert(5, 9)
    assert_true(
        inserted == IntList(2, 3, 4, 5, 6, 9),
        "IntList -> insert at 3 assertion failed",
    )


fn test_copy() raises:
    print("test_copy")
    il = IntList(1, 2)
    copied = il.copy()
    assert_true(copied == IntList(1, 2), "copy assertion failed")


fn test_reverse() raises:
    print("test_reverse")
    il = IntList(1, 2)
    il.reverse()
    assert_true(il == IntList(2, 1), "reverse assertion failed")


fn test_pop() raises:
    print("test_pop")
    il = IntList(1, 2, 3)
    assert_true(
        il.pop() == 3 and il.pop() == 2 and il.pop() == 1 and len(il) == 0,
        "pop assertion failed",
    )


fn test_zip() raises:
    print("test_zip")
    l1 = IntList(1, 2, 3)
    l2 = IntList(4, 5, 6, 7)
    zipped = l1.zip(l2)
    i = 0
    for each in zipped:
        if i == 0:
            assert_true(
                each[0] == 1 and each[1] == 4,
                "zip iteration 0 - assertion failed",
            )
        if i == 1:
            assert_true(
                each[0] == 2 and each[1] == 5,
                "zip iteration 1 - assertion failed",
            )
        if i == 2:
            assert_true(
                each[0] == 3 and each[1] == 6,
                "zip iteration 2 - assertion failed",
            )
        i += 1


fn test_zip_reversed() raises:
    print("test_zip_reversed")
    l1 = IntList(1, 2, 3)
    l2 = IntList(4, 5, 6, 7)
    zipped = l1.zip_reversed(l2)

    i = 0
    for each in zipped:
        if i == 0:
            assert_true(
                each[0] == 3 and each[1] == 7,
                "zip reverse iteration 0 - assertion failed",
            )
        if i == 1:
            assert_true(
                each[0] == 2 and each[1] == 6,
                "zip reverse iteration 1 - assertion failed",
            )
        if i == 2:
            assert_true(
                each[0] == 1 and each[1] == 5,
                "zip reverse iteration 2 - assertion failed",
            )
        i += 1

    l1 = IntList(1, 2, 3)
    l2 = IntList(4, 5, 6)
    zipped = l1.zip_reversed(l2)

    i = 0
    for each in zipped:
        if i == 0:
            assert_true(
                each[0] == 3 and each[1] == 6,
                (
                    "equal length IntList zip reverse iteration 0 - assertion"
                    " failed"
                ),
            )
        if i == 1:
            assert_true(
                each[0] == 2 and each[1] == 5,
                (
                    "equal length IntList zip reverse iteration 1 - assertion"
                    " failed"
                ),
            )
        if i == 2:
            assert_true(
                each[0] == 1 and each[1] == 4,
                (
                    "Equal length IntList zip reverse iteration 2 - assertion"
                    " failed"
                ),
            )
        i += 1


fn test_product() raises:
    print("test_product")
    il = IntList(1, 3, 4, 10)
    assert_true(il.product() == 120, "product assertion failed")


fn test_replace() raises:
    print("test_replace")
    il = IntList(1, 2, 4)
    il = il.replace(2, 3)
    assert_true(il == IntList(1, 2, 3), "replace assertion failed")


fn test_init() raises:
    print("test_init")

    fn create_2(*elems: Int) raises:
        il = IntList.with_capacity(len(elems))
        for each in elems:
            il.append(each)
        assert_true(
            len(il) == 3 and il == IntList(1, 2, 3),
            "IntList init from runtime variadic elems assertion 1 failed",
        )

    create_2(1, 2, 3)

    fn create_1(*elems: Int) raises:
        length = len(elems)
        il = IntList(length)
        assert_true(
            len(il) == 1 and il == IntList(3),
            "IntList init from runtime variadic elems assertion 2 failed",
        )

    create_1(1, 2, 3)


fn test_clear() raises:
    print("test_clear")
    ll = IntList.new([1, 2, 3, 4, 5])
    ll.clear()
    assert_true(ll == IntList(), "clear assertion 1 failed")
    ll.append(200)
    ll.prepend(100)
    assert_true(ll == IntList(100, 200), "clear assertion 2 failed")


fn test_any() raises:
    print("test_any")
    l = IntList(1, 3, 0, -1)

    fn check(e: Int) -> Bool:
        return e == -1

    assert_true(l.any(check), "Any assertion for -1 failed")


fn test_negative_indices() raises:
    print("test_negative_indices")
    l1 = IntList(1, 2, 3)
    assert_true(
        l1[-1] == 3 and l1[-2] == 2 and l1[-3] == 1,
        "IntList negative indices assertion failed",
    )


fn test_permute() raises:
    print("test_permute")
    a = IntList(1, 2, 3)
    perm = IntList(1, 2, -3)
    permuted = a.permute(perm)
    assert_true(permuted == IntList(2, 3, 1), "permute assertion failed")


fn main() raises:
    run_all_intlist_tests()
    test_large_list()
    test_contains_unsorted()
    test_negative_and_duplicates()
    test_edge_cases()
    test_sorted_contains()
    test_prepend()
    test_slice()
    test_deduplicate()
    test_init()
    test_range_list()
    test_new()
    test_has_duplicates()
    test_indices_of()
    test_bulk_replace()
    test_count()
    test_bulk_insert()
    test_with_capacity_fill()
    test_select()
    test_sorted()
    test_of()
    test_sort()
    test_replace()
    test_product()
    test_insert()
    test_reverse()
    test_copy()
    test_pop()
    test_zip()
    test_zip_reversed()
    test_clear()
    test_any()
    test_negative_indices()
    test_permute()

    print("Done running IntList test cases")
