from intlist import IntList
from testing import assert_true, assert_false


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
    assert_true(indices == IntList.Empty, "indices_of assertion 4 failed")


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
    print("Running IntList test cases")
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
