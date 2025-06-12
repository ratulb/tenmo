from testing import assert_false
from testing import assert_true
from dynarray import DynArray

fn test_flexarray__add__() raises:
    fa = DynArray(1, 2, 3)
    result = fa + fa
    assert_true(
        result == DynArray(1, 2, 3, 1, 2, 3),
        "DynArray __add__ assertion failed",
    )
    result = result + DynArray()
    assert_true(
        result == result,
        "DynArray __add__ empty DynArray assertion failed",
    )
    result = DynArray(1) + DynArray(2)
    assert_true(
        result == DynArray(1, 2),
        "DynArray __add__ 1 2 assertion failed",
    )


fn test_flexarray__mul__() raises:
    fa = DynArray(1, 2, 3, 4)
    result = fa * 2
    assert_true(
        result == DynArray(1, 2, 3, 4, 1, 2, 3, 4),
        "DynArray __mul__ assertion failed",
    )
    result = fa * 3
    assert_true(
        result == DynArray(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4),
        "DynArray 2nd __mul__ assertion failed",
    )



fn test_flexarray_copy() raises:
    fa = DynArray(1.1, 2.2, 3.3)
    #assert_true(fa.T == Float64, "DynArray dtype assertion failed")
    fa.print()
    cp = fa.copy()
    assert_true(fa[0] == cp[0], "DynArray copy assertion failed")


fn test_flexarray_iter() raises:
    iterator = DynArray(11, 22, 33, 44)
    index = 0
    fa = DynArray(11, 22, 33, 44)
    for each in fa:
        assert_true(
            each == iterator[index], "Forward iterator assertion failed"
        )
        index += 1
    for each in fa.__reversed__():
        index -= 1
        assert_true(
            each == iterator[index], "Backward iterator assertion failed"
        )
    fab = DynArray(True, True, True, False, False, False)
    for each in fab:
        if index <= 2:
            assert_true(each, "Forward bool iterator assertion failed")
        else:
            assert_false(each, "Forward bool iterator assertion failed")

        index += 1
    for each in fab.__reversed__():
        index -= 1
        if index > 2:
            assert_false(each, "Forward bool iterator assertion failed")
        else:
            assert_true(each, "Forward bool iterator assertion failed")


fn test_flexarray() raises:
    fa = DynArray()
    assert_false(fa.data, "Uninitialized data pointer assertion failed")
    fa.append(100)
    assert_true(
        fa.size == 1 and fa.capacity == 1, "size and capacity assertion failed"
    )
    assert_true(100 in fa, "contains assertion failed")
    copy1 = fa
    assert_true(
        100 in copy1, "Post __copyint__ destination contains assertion failed"
    )
    assert_true(100 in fa, "Post __copyint__ source contains assertion failed")
    fa2 = DynArray()
    fa2.copy_from(0, fa, 1)
    assert_true(
        100 in fa2, "Post copy_from destination contains assertion failed"
    )
    fa3 = DynArray(True, True, True, False)
    fa4 = DynArray[Bool]()
    fa4.copy_from(4, fa3, 4)
    assert_true(
        len(fa4) == 8,
        "Copy from source - capacity initialization assertion failed",
    )
    fa5 = DynArray[Bool].with_capacity(12)
    assert_true(len(fa5) == 0, "with_capacity len assertion failed")
    fa5.copy_from(4, fa3, 4)
    assert_true(len(fa5) == 8, "post with_capacity len assertion failed")


fn main() raises:
    test_flexarray_copy()
    test_flexarray()
    test_flexarray_iter()
    test_flexarray__mul__()
    test_flexarray__add__()
