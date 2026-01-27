from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout import Layout
from sys import argv


fn main() raises:
    t1 = IntTuple(1, 3, 1)
    print(t1)
    t2 = create(9, 8, 1)
    print("UNKNOWN_VALUE: ", UNKNOWN_VALUE)
    t2 = t2.replace_entry(0, UNKNOWN_VALUE)
    print(t2.flatten().all_known(), t2.is_value(), t2.is_tuple())
    print(t2.flatten())
    var values = argv()
    for e in values:
        print("E: ", e)

    var (x, y, z) = Int(values[1]), Int(values[2]), Int(values[3])
    var t3 = create(x, y, z)
    print(t3.size())
    print("After size ")
    for e in t3:
        print("Guru print: ", e)
    t3 = t3.replace_entry(2, 999)
    print("At last: ", t3[2].count_values(), t3[2][0], t3[2][0].is_value())
    print(IntTuple(1, 2, 3, 4, 5, 6).size())
    print(IntTuple(1, 2, 3, 4, 5, 6).count_values())

    var layout = Layout()
    layout.append(Layout(x))
    print(layout)

    var tup = IntTuple()
    tup.extend(IntTuple(x))
    tup.append(IntTuple(y), IntTuple(z))
    print("Finally: ", tup, tup.product_flatten())
    var tupp = IntTuple(1, IntTuple(2, 3))
    print("tupp: ", tupp, tupp.product_flatten(), tupp.flatten())

    var layout2 = Layout(IntTuple(2, 3, 6), IntTuple(6, 2, 1))
    # print(layout2, layout.shape, layout.stride)
    print(layout2, layout2.shape, layout2.stride)

    var sub_layout = layout2[1]
    print("Sub layout: ", sub_layout, sub_layout.cosize())
    var one_d = Layout(IntTuple(7), IntTuple(1))
    print(
        one_d,
        one_d.shape,
        one_d.stride,
        one_d.cosize(),
        one_d.size(),
        one_d(IntTuple(42)),
        one_d.rank(),
    )


fn create(*entries: Int) -> IntTuple:
    return IntTuple(entries)
