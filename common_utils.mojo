from shapes import Shape
from testing import assert_true


fn int_varia_list_to_str(list: VariadicList[Int]) -> String:
    s = String("[")
    for idx in range(len(list)):
        s += list[idx].__str__()
        if idx != len(list) - 1:
            s += ", "
    s += "]"
    return s


# Convert a VariadicList to List
fn varia_list_to_list(vlist: VariadicList[Int]) -> List[Int]:
    list = List[Int](capacity=len(vlist))
    for each in vlist:
        list.append(each)
    return list


# Create a single element VariadicList
fn passthrough(n: Int) -> VariadicList[Int]:
    fn single_elem_list(*single: Int) -> VariadicList[Int]:
        return single

    return single_elem_list(n)


fn validate_shape(shape: Shape) raises:
    for idx in range(shape.ndim):
        # print("Validating shape: ", shape.axes_spans[idx])
        assert_true(shape.axes_spans[idx] > 0, "Shape dimension not valid")


# Create a single element VariadicList
fn single_elem_list(elem: Int) raises -> VariadicList[Int]:
    alias int_type = __mlir_type[`!kgen.variadic<`, Int, `>`]
    typed_list = __mlir_op.`pop.variadic.create`[_type=int_type](elem)
    result = VariadicList(typed_list)
    assert_true(len(result) == 1, "Single element list length assertion failed")
    assert_true(
        result[0] == elem, "Single element list element assertion failed"
    )
    return result


fn is_power_of_2(n: Int) -> Bool:
    return (n & (n - 1)) == 0


# Get next power of 2 for n
fn next_power_of_2(n: Int) raises -> Int:
    assert_true(n > 0, "Next power of 2 is supported for >= 1")
    if is_power_of_2(n):
        return n
    if n == 1:
        return 1
    var power = 1
    while power < n:
        power *= 2
    return power


from os import Atomic
from memory import UnsafePointer


@value
struct IdGen:
    var tensor_ids: UnsafePointer[UInt64]

    fn __init__(out self):
        self.tensor_ids = UnsafePointer[UInt64].alloc(1)

    fn __enter__(mut self) -> Self:
        self.tensor_ids[] = 0
        return self

    fn __exit__(mut self):
        print("Exiting IdGen")

    fn next(self) -> UInt64:
        return Atomic.fetch_add(self.tensor_ids, 1)


fn main():
    print("Hello")
