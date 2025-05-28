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
    for shape_dim in shape.axes_spans:
        _ = String(shape_dim)
        assert_true(shape_dim > 0, "Shape dimension not valid")
