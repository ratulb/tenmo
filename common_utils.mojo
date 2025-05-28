from tensors import Tensor

fn int_varia_list_to_str(list: VariadicList[Int]) -> String:
    s = String("[")
    for idx in range(len(list)):
        s += list[idx].__str__()
        if idx != len(list) - 1:
            s += ", "
    s += "]"
    return s
fn varia_list_to_list(vlist: VariadicList[Int]) -> List[Int]:
    list = List[Int](capacity = len(vlist))
    for each in vlist:
        list.append(each)
    return list

