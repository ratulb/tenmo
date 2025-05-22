fn int_varia_list_to_str(list: VariadicList[Int]) -> String:
    s = String("[")
    for idx in range(len(list)):
        s += list[idx].__str__()
        if idx != len(list) -1:
            s += ", "
    s += "]"
    return s
