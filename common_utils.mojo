from shapes import Shape
from tenmo import Tensor
from gradbox import Gradbox
from sys.param_env import env_get_string
from logger import Level, Logger
from intlist import IntList
from layers import Sequential, Linear, ReLU

# from strides import Strides
from os import abort
from utils import Variant
from testing import assert_true
from layout.int_tuple import IntArray

alias LOG_LEVEL = env_get_string["LOGGING_LEVEL", "INFO"]()
alias log = Logger[Level._from_str(LOG_LEVEL)]()

# Color codes
alias RED: String = "\033[31m"
alias CYAN: String = "\033[36m"
alias MAGENTA: String = "\033[35m"
alias BLUE: String = "\033[34m"  # Standard blue
alias YELLOW: String = "\033[33m"  # Standard yellow
alias RESET: String = "\033[0m"

# Bright variants (more vibrant)
alias BRIGHT_BLUE: String = "\033[94m"


fn log_debug(msg: String, color: String = RED):
    log.debug(color + msg + RESET.__str__())


fn log_info(msg: String, color: String = BRIGHT_BLUE):
    log.info(color + msg + RESET.__str__())


fn log_warning(msg: String, color: String = YELLOW):
    log.warning(color + msg + RESET.__str__())


@always_inline("nodebug")
fn panic(*s: String):
    var message = String(capacity=len(s))
    if len(s) > 0:
        message += s[0].strip()
        for i in range(1, len(s)):
            message += " " + s[i].strip()
    abort(RED + message + RESET.__str__())


fn id[type: AnyType, //](t: type) -> Int:
    return Int(UnsafePointer(to=t))


fn addr[type: AnyType, //](t: type) -> UnsafePointer[type]:
    return UnsafePointer(to=t)


fn addrs[type: AnyType, //](*ts: type) -> List[UnsafePointer[type]]:
    l = List[UnsafePointer[type]](capacity=len(ts))
    for t in ts:
        l.append(UnsafePointer(to=t))
    return l


fn is_null[type: AnyType, //](ptr: UnsafePointer[type]) -> Bool:
    return ptr.__as_bool__() == False


# Helper
def do_assert[
    dtype: DType, //
](a: Tensor[dtype], b: Tensor[dtype], msg: String):
    shape_mismatch = String("{0}: shape mismatch {1} vs {2}")
    tensors_not_equal = String("{}: values mismatch")
    assert_true(
        a.shape == b.shape, shape_mismatch.format(msg, a.shape, b.shape)
    )
    assert_true((a == b).all_true(), tensors_not_equal.format(msg))


# Helper
def assert_grad[
    dtype: DType, //
](t: Tensor[dtype], expected: Tensor[dtype], label: String):
    assert_true(
        (t.gradbox[] == expected).all_true(),
        String("grad assertion failed for {0}").format(label),
    )


fn variadiclist_as_str1(list: VariadicList[Int]) -> String:
    s = String("[")
    for idx in range(len(list)):
        s += list[idx].__str__()
        if idx != len(list) - 1:
            s += ", "
    s += "]"
    return s


# Create a single or two element(s) VariadicList
fn variadic1or2(m: Int, n: Int = -1) -> VariadicList[Int]:
    fn create_variadic_list(*elems: Int) -> VariadicList[Int]:
        return elems

    if n == -1:
        return create_variadic_list(m)
    return create_variadic_list(m, n)


@fieldwise_init
struct NewAxis(Copyable & Movable):  # Empty struct as a sentinel
    pass


alias Idx = Variant[Int, IntArray, Slice, NewAxis]

alias newaxis = Idx(NewAxis())


fn i(value: Int) -> Idx:
    return Idx(value)


fn il(index_list: IntArray) -> Idx:
    return Idx(index_list)


fn il(*indices: Int) -> Idx:
    intarray = IntArray(size=len(indices))
    for i in range(len(indices)):
        intarray[i] = indices[i]
    return Idx(intarray)


fn s() -> Idx:
    return s(None, None, None)


fn s(end: Int) -> Idx:
    return Idx(slice(end))


fn s(start: Int, end: Int) -> Idx:
    return Idx(slice(start, end))


fn s(start: Optional[Int], end: Optional[Int], step: Optional[Int]) -> Idx:
    return Idx(slice(start, end, step))


struct Slicer:
    @staticmethod
    fn slice(
        slice: Slice, end: Int, start: Int = 0, step: Int = 1
    ) -> Tuple[Int, Int, Int]:
        _start, _end, _step = (
            slice.start.or_else(start),
            slice.end.or_else(end),
            slice.step.or_else(step),
        )
        return _start, _end, _step


fn print_gradbox_recursive[
    dtype: DType
](
    grad_ptr: UnsafePointer[Gradbox[dtype]],
    mut indices: List[Int],
    level: Int,
    num_first: Int = 10,
    num_last: Int = 10,
):
    if grad_ptr[].rank() == 0:  # Tensor with Shape ()
        print(grad_ptr[][[]])
        return
    current_dim = len(indices)
    indent = " " * (level * 2)

    if current_dim >= grad_ptr[].rank():
        print(
            "ERROR: current_dim (",
            current_dim,
            ") >= ndim (",
            grad_ptr[].rank(),
            ")",
        )
        return

    size = grad_ptr[].shape()[current_dim]

    if size < 0 or size > 1_000_000:
        print(
            "ERROR: suspicious size: ",
            size,
            "at dim ",
            current_dim,
            grad_ptr[].shape().__str__(),
        )
        return

    # Base case: last dimension (print actual elements)
    if current_dim == grad_ptr[].rank() - 1:
        print(indent + "[", end="")

        for i in range(size):
            if i < num_first:
                indices.append(i)
                print(grad_ptr[][indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")
            elif i == num_first and size > num_first + num_last:
                print("..., ", end="")
            elif i >= size - num_last:
                indices.append(i)
                print(grad_ptr[][indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")

        print("]", end="")

    else:
        print(indent + "[")
        for i in range(size):
            if i < num_first:
                indices.append(i)
                print_gradbox_recursive(
                    grad_ptr, indices, level + 1, num_first, num_last
                )
                _ = indices.pop()
            elif i == num_first and size > num_first + num_last:
                print(indent + "  ...,")
            elif i >= size - num_last:
                indices.append(i)
                print_gradbox_recursive(
                    grad_ptr, indices, level + 1, num_first, num_last
                )
                _ = indices.pop()

            # Print comma and newline for all but last element
            if i != size - 1 and (i < num_first or i >= size - num_last):
                print(",")
            # Special case: last element needs newline before closing bracket
            elif i == size - 1:
                print()  # Newline before closing bracket

        print(indent + "]", end="")


fn print_tensor_recursive[
    dtype: DType
](
    read tensor_ptr: Tensor[dtype],
    mut indices: List[Int],
    level: Int,
    num_first: Int = 10,
    num_last: Int = 10,
):
    if tensor_ptr.rank() == 0:  # Tensor with Shape ()
        print(tensor_ptr[[]])
        return
    current_dim = len(indices)
    indent = " " * (level * 2)

    if current_dim >= tensor_ptr.rank():
        print(
            "ERROR: current_dim (",
            current_dim,
            ") >= ndim (",
            tensor_ptr.rank(),
            ")",
        )
        return

    size = tensor_ptr.shape()[current_dim]

    if size < 0 or size > 1_000_000:
        print(
            "ERROR: suspicious size: ",
            size,
            "at dim ",
            current_dim,
            tensor_ptr.shape().__str__(),
        )
        return

    # Base case: last dimension (print actual elements)
    if current_dim == tensor_ptr.rank() - 1:
        print(indent + "[", end="")

        for i in range(size):
            if i < num_first:
                indices.append(i)
                print(tensor_ptr[indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")
            elif i == num_first and size > num_first + num_last:
                print("..., ", end="")
            elif i >= size - num_last:
                indices.append(i)
                print(tensor_ptr[indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")

        print("]", end="")

    else:
        print(indent + "[")
        for i in range(size):
            if i < num_first:
                indices.append(i)
                print_tensor_recursive(
                    tensor_ptr, indices, level + 1, num_first, num_last
                )
                _ = indices.pop()
            elif i == num_first and size > num_first + num_last:
                print(indent + "  ...,")
            elif i >= size - num_last:
                indices.append(i)
                print_tensor_recursive(
                    tensor_ptr, indices, level + 1, num_first, num_last
                )
                _ = indices.pop()

            # Print comma and newline for all but last element
            if i != size - 1 and (i < num_first or i >= size - num_last):
                print(",")
            # Special case: last element needs newline before closing bracket
            elif i == size - 1:
                print()  # Newline before closing bracket

        print(indent + "]", end="")


@register_passable
struct IntArrayHelper(Copyable):
    @staticmethod
    fn print(arr: IntArray):
        for i in range(arr.size()):
            end = ", " if i < arr.size() - 1 else "\n"
            print(arr[i], end=end)

    @staticmethod
    fn extend(arr: IntArray, *elems: Int) -> IntArray:
        out = IntArray(arr.size() + len(elems))
        out.copy_from(0, arr, arr.size())
        idx = 0
        for i in range(arr.size(), arr.size() + len(elems)):
            out[i] = elems[idx]
            idx += 1
        return out

    @staticmethod
    fn from_list(l: List[Int]) -> IntArray:
        out = IntArray(size=len(l))
        for i in range(len(l)):
            out[i] = l[i]
        return out


# Utility repeat function
fn str_repeat(s: String, n: Int) -> String:
    if n <= 0:
        return ""
    var parts = List[String]()
    for _ in range(n):
        parts.append(s)
    return StringSlice("").join(parts)


fn print_summary[
    dtype: DType
](mod: Sequential[dtype], sample_input: Optional[Tensor[dtype]] = None):
    # Table headers
    var headers = List[String]()
    headers.append("Name")
    headers.append("Type")
    headers.append("Input Shape")
    headers.append("Output Shape")
    headers.append("Params")
    headers.append("Trainable")

    var rows = List[List[String]]()
    rows.append(headers)

    var total_params = 0
    var trainable_params = 0

    # If sample_input is provided → run a dry forward pass to get shapes
    var x = sample_input
    var current_shape = "(?, ?)"

    for i in range(len(mod.modules)):
        m = mod.modules[i]
        var name = "Layer" + i.__str__()

        if m.layer.isa[Linear[dtype]]():
            var l = m.layer[Linear[dtype]]

            # Infer input/output shapes
            var in_features = l.weights.shape[0]
            var out_features = l.weights.shape[1]

            var input_shape = "(?, " + in_features.__str__() + ")"
            var output_shape = "(?, " + out_features.__str__() + ")"

            if x:
                input_shape = x.value().shape.__str__()
                x = Optional(m(x.value()))
                output_shape = x.value().shape.__str__()

            current_shape = output_shape

            # Params
            var params = (
                l.weights.shape.num_elements() + l.bias.shape.num_elements()
            )
            total_params += params
            if l.weights.requires_grad or l.bias.requires_grad:
                trainable_params += params

            rows.append(
                [
                    name,
                    "Linear",
                    input_shape,
                    output_shape,
                    params.__str__(),
                    (l.weights.requires_grad or l.bias.requires_grad).__str__(),
                ]
            )

        elif m.layer.isa[ReLU[dtype]]():
            var input_shape = current_shape
            var output_shape = current_shape

            if x:
                input_shape = x.value().shape.__str__()
                x = Optional(m(x.value()))
                output_shape = x.value().shape.__str__()
                current_shape = output_shape

            rows.append([name, "ReLU", input_shape, output_shape, "0", "False"])

    # Compute column widths
    var widths = List[Int]()
    for j in range(len(headers)):
        var maxw = 0
        for row in rows:
            if len(row[j]) > maxw:
                maxw = len(row[j])
        widths.append(maxw)

    # Print horizontal rule
    fn print_rule():
        var line = ""
        for w in widths:
            line += "+" + str_repeat("-", w + 2)
        line += "+"
        print(line)

    # Print table
    print_rule()
    for idx in range(len(rows)):
        var line = ""
        for j in range(len(rows[idx])):
            var val = rows[idx][j]
            line += "| " + val + str_repeat(" ", widths[j] - len(val)) + " "
        line += "|"
        print(line)
        print_rule()

    # Footer summary
    var non_trainable_params = total_params - trainable_params
    print("\nSummary:")
    print("  Total params:        ", total_params)
    print("  Trainable params:    ", trainable_params)
    print("  Non-trainable params:", non_trainable_params)

fn main() raises:
    _ = """a = Tensor.arange(24, requires_grad=True).reshape(2, 3, 4)
    b = Tensor.scalar(10)
    l = addrs(a, b)
    for e in l:
        e[].print()"""
    l = List(8, 9, 0)
    IntArrayHelper.print(IntArrayHelper.from_list(l))
