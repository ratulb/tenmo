from shapes import Shape
from tensors import Tensor
from sys.param_env import env_get_string
from logger import Level, Logger
from intlist import IntList
from layers import Sequential, Linear, ReLU
# from strides import Strides
from os import abort
from utils import Variant
from builtin._location import __call_location, _SourceLocation
from testing import assert_true

alias LOG_LEVEL = env_get_string["LOGGING_LEVEL", "INFO"]()
alias log = Logger[Level._from_str(LOG_LEVEL)]()

# Color codes
alias RED: String = "\033[31m"
alias CYAN: String = "\033[36m"
alias MAGENTA: String = "\033[35m"


@always_inline
fn is_power_of_two(x: Int) -> Bool:
    return x > 0 and (x & (x - 1)) == 0


fn id[type: AnyType, //](t: type) -> Int:
    return Int(UnsafePointer(to=t))


fn is_null[type: AnyType, //](ptr: UnsafePointer[type]) -> Bool:
    return ptr.__as_bool__() == False


@always_inline("nodebug")
fn panic[depth: Int = 1](*s: String):
    var message = String(capacity=len(s))
    if len(s) > 0:
        message += s[0].strip()
        for i in range(1, len(s)):
            message += " " + s[i].strip()

    loc = __call_location[inline_count=depth]()
    message += (
        " at "
        + loc.file_name
        + ":"
        + loc.line.__str__()
        + ":"
        + loc.col.__str__()
    )
    abort(message)


fn log_debug(msg: String, color: String = ""):
    var reset = "\033[0m"
    log.debug(color + msg + reset)


fn log_info(msg: String):
    log.info(msg)


fn log_warning(msg: String):
    log.warning(msg)


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


fn variadiclist_as_str(list: VariadicList[Int]) -> String:
    s = String("[")
    for idx in range(len(list)):
        s += list[idx].__str__()
        if idx != len(list) - 1:
            s += ", "
    s += "]"
    return s


# Convert a VariadicList to List
fn variadiclist_as_intlist(vlist: VariadicList[Int]) -> IntList:
    list = IntList.with_capacity(capacity=len(vlist))
    for each in vlist:
        list.append(each)
    return list^


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


alias Idx = Variant[Int, IntList, Slice, NewAxis]

alias newaxis = Idx(NewAxis())


fn i(value: Int) -> Idx:
    return Idx(value)


fn il(index_list: IntList) -> Idx:
    return Idx(index_list)


fn il(*indices: Int) -> Idx:
    return Idx(IntList(indices))


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
    ) -> (Int, Int, Int):
        _start, _end, _step = (
            slice.start.or_else(start),
            slice.end.or_else(end),
            slice.step.or_else(step),
        )
        return _start, _end, _step


fn compute_output_shape(
    original_shape: Shape, normalized_axes: IntList, keepdims: Bool
) -> Shape:
    """Compute the output shape after reduction along specified axes.

    Args:
        original_shape: Shape of the tensor before reduction.
        normalized_axes: Sorted list of axes to reduce over (must be valid for shape).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        Shape after reduction

    Behavior:
        - If reducing all axes and keepdims=False → returns Shape.Void (scalar)
        - Otherwise:
            - For reduced axes: keep as 1 if keepdims=True, else remove
            - For non-reduced axes: keep original size.
    """
    rank = original_shape.rank()

    # Full reduction case (return scalar shape if not keeping dims)
    if rank == 0 or (len(normalized_axes) == rank and not keepdims):
        return Shape.Void

    var spans = IntList.with_capacity(rank)
    for dim in range(rank):
        if dim in normalized_axes:
            if keepdims:
                spans.append(1)  # Keep reduced dim as size 1
        else:
            spans.append(original_shape[dim])  # Keep original size

    return Shape(spans)


fn print_tensor_recursive[
    dtype: DType, //
](
    tensor_ptr: UnsafePointer[Tensor[dtype]],
    mut indices: IntList,
    level: Int,
    num_first: Int = 10,
    num_last: Int = 10,
):
    if tensor_ptr[].rank() == 0:  # Tensor with Shape ()
        print(tensor_ptr[][IntList.Empty])
        return
    current_dim = len(indices)
    indent = " " * (level * 2)

    if current_dim >= tensor_ptr[].rank():
        print(
            "ERROR: current_dim (",
            current_dim,
            ") >= ndim (",
            tensor_ptr[].rank(),
            ")",
        )
        return

    size = tensor_ptr[].shape[current_dim]

    if size < 0 or size > 1_000_000:
        print(
            "ERROR: suspicious size: ",
            size,
            "at dim ",
            current_dim,
            tensor_ptr[].shape.__str__(),
        )
        return

    # Base case: last dimension (print actual elements)
    if current_dim == tensor_ptr[].rank() - 1:
        print(indent + "[", end="")

        for i in range(size):
            if i < num_first:
                indices.append(i)
                print(tensor_ptr[][indices], end="")
                _ = indices.pop()
                if i != size - 1:
                    print(", ", end="")
            elif i == num_first and size > num_first + num_last:
                print("..., ", end="")
            elif i >= size - num_last:
                indices.append(i)
                print(tensor_ptr[][indices], end="")
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


# Utility repeat function
fn str_repeat(s: String, n: Int) -> String:
    if n <= 0:
        return ""
    var parts = List[String]()
    for _ in range(n):
        parts.append(s)
    return StringSlice("").join(parts)

fn print_summary[dtype: DType](mod: Sequential[dtype], 
                               sample_input: Optional[Tensor[dtype]] = None):
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
            var params = l.weights.shape.num_elements() + l.bias.shape.num_elements()
            total_params += params
            if l.weights.requires_grad or l.bias.requires_grad:
                trainable_params += params

            rows.append([name, "Linear", input_shape, output_shape,
                         params.__str__(),
                         (l.weights.requires_grad or l.bias.requires_grad).__str__()])

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
    a = Tensor.arange(24, requires_grad=True).reshape(2, 3, 4)
    l = IntList.Empty
    print_tensor_recursive(UnsafePointer(to=a), l, 2)
