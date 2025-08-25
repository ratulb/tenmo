from shapes import Shape
from tensors import Tensor
from sys.param_env import env_get_string
from logger import Level, Logger
from intlist import IntList

# from strides import Strides
from os import abort
from utils import Variant
from builtin._location import __call_location, _SourceLocation
from testing import assert_true

alias LOG_LEVEL = env_get_string["LOGGING_LEVEL", "INFO"]()
alias log = Logger[Level._from_str(LOG_LEVEL)]()


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


fn log_debug(msg: String):
    log.debug(msg)


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


fn main() raises:
    pass
