from shapes import Shape
from tensors import Tensor
from testing import assert_true
from sys.param_env import env_get_string
from logger import Level, Logger
from intlist import IntList
from os import abort
from utils import Variant

alias LOG_LEVEL = env_get_string["LOGGING_LEVEL", "INFO"]()
alias log = Logger[Level._from_str(LOG_LEVEL)]()

@always_inline
fn is_power_of_two(x: Int) -> Bool:
    return x > 0 and (x & (x - 1)) == 0


fn id[type: AnyType, //](t: type) -> Int:
    return Int(UnsafePointer(to=t))


fn is_null[type: AnyType, //](ptr: UnsafePointer[type]) -> Bool:
    return ptr.__as_bool__() == False


fn panic(*s: String):
    abrt = String(capacity=len(s))
    abrt += s[0].strip()
    for i in range(1, len(s)):
        stripped = " " + s[i].strip()
        abrt += stripped
    abort(abrt)


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
        (t.grad[] == expected).all_true(),
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


alias Idx = Variant[Int, Slice, NewAxis]

alias newaxis = Idx(NewAxis())


fn i(value: Int) -> Idx:
    return Idx(value)


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


struct Validator:
    @staticmethod
    fn validate_dtype_consistency(
        dtype: DType, requires_grad: Bool, label: String
    ):
        if requires_grad:
            if not (dtype.is_floating_point()):
                abort(
                    "Tensor → "
                    + label
                    + " → requires_grad=True is only supported for floating"
                    " point types. "
                )

    @staticmethod
    fn validate_and_normalize_axes(shape: Shape, axes: IntList) -> IntList:
        """Validate and normalize axes for reduction operations.
        Args:
            shape: Tensor shape to validate against
            axes: Input axes to normalize
        Returns:
            Normalized, sorted, and deduplicated axes
        Behavior:
            - For scalar tensors (rank=0):
                - `[-1]` → empty list (reduce all)
                - Any other non-empty axes → error
            - For non-scalar tensors:
                - Empty list → reduce all axes (return 0..rank-1)
                - Normalize negative indices
                - Validate bounds
                - Sort and deduplicate.
        """
        rank = shape.rank()
        # Handle scalar case (rank=0)
        if rank == 0:
            if axes.is_empty() or axes == IntList(-1):
                return (
                    IntList()
                )  # Special case: [-1] means reduce all/# Empty axes for scalar is valid
            if len(axes) > 0:
                abort(
                    "Tensor → validate_and_normalize_axes - cannot reduce over"
                    " axes "
                    + axes.__str__()
                    + " for scalar tensor with shape: "
                    + shape.__str__()
                )

        # Default case: reduce all axes
        if len(axes) == 0:
            return IntList.range_list(rank)

        # Normalize and validate axes
        var normalized = IntList.with_capacity(len(axes))
        for axis in axes:
            normalized_axis = axis if axis >= 0 else axis + rank
            if normalized_axis < 0 or normalized_axis >= rank:
                abort(
                    "Tensor → validate_and_normalize_axes - invalid axis: "
                    + String(axis)
                    + " for tensor shape: "
                    + shape.__str__()
                )
            normalized.append(normalized_axis)

        # Ensure uniqueness and sorted order
        normalized.sort_and_deduplicate()
        return normalized

    @staticmethod
    fn validate_axes(axes: IntList, shape: Shape) -> IntList:
        rank = shape.rank()
        var normalized_axes = IntList.with_capacity(rank)
        if axes.len() != rank:
            panic(
                "Validator → validate_axes: transpose axes must have length",
                String(rank) + ",",
                "but got",
                String(axes.len()),
            )
        var seen = IntList.filled(rank, 0)
        # Normalize/validate/check duplicate
        for axis in axes:
            normalized_axis = axis if axis >= 0 else axis + rank
            if normalized_axis < 0 or normalized_axis >= rank:
                panic(
                    "Validator → validate_axes: invalid axis",
                    String(axis),
                    "in transpose: must be in range [0,"
                    + String(rank - 1)
                    + "]",
                )

            if seen[normalized_axis] == 1:
                panic(
                    "Validator → validate_axes: duplicate axis",
                    String(axis),
                    "in transpose axes",
                )

            seen[normalized_axis] = 1
            normalized_axes.append(normalized_axis)
        return normalized_axes

    @staticmethod
    fn validate_indices(
        indices: IntList,
        shape: Shape,
        prefix: String = "",
        do_panic: Bool = True,
    ) -> Bool:
        # Check rank match
        if len(indices) != shape.rank():
            if do_panic:
                panic(
                    prefix + " →" if prefix else "",
                    "Incorrect number of indices: expected "
                    + String(shape.rank())
                    + ", got "
                    + String(len(indices)),
                )
            return False

        # Check each index
        for i in range(shape.rank()):
            if indices[i] < 0:
                if do_panic:
                    panic(
                        prefix + " →" if prefix else "",
                        "Negative index at dimension "
                        + String(i)
                        + ": "
                        + String(indices[i]),
                    )
                return False
            if indices[i] >= shape[i]:
                if do_panic:
                    panic(
                        prefix + " →" if prefix else "",
                        "Index out of bounds at dimension "
                        + String(i)
                        + ": "
                        + String(indices[i])
                        + " >= "
                        + String(shape[i]),
                    )
                return False

        return True


fn main() raises:
    pass
