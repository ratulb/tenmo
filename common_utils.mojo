from shapes import Shape
from tensors import Tensor
from testing import assert_true
from sys.param_env import env_get_string
from logger import Level, Logger
from intlist import IntList
from os import abort

alias LOG_LEVEL = env_get_string["LOGGING_LEVEL", "INFO"]()
alias log = Logger[Level._from_str(LOG_LEVEL)]()


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
        # Ensure axes are unique, sorted, and within bounds.
        rank = shape.rank()

        if rank == 0:
            if len(axes) == 1 and axes[0] == -1:
                return (
                    IntList()
                )  # Interpret `[-1]` as "reduce all axes" for scalars
            if len(axes) > 0:
                abort(
                    "Tensor → validate_and_normalize_axes - cannot reduce over"
                    " axes "
                    + axes.__str__()
                    + " for scalar tensor with shape: "
                    + shape.__str__()
                )
            return IntList()  # Scalar sum over [] is valid

        if len(axes) == 0:
            return IntList.range_list(rank)
        normalized = IntList.with_capacity(len(axes))
        for _axis in axes:
            axis = _axis
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                abort(
                    "Tensor → validate_and_normalize_axes - invalid axis: "
                    + String(_axis)
                    + " for tensor shape: "
                    + shape.__str__()
                )
            normalized.append(axis)
        # Sort and deduplicate
        normalized.sort_and_deduplicate()
        return normalized

    @staticmethod
    fn validate_axes(axes: IntList, rank: Int) -> IntList:
        var normalized_axes = IntList.Empty
        try:
            if axes.len() != rank:
                raise (
                    String(
                        "transpose axes must have length {0}, but got {1}"
                    ).format(rank, axes.len())
                )
            var seen = IntList.filled(rank, 0)
            # Normalize/validate/check duplicate
            for axis in axes:
                normalized_axis = axis
                if normalized_axis < 0:
                    normalized_axis += rank
                if normalized_axis < 0 or normalized_axis >= rank:
                    raise (
                        String(
                            "Invalid axis {0} in transpose: must be in range"
                            " [0, {1}]"
                        ).format(axis, rank - 1)
                    )

                if seen[normalized_axis] == 1:
                    raise (
                        String(
                            "Duplicate axis {0} found in transpose axes"
                        ).format(axis)
                    )

                seen[normalized_axis] = 1
                normalized_axes.append(normalized_axis)
        except e:
            abort(e.__str__())
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
    indices = IntList(1)
    shape = Shape.of(1, 2)
    #_= Validator.validate_indices(indices, shape, "Tensor")
    _= Validator.validate_indices(indices, shape)
