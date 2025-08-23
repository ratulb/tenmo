from shapes import Shape
from tensors import Tensor
from sys.param_env import env_get_string
from logger import Level, Logger
from intlist import IntList
from strides import Strides
from os import abort
from utils import Variant
from builtin._location import __call_location, _SourceLocation

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
fn panic[depth: Int=1](*s: String):
    var message = String(capacity=len(s))
    if len(s) > 0:
        message += s[0].strip()
        for i in range(1, len(s)):
            message += " " + s[i].strip()

    loc = __call_location[inline_count=depth]()
    message += " at " + loc.file_name + ":" + loc.line.__str__() + ":" + loc.col.__str__()
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

    @staticmethod
    fn validate_new_shape(curr_dims: IntList, new_dims: IntList) -> Shape:
        """

        Validates if a tensor can be reshaped from `current_shape` to `new_shape`.

        Args:
            curr_dims: Original shape of the tensor (e.g., `3, 4, 5`).
            new_dims: Requested new shape (e.g., `2, -1, 10`). May contain at most one `-1`.

        Returns:
            Shape: Validated concrete shape (e.g., `Shape(2, 6, 10)`).

        """
        var concrete_dims: IntList

        # --- Step 1: Check for invalid values in `new_shape` ---
        if new_dims.any(Self.invalid_dim):
            panic(
                "Shape dimensions must be positive or -1 got ",
                new_dims.__str__(),
            )

        # --- Step 2: Count `-1` entries (only one allowed) ---
        neg_one_count = new_dims.count(-1)
        if neg_one_count > 1:
            panic(
                "At most one -1 allowed in new_shape got ", new_dims.__str__()
            )

        # Calculate concrete shape (replacing -1 if needed)
        curr_product = curr_dims.product()
        if neg_one_count == 1:
            # Infer the dimension marked as -1
            known_dims_product = 1
            for dim in new_dims:
                if dim != -1:
                    known_dims_product *= dim
            if curr_product % known_dims_product != 0:
                panic(
                    "Cannot infer -1:",
                    String(curr_product),
                    "elements not divisible by",
                    String(known_dims_product),
                )
            inferred_dim = curr_product // known_dims_product
            concrete_dims = IntList.new(
                [inferred_dim if dim == -1 else dim for dim in new_dims]
            )
        else:
            concrete_dims = new_dims.copy()

        if concrete_dims.product() != curr_product:
            panic(
                "Shape mismatch: ",
                String(curr_product),
                " elements vs. ",
                String(concrete_dims.product()),
            )

        return Shape(concrete_dims)

    @always_inline
    @staticmethod
    fn invalid_dim(dim: Int) -> Bool:
        return dim == 0 or dim < -1

    @always_inline
    @staticmethod
    fn validate_and_compute_view_metadata(
        original_shape: Shape,
        original_strides: Strides,
        slices: VariadicListMem[Slice],
    ) -> Tuple[Shape, Strides, Int]:
        """
        Computes the new shape, strides, and offset for a tensor view after slicing.

        Args:
            original_shape: The shape of the original tensor.
            original_strides: The strides of the original tensor.
            slices: VariadicList of slice objects for each dimension.

        Returns:
            Tuple[Shape, Strides, int]: New shape, strides, and offset.

        """
        rank = original_shape.rank()
        if len(slices) != rank:
            abort("Number of slices must match tensor rank")

        new_shape = IntList.with_capacity(rank)
        new_strides = IntList.with_capacity(rank)
        new_offset = 0

        for i in range(rank):
            axis = original_shape[i]
            stride = original_strides[i]

            start, end, step = Slicer.slice(slices[i], axis)

            # Negative index adjustment
            start = start + axis if start < 0 else start
            end = end + axis if end < 0 else end

            # Clamp to bounds
            start = max(0, min(start, axis))
            end = max(0, min(end, axis))

            # Calculate length (ceil division)
            span = end - start
            length = (span + (step - 1)) // step

            new_shape.append(length)
            new_strides.append(stride * step)
            new_offset += start * stride

        return Shape(new_shape), Strides(new_strides), new_offset

    @always_inline
    @staticmethod
    fn validate_and_compute_advanced_indexing_metadata(
        original_shape: Shape,
        original_strides: Strides,
        indices: VariadicListMem[Idx],
    ) -> Tuple[Shape, Strides, Int]:
        """
        Computes view metadata (shape, strides, offset) for advanced indexing operations.
        Args:
            original_shape: Shape of the original tensor.
            original_strides: Strides of the original tensor.
            indices: VariadicListMem of Idx variants (NewAxis/Int/Slice).
        Returns:
            Tuple[Shape, Strides, int]: New shape, strides, and offset.
        """
        # Validate rank vs non-newaxis indices count
        var required_rank = 0
        for idx in indices:
            if not idx.isa[NewAxis]():  # Only count non-newaxis indices
                required_rank += 1
        if required_rank != original_shape.rank():
            panic(
                "Tensor indexing: axes count(",
                String(original_shape.rank()),
                ") and ",
                "non-newaxis indices count(",
                String(required_rank),
                ") mismatch",
            )

        new_shape = IntList.with_capacity(len(indices))
        new_strides = IntList.with_capacity(len(indices))
        offset = 0
        dim_counter = 0  # Tracks original tensor dimensions

        for idx in indices:
            if idx.isa[NewAxis]():
                # Case 1: NewAxis insertion
                new_shape.append(1)
                new_strides.append(0)
            elif idx.isa[Int]():
                # Case 2: Integer indexing (dimension reduction)
                axis = idx[Int]
                shape_dim = original_shape[dim_counter]
                stride_dim = original_strides[dim_counter]
                dim_counter += 1
                if axis < 0:
                    axis += shape_dim
                if not 0 <= axis < shape_dim:
                    panic(
                        "Index",
                        String(axis),
                        "out of bounds for dimension",
                        String(shape_dim),
                    )
                offset += axis * stride_dim
                # No shape/strides append (reduces rank)
            elif idx.isa[Slice]():
                # Case 3: Slicing
                s = idx[Slice]
                shape_dim = original_shape[dim_counter]
                stride_dim = original_strides[dim_counter]
                dim_counter += 1
                start, end, step = Slicer.slice(s, shape_dim)
                start = max(0, min(start, shape_dim))
                end = max(0, min(end, shape_dim))
                if step == 0:
                    panic("Slice step cannot be zero")
                if (step > 0 and start >= end) or (step < 0 and start <= end):
                    panic(
                        "Invalid slice range [",
                        String(start),
                        ":",
                        String(end),
                        ":",
                        String(step),
                        "]",
                    )

                new_length = (end - start + step - 1) // step
                new_shape.append(new_length)
                new_strides.append(stride_dim * step)
                offset += start * stride_dim

        return Shape(new_shape), Strides(new_strides), offset

    @always_inline
    @staticmethod
    fn validate_matrix_shapes[
        dtype: DType, //
    ](lhs: Tensor[dtype], rhs: Tensor[dtype]):
        if not lhs.rank() == 2:
            abort("Tensor → matmul(Tensor): Only supports 2D matmul")
        if not rhs.rank() == 2:
            abort("Tensor → matmul(Tensor): Other must be 2D")
        if not lhs.shape[1] == rhs.shape[0]:
            abort("Tensor → matmul(Tensor): Incompatible shapes")



from testing import assert_true


fn test_validate_new_shape() raises:
    print("test_validate_new_shape")
    curr_dims = IntList.new([3, 4, 5])
    new_dims = IntList.new([2, -1, 10])
    concrete_shape = Validator.validate_new_shape(curr_dims, new_dims)
    assert_true(
        concrete_shape == Shape.of(2, 3, 10),
        "validate_new_shape assertion 1 failed",
    )
    new_dims = IntList.new([-1])
    concrete_shape = Validator.validate_new_shape(curr_dims, new_dims)
    assert_true(
        concrete_shape == Shape.of(60), "validate_new_shape assertion 2 failed"
    )


fn main() raises:
    #test_validate_new_shape()
    pass
