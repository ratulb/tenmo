from shapes import Shape
from strides import Strides
from common_utils import (
    Slicer,
    panic,
    Idx,
    NewAxis,
    i,
    s,
    il,
    newaxis,
    log_warning,
)
from tenmo import Tensor
from intarray import IntArray


struct Validator:
    @staticmethod
    fn validate_repeat_args(
        original_shape: Shape,
        repeat: IntArray,
    ):
        if len(repeat) != original_shape.rank():
            panic(
                "repeat expects repeat length = rank. Got ",
                String(len(repeat)),
                " vs rank ",
                String(original_shape.rank()),
            )

        for i in range(original_shape.rank()):
            if repeat[i] <= 0:
                panic("repeat expects values > 0, got ", String(repeat[i]))

    @staticmethod
    fn check_permutation(permutation: List[Int], axis_length: Int):
        # Must have correct length
        if len(permutation) != axis_length:
            panic(
                (
                    "Tensor → check_permutation: permutation length must match"
                    " axis length."
                ),
                "perm length",
                len(permutation).__str__(),
                "and axis length",
                axis_length.__str__(),
            )

        # Must contain all indices 0..axis_len-1 exactly once
        var seen = IntArray.with_capacity(axis_length)
        for v in permutation:
            if v < 0 or v >= axis_length:
                panic(
                    "Tensor → check_permutation: permutation index out of"
                    " range."
                )
            if v in seen:
                panic(
                    "Tensor → check_permutation: permutation contains"
                    " duplicates."
                )
            seen.append(v)

    @staticmethod
    fn validate_dtype_consistency(
        dtype: DType, requires_grad: Bool, label: String
    ):
        if requires_grad:
            if not (dtype.is_floating_point()):
                panic(
                    "Tensor → "
                    + label
                    + " → requires_grad=True is only supported for floating"
                    " point types. "
                )

    # ============================================
    # OPTIMIZED VALIDATOR METHODS USING INTARRAY
    # ============================================
    @always_inline
    @staticmethod
    fn validate_and_normalize_axes(
        shape: Shape,
        axes: IntArray,
        ordered: Bool = True,
        fill_missing: Bool = False,
    ) -> IntArray:
        """Validate and normalize axes for reduction operations.

        Args:
            shape: Tensor shape to validate against.
            axes: Input axes to normalize.
            ordered: Whether to sort the axes.
            fill_missing: Whether to add missing axes.

        Returns:
            Normalized, sorted (default) axes.

        Behavior:
            - For scalar tensors (rank=0):
                - Empty or `[-1]` → empty array (reduce all).
                - Any other non-empty axes → error.
            - For non-scalar tensors:
                - Empty array → reduce all axes (return 0..rank-1).
                - Normalize negative indices.
                - Validate bounds.
                - Error on duplicate axes.
                - Fill missing axes if explicitly asked.
        """
        var rank = shape.rank()

        # Handle scalar case (rank=0)
        if rank == 0:
            if len(axes) == 0:
                return IntArray()
            if len(axes) == 1 and axes[0] == -1:
                return IntArray()
            # Any other axes for scalar is invalid
            panic(
                "Validator → validate_and_normalize_axes: cannot reduce over"
                " axes "
                + axes.__str__()
                + " for scalar tensor with shape: "
                + shape.__str__()
            )

        # Default case: reduce all axes
        if len(axes) == 0:
            return IntArray.range(0, rank)

        # Normalize negative indices and validate bounds
        var normalized = IntArray.with_capacity(len(axes))
        for i in range(len(axes)):
            var axis = axes[i]
            var normalized_axis = axis if axis >= 0 else axis + rank

            if normalized_axis < 0 or normalized_axis >= rank:
                panic(
                    "Validator → validate_and_normalize_axes: invalid axis "
                    + String(axis)
                    + " for tensor shape "
                    + shape.__str__()
                )

            normalized.append(normalized_axis)

        # Validate uniqueness using a simple seen array (O(n²) but n is small)
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                if normalized[i] == normalized[j]:
                    panic(
                        "Validator → validate_and_normalize_axes: duplicate"
                        " axis "
                        + String(normalized[i])
                    )

        # Sort if ordered
        if ordered:
            # Simple insertion sort (axes are typically small)
            for i in range(1, len(normalized)):
                var key = normalized[i]
                var j = i - 1
                while j >= 0 and normalized[j] > key:
                    normalized[j + 1] = normalized[j]
                    j -= 1
                normalized[j + 1] = key

        # Fill missing axes if requested
        if not fill_missing:
            return normalized^

        # Add missing axes: insert each missing axis at its own position
        var result = normalized  # Start with specified axes
        seen = normalized
        for i in range(rank):
            if i not in seen:
                result = result.insert(i, i)
        return result^

    @always_inline
    @staticmethod
    fn validate_and_construct_new_shape(
        current_shape: Shape, newdims: IntArray
    ) -> Shape:
        """Validate if a tensor can be reshaped from current_shape to new shape.

        Args:
            current_shape: Original shape of the tensor (e.g., `(3, 4, 5)`).
            newdims: Requested new shape (e.g., `[2, -1, 10]`). May contain at most one `-1`.

        Returns:
            Validated concrete shape (e.g., `Shape(2, 6, 10)`).

        Behavior:
            - Infers dimension marked with `-1`.
            - Validates total element count matches.
            - Handles scalar ↔ 1-element tensor conversions.
        """
        var current_numels = current_shape.num_elements()

        # Special case: scalar → scalar
        if current_shape.rank() == 0 and len(newdims) == 0:
            return Shape()

        # Special case: (1,) → scalar or scalar → (1,)
        if current_numels == 1:
            if len(newdims) == 0:
                return Shape()
            if len(newdims) == 1 and (newdims[0] == 1 or newdims[0] == -1):
                return Shape(1)

        # Special case: scalar → (1,)
        if current_shape.rank() == 0:
            if len(newdims) == 1 and (newdims[0] == 1 or newdims[0] == -1):
                return Shape(1)

        # Validate newdims and find -1 position
        var estimated_size = 1
        var infer_index = -1
        var concrete_dims = IntArray.with_capacity(len(newdims))

        for i in range(len(newdims)):
            var dim = newdims[i]

            if dim == -1:
                if infer_index != -1:
                    panic(
                        "Validator → validate_and_construct_new_shape: only one"
                        " -1 allowed in reshape"
                    )
                infer_index = i
                concrete_dims.append(1)  # Temporary placeholder

            elif dim == 0 or dim < -1:
                panic(
                    "Validator → validate_and_construct_new_shape: invalid"
                    " dimension "
                    + String(dim)
                )

            else:
                concrete_dims.append(dim)
                estimated_size *= dim

        # Infer the -1 dimension
        if infer_index != -1:
            if estimated_size == 0:
                panic(
                    "Validator → validate_and_construct_new_shape: cannot infer"
                    " dimension when other dimensions are 0"
                )

            var inferred_dim = current_numels // estimated_size

            # Check if division is exact
            if inferred_dim * estimated_size != current_numels:
                panic(
                    "Validator → validate_and_construct_new_shape: cannot infer"
                    " dimension exactly. "
                    + "Total elements: "
                    + String(current_numels)
                    + ", estimated size: "
                    + String(estimated_size)
                )

            concrete_dims[infer_index] = inferred_dim

        # Validate total element count
        var new_numels = concrete_dims.product()
        if new_numels != current_numels:
            panic(
                "Validator → validate_and_construct_new_shape: cannot reshape"
                " tensor with "
                + String(current_numels)
                + " elements to shape with "
                + String(new_numels)
                + " elements"
            )

        return Shape(concrete_dims)

    @always_inline
    @staticmethod
    fn validate_and_compute_slice_metadata(
        original_shape: Shape,
        original_strides: Strides,
        axis: Int,
        start: Int,
        end: Int,
        step: Int = 1,
    ) -> Tuple[Shape, Strides, Int]:
        """
        Compute new shape, strides, and offset for a single-axis slice.

        Args:
            original_shape: Shape of the tensor.
            original_strides: Strides of the tensor.
            axis: Axis to slice (can be negative).
            start: Slice start index.
            end: Slice end index.
            step: Slice step (cannot be 0).

        Returns:
            Tuple of (new_shape, new_strides, new_offset).
        """
        if start == end:
            panic("Slice start and end can not be same")
        # Validate step
        if step == 0:
            panic("Slice step cannot be zero")

        # Normalize axis
        var actual_axis = axis
        if actual_axis < 0:
            actual_axis += original_shape.rank()
        if actual_axis < 0 or actual_axis >= original_shape.rank():
            panic(
                "Axis ",
                String(axis),
                " out of bounds for tensor of rank ",
                String(original_shape.rank()),
            )

        # Prepare new shape, strides, offset
        var new_shape: IntArray = IntArray.with_capacity(original_shape.rank())
        var new_strides: IntArray = IntArray.with_capacity(
            original_shape.rank()
        )
        var new_offset: Int = 0

        for i in range(original_shape.rank()):
            var dim = original_shape[i]
            var stride = original_strides[i]

            if i == actual_axis:
                # Adjust negative indices
                var _start = start + dim if start < 0 else start
                var _end = end + dim if end < 0 else end

                # Clamp to bounds
                _start = max(0, min(_start, dim))
                _end = max(0, min(_end, dim))

                # Compute length like Python slice
                var length = 0
                if step > 0 and _start < _end:
                    length = (_end - _start + step - 1) // step
                elif step < 0 and _start > _end:
                    length = (_start - _end - step - 1) // (-step)

                new_shape.append(length)
                new_strides.append(stride * step)
                new_offset += _start * stride
            else:
                new_shape.append(dim)
                new_strides.append(stride)
        return Shape(new_shape), Strides(new_strides), new_offset

    @always_inline
    @staticmethod
    fn validate_and_compute_slice_metadata_multi(
        original_shape: Shape,
        original_strides: Strides,
        axes: List[Int],
        starts: List[Int],
        ends: List[Int],
        steps: IntArray,
    ) -> Tuple[Shape, Strides, Int]:
        """
        Compute new shape, strides, offset for multi-axis slicing.

        Args:
            original_shape: Shape of the tensor.
            original_strides: Strides of the tensor.
            axes: List of axes to slice (can be negative).
            starts: List of start indices.
            ends: List of end indices.
            steps: List of steps (cannot be 0).

        Returns:
            Tuple of (new_shape, new_strides, new_offset).
        """

        var rank = original_shape.rank()
        if (
            len(axes) != len(starts)
            or len(axes) != len(ends)
            or len(axes) != len(steps)
        ):
            panic("axes, starts, ends, steps must all have same length")

        # Normalize axes
        var actual_axes: IntArray = IntArray.with_capacity(len(axes))
        for a in axes:
            var axis = a
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                panic(
                    "Axis ",
                    String(a),
                    " out of bounds for tensor of rank ",
                    String(rank),
                )
            actual_axes.append(axis)

        # Prepare new shape, strides, offset
        var new_shape: IntArray = IntArray.with_capacity(rank)
        var new_strides: IntArray = IntArray.with_capacity(rank)
        var new_offset: Int = 0

        for i in range(rank):
            var dim = original_shape[i]
            var stride = original_strides[i]

            # Check if this axis is sliced
            var idx = -1
            for j in range(len(actual_axes)):
                if actual_axes[j] == i:
                    idx = j
                    break

            if idx != -1:
                # This axis is sliced
                var start = starts[idx]
                var end = ends[idx]
                var step = steps[idx]

                if step == 0:
                    panic("Slice step cannot be zero")

                # Adjust negative indices
                var _start = start + dim if start < 0 else start
                var _end = end + dim if end < 0 else end

                # Clamp to bounds
                _start = max(0, min(_start, dim))
                _end = max(0, min(_end, dim))

                # Compute length
                var length = 0
                if step > 0 and _start < _end:
                    length = (_end - _start + step - 1) // step
                elif step < 0 and _start > _end:
                    length = (_start - _end - step - 1) // (-step)
                # else length remains 0

                new_shape.append(length)
                new_strides.append(stride * step)
                new_offset += _start * stride
            else:
                # Full slice for other axes
                new_shape.append(dim)
                new_strides.append(stride)

        return Shape(new_shape), Strides(new_strides), new_offset

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
            panic("Number of slices must match tensor rank")

        new_shape = IntArray.with_capacity(rank)
        new_strides = IntArray.with_capacity(rank)
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
        # Count required rank: Int contributes 1; IntArray contributes len(list); Slice contributes 1; NewAxis contributes 0
        var required_rank = 0

        for idx in indices:
            if idx.isa[NewAxis]():
                continue
            elif idx.isa[Int]():
                required_rank += 1
            elif idx.isa[IntArray]():
                required_rank += idx[IntArray].size()
            elif idx.isa[Slice]():
                required_rank += 1

        if required_rank != original_shape.rank():
            panic(
                "Tensor indexing: axes count(",
                String(original_shape.rank()),
                ") and non-newaxis indices count(",
                String(required_rank),
                ") mismatch",
            )

        new_shape = IntArray.with_capacity(len(indices))
        new_strides = IntArray.with_capacity(len(indices))
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
            elif idx.isa[IntArray]():
                list = idx[IntArray]
                for t in range(list.size()):
                    shape_dim = original_shape[dim_counter]
                    stride_dim = original_strides[dim_counter]
                    dim_counter += 1

                    var ai = list[t]
                    if ai < 0:
                        ai += shape_dim
                    if not 0 <= ai < shape_dim:
                        panic(
                            "Index ",
                            String(ai),
                            " out of bounds for dim ",
                            String(shape_dim),
                        )

                    offset += ai * stride_dim
                # Multiple dims consumed; rank reduced by len(list)

            elif idx.isa[Slice]():
                # Case 3: Slicing
                s = idx[Slice]
                shape_dim = original_shape[dim_counter]
                stride_dim = original_strides[dim_counter]
                dim_counter += 1
                start, end, step = Slicer.slice(s, shape_dim)
                if step > 0 and end > start and end < 0:
                    start += shape_dim
                    end += shape_dim
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
    fn validate_view_params(
        storage_size: Int,
        shape: Shape,
        strides: Strides,
        offset: Int,
    ) -> Tuple[Int, Strides]:
        """
        Validate absolute view parameters.
        Both strides and offset are absolute w.r.t. the underlying buffer.

        Precondition: All shape dimensions are >= 1 (enforced by Shape constructor).
        """

        # --- 1. Validate offset ---
        if offset < 0 or offset >= storage_size:
            print("Offset validation failed:")
            print("  Storage size:", storage_size)
            print("  Offset:", offset)
            panic("Tensor → view: offset out of storage bounds")

        # --- 2. Validate stride rank ---
        if shape.rank() != len(strides):
            panic("Tensor → view: stride rank mismatch with shape rank")

        # --- 3. Validate strides & compute memory range ---
        var min_index = offset
        var max_index = offset

        for i in range(shape.rank()):
            var stride = strides[i]
            var size = shape[i]  # Guaranteed >= 1

            # Zero stride only valid for singleton dims
            if stride == 0:
                if size > 1:
                    panic(
                        "Tensor → view: zero stride only allowed for singleton"
                        " dims"
                    )
                continue  # size == 1, no span contribution

            # Compute bounds (size >= 1, stride != 0)
            var span = (size - 1) * stride
            if stride > 0:
                max_index += span
            else:
                min_index += span

        # --- 4. Validate against storage ---
        if min_index < 0 or max_index >= storage_size:
            print("Strides validation failed:")
            print("  Storage size:", storage_size)
            print("  Offset:", offset)
            print("  Shape:", shape)
            print("  Strides:", strides)
            print("  Min index:", min_index)
            print("  Max index:", max_index)
            panic("Tensor → view: strides access out of bounds")

        # --- 5. Check for self-overlapping layout ---
        if not Self.is_non_overlapping(shape, strides):
            log_warning("Tensor → view: self-overlapping layout detected")
            log_warning("  Shape:", shape.__str__())
            log_warning("  Strides:", strides.__str__())

        return (offset, strides)

    @staticmethod
    @always_inline
    fn is_non_overlapping(shape: Shape, strides: Strides) -> Bool:
        """Check if view has self-overlapping positions."""
        rank = shape.rank()
        if rank == 0:
            return True

        var pairs = List[Tuple[Int, Int]](capacity=rank)
        for i in range(rank):
            pairs.append((abs(strides[i]), i))

        # Sort by stride ascending
        fn comp_fn(
            pair_a: Tuple[Int, Int], pair_b: Tuple[Int, Int]
        ) capturing -> Bool:
            return pair_a[0] < pair_b[0]

        sort[comp_fn](pairs)  # Sort by stride ascending

        var required_stride = 1
        for abs_stride, dim in pairs:
            # Skip dimensions with no elements or broadcast dims
            if shape[dim] <= 1 or strides[dim] == 0:
                continue

            if abs_stride < required_stride:
                return False  # Self-overlap detected

            required_stride *= shape[dim]

        return True


fn main() raises:
    pass
