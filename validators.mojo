from shapes import Shape
from intlist import IntList
from strides import Strides
from common_utils import Slicer, panic, Idx, NewAxis, i, s, il, newaxis
from tensors import Tensor


struct Validator:
    @staticmethod
    fn validate_repeat_args(
        original_shape: Shape,
        repeat: IntList,
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
        var seen = IntList.with_capacity(axis_length)
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

    @staticmethod
    fn validate_and_normalize_axes(
        shape: Shape,
        axes: IntList,
        ordered: Bool = True,
        fill_missing: Bool = False,
    ) -> IntList:
        """Validate and normalize axes for reduction operations.
        Args:
            shape: Tensor shape to validate against.
            axes: Input axes to normalize.
            ordered: Wheather to sort the axes.
            fill_missing: Wheather add missing axes.
        Returns:
            Normalized, sorted(default) axes.
        Behavior:
            - For scalar tensors (rank=0):
                - `[-1]` → empty list (reduce all).
                - Any other non-empty axes → error.
            - For non-scalar tensors:
                - Empty list → reduce all axes (return 0..rank-1).
                - Normalize negative indices.
                - Validate bounds.
                - Error on deduplicate axes.
                - Fill missing axes if explicitly asked.

        """
        rank = shape.rank()
        # Handle scalar case (rank=0)
        if rank == 0:
            if axes.is_empty() or axes == IntList(-1):
                return (
                    IntList()
                )  # Special case: [-1] means reduce all/# Empty axes for scalar is valid
            if len(axes) > 0:
                panic(
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
                panic(
                    "Validator → validate_and_normalize_axes - invalid axis: "
                    + String(axis)
                    + " for tensor shape: "
                    + shape.__str__()
                )
            normalized.append(normalized_axis)

        # Validate uniqueness always
        unique = IntList()
        for axis in normalized:
            if axis in unique:
                panic(
                    "validate_and_normalize_axes → duplicate axis",
                    axis.__str__(),
                )
            unique.append(axis)

        # Sorted order
        if ordered:
            normalized.sort()
        if not fill_missing:
            return normalized
        # If partial: specified axes go to front, others follow in original order
        seen = normalized
        result = normalized  # Start with specified axes
        for i in range(rank):
            if i not in seen:
                result = result.insert(i, i)
        return result

    @always_inline
    @staticmethod
    fn validate_and_construct_new_shape(
        current_shape: Shape, newdims: IntList
    ) -> Shape:
        """

        Validates if a tensor can be reshaped from `current_shape` to `new_shape`.

        Args:
            current_shape: Original shape of the tensor (e.g., `3, 4, 5`).
            newdims: Requested new shape (e.g., `2, -1, 10`). May contain at most one `-1`.

        Returns:
            Shape: Validated concrete shape (e.g., `Shape(2, 6, 10)`).

        """
        if current_shape == Shape(1) and (
            newdims == IntList() or newdims == IntList(-1)
        ):
            return Shape()

        if current_shape == Shape() and newdims == IntList():
            return Shape()

        if current_shape == Shape() and (
            newdims == IntList(1) or newdims == IntList(-1)
        ):
            return Shape(1)

        if current_shape.num_elements() == newdims.product():
            return Shape(newdims)

        var estimated_size = 1
        var concrete_dims = IntList.with_capacity(len(newdims))
        var infer_index = -1

        for i in range(len(newdims)):
            if newdims[i] == -1:
                if infer_index != -1:
                    panic("Tensor → reshape: only one -1 allowed in reshape")
                infer_index = i
                concrete_dims.append(1)  # temporary placeholder

            elif newdims[i] == 0 or newdims[i] < -1:
                panic("Tensor → reshape: invalid dim: ", newdims[i].__str__())

            else:
                concrete_dims.append(newdims[i])
                estimated_size *= newdims[i]

        if infer_index != -1:
            concrete_dims[infer_index] = Int(
                current_shape.num_elements() / estimated_size
            )

        if concrete_dims.product() != current_shape.num_elements():
            panic(
                "Tensor → reshape: can't reshape tensor containing ",
                current_shape.num_elements().__str__(),
                "elements to a tensor of ",
                concrete_dims.product().__str__(),
                "elements",
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

        # Validate step
        if step == 0:
            panic("Slice step cannot be zero")

        # Prepare new shape, strides, offset
        var new_shape: IntList = IntList.with_capacity(original_shape.rank())
        var new_strides: IntList = IntList.with_capacity(original_shape.rank())
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
        axes: IntList,
        starts: IntList,
        ends: IntList,
        steps: IntList,
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
            axes.len() != starts.len()
            or axes.len() != ends.len()
            or axes.len() != steps.len()
        ):
            panic("axes, starts, ends, steps must all have same length")

        # Normalize axes
        var actual_axes: IntList = IntList.with_capacity(axes.len())
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
        var new_shape: IntList = IntList.with_capacity(rank)
        var new_strides: IntList = IntList.with_capacity(rank)
        var new_offset: Int = 0

        for i in range(rank):
            var dim = original_shape[i]
            var stride = original_strides[i]

            # Check if this axis is sliced
            var idx = -1
            for j in range(actual_axes.len()):
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
        # Count required rank: Int contributes 1; IntList contributes len(list); Slice contributes 1; NewAxis contributes 0
        var required_rank = 0

        for idx in indices:
            if idx.isa[NewAxis]():
                continue
            elif idx.isa[Int]():
                required_rank += 1
            elif idx.isa[IntList]():
                required_rank += len(idx[IntList])
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
            elif idx.isa[IntList]():
                list = idx[IntList]
                for t in range(len(list)):
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
    fn validate_view_params[
        dtype: DType, //
    ](
        this: Tensor[dtype],
        shape: Shape,
        strides: Strides,
        offset: Int,
        # ) -> Tuple[Int, Int, Int]:
    ) -> Int:
        """
        Validate view parameters and compute absolute bounds.

        Args:
            this: Tensor - owning or non-owning.
            shape: The new shape for the view.
            strides: The new strides for the view.
            offset: The offset for the view.

        Returns:
            #Tuple of (abs_min, abs_max, abs_offset) absolute coordinates.
            Absolute offest with respect to base tensor.

        """
        # Calculate logical bounds of new view (relative to parent)
        var min_index = offset
        var max_index = offset

        for i in range(shape.rank()):
            stride = strides[i]
            if stride == 0:
                panic("Tensor → view: stride cannot be 0 in a view")
            extent = (shape[i] - 1) * stride
            if extent >= 0:
                max_index += extent
            else:
                min_index += extent  # negative stride

        # Convert to absolute coordinates (relative to base tensor)
        abs_min = this.offset + min_index
        abs_max = this.offset + max_index
        abs_offset = this.offset + offset

        # Normalize bounds (account for negative strides)
        lo = min(abs_min, abs_max)
        hi = max(abs_min, abs_max)

        # Bounds checking - PyTorch style
        if this.owns_data:
            # For root tensor, check against storage size
            if lo < 0 or hi >= this.numels():
                panic("Tensor → view: exceeds tensor's memory bounds")
        else:
            # For views, check logical range is contained in parent's logical range
            parent_lo = this.offset
            parent_hi = this.offset + this.max_index()
            if lo < parent_lo or hi > parent_hi:
                panic("Tensor → view: exceeds parent tensor's memory bounds")

        # return abs_min, abs_max, abs_offset
        return abs_offset


fn main() raises:
    pass
