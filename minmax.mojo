from tenmo import Tensor
from operators import AddTensor
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from common_utils import panic
from validators import Validator
from utils.numerics import min_finite, max_finite
from intarray import IntArray
from gradbox import Gradbox
from ancestry import Ancestor
from indexhelper import IndexCalculator

alias Gradbag[dtype: DType] = List[Tuple[IntArray, Scalar[dtype]]]


@fieldwise_init
struct MinMaxBackward[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    var axes: IntArray
    var keepdims: Bool
    var gradbag: Gradbag[dtype]

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.keepdims = other.keepdims
        self.gradbag = other.gradbag.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.keepdims = other.keepdims
        self.gradbag = other.gradbag^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()
        var ancestor = output.ancestry().get(0)
        var mask = Gradbox[dtype].zeros(ancestor.shape(), share=False)

        # Build mask from saved gradient contributions
        for grad in self.gradbag:
            mask[grad[0]] = grad[1]

        var shape = ancestor.shape()
        var rank = shape.rank()

        if rank == 0:
            return [(ancestor^, gradbox^, AddTensor)]

        if gradbox.shape() == Shape():
            # Scalar upstream gradient
            var filled = Gradbox[dtype].full(shape, gradbox.item(), share=False)
            var grad_contrib = filled * mask
            return [(ancestor^, grad_contrib^, AddTensor)]
        else:
            # Non-scalar upstream gradient
            var gradbox_like_input: Gradbox[dtype]
            if not self.keepdims:
                gradbox_like_input = gradbox.unsqueeze(self.axes).broadcast_to(
                    shape, share=False
                )
            else:
                gradbox_like_input = gradbox.broadcast_to(shape, share=False)

            var grad_contrib = mask * gradbox_like_input
            return [(ancestor^, grad_contrib^, AddTensor)]


@fieldwise_init
@register_passable
struct MinMax[dtype: DType = DType.float32]:
    @staticmethod
    fn forward[
        max: Bool, track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var out_shape = shape.compute_output_shape(normalized_axes, keepdims)
        var result = Tensor[dtype].zeros(out_shape)
        var gradbag: Gradbag[dtype] = Gradbag[dtype]()

        # ===== FAST PATH: Scalar Input =====
        if rank == 0:
            var v = self[IntArray()]
            result[IntArray()] = v

            @parameter
            if track_grad:
                grad_required = requires_grad.or_else(self.requires_grad)
                if grad_required:
                    result.requires_grad_(True)
                    var backward_fn = MinMaxBackward[dtype](
                        normalized_axes, keepdims, gradbag^
                    ).into_backward_fn()
                    result.backwardFn = Optional(backward_fn^)
                    result.add_ancestry(self)
            return result^

        # ===== FAST PATH: Full Reduction to Scalar =====
        if out_shape == Shape():
            return Self._full_reduction_vectorized[max, track_grad](
                self,
                shape,
                normalized_axes,
                keepdims,
                result^,
                gradbag^,
                requires_grad,
            )

        # ===== GENERAL CASE: Partial Reduction (Parallelized) =====
        return Self._partial_reduction_parallel[max, track_grad](
            self,
            shape,
            normalized_axes,
            keepdims,
            out_shape,
            result^,
            gradbag^,
            requires_grad,
        )

    # ===== VECTORIZED FULL REDUCTION (All axes → scalar) =====

    @always_inline
    @staticmethod
    fn _full_reduction_vectorized[
        max: Bool, track_grad: Bool
    ](
        self: Tensor[dtype],
        shape: Shape,
        normalized_axes: IntArray,
        keepdims: Bool,
        var result: Tensor[dtype],
        var gradbag: Gradbag[dtype],
        requires_grad: Optional[Bool],
    ) -> Tensor[dtype]:
        var total_elements = shape.num_elements()

        # Initialize with first element
        var best_value = self[shape.first_index()]
        var best_positions = List[IntArray]()

        @parameter
        if track_grad:
            best_positions.append(shape.first_index())

        # ===== SEQUENTIAL SCAN (No vectorization for gradient tracking) =====
        # Vectorization doesn't help much for min/max with gradient tracking
        # because we need to track positions, which requires branching

        for flat_idx in range(
            1, total_elements
        ):  # Start from 1, we already have element 0
            var idx = IndexCalculator.index_to_coord(shape, flat_idx)
            var cur = self[idx]

            @parameter
            if max:
                if cur > best_value:
                    best_value = cur

                    @parameter
                    if track_grad:
                        best_positions.clear()
                        best_positions.append(idx)
                elif cur == best_value:

                    @parameter
                    if track_grad:
                        best_positions.append(idx)
            else:
                if cur < best_value:
                    best_value = cur

                    @parameter
                    if track_grad:
                        best_positions.clear()
                        best_positions.append(idx)
                elif cur == best_value:

                    @parameter
                    if track_grad:
                        best_positions.append(idx)

        # Write result
        result[IntArray()] = best_value

        @parameter
        if track_grad:
            # Split gradient among all tied positions
            var count = len(best_positions)
            if count > 0:
                var inv = Scalar[dtype](1) / count
                for p in best_positions:
                    gradbag.append((p, inv))

            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var backward_fn = MinMaxBackward[dtype](
                    normalized_axes, keepdims, gradbag^
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^

    # ===== PARALLELIZED PARTIAL REDUCTION =====
    @always_inline
    @staticmethod
    fn _partial_reduction_parallel[
        max: Bool, track_grad: Bool
    ](
        self: Tensor[dtype],
        shape: Shape,
        normalized_axes: IntArray,
        keepdims: Bool,
        out_shape: Shape,
        var result: Tensor[dtype],
        var gradbag: Gradbag[dtype],
        requires_grad: Optional[Bool],
    ) -> Tensor[dtype]:
        from algorithm import parallelize

        var reduced_shape = shape.reduced_shape(normalized_axes)
        var num_output_elements = out_shape.num_elements()

        # Thread-local storage for gradient bags (one per output element)
        # Note: In real Mojo, you'd need proper synchronization primitives
        var local_gradbags = List[Gradbag[dtype]]()
        for _ in range(num_output_elements):
            local_gradbags.append(Gradbag[dtype]())

        # ===== PARALLEL PROCESSING: Each output element computed independently =====
        @parameter
        fn compute_output_element(out_flat_idx: Int):
            # Convert flat index to multidimensional index
            var out_idx = IndexCalculator.index_to_coord(
                out_shape, out_flat_idx
            )
            var best_value: Scalar[dtype]

            # Initialize best value for this output element
            @parameter
            if max:
                best_value = min_finite[dtype]()
            else:
                best_value = max_finite[dtype]()

            var best_positions = List[IntArray]()
            var first_iteration = True

            # ===== VECTORIZED INNER LOOP: Scan reduced dimensions =====
            # For each output position, scan all elements in the reduced block
            var num_reduced_elements = reduced_shape.num_elements()

            for red_flat_idx in range(num_reduced_elements):
                var red_idx = IndexCalculator.index_to_coord(
                    reduced_shape, red_flat_idx
                )

                # Compute full input index
                var full_idx = out_idx.replace(
                    normalized_axes, red_idx
                ) if keepdims else out_idx.insert(normalized_axes, red_idx)

                var cur = self[full_idx]

                if first_iteration:
                    best_value = cur
                    first_iteration = False

                    @parameter
                    if track_grad:
                        best_positions.append(full_idx)
                else:

                    @parameter
                    if max:
                        if cur > best_value:
                            best_value = cur

                            @parameter
                            if track_grad:
                                best_positions.clear()
                                best_positions.append(full_idx)
                        elif cur == best_value:

                            @parameter
                            if track_grad:
                                best_positions.append(full_idx)
                    else:
                        if cur < best_value:
                            best_value = cur

                            @parameter
                            if track_grad:
                                best_positions.clear()
                                best_positions.append(full_idx)
                        elif cur == best_value:

                            @parameter
                            if track_grad:
                                best_positions.append(full_idx)

            # Write result to output
            result[out_idx] = best_value

            @parameter
            if track_grad:
                # Store gradient contributions in thread-local storage
                var count = len(best_positions)
                if count > 0:
                    var inv = Scalar[dtype](1) / count
                    for p in best_positions:
                        local_gradbags[out_flat_idx].append((p, inv))

        # Execute in parallel across all output elements
        # Each thread handles a subset of output positions independently
        parallelize[compute_output_element](num_output_elements)

        # ===== MERGE: Combine thread-local gradient bags =====
        @parameter
        if track_grad:
            for i in range(num_output_elements):
                for item in local_gradbags[i]:
                    gradbag.append(item)

            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var backward_fn = MinMaxBackward[dtype](
                    normalized_axes, keepdims, gradbag^
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^


@fieldwise_init
struct MinMaxBackward_orig[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    var axes: IntArray
    var keepdims: Bool
    var gradbag: Gradbag[dtype]

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.keepdims = other.keepdims
        self.gradbag = other.gradbag.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.keepdims = other.keepdims
        self.gradbag = other.gradbag^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        # Retrieve upstream grad and saved tensors
        var gradbox = output.grad()
        var ancestor = output.ancestry().get(0)  # original input
        var mask = Gradbox[dtype].zeros(ancestor.shape(), share=False)
        for grad in self.gradbag:
            mask[grad[0]] = grad[1]
        var shape = ancestor.shape()
        var rank = shape.rank()

        # If input was scalar, just pass gradient through
        if rank == 0:
            return [(ancestor^, gradbox^, AddTensor)]

        if gradbox.shape() == Shape():
            # Scalar upstream grad → same scalar everywhere that was max
            # Build a tensor of that scalar, then mask it
            var filled = Gradbox[dtype].full(shape, gradbox.item(), share=False)
            # Apply mask: grad_contrib = filled * mask
            var grad_contrib = filled * mask
            return [(ancestor^, grad_contrib^, AddTensor)]
        else:
            # Build gradient broadcasted to input shape
            # then mask it so only argmax positions receive gradient.
            var gradbox_like_input: Gradbox[dtype]
            # Non-scalar upstream grad
            if not self.keepdims:
                gradbox_like_input = gradbox.unsqueeze(self.axes).broadcast_to(
                    shape, share=False
                )
            else:
                # keepdims=True: just broadcast to input shape
                gradbox_like_input = gradbox.broadcast_to(shape, share=False)

            # Apply mask
            var grad_contrib = mask * gradbox_like_input

            return [(ancestor^, grad_contrib^, AddTensor)]


@fieldwise_init
@register_passable
struct MinMax_orig[dtype: DType = DType.float32]:
    @staticmethod
    fn forward[
        max: Bool, track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var out_shape = shape.compute_output_shape(normalized_axes, keepdims)
        var out = Tensor[dtype].zeros(out_shape)
        # Mask stores fractional responsibility: 1/count_of_maxes for positions that are maxima
        # Keep grad shares in gradbag which contains index, grad value(IntArray, Scalar)
        var gradbag: Gradbag[dtype] = Gradbag[dtype]()

        if out_shape == Shape():
            if rank == 0:
                # scalar input -> min/max is the value itself; mask = 1
                var v = self[IntArray()]
                out[IntArray()] = v
            elif rank == len(normalized_axes) and not keepdims:
                # reduce all dims -> scalar: find all positions equal to global max
                var first_iter = True
                var best_value = self[shape.first_index()]

                var best_positions = List[IntArray]()
                for idx in shape:
                    var cur = self[idx]
                    if first_iter:
                        best_value = cur
                        first_iter = False

                        @parameter
                        if track_grad:
                            best_positions.append(idx)
                    else:

                        @parameter
                        if max:
                            if cur > best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(idx)

                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(idx)
                                pass

                        else:
                            if cur < best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(idx)
                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(idx)
                                pass

                out[IntArray()] = best_value

                @parameter
                if track_grad:
                    # Split responsibility among ties
                    var count = len(best_positions)
                    if count > 0:
                        var inv = Scalar[dtype](1) / count
                        for p in best_positions:
                            gradbag.append((p, inv))
        else:
            # Partial reduction
            var reduced_shape = shape.reduced_shape(normalized_axes)

            for out_idx in out_shape:
                # Track best value and all positions with that best (in the reduced block)
                @parameter
                if max:
                    best_value = min_finite[dtype]()
                else:
                    best_value = max_finite[dtype]()

                var best_positions = List[IntArray]()
                var first_iteration = True

                for red_idx in reduced_shape:
                    var full_idx = out_idx.replace(
                        normalized_axes, red_idx
                    ) if keepdims else out_idx.insert(normalized_axes, red_idx)
                    var cur = self[full_idx]

                    if first_iteration:
                        best_value = cur
                        first_iteration = False

                        @parameter
                        if track_grad:
                            best_positions.append(full_idx)
                    else:

                        @parameter
                        if max:
                            if cur > best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(full_idx)
                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(full_idx)
                                pass
                        else:
                            if cur < best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(full_idx)
                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(full_idx)
                                pass

                # write result to output
                out[out_idx] = best_value

                @parameter
                if track_grad:
                    # split responsibility among ties in this reduced block
                    var count = len(best_positions)
                    if count > 0:
                        var inv = Scalar[dtype](1) / count
                        for p in best_positions:
                            gradbag.append((p, inv))

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backward_fn = MinMaxBackward[dtype](
                    normalized_axes, keepdims, gradbag^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
