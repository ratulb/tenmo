from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import BackwardFn, BackwardFnType

from std.sys import has_accelerator
from .shared.shapes import Shape
from .validators import Validator
from .ndbuffer import NDBuffer
from .shared.intarray import IntArray
from .shared.strides import Strides
from .shared.indexhelper import IndexCalculator
from .ancestry import Ancestor


def try_reshape_strides(
    old_shape: Shape, old_strides: Strides, new_shape: Shape
) -> Tuple[Bool, Strides]:
    """Infer view strides for a reshape, or report that a copy is required.

    High-level idea: a reshape is pure re-addressing — same storage, new
    shape — exactly when constant strides exist that visit the logical
    elements in order. The algorithm finds those strides by grouping:
    walk the old and new dims together, accumulating consecutive dims on
    each side until both groups hold the same element count, then check
    the old group is laid out densely and split its innermost stride
    across the new group. Worked example: a (2,4) column slice with
    strides (12,1) reshaped to (2,2,2) groups dim 0 with dim 0 (two
    elements each, base stride 12) and old dim 1 with new dims 1..2
    (four each, base stride 1, subdivided innermost-first to (2,1)) —
    yielding (12,2,1), a valid view, no copy. The merge pattern instead
    fails: (2,2) with strides (1,2) reshaped to (4,) groups every dim at
    once, and the density check 1 == 2*2 fails — no constant stride
    reads 0,2,1,3 in order, so it copies.

    Returns (True, strides) on success, else (False, _) meaning the
    caller must materialize. Conservative bails (copy): negative
    strides, broadcast stride-0 dims with size > 1, element-count
    mismatch, trailing dense leftovers. Callers must still bounds-check
    (offset + max index < buffer size): compatible strides can still
    overrun a short buffer.
    """
    var old_rank = old_shape.rank()
    var new_rank = new_shape.rank()
    # Guard pass: a view must step forward through storage, one element
    # per stride unit. Negative strides would walk backward (flip, not
    # reshape), and a stride-0 dim of size > 1 re-reads one element
    # (broadcast, not reshape) — both bail to a copy up front.
    for i in range(old_rank):
        var s = old_strides[i]
        if s < 0:
            return (False, Strides())
        if s == 0 and old_shape[i] > 1:
            return (False, Strides())
    # Same-shape reshape changes nothing about addressing, so it is
    # always a view carrying the identical strides (even strided ones).
    if old_shape == new_shape:
        return (True, Strides(copy=old_strides))
    # Output strides under construction, one slot per new dim. Zero is a
    # placeholder: every slot is written exactly once below — either by
    # its group's subdivision or, for trailing size-1 dims, by the
    # leftovers loop.
    var out = IntArray.with_capacity(new_rank)
    for _ in range(new_rank):
        out.append(0)
    # o/n walk the old/new dims; each loop pass consumes one
    # equal-element group from both sides.
    var o = 0
    var n = 0
    while o < old_rank and n < new_rank:
        var o_start = o
        var n_start = n
        var o_prod = old_shape[o]
        var n_prod = new_shape[n]
        # Grow the lagging side until both groups hold the same element
        # count. Running past either rank means the total counts differ
        # (numel mismatch) — no addressing exists, copy.
        while o_prod != n_prod:
            if o_prod < n_prod:
                o += 1
                if o >= old_rank:
                    return (False, Strides())
                o_prod *= old_shape[o]
            else:
                n += 1
                if n >= new_rank:
                    return (False, Strides())
                n_prod *= new_shape[n]
        # Density check: the old group's dense dims must already sit back
        # to back in storage — dim a's stride must equal the next dense
        # dim b's stride times every size between them (so stepping one
        # in a skips exactly one full b-block). Size-1 dims address a
        # single element, so any stride works for them: skip. A single
        # dense dim is trivially dense. Failure (the merge pattern) bails
        # to a copy.
        var prev = -1
        for k in range(o_start, o + 1):
            if old_shape[k] == 1:
                continue
            if prev != -1:
                var between = 1
                for t in range(prev + 1, k + 1):
                    between *= old_shape[t]
                if old_strides[prev] != old_strides[k] * between:
                    return (False, Strides())
            prev = k
        # Base stride: one step in the new group's outermost dim must move
        # past one whole inner block, i.e. the innermost dense old stride
        # (the fastest-varying step). An all-size-1 group holds one
        # element, so any base works — use 1.
        var base = old_strides[prev] if prev != -1 else 1
        # Subdivision: hand out strides innermost-first. The innermost new
        # dim steps by base; each dim outward multiplies by the sizes
        # inside it (one step outward skips one full inner block).
        var acc = base
        var r = n
        while True:
            out[r] = acc
            if r == n_start:
                break
            acc *= new_shape[r]
            r -= 1
        o += 1
        n += 1
    # Leftover dims past the last full group: numel equality means one
    # side cannot hide dense dims while the other ran out, so any
    # leftover must be size-1 — else the counts never matched, copy.
    # Trailing size-1 new dims address one element each, so stride 1.
    while o < old_rank:
        if old_shape[o] != 1:
            return (False, Strides())
        o += 1
    while n < new_rank:
        if new_shape[n] != 1:
            return (False, Strides())
        out[n] = 1
        n += 1
    return (True, Strides(out))


@fieldwise_init
struct ReshapeBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var reshaped = gradbox.reshape(ancestor.shape())
        if ancestor.requires_grad:
            ancestor.update_grad(reshaped^, AddTensor, None)
        parent_ids.append(ancestor._id)
        # View conduit: grad already forwarded to base above; always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Reshape[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            tensor.shape(), new_shape.intarray()
        )
        # View-or-copy decision, two gates. Gate 1: try_reshape_strides
        # infers constant strides for the target shape — contiguous inputs
        # keep their default-strides fast path, and splittable
        # non-contiguous inputs (a column slice of a fused row reshaped to
        # heads) get subdivided strides instead of a copy. Transposes,
        # permute merges, negative strides, and broadcast dims fail gate 1
        # and materialize below. Gate 2: the inferred view must fit the
        # storage — offset plus highest addressed index inside the buffer
        # — since compatible strides can still overrun a short buffer.
        # The source offset is preserved throughout, so slices keep
        # pointing at their data.
        var src_offset = tensor.offset()
        var buffer_size: Int
        comptime if has_accelerator():
            if tensor.buffer.is_on_gpu():
                buffer_size = len(tensor.buffer.device_state.value())
            else:
                buffer_size = len(tensor.buffer.buffer)
        else:
            buffer_size = len(tensor.buffer.buffer)
        var ndb: NDBuffer[Self.dtype]
        var view = try_reshape_strides(
            tensor.buffer.shape, tensor.buffer.strides, shape
        )
        if view[0] and IndexCalculator.max_storage_index(
            shape, view[1], src_offset
        ) < buffer_size:
            ndb = tensor.buffer.share(shape, view[1], offset=src_offset)
        else:
            ndb = tensor.buffer.contiguous(shape)

        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    ReshapeBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, tensor)

        return out^
