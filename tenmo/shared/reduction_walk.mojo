# ReductionWalk / ReductionOdometer — tenmo/shared/reduction_walk.mojo
#
# Pure stride arithmetic for NDBuffer reductions over a set of
# axes. Historically each reduced element was addressed by building a fresh
# IntArray (replace/insert) + bounds-checked flatten — a per-element allocation
# for every reduction over a non-suffix axis.
#
# These two types replace that with pure pointer arithmetic:
#
#   ReductionWalk       built ONCE per reduction (hoists strides/masks/volume)
#       base_offset()   map a (single, reused) output coordinate → source-buffer
#                       offset: sum of the non-reduced dims' stride products
#                       (O(rank), zero allocation)
#       base_logical()  same mapping to the logical row-major flat index (used
#                       when writing an input-shaped contiguous output such as
#                       excl-product / minmax mask)
#       make_odometer() O(1)-per-element advancing cursor over the reduced
#                       volume
#   ReductionOdometer   mutable cursor tracking BOTH the source-buffer position
#                       (`off`) and the logical flat index (`logical_pos`)

from .intarray import IntArray
from .shapes import Shape
from .strides import Strides


@fieldwise_init
struct ReductionWalk(ImplicitlyCopyable):
    var rank: Int
    var num_red: Int
    var strides: IntArray          # source per-dim buffer strides (rank)
    var flat_strides: IntArray     # logical row-major strides (rank)
    var red_mask: IntArray         # 1 where the dim is reduced (rank)
    var src_to_out: IntArray       # out_coord slot for each non-reduced dim
    var red_sizes: IntArray        # size of each reduced dim
    var red_strides: IntArray      # source buffer stride of each reduced dim
    var red_flat_strides: IntArray  # logical stride of each reduced dim
    var volume: Int                # product of the reduced dims

    @staticmethod
    def build(
        shape: Shape, axes: IntArray, strides: Strides, keepdims: Bool
    ) -> ReductionWalk:
        var rank = shape.rank()
        var num_red = axes.size()
        var strides_c = IntArray.with_capacity(rank)
        var flat_c = IntArray.with_capacity(rank)
        var red_mask_c = IntArray.with_capacity(rank)
        var src_to_out_c = IntArray.with_capacity(rank)
        var red_sizes_c = IntArray.with_capacity(num_red)
        var red_strides_c = IntArray.with_capacity(num_red)
        var red_flat_c = IntArray.with_capacity(num_red)

        # Logical row-major strides of the parent shape:
        # suffix_prod[d] = product of all sizes strictly after dim d.
        var suffix_prod = IntArray.with_capacity(rank)
        var acc = 1
        for d in range(rank - 1, -1, -1):
            suffix_prod.append(acc)
            acc *= shape[d]
        for d in range(rank):
            flat_c.append(suffix_prod[rank - 1 - d])

        var out_cursor = 0
        var vol = 1
        for d in range(rank):
            strides_c.append(strides[d])
            var is_red = False
            for a in range(num_red):
                if axes[a] == d:
                    is_red = True
                    break
            if is_red:
                red_mask_c.append(1)
                src_to_out_c.append(-1)
                red_sizes_c.append(shape[d])
                red_strides_c.append(strides[d])
                red_flat_c.append(flat_c[d])
                vol *= shape[d]
            elif keepdims:
                # Kept dims stay at their source slot in the output coordinate.
                red_mask_c.append(0)
                src_to_out_c.append(d)
            else:
                red_mask_c.append(0)
                src_to_out_c.append(out_cursor)
                out_cursor += 1

        return ReductionWalk(
            rank,
            num_red,
            strides_c^,
            flat_c^,
            red_mask_c^,
            src_to_out_c^,
            red_sizes_c^,
            red_strides_c^,
            red_flat_c^,
            vol,
        )

    @always_inline
    def base_offset(self, out_coord: IntArray, source_offset: Int) -> Int:
        var base = source_offset
        for d in range(self.rank):
            if self.red_mask[d] == 0:
                base += out_coord[self.src_to_out[d]] * self.strides[d]
        return base

    @always_inline
    def base_logical(self, out_coord: IntArray) -> Int:
        var base = 0
        for d in range(self.rank):
            if self.red_mask[d] == 0:
                base += out_coord[self.src_to_out[d]] * self.flat_strides[d]
        return base

    # NOTE: make_odometer / make_odometer_at allocate ONLY the mutable per-worker
    # coordinate cursor (`coords`, a single owning IntArray). The read-only
    # reduced sizes/strides are NOT copied into the odometer: `advance` reads
    # them directly from the walk reference passed in. This keeps per-worker
    # allocation at a single IntArray (no per-reduced-dim copies, no stack
    # aggregates, no borrowed pointers that dangle across closure copies).
    @always_inline
    def make_odometer(
        self, start_off: Int, start_logical: Int
    ) -> ReductionOdometer:
        return ReductionOdometer(
            start_off,
            start_logical,
            IntArray.filled(self.num_red, 0),
        )

    @always_inline
    def make_odometer_at(
        self, start_off: Int, start_logical: Int, start_coords: IntArray
    ) -> ReductionOdometer:
        # Build an odometer whose reduced-domain coordinates start at
        # start_coords (used to seed a window into the middle of the walk).
        return ReductionOdometer(start_off, start_logical, start_coords)

    # Zero-allocation flat-index → offset mapping, mirroring
    # IndexCalculator.index_to_coord but folding straight into a scalar base.
    # out_flat_idx is decomposed over out_shape (trailing out dim first); each
    # recovered out coordinate contributes to exactly one non-reduced input dim
    # (src_to_out[d] == o). Works for both keepdims (reduced out dims are size-1
    # and contribute 0) and non-keepdims (reduced out dims are removed).
    @always_inline
    def base_offset_from_flat(
        self, out_shape: Shape, out_flat_idx: Int, source_offset: Int
    ) -> Int:
        var base = source_offset
        var remaining = out_flat_idx
        for o in range(out_shape.rank() - 1, -1, -1):
            var dim = out_shape[o]
            var coord = remaining % dim
            remaining //= dim
            for d in range(self.rank):
                if self.red_mask[d] == 0 and self.src_to_out[d] == o:
                    base += coord * self.strides[d]
                    break
        return base

    @always_inline
    def base_logical_from_flat(
        self, out_shape: Shape, out_flat_idx: Int
    ) -> Int:
        var base = 0
        var remaining = out_flat_idx
        for o in range(out_shape.rank() - 1, -1, -1):
            var dim = out_shape[o]
            var coord = remaining % dim
            remaining //= dim
            for d in range(self.rank):
                if self.red_mask[d] == 0 and self.src_to_out[d] == o:
                    base += coord * self.flat_strides[d]
                    break
        return base


@fieldwise_init
struct ReductionOdometer(ImplicitlyCopyable):
    var off: Int                # source-buffer offset of the CURRENT element
    var logical_pos: Int        # logical row-major flat index of the element
    var coords: IntArray        # reduced-dim coordinates (mutated in place)

    # Reads the read-only reduced sizes/strides from `walk` (passed by ref) so
    # that no per-worker copies or borrowed pointers into the walk are stored.
    # `walk` must be the same ReductionWalk this odometer was made from and must
    # outlive the odometer's use (true within a reduction worker closure).
    @always_inline
    def advance(mut self, ref walk: ReductionWalk) -> Bool:
        for d in range(walk.red_sizes.size() - 1, -1, -1):
            var c = self.coords[d] + 1
            if c < walk.red_sizes[d]:
                self.coords[d] = c
                self.off += walk.red_strides[d]
                self.logical_pos += walk.red_flat_strides[d]
                return True
            else:
                self.off -= (walk.red_sizes[d] - 1) * walk.red_strides[d]
                self.logical_pos -= (walk.red_sizes[d] - 1) * walk.red_flat_strides[
                    d
                ]
                self.coords[d] = 0
        return False