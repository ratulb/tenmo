"""Flat layout descriptor — shape/strides/offset for a dense strided view.

Describes a (possibly strided) view of a flat buffer purely in terms of
shape/strides/offset, independent of any buffer or device type.

Depends only on `shared.shapes`, `shared.strides`, and the
stdlib. `contiguous` is computed once at construction using the same
`strides.is_contiguous(shape)` semantics as `NDBuffer.is_contiguous()`
(size-1 dimensions are wildcards) — this is the single source of truth for
contiguity after the storage refactor.
"""

from .shapes import Shape
from .strides import Strides


struct Layout(RegisterPassable & ImplicitlyCopyable):
    """A strided view descriptor.

        element_{i0..ik-1} = flat[offset + i0*strides[0] + ... + ik-1*strides[k-1]]

    `contiguous` is computed once at construction (wildcard-aware
    C-contiguity, matching `NDBuffer.is_contiguous()`).
    """

    var shape: Shape
    var strides: Strides
    var offset: Int
    var contiguous: Bool

    def __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.shape = shape
        self.strides = strides.or_else(Strides.default(shape))
        self.offset = offset
        self.contiguous = self.strides.is_contiguous(shape)

    @always_inline
    def numel(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    def rank(self) -> Int:
        return self.shape.rank()

    @always_inline
    def is_contiguous(self) -> Bool:
        return self.contiguous
