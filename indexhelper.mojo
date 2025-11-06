from shapes import Shape
from intlist import IntList
from strides import Strides
from layout.int_tuple import IntArray
from common_utils import panic #, IntArrayHelper


fn main():
    pass

@register_passable
struct IndexCalculator:
    """Utility for calculating flat indices and coordinates in multi-dimensional arrays."""

    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: IntArray, strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat index from IntArray indices."""
        # 1. Rank check
        var rank = shape.rank()
        if indices.size() != rank or len(strides) != rank:
            if indices.size() != rank and len(strides) != rank:
                panic("flatten_index: indices(", indices.size().__str__(), ") and strides(", len(strides).__str__(), ") both ≠ required rank(", rank.__str__(), ")")
            elif indices.size() != rank:
                panic("flatten_index: indices(", indices.size().__str__(), ") ≠ required rank(", rank.__str__(), ")")
            else:
                panic("flatten_index: strides(", len(strides).__str__(), ") ≠ required rank(", rank.__str__(), ")")

        var flat = offset  # absolute base offset

        #print("var flat: ", flat, indices.size(), shape)
        #IntArrayHelper.print(indices)
        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(indices.size()):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]
            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate

            if idx < 0 or idx >= dim_size:
                panic("flatten_index: index out of bounds: axis", dim_idx.__str__(),", got",indices[dim_idx].__str__(),", size", dim_size.__str__())


            flat = flat + idx * strides[dim_idx]

        return flat


    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: IntList, strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat index from IntList indices."""
        # 1. Rank check
        var rank = shape.rank()
        if indices.size() != rank or len(strides) != rank:
            if indices.size() != rank and len(strides) != rank:
                panic("flatten_index: indices(", indices.size().__str__(), ") and strides(", len(strides).__str__(), ") both ≠ required rank(", rank.__str__(), ")")
            elif indices.size() != rank:
                panic("flatten_index: indices(", indices.size().__str__(), ") ≠ required rank(", rank.__str__(), ")")
            else:
                panic("flatten_index: strides(", len(strides).__str__(), ") ≠ required rank(", rank.__str__(), ")")

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(indices.size()):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic("flatten_index: index out of bounds: axis", dim_idx.__str__(),", got",indices[dim_idx].__str__(),", size", dim_size.__str__())

            flat = flat + idx * strides[dim_idx]

        return flat


    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: List[Int], strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat index from List[Int] indices."""
        # 1. Rank check
        var rank = shape.rank()
        if len(indices) != rank or len(strides) != rank:
            if len(indices) != rank and len(strides) != rank:
                panic("flatten_index: indices(", len(indices).__str__(), ") and strides(", len(strides).__str__(), ") both ≠ required rank(", rank.__str__(), ")")
            elif len(indices) != rank:
                panic("flatten_index: indices(", len(indices).__str__(), ") ≠ required rank(", rank.__str__(), ")")
            else:
                panic("flatten_index: strides(", len(strides).__str__(), ") ≠ required rank(", rank.__str__(), ")")

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic("flatten_index: index out of bounds: axis", dim_idx.__str__(),", got",indices[dim_idx].__str__(),", size", dim_size.__str__())

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: VariadicList[Int], strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat index from List[Int] indices."""
        # 1. Rank check
        var rank = shape.rank()
        if len(indices) != rank or len(strides) != rank:
            if len(indices) != rank and len(strides) != rank:
                panic("flatten_index: indices(", len(indices).__str__(), ") and strides(", len(strides).__str__(), ") both ≠ required rank(", rank.__str__(), ")")
            elif len(indices) != rank:
                panic("flatten_index: indices(", len(indices).__str__(), ") ≠ required rank(", rank.__str__(), ")")
            else:
                panic("flatten_index: strides(", len(strides).__str__(), ") ≠ required rank(", rank.__str__(), ")")

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic("flatten_index: index out of bounds: axis", dim_idx.__str__(),", got",indices[dim_idx].__str__(),", size", dim_size.__str__())

            flat = flat + idx * strides[dim_idx]

        return flat


    @always_inline
    @staticmethod
    fn index_to_coord(shape: Shape, flat_index: Int) -> IntList:
        """Convert flat index to multi-dimensional coordinates."""
        if flat_index < 0 or flat_index >= shape.num_elements():
            panic(
                "IndexCalculator → unravel_index: flat_index",
                flat_index.__str__(),
                "out of bounds.",
                "Should be between 0 <= and <",
                shape.num_elements().__str__(),
            )
        var rank = shape.rank()
        var indices = IntList.filled(rank, 0)
        var remaining = flat_index
        for i in range(rank - 1, -1, -1):  # from last axis backward
            var dim = shape[i]
            indices[i] = remaining % dim
            remaining //= dim

        return indices^
