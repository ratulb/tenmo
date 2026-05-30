from .ndbuffer import NDBuffer
from .intarray import IntArray
from .validators import Validator
from std.sys.info import has_accelerator
from .minmax_kernel import ReductionMinMax
from .minmax_reducer import MinMaxReducer
from .common_utils import panic


@fieldwise_init
struct MinmaxNdBuffer[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def minmax[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        paired: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        ref shape = ndb.shape
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var (result_ndb, mask_ndb) = ReductionMinMax[
                        Self.dtype
                    ].launch[is_max=is_max](ndb, normalized_axes, keepdims)
                    return result_ndb, mask_ndb
                except e:
                    print(e)
                    panic("MinmaxNdBuffer minmax: gpu path failed")
                    return (
                        NDBuffer[Self.dtype].Empty(),
                        NDBuffer[Self.dtype].Empty(),
                    )
            else:
                return Self.minmax_cpu[is_max](
                    ndb, normalized_axes, keepdims, paired
                )
        else:
            return Self.minmax_cpu[is_max](
                ndb, normalized_axes, keepdims, paired
            )

    @staticmethod
    def minmax_cpu[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
        paired: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var result_ndb = MinMaxReducer[Self.dtype].reduce_minmax[is_max](
            ndb, normalized_axes, keepdims
        )
        if paired:
            var mask_ndb = MinMaxReducer[Self.dtype].build_minmax_mask[is_max](
                ndb, result_ndb, normalized_axes, keepdims
            )
            return result_ndb, mask_ndb
        else:
            return result_ndb, NDBuffer[Self.dtype].Empty()
