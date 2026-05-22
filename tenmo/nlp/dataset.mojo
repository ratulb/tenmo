from bpe import Tokenizer
from tenmo.dataloader import Dataset, DataLoader
from tenmo.shapes import Shape
from tenmo.common_utils import panic
from std.memory import memcpy


struct LLMDataset[
    TokType: Tokenizer,
    dtype: DType = DType.int64,
](
    Sized & Copyable & Movable & Dataset
):
    """
    A sliding-window language modelling dataset.

    Encodes text with a Tokenizer, then creates input/target pairs
    using a sliding window. Stores IDs in two flat ``List[Scalar[dtype]]``
    arrays (input and target) so that ``DataLoader`` can access them via
    fixed-offset pointer arithmetic.
    """

    comptime _feature_dtype = Self.dtype
    comptime _label_dtype = Self.dtype
    comptime _TokType = Self.TokType

    var _input_data: List[Scalar[Self.dtype]]
    var _target_data: List[Scalar[Self.dtype]]
    var _num_samples: Int
    var _max_length: Int

    def __init__(
        out self,
        txt: String,
        mut tokenizer: Self._TokType,
        max_length: Int,
        stride: Int,
    ) raises:
        var ids = tokenizer.encode(txt)
        var n = len(ids)

        self._max_length = max_length
        if n <= max_length:
            self._num_samples = 0
        else:
            self._num_samples = (n - max_length + stride - 1) // stride

        var cap = self._num_samples * max_length
        self._input_data = List[Scalar[Self.dtype]](capacity=cap)
        self._target_data = List[Scalar[Self.dtype]](capacity=cap)

        for i in range(self._num_samples):
            var start = i * stride
            for j in range(max_length):
                self._input_data.append(Scalar[Self.dtype](ids[start + j]))
                self._target_data.append(Scalar[Self.dtype](ids[start + 1 + j]))

    def __len__(self) -> Int:
        return self._num_samples

    def get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self._feature_dtype], ImmutAnyOrigin]:
        return self._input_data.unsafe_ptr().as_immutable()

    def get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self._label_dtype], ImmutAnyOrigin]:
        return self._target_data.unsafe_ptr().as_immutable()

    def get_feature_shape(self) -> Shape:
        return Shape(self._max_length)

    def get_label_shape(self) -> Shape:
        return Shape(self._max_length)

    def get_features_per_sample(self) -> Int:
        return self._max_length

    def get_labels_per_sample(self) -> Int:
        return self._max_length

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(Pointer(to=self), batch_size, shuffle, drop_last)
