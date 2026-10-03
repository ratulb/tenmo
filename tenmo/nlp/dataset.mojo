from ..tensor import Tensor
from ..dataloader import Dataset, NativeLoader
from ..shared.shapes import Shape
from bpe.tokenizer_trait import Tokenizer
from std.memory import unsafe_memcpy
from std.random import random_float64


struct LLMDataset[
    TokType: Tokenizer,
    dtype: DType = DType.int64,
](Dataset):
    """
        A sliding-window language modelling dataset.

    Encodes text with a Tokenizer, then creates input/target pairs
    using a sliding window. Stores IDs in two flat 1-D tensors
    (input and target) so that `NativeLoader` can access them via
    fixed-offset pointer arithmetic.
    """

    comptime _sample_dtype = Self.dtype
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
    ) -> Pointer[Scalar[Self._sample_dtype], ImmutAnyOrigin]:
        return self._input_data.unsafe_ptr().as_unsafe_any_origin().as_imm()

    def get_labels_ptr(
        ref self,
    ) -> Pointer[Scalar[Self._label_dtype], ImmutAnyOrigin]:
        return self._target_data.unsafe_ptr().as_unsafe_any_origin().as_imm()

    def get_feature_shape(self) -> Shape:
        return Shape(self._max_length)

    def get_label_shape(self) -> Shape:
        return Shape(self._max_length)

    def get_features_per_sample(self) -> Int:
        return self._max_length

    def get_labels_per_sample(self) -> Int:
        return self._max_length

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self._sample_dtype], Tensor[Self._label_dtype]]:
        var index = idx.value() if idx else Int(
            random_float64() * Float64(self._num_samples)
        )
        var features = Tensor[Self._sample_dtype].zeros(
            Shape(self._max_length)
        )
        var labels = Tensor[Self._label_dtype].zeros(Shape(self._max_length))
        var src_feat = self._input_data.unsafe_ptr().unsafe_offset(
            index * self._max_length
        )
        var src_label = self._target_data.unsafe_ptr().unsafe_offset(
            index * self._max_length
        )
        var dst_feat = (
            features.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var dst_label = (
            labels.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        unsafe_memcpy(
            dest=dst_feat, src=src_feat.as_imm(), count=self._max_length
        )
        unsafe_memcpy(
            dest=dst_label, src=src_label.as_imm(), count=self._max_length
        )
        return features^, labels^

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[Scalar[Self._sample_dtype]] = None,
        normalize_std: Optional[Scalar[Self._sample_dtype]] = None,
    ) -> NativeLoader[Self, origin_of(self)]:
        return NativeLoader(
            Pointer(to=self),
            batch_size,
            shuffle,
            drop_last,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )


@fieldwise_init
struct RandomSlidingWindowDataset[
    dtype: DType = DType.int64,
](Dataset):
    """Sliding-window dataset from a 1D token tensor.

    Given a 1D tensor of token IDs and a window size, pre-computes all
    overlapping context/target pairs so that `NativeLoader` can access
    them via fixed-offset pointer arithmetic.
    """

    comptime _sample_dtype = Self.dtype
    comptime _label_dtype = Self.dtype

    var _input_data: List[Scalar[Self.dtype]]
    var _target_data: List[Scalar[Self.dtype]]
    var _num_samples: Int
    var _seq_length: Int

    def __init__(
        out self,
        data: Tensor[Self.dtype],
        seq_length: Int,
    ):
        self._seq_length = seq_length
        var n = len(data)
        self._num_samples = n - seq_length if n > seq_length else 0

        var cap = self._num_samples * seq_length
        self._input_data = List[Scalar[Self.dtype]](capacity=cap)
        self._target_data = List[Scalar[Self.dtype]](capacity=cap)

        for i in range(self._num_samples):
            for j in range(seq_length):
                self._input_data.append(data[i + j])
                self._target_data.append(data[i + 1 + j])

    def __len__(self) -> Int:
        return self._num_samples

    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self._sample_dtype], ImmutAnyOrigin]:
        return self._input_data.unsafe_ptr().as_unsafe_any_origin().as_imm()

    def get_labels_ptr(
        ref self,
    ) -> Pointer[Scalar[Self._label_dtype], ImmutAnyOrigin]:
        return self._target_data.unsafe_ptr().as_unsafe_any_origin().as_imm()

    def get_feature_shape(self) -> Shape:
        return Shape(self._seq_length)

    def get_label_shape(self) -> Shape:
        return Shape(self._seq_length)

    def get_features_per_sample(self) -> Int:
        return self._seq_length

    def get_labels_per_sample(self) -> Int:
        return self._seq_length

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self._sample_dtype], Tensor[Self._label_dtype]]:
        var index = idx.value() if idx else Int(
            random_float64() * Float64(self._num_samples)
        )
        var features = Tensor[Self._sample_dtype].zeros(
            Shape(self._seq_length)
        )
        var labels = Tensor[Self._label_dtype].zeros(Shape(self._seq_length))
        var src_feat = self._input_data.unsafe_ptr().unsafe_offset(
            index * self._seq_length
        )
        var src_label = self._target_data.unsafe_ptr().unsafe_offset(
            index * self._seq_length
        )
        var dst_feat = (
            features.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var dst_label = (
            labels.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        unsafe_memcpy(
            dest=dst_feat, src=src_feat.as_imm(), count=self._seq_length
        )
        unsafe_memcpy(
            dest=dst_label, src=src_label.as_imm(), count=self._seq_length
        )
        return features^, labels^

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[Scalar[Self._sample_dtype]] = None,
        normalize_std: Optional[Scalar[Self._label_dtype]] = None,
    ) -> NativeLoader[Self, origin_of(self)]:
        return NativeLoader(
            Pointer(to=self),
            batch_size,
            shuffle,
            drop_last,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
