from tenmo import Tensor
from common_utils import panic
from random import shuffle as reshuffle
from python import PythonObject
from numpy_interop import from_ndarray, numpy_dtype
from memory import memcpy, Pointer
from shapes import Shape

# MNIST
alias MNIST_MEAN = 0.1307
alias MNIST_STD = 0.3081

# Fashion-MNIST
alias FASHION_MNIST_MEAN = 0.2860
alias FASHION_MNIST_STD = 0.3530

# CIFAR-10 (per-channel)
alias CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
alias CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ImageNet (per-channel)
alias IMAGENET_MEAN = (0.485, 0.456, 0.406)
alias IMAGENET_STD = (0.229, 0.224, 0.225)


@fieldwise_init
struct Batch[feature_dtype: DType, label_dtype: DType](
    ImplicitlyCopyable & Movable
):
    var features: Tensor[Self.feature_dtype]
    var labels: Tensor[Self.label_dtype]
    var batch_size: Int

    fn __init__(
        out self,
        features: Tensor[Self.feature_dtype],
        labels: Tensor[Self.label_dtype],
    ):
        self.features = features
        self.labels = labels
        self.batch_size = features.shape()[0]


trait Dataset(Sized & Copyable & Movable):
    alias _feature_dtype: DType
    alias _label_dtype: DType

    fn __len__(self) -> Int:
        ...

    # NOTE: __getitem__ can be used for single-sample access if needed
    # But DataLoader should NOT use it for batching!
    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self._feature_dtype], Tensor[Self._label_dtype]]:
        ...

    # KEY: Direct access to underlying data (no copies)
    fn get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self._feature_dtype], ImmutAnyOrigin]:
        """Get raw pointer to feature data."""
        ...

    fn get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self._label_dtype], ImmutAnyOrigin]:
        """Get raw pointer to label data."""
        ...

    fn get_feature_dim(self) -> Int:
        """Number of features per sample."""
        ...

    fn get_label_dim(self) -> Int:
        """Number of labels per sample (1 for scalar labels)."""
        ...

    fn is_labels_scalar(self) -> Bool:
        """True if labels are scalar (rank 1), False if multi-dimensional."""
        ...

    fn into_loader(
        ref self, batch_size: Int, shuffle: Bool = True, drop_last: Bool = False
    ) -> DataLoader[Self, origin_of(self)]:
        ...


# ==============================================================================
# OPTIMIZED DATALOADER
# ==============================================================================


@fieldwise_init
struct DataLoader[DatasetSource: Dataset, origin: ImmutOrigin](
    ImplicitlyCopyable & Movable & Sized
):
    """Zero-copy batched data loading with optimized bulk memcpy."""

    var dataset: Pointer[DatasetSource, Self.origin]
    var batch_size: Int
    var shuffle_data: Bool
    var drop_last: Bool
    var _current_idx: Int
    var _indices: List[Int]
    var _num_batches: Int

    # Cached dataset metadata
    var _feature_dim: Int
    var _label_dim: Int
    var _labels_scalar: Bool

    # Pre-allocated batch buffers (full size)
    var _batch: Batch[DatasetSource._feature_dtype, DatasetSource._label_dtype]

    # Optional last batch for smaller remainder (if drop_last=False)
    var _last_batch: Optional[
        Batch[DatasetSource._feature_dtype, DatasetSource._label_dtype]
    ]
    var _last_batch_size: Int

    fn __copyinit__(out self, other: Self):
        self.dataset = other.dataset
        self.batch_size = other.batch_size
        self.shuffle_data = other.shuffle_data
        self.drop_last = other.drop_last
        self._current_idx = other._current_idx
        self._indices = other._indices.copy()
        self._num_batches = other._num_batches
        self._feature_dim = other._feature_dim
        self._label_dim = other._label_dim
        self._labels_scalar = other._labels_scalar
        self._batch = other._batch
        self._last_batch = other._last_batch
        self._last_batch_size = other._last_batch_size

    fn __init__(
        out self,
        dataset: Pointer[DatasetSource, Self.origin],
        batch_size: Int,
        shuffle: Bool = False,
        drop_last: Bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.drop_last = drop_last
        self._current_idx = 0

        var total_samples = len(self.dataset[])
        self._indices = List[Int](capacity=total_samples)

        for i in range(total_samples):
            self._indices.append(i)

        # Shuffle if needed (first epoch)
        if self.shuffle_data:
            # reshuffle(self._indices)
            pass

        # Calculate number of batches
        if drop_last:
            self._num_batches = total_samples // batch_size
        else:
            self._num_batches = (total_samples + batch_size - 1) // batch_size

        # Cache dataset metadata
        ref dataset_ref = dataset[]
        self._feature_dim = dataset_ref.get_feature_dim()
        self._label_dim = dataset_ref.get_label_dim()
        self._labels_scalar = dataset_ref.is_labels_scalar()

        # Pre-allocate full-size batch
        var batch_features = Tensor[DatasetSource._feature_dtype].zeros(
            batch_size, self._feature_dim
        )
        var batch_labels: Tensor[DatasetSource._label_dtype]

        if self._labels_scalar:
            batch_labels = Tensor[DatasetSource._label_dtype].zeros(batch_size)
        else:
            batch_labels = Tensor[DatasetSource._label_dtype].zeros(
                batch_size, self._label_dim
            )

        self._batch = Batch[
            DatasetSource._feature_dtype, DatasetSource._label_dtype
        ](batch_features^, batch_labels^)

        # Pre-allocate optional last batch if needed
        if not drop_last:
            var remainder = total_samples % batch_size
            if remainder != 0:
                self._last_batch_size = remainder
                var last_features = Tensor[DatasetSource._feature_dtype].zeros(
                    remainder, self._feature_dim
                )
                var last_labels: Tensor[DatasetSource._label_dtype]

                if self._labels_scalar:
                    last_labels = Tensor[DatasetSource._label_dtype].zeros(
                        remainder
                    )
                else:
                    last_labels = Tensor[DatasetSource._label_dtype].zeros(
                        remainder, self._label_dim
                    )

                self._last_batch = Batch[
                    DatasetSource._feature_dtype, DatasetSource._label_dtype
                ](last_features^, last_labels^)
            else:
                self._last_batch_size = 0
                self._last_batch = None
        else:
            self._last_batch_size = 0
            self._last_batch = None

    fn __iter__(mut self) -> ref [self] Self:
        self._current_idx = 0
        if self.shuffle_data:
            reshuffle(self._indices)
        return self

    fn __next__(
        mut self,
    ) -> ref [self._batch, self._last_batch.value()] Batch[
        DatasetSource._feature_dtype, DatasetSource._label_dtype
    ]:
        """Get next batch with optimized bulk memcpy for sequential access."""
        var start_idx = self._current_idx
        var end_idx = min(start_idx + self.batch_size, len(self._indices))
        var actual_batch_size = end_idx - start_idx

        # Determine which batch to use
        var is_last_batch = actual_batch_size < self.batch_size
        ref dataset_ref = self.dataset[]
        # Get appropriate batch reference
        if is_last_batch and self._last_batch:
            ref current_batch = self._last_batch.value()

            # Get pointers
            var dataset_features_ptr = dataset_ref.get_features_ptr()
            var dataset_labels_ptr = dataset_ref.get_labels_ptr()
            var batch_features_ptr = (
                current_batch.features.buffer.data_buffer().data
            )
            var batch_labels_ptr = (
                current_batch.labels.buffer.data_buffer().data
            )

            # Fill last batch (always use row-by-row for last batch)
            for i in range(actual_batch_size):
                var sample_idx = self._indices[start_idx + i]

                # Copy feature row
                var src_offset = sample_idx * self._feature_dim
                var dst_offset = i * self._feature_dim
                memcpy(
                    dest=batch_features_ptr + dst_offset,
                    src=dataset_features_ptr + src_offset,
                    count=self._feature_dim,
                )

                # Copy label
                if self._labels_scalar:
                    batch_labels_ptr[i] = dataset_labels_ptr[sample_idx]
                else:
                    var src_label_offset = sample_idx * self._label_dim
                    var dst_label_offset = i * self._label_dim
                    memcpy(
                        dest=batch_labels_ptr + dst_label_offset,
                        src=dataset_labels_ptr + src_label_offset,
                        count=self._label_dim,
                    )

            self._current_idx = end_idx
            return current_batch

        else:
            # Use full-size batch
            ref current_batch = self._batch

            # Get pointers
            var dataset_features_ptr = dataset_ref.get_features_ptr()
            var dataset_labels_ptr = dataset_ref.get_labels_ptr()
            var batch_features_ptr = (
                current_batch.features.buffer.data_buffer().data
            )
            var batch_labels_ptr = (
                current_batch.labels.buffer.data_buffer().data
            )

            # OPTIMIZATION: Bulk memcpy if not shuffled (contiguous indices)
            if not self.shuffle_data:
                # Fast path: Single bulk memcpy for contiguous data
                var first_sample_idx = self._indices[start_idx]

                # Copy all features in one go
                var src_features_offset = first_sample_idx * self._feature_dim
                var total_feature_elements = (
                    actual_batch_size * self._feature_dim
                )
                memcpy(
                    dest=batch_features_ptr,
                    src=dataset_features_ptr + src_features_offset,
                    count=total_feature_elements,
                )

                # Copy all labels in one go
                if self._labels_scalar:
                    memcpy(
                        dest=batch_labels_ptr,
                        src=dataset_labels_ptr + first_sample_idx,
                        count=actual_batch_size,
                    )
                else:
                    var src_labels_offset = first_sample_idx * self._label_dim
                    var total_label_elements = (
                        actual_batch_size * self._label_dim
                    )
                    memcpy(
                        dest=batch_labels_ptr,
                        src=dataset_labels_ptr + src_labels_offset,
                        count=total_label_elements,
                    )
            else:
                # Slow path: Row-by-row memcpy for shuffled (non-contiguous) data
                for i in range(actual_batch_size):
                    var sample_idx = self._indices[start_idx + i]

                    # Copy feature row
                    var src_offset = sample_idx * self._feature_dim
                    var dst_offset = i * self._feature_dim
                    memcpy(
                        dest=batch_features_ptr + dst_offset,
                        src=dataset_features_ptr + src_offset,
                        count=self._feature_dim,
                    )

                    # Copy label
                    if self._labels_scalar:
                        batch_labels_ptr[i] = dataset_labels_ptr[sample_idx]
                    else:
                        var src_label_offset = sample_idx * self._label_dim
                        var dst_label_offset = i * self._label_dim
                        memcpy(
                            dest=batch_labels_ptr + dst_label_offset,
                            src=dataset_labels_ptr + src_label_offset,
                            count=self._label_dim,
                        )

            self._current_idx = end_idx
            return current_batch

    fn __has_next__(self) -> Bool:
        if self.drop_last:
            return (self._current_idx + self.batch_size) <= len(self._indices)
        else:
            return self._current_idx < len(self._indices)

    fn __len__(self) -> Int:
        return self._num_batches

    fn reset(mut self):
        """Reset for new epoch."""
        self._current_idx = 0
        if self.shuffle_data:
            reshuffle(self._indices)


# ==============================================================================
# DATASET IMPLEMENTATIONS
# ==============================================================================


@fieldwise_init
struct NumpyDataset[feature_dtype: DType, label_dtype: DType = feature_dtype](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset backed by NumPy arrays. Owns the data."""

    alias _feature_dtype = feature_dtype
    alias _label_dtype = label_dtype

    var _features: Tensor[Self.feature_dtype]
    var _labels: Tensor[Self.label_dtype]
    var _size: Int
    var _feature_dim: Int
    var _label_dim: Int
    var _labels_scalar: Bool

    fn __init__(
        out self,
        features_numpy: PythonObject,
        labels_numpy: PythonObject,
        copy: Bool = True,
    ) raises:
        """Create dataset from NumPy arrays. Copies data once."""
        self._features = from_ndarray[Self.feature_dtype](
            features_numpy, requires_grad=False, copy=copy
        )
        self._labels = from_ndarray[Self.label_dtype](
            labels_numpy, requires_grad=False, copy=copy
        )

        self._size = self._features.shape()[0]

        if self._labels.shape()[0] != self._size:
            panic(
                "NumpyDataset: features and labels must have same number of"
                " samples"
            )

        # Cache metadata
        var features_shape = self._features.shape()
        var labels_shape = self._labels.shape()

        self._feature_dim = (
            features_shape[1] if features_shape.rank() > 1 else 1
        )
        self._label_dim = labels_shape[1] if labels_shape.rank() > 1 else 1
        self._labels_scalar = labels_shape.rank() == 1

    fn __init__(
        out self,
        features: Tensor[Self.feature_dtype],
        labels: Tensor[Self.label_dtype],
    ):
        """Create dataset from existing Mojo tensors."""
        self._features = features
        self._labels = labels
        self._size = features.shape()[0]

        if labels.shape()[0] != self._size:
            panic(
                "NumpyDataset: features and labels must have same number of"
                " samples"
            )

        # Cache metadata
        var features_shape = self._features.shape()
        var labels_shape = self._labels.shape()

        self._feature_dim = (
            features_shape[1] if features_shape.rank() > 1 else 1
        )
        self._label_dim = labels_shape[1] if labels_shape.rank() > 1 else 1
        self._labels_scalar = labels_shape.rank() == 1

    fn __len__(self) -> Int:
        return self._size

    fn get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.feature_dtype], ImmutAnyOrigin]:
        return self._features.buffer.data_buffer().data.as_immutable()

    fn get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.label_dtype], ImmutAnyOrigin]:
        return self._labels.buffer.data_buffer().data.as_immutable()

    fn get_feature_dim(self) -> Int:
        return self._feature_dim

    fn get_label_dim(self) -> Int:
        return self._label_dim

    fn is_labels_scalar(self) -> Bool:
        return self._labels_scalar

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        """Get single sample - creates a copy."""
        if idx < 0 or idx >= self._size:
            panic("NumpyDataset: index out of bounds")

        var sample_feature = Tensor[Self.feature_dtype].zeros(self._feature_dim)
        var dataset_features_ptr = self.get_features_ptr()
        var src_offset = idx * self._feature_dim

        for j in range(self._feature_dim):
            sample_feature[j] = dataset_features_ptr[src_offset + j]

        var sample_label: Tensor[Self.label_dtype]
        var dataset_labels_ptr = self.get_labels_ptr()

        if self._labels_scalar:
            sample_label = Tensor[Self.label_dtype].d1(
                [dataset_labels_ptr[idx]]
            )
        else:
            sample_label = Tensor[Self.label_dtype].zeros(self._label_dim)
            var src_label_offset = idx * self._label_dim
            for j in range(self._label_dim):
                sample_label[j] = dataset_labels_ptr[src_label_offset + j]

        return (sample_feature^, sample_label^)

    fn into_loader(
        ref self, batch_size: Int, shuffle: Bool = True, drop_last: Bool = False
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(Pointer(to=self), batch_size, shuffle, drop_last)


@fieldwise_init
struct TensorDataset[feature_dtype: DType, label_dtype: DType = feature_dtype](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset from tensors. References existing data (no copy)."""

    alias _feature_dtype = feature_dtype
    alias _label_dtype = label_dtype

    var _features: Tensor[Self.feature_dtype]
    var _labels: Tensor[Self.label_dtype]
    var _size: Int
    var _feature_dim: Int
    var _label_dim: Int
    var _labels_scalar: Bool

    fn __init__(
        out self,
        features: Tensor[Self.feature_dtype],
        labels: Tensor[Self.label_dtype],
    ):
        """Create dataset from feature and label tensors (no copy)."""
        self._features = features
        self._labels = labels
        self._size = features.shape()[0]

        if labels.shape()[0] != self._size:
            panic(
                "TensorDataset: features and labels must have same number of"
                " samples"
            )

        # Cache metadata
        var features_shape = self._features.shape()
        var labels_shape = self._labels.shape()

        self._feature_dim = (
            features_shape[1] if features_shape.rank() > 1 else 1
        )
        self._label_dim = labels_shape[1] if labels_shape.rank() > 1 else 1
        self._labels_scalar = labels_shape.rank() == 1

    fn __len__(self) -> Int:
        return self._size

    fn get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.feature_dtype], ImmutAnyOrigin]:
        return self._features.buffer.data_buffer().data.as_immutable()

    fn get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.label_dtype], ImmutAnyOrigin]:
        return self._labels.buffer.data_buffer().data.as_immutable()

    fn get_feature_dim(self) -> Int:
        return self._feature_dim

    fn get_label_dim(self) -> Int:
        return self._label_dim

    fn is_labels_scalar(self) -> Bool:
        return self._labels_scalar

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        """Get single sample."""
        if idx < 0 or idx >= self._size:
            panic("TensorDataset: index out of bounds")

        var sample_feature = Tensor[Self.feature_dtype].zeros(self._feature_dim)
        var dataset_features_ptr = self.get_features_ptr()
        var src_offset = idx * self._feature_dim

        for j in range(self._feature_dim):
            sample_feature[j] = dataset_features_ptr[src_offset + j]

        var sample_label: Tensor[Self.label_dtype]
        var dataset_labels_ptr = self.get_labels_ptr()

        if self._labels_scalar:
            sample_label = Tensor[Self.label_dtype].d1(
                [dataset_labels_ptr[idx]]
            )
        else:
            sample_label = Tensor[Self.label_dtype].zeros(self._label_dim)
            var src_label_offset = idx * self._label_dim
            for j in range(self._label_dim):
                sample_label[j] = dataset_labels_ptr[src_label_offset + j]

        return (sample_feature^, sample_label^)

    fn into_loader(
        ref self, batch_size: Int, shuffle: Bool = True, drop_last: Bool = False
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(Pointer(to=self), batch_size, shuffle, drop_last)


fn main() raises:
    test_dataloader_shuffle_preserves_all_data_dl()


from testing import assert_true


fn test_dataloader_shuffle_preserves_all_data_dl() raises:
    """Test that shuffle doesn't lose or duplicate data."""
    print("test_dataloader_shuffle_preserves_all_data_dl")
    var features = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    ).float()
    var labels = Tensor.d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    # Collect all batches
    var all_features = Tensor.d2(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    ).float()
    var all_labels = Tensor.d1([0.0, 0.0, 0.0, 0.0]).float()

    var idx = 0
    for batch in loader:
        for i in range(batch.batch_size):
            all_features[idx, 0] = batch.features[i, 0]
            all_features[idx, 1] = batch.features[i, 1]
            all_labels[idx] = batch.labels[i]
            idx += 1

    # Check sum (order-independent verification)
    var sum_feat = all_features.sum()
    var expected_sum_feat = features.sum()
    assert_true(sum_feat.all_close(expected_sum_feat))

    var sum_labels = all_labels.sum()
    var expected_sum_labels = labels.sum()
    assert_true(sum_labels.all_close(expected_sum_labels))
