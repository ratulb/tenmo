from tenmo import Tensor
from common_utils import panic
from random import random_ui64
from python import PythonObject
from numpy_interop import from_ndarray


trait Dataset(Sized & Copyable & Movable):
    alias _feature_dtype: DType
    alias _label_dtype: DType

    """Interface for datasets with potentially different feature and label dtypes."""

    fn __len__(self) -> Int:
        """Total number of samples."""
        ...

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self._feature_dtype], Tensor[Self._label_dtype]]:
        """Get single sample (features, label)."""
        ...

    fn features(ref self) -> Tensor[Self._feature_dtype]:
        ...

    fn labels(ref self) -> Tensor[Self._label_dtype]:
        ...


@fieldwise_init
struct Batch[feature_dtype: DType, label_dtype: DType](
    ImplicitlyCopyable & Movable
):
    """Container for a batch of data with different feature/label dtypes."""

    var features: Tensor[Self.feature_dtype]
    var labels: Tensor[Self.label_dtype]
    var batch_size: Int

    fn __init__(
        out self, features: Tensor[Self.feature_dtype], labels: Tensor[Self.label_dtype]
    ):
        self.features = features
        self.labels = labels
        self.batch_size = features.shape()[0]


@fieldwise_init
struct DataLoader[DatasetSource: Dataset, //](Copyable & Movable & Sized):
    """Batched data loading with heterogeneous dtypes."""

    var dataset: DatasetSource
    var batch_size: Int
    var shuffle_data: Bool
    var drop_last: Bool
    var _current_idx: Int
    var _indices: List[Int]
    var _num_batches: Int
    var _feature_dim: Int
    var _label_dim: Int
    var _label_rank: Int

    fn __init__(
        out self,
        dataset: DatasetSource,
        batch_size: Int,
        reshuffle: Bool = False,
        drop_last: Bool = False,
    ):
        self.dataset = dataset.copy()
        self.batch_size = batch_size
        self.shuffle_data = reshuffle
        self.drop_last = drop_last
        self._current_idx = 0

        var total_samples = len(self.dataset)
        self._indices = List[Int](capacity=total_samples)

        if drop_last:
            self._num_batches = total_samples // batch_size
        else:
            self._num_batches = (total_samples + batch_size - 1) // batch_size

        # Cache dimensions
        var features_shape = self.dataset.features().shape()
        var labels_shape = self.dataset.labels().shape()

        self._feature_dim = (
            features_shape[1] if features_shape.rank() > 1 else 1
        )
        self._label_dim = labels_shape[1] if labels_shape.rank() > 1 else 1
        self._label_rank = labels_shape.rank()

        self._reset_indices()

    fn _reset_indices(mut self):
        self._indices.clear()
        var n = len(self.dataset)
        for i in range(n):
            self._indices.append(i)
        if self.shuffle_data:
            self._shuffle_indices()

    fn _shuffle_indices(mut self):
        var n = len(self._indices)
        for i in range(n - 1, 0, -1):
            var rand_val = random_ui64(0, UInt64(i + 1))
            var j = Int(rand_val % UInt64(i + 1))
            var temp = self._indices[i]
            self._indices[i] = self._indices[j]
            self._indices[j] = temp

    fn __iter__(mut self) -> Self:
        self._current_idx = 0
        self._reset_indices()
        return self.copy()

    fn __next__(
        mut self,
    ) -> Batch[DatasetSource._feature_dtype, DatasetSource._label_dtype]:
        """Get next batch with proper label shape for CrossEntropyLoss."""
        var start_idx = self._current_idx
        var end_idx = min(start_idx + self.batch_size, len(self._indices))
        var actual_batch_size = end_idx - start_idx

        var feature_dim = self._feature_dim
        var label_rank = self._label_rank

        # Create batch tensors
        var batch_features = Tensor[DatasetSource._feature_dtype].zeros(
            actual_batch_size, feature_dim
        )

        # ✅ KEY FIX: Create labels with correct shape for CrossEntropyLoss
        var batch_labels: Tensor[DatasetSource._label_dtype]
        if label_rank == 1:
            # Dataset has (N,) labels → batch should be (batch_size,)
            batch_labels = Tensor[DatasetSource._label_dtype].zeros(
                actual_batch_size
            )
        else:
            # Dataset has (N, label_dim) labels → batch should be (batch_size, label_dim)
            batch_labels = Tensor[DatasetSource._label_dtype].zeros(
                actual_batch_size, self._label_dim
            )

        # Get dataset tensors ONCE
        ref dataset_features = self.dataset.features()
        ref dataset_labels = self.dataset.labels()

        # Fill batch
        for i in range(actual_batch_size):
            var sample_idx = self._indices[start_idx + i]

            # Copy features
            for j in range(feature_dim):
                batch_features[i, j] = dataset_features[sample_idx, j]

            # Copy labels with correct shape
            if label_rank == 1:
                # (batch_size,) shape
                batch_labels[i] = dataset_labels[sample_idx]
            else:
                # (batch_size, label_dim) shape
                for j in range(self._label_dim):
                    batch_labels[i, j] = dataset_labels[sample_idx, j]

        self._current_idx = end_idx
        return Batch[DatasetSource._feature_dtype, DatasetSource._label_dtype](
            batch_features^, batch_labels^
        )

    fn __has_next__(self) -> Bool:
        if self.drop_last:
            return (self._current_idx + self.batch_size) <= len(self._indices)
        else:
            return self._current_idx < len(self._indices)

    fn __len__(self) -> Int:
        return self._num_batches


@fieldwise_init
struct TensorDataset[feature_dtype: DType, label_dtype: DType = feature_dtype](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset from tensors with potentially different feature/label dtypes."""

    alias _feature_dtype = feature_dtype
    alias _label_dtype = label_dtype

    var _features: Tensor[Self.feature_dtype]
    var _labels: Tensor[Self.label_dtype]
    var _size: Int

    fn __init__(
        out self, features: Tensor[Self.feature_dtype], labels: Tensor[Self.label_dtype]
    ):
        """Create dataset from feature and label tensors.

        Args:
            features: Shape (N, feature_dim).
            labels: Shape (N,) or (N, label_dim).
        """
        self._features = features
        self._labels = labels
        self._size = features.shape()[0]

        if labels.shape()[0] != self._size:
            panic(
                "TensorDataset: features and labels must have same number of"
                " samples"
            )

    fn features(ref self) -> Tensor[Self.feature_dtype]:
        return self._features

    fn labels(ref self) -> Tensor[Self.label_dtype]:
        return self._labels

    fn __len__(self) -> Int:
        return self._size

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        """Get single sample."""
        if idx < 0 or idx >= self._size:
            panic("TensorDataset: index out of bounds")

        # Extract single row for features
        ref feature_shape = self._features.shape()
        var feature_dim = feature_shape[1] if feature_shape.rank() > 1 else 1
        var sample_feature = Tensor[Self.feature_dtype].zeros(feature_dim)

        for j in range(feature_dim):
            sample_feature[j] = self._features[idx, j]

        # Extract single label
        var label_shape = self._labels.shape()
        var sample_label: Tensor[Self.label_dtype]

        if label_shape.rank() == 1:
            # Labels are (N,) - return single scalar
            sample_label = Tensor[Self.label_dtype].d1([self._labels[idx]])
        else:
            # Labels are (N, label_dim)
            var label_dim = label_shape[1]
            sample_label = Tensor[Self.label_dtype].zeros(label_dim)
            for j in range(label_dim):
                sample_label[j] = self._labels[idx, j]

        return (sample_feature^, sample_label^)


@fieldwise_init
struct NumpyDataset[feature_dtype: DType, label_dtype: DType = feature_dtype](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset backed by NumPy arrays with heterogeneous dtypes."""

    alias _feature_dtype = feature_dtype
    alias _label_dtype = label_dtype

    var _features: Tensor[Self.feature_dtype]
    var _labels: Tensor[Self.label_dtype]
    var _size: Int
    var _owns_data: Bool

    fn __init__(
        out self,
        features_numpy: PythonObject,
        labels_numpy: PythonObject,
        copy: Bool = True,
    ) raises:
        """Create dataset from NumPy arrays."""
        self._features = from_ndarray[Self.feature_dtype](
            features_numpy, requires_grad=False, copy=copy
        )
        self._labels = from_ndarray[Self.label_dtype](
            labels_numpy, requires_grad=False, copy=copy
        )
        self._size = self._features.shape()[0]
        self._owns_data = copy

        if self._labels.shape()[0] != self._size:
            panic(
                "NumpyDataset: features and labels must have same number of"
                " samples"
            )

    fn __init__(
        out self, features: Tensor[Self.feature_dtype], labels: Tensor[Self.label_dtype]
    ):
        """Create dataset from existing Mojo tensors."""
        self._features = features
        self._labels = labels
        self._size = features.shape()[0]
        self._owns_data = True

        if labels.shape()[0] != self._size:
            panic(
                "NumpyDataset: features and labels must have same number of"
                " samples"
            )

    fn features(ref self) -> Tensor[Self.feature_dtype]:
        return self._features

    fn labels(ref self) -> Tensor[Self.label_dtype]:
        return self._labels

    fn __len__(self) -> Int:
        return self._size

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        """Get single sample."""
        if idx < 0 or idx >= self._size:
            panic("NumpyDataset: index out of bounds")

        # Extract features
        ref feature_shape = self._features.shape()
        var feature_dim = feature_shape[1] if feature_shape.rank() > 1 else 1
        var sample_feature = Tensor[Self.feature_dtype].zeros(feature_dim)

        if feature_shape.rank() == 1:
            sample_feature[0] = self._features[idx]
        else:
            for j in range(feature_dim):
                sample_feature[j] = self._features[idx, j]

        # Extract label
        var label_shape = self._labels.shape()
        var sample_label: Tensor[Self.label_dtype]

        if label_shape.rank() == 1:
            # Return single value
            sample_label = Tensor[Self.label_dtype].d1([self._labels[idx]])
        else:
            var label_dim = label_shape[1]
            sample_label = Tensor[Self.label_dtype].zeros(label_dim)
            for j in range(label_dim):
                sample_label[j] = self._labels[idx, j]

        return (sample_feature^, sample_label^)


fn main() raises:
    pass
