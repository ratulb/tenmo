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


# ==============================================================================
# UPDATED TRAIT - Now includes shape information
# ==============================================================================


trait Dataset(Sized & Copyable & Movable):
    alias _feature_dtype: DType
    alias _label_dtype: DType

    fn __len__(self) -> Int:
        ...

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

    fn get_feature_shape(self) -> Shape:
        """Get the shape of a single feature sample (excluding batch dimension).
        """
        ...

    fn get_label_shape(self) -> Shape:
        """Get the shape of a single label (excluding batch dimension)."""
        ...

    fn get_features_per_sample(self) -> Int:
        """Total number of elements per feature sample."""
        ...

    fn get_labels_per_sample(self) -> Int:
        """Total number of elements per label."""
        ...

    fn into_loader(
        ref self, batch_size: Int, shuffle: Bool = True, drop_last: Bool = False
    ) -> DataLoader[Self, origin_of(self)]:
        ...


# ==============================================================================
# UPDATED DATALOADER - Preserves multi-dimensional shapes
# ==============================================================================


@fieldwise_init
struct DataLoader[DatasetSource: Dataset, origin: ImmutOrigin](
    ImplicitlyCopyable & Movable & Sized
):
    """Zero-copy batched data loading that preserves tensor shapes."""

    var dataset: Pointer[DatasetSource, Self.origin]
    var batch_size: Int
    var shuffle_data: Bool
    var drop_last: Bool
    var _current_idx: Int
    var _indices: List[Int]
    var _num_batches: Int

    # Cached dataset metadata
    var _feature_shape: Shape  # Shape of single sample (e.g., [1, 28, 28])
    var _label_shape: Shape  # Shape of single label (e.g., [])
    var _features_per_sample: Int
    var _labels_per_sample: Int

    # Pre-allocated batch buffers
    var _batch: Batch[DatasetSource._feature_dtype, DatasetSource._label_dtype]
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
        self._feature_shape = other._feature_shape
        self._label_shape = other._label_shape
        self._features_per_sample = other._features_per_sample
        self._labels_per_sample = other._labels_per_sample
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

        # Calculate number of batches
        if drop_last:
            self._num_batches = total_samples // batch_size
        else:
            self._num_batches = (total_samples + batch_size - 1) // batch_size

        # Cache dataset metadata
        ref dataset_ref = dataset[]
        self._feature_shape = dataset_ref.get_feature_shape()
        self._label_shape = dataset_ref.get_label_shape()
        self._features_per_sample = dataset_ref.get_features_per_sample()
        self._labels_per_sample = dataset_ref.get_labels_per_sample()

        # Build batch shape: [batch_size, *feature_shape]
        var batch_feature_dims = List[Int](
            capacity=self._feature_shape.rank() + 1
        )
        batch_feature_dims.append(batch_size)
        for i in range(self._feature_shape.rank()):
            batch_feature_dims.append(self._feature_shape[i])

        # Build label shape: [batch_size, *label_shape]
        var batch_label_dims = List[Int](capacity=self._label_shape.rank() + 1)
        batch_label_dims.append(batch_size)
        for i in range(self._label_shape.rank()):
            batch_label_dims.append(self._label_shape[i])

        # Allocate full-size batch
        var batch_features = Tensor[DatasetSource._feature_dtype].zeros(
            Shape(batch_feature_dims)
        )
        var batch_labels = Tensor[DatasetSource._label_dtype].zeros(
            Shape(batch_label_dims)
        )

        self._batch = Batch[
            DatasetSource._feature_dtype, DatasetSource._label_dtype
        ](batch_features^, batch_labels^)

        # Allocate last batch if needed
        if not drop_last:
            var remainder = total_samples % batch_size
            if remainder != 0:
                self._last_batch_size = remainder

                var last_feature_dims = List[Int](
                    capacity=self._feature_shape.rank() + 1
                )
                last_feature_dims.append(remainder)
                for i in range(self._feature_shape.rank()):
                    last_feature_dims.append(self._feature_shape[i])

                var last_label_dims = List[Int](
                    capacity=self._label_shape.rank() + 1
                )
                last_label_dims.append(remainder)
                for i in range(self._label_shape.rank()):
                    last_label_dims.append(self._label_shape[i])

                var last_features = Tensor[DatasetSource._feature_dtype].zeros(
                    Shape(last_feature_dims)
                )
                var last_labels = Tensor[DatasetSource._label_dtype].zeros(
                    Shape(last_label_dims)
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
        """Get next batch with proper shape preservation."""
        var start_idx = self._current_idx
        var end_idx = min(start_idx + self.batch_size, len(self._indices))
        var actual_batch_size = end_idx - start_idx

        var is_last_batch = actual_batch_size < self.batch_size
        ref dataset_ref = self.dataset[]

        # Choose appropriate batch
        if is_last_batch and self._last_batch:
            ref current_batch = self._last_batch.value()
            self._fill_batch(
                current_batch, start_idx, actual_batch_size, dataset_ref
            )
            self._current_idx = end_idx
            return current_batch
        else:
            ref current_batch = self._batch
            self._fill_batch(
                current_batch, start_idx, actual_batch_size, dataset_ref
            )
            self._current_idx = end_idx
            return current_batch

    fn _fill_batch(
        self,
        batch: Batch[DatasetSource._feature_dtype, DatasetSource._label_dtype],
        start_idx: Int,
        actual_batch_size: Int,
        ref dataset_ref: DatasetSource,
    ):
        """Fill batch with data, preserving multi-dimensional structure."""
        var dataset_features_ptr = dataset_ref.get_features_ptr()
        var dataset_labels_ptr = dataset_ref.get_labels_ptr()
        var batch_features_ptr = batch.features.buffer.data_buffer().data
        var batch_labels_ptr = batch.labels.buffer.data_buffer().data

        # OPTIMIZATION: Bulk copy if not shuffled
        if not self.shuffle_data:
            var first_sample_idx = self._indices[start_idx]

            # Copy features in bulk
            var src_features_offset = (
                first_sample_idx * self._features_per_sample
            )
            var total_feature_elements = (
                actual_batch_size * self._features_per_sample
            )
            memcpy(
                dest=batch_features_ptr,
                src=dataset_features_ptr + src_features_offset,
                count=total_feature_elements,
            )

            # Copy labels in bulk
            var src_labels_offset = first_sample_idx * self._labels_per_sample
            var total_label_elements = (
                actual_batch_size * self._labels_per_sample
            )
            memcpy(
                dest=batch_labels_ptr,
                src=dataset_labels_ptr + src_labels_offset,
                count=total_label_elements,
            )
        else:
            # Row-by-row copy for shuffled data
            for i in range(actual_batch_size):
                var sample_idx = self._indices[start_idx + i]

                # Copy feature sample
                var src_offset = sample_idx * self._features_per_sample
                var dst_offset = i * self._features_per_sample
                memcpy(
                    dest=batch_features_ptr + dst_offset,
                    src=dataset_features_ptr + src_offset,
                    count=self._features_per_sample,
                )

                # Copy label
                var src_label_offset = sample_idx * self._labels_per_sample
                var dst_label_offset = i * self._labels_per_sample
                memcpy(
                    dest=batch_labels_ptr + dst_label_offset,
                    src=dataset_labels_ptr + src_label_offset,
                    count=self._labels_per_sample,
                )

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

# ==============================================================================
# UPDATED NUMPYDATASET - Properly handles multi-dimensional data
# ==============================================================================


@fieldwise_init
struct NumpyDataset[feature_dtype: DType, label_dtype: DType = feature_dtype](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset that preserves original tensor shapes."""

    alias _feature_dtype = feature_dtype
    alias _label_dtype = label_dtype

    var _features: Tensor[Self.feature_dtype]
    var _labels: Tensor[Self.label_dtype]
    var _size: Int
    var _feature_shape: Shape  # Shape without batch dimension
    var _label_shape: Shape  # Shape without batch dimension
    var _features_per_sample: Int
    var _labels_per_sample: Int

    fn __init__(
        out self,
        features_numpy: PythonObject,
        labels_numpy: PythonObject,
        copy: Bool = True,
    ) raises:
        """Create dataset from NumPy arrays. Copies data once."""
        features = from_ndarray[Self.feature_dtype](
            features_numpy, requires_grad=False, copy=copy
        )
        labels = from_ndarray[Self.label_dtype](
            labels_numpy, requires_grad=False, copy=copy
        )
        self = Self(features, labels)

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

        # Extract shapes (excluding batch dimension)
        var features_shape = features.shape()
        var labels_shape = labels.shape()

        # Feature shape: everything after batch dimension
        var feature_dims = List[Int](capacity=features_shape.rank() - 1)
        for i in range(1, features_shape.rank()):
            feature_dims.append(features_shape[i])
        self._feature_shape = Shape(feature_dims)

        # Label shape: everything after batch dimension
        var label_dims = List[Int](capacity=labels_shape.rank() - 1)
        for i in range(1, labels_shape.rank()):
            label_dims.append(labels_shape[i])
        self._label_shape = Shape(label_dims)

        # Calculate total elements per sample
        self._features_per_sample = 1
        for i in range(self._feature_shape.rank()):
            self._features_per_sample *= self._feature_shape[i]

        self._labels_per_sample = 1
        for i in range(self._label_shape.rank()):
            self._labels_per_sample *= self._label_shape[i]

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

    fn get_feature_shape(self) -> Shape:
        return self._feature_shape

    fn get_label_shape(self) -> Shape:
        return self._label_shape

    fn get_features_per_sample(self) -> Int:
        return self._features_per_sample

    fn get_labels_per_sample(self) -> Int:
        return self._labels_per_sample

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        """Get single sample - not used by DataLoader."""
        if idx < 0 or idx >= self._size:
            panic("NumpyDataset: index out of bounds")

        # Create tensors with proper shape
        var sample_feature = Tensor[Self.feature_dtype].zeros(
            self._feature_shape
        )
        var sample_label = Tensor[Self.label_dtype].zeros(self._label_shape)

        var dataset_features_ptr = self.get_features_ptr()
        var dataset_labels_ptr = self.get_labels_ptr()
        var sample_feature_ptr = sample_feature.buffer.data_buffer().data
        var sample_label_ptr = sample_label.buffer.data_buffer().data

        # Copy data
        var src_feature_offset = idx * self._features_per_sample
        memcpy(
            dest=sample_feature_ptr,
            src=dataset_features_ptr + src_feature_offset,
            count=self._features_per_sample,
        )

        var src_label_offset = idx * self._labels_per_sample
        memcpy(
            dest=sample_label_ptr,
            src=dataset_labels_ptr + src_label_offset,
            count=self._labels_per_sample,
        )

        return (sample_feature^, sample_label^)

    fn into_loader(
        ref self, batch_size: Int, shuffle: Bool = True, drop_last: Bool = False
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(Pointer(to=self), batch_size, shuffle, drop_last)


# """
# Updated TensorDataset that properly handles multi-dimensional tensors.
# Works with both the old API (for simple 2D data) and new API (for any dimensions).
# """


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
    var _feature_shape: Shape  # Shape of single sample (excluding batch dim)
    var _label_shape: Shape  # Shape of single label (excluding batch dim)
    var _features_per_sample: Int
    var _labels_per_sample: Int

    # Legacy fields for backward compatibility (if needed)
    var _feature_dim: Int  # For 2D data: equals _features_per_sample
    var _label_dim: Int  # For 2D data: equals _labels_per_sample
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

        # Extract shapes (excluding batch dimension)
        var features_shape = features.shape()
        var labels_shape = labels.shape()

        # Feature shape: everything after batch dimension
        var feature_dims = List[Int](capacity=features_shape.rank() - 1)
        for i in range(1, features_shape.rank()):
            feature_dims.append(features_shape[i])
        self._feature_shape = Shape(feature_dims)

        # Label shape: everything after batch dimension
        var label_dims = List[Int](capacity=labels_shape.rank() - 1)
        for i in range(1, labels_shape.rank()):
            label_dims.append(labels_shape[i])
        self._label_shape = Shape(label_dims)

        # Calculate total elements per sample
        self._features_per_sample = 1
        for i in range(self._feature_shape.rank()):
            self._features_per_sample *= self._feature_shape[i]

        self._labels_per_sample = 1
        for i in range(self._label_shape.rank()):
            self._labels_per_sample *= self._label_shape[i]

        # Legacy compatibility fields
        self._feature_dim = self._features_per_sample
        self._label_dim = (
            self._labels_per_sample if self._label_shape.rank() > 0 else 1
        )
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

    # New API methods (required by updated Dataset trait)
    fn get_feature_shape(self) -> Shape:
        """Get the shape of a single feature sample (excluding batch dimension).
        """
        return self._feature_shape

    fn get_label_shape(self) -> Shape:
        """Get the shape of a single label (excluding batch dimension)."""
        return self._label_shape

    fn get_features_per_sample(self) -> Int:
        """Total number of elements per feature sample."""
        return self._features_per_sample

    fn get_labels_per_sample(self) -> Int:
        """Total number of elements per label."""
        return self._labels_per_sample

    # Legacy API methods (for backward compatibility)
    fn get_feature_dim(self) -> Int:
        """Legacy: Returns total feature elements (same as get_features_per_sample).
        """
        return self._feature_dim

    fn get_label_dim(self) -> Int:
        """Legacy: Returns total label elements."""
        return self._label_dim

    fn is_labels_scalar(self) -> Bool:
        """Legacy: True if labels are scalar (rank 1)."""
        return self._labels_scalar

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        """Get single sample - preserves original shape."""
        if idx < 0 or idx >= self._size:
            panic("TensorDataset: index out of bounds")

        # Create tensors with proper shape
        var sample_feature = Tensor[Self.feature_dtype].zeros(
            self._feature_shape
        )
        var sample_label = Tensor[Self.label_dtype].zeros(self._label_shape)

        var dataset_features_ptr = self.get_features_ptr()
        var dataset_labels_ptr = self.get_labels_ptr()
        var sample_feature_ptr = sample_feature.buffer.data_buffer().data
        var sample_label_ptr = sample_label.buffer.data_buffer().data

        # Copy feature data
        var src_feature_offset = idx * self._features_per_sample
        memcpy(
            dest=sample_feature_ptr,
            src=dataset_features_ptr + src_feature_offset,
            count=self._features_per_sample,
        )

        # Copy label data
        var src_label_offset = idx * self._labels_per_sample
        memcpy(
            dest=sample_label_ptr,
            src=dataset_labels_ptr + src_label_offset,
            count=self._labels_per_sample,
        )

        return (sample_feature^, sample_label^)

    fn into_loader(
        ref self, batch_size: Int, shuffle: Bool = True, drop_last: Bool = False
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(Pointer(to=self), batch_size, shuffle, drop_last)
