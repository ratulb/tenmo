from .tensor import Tensor
from .common_utils import panic
from std.random import shuffle as reshuffle, random_si64
from std.python import PythonObject
from .numpy_interop import from_ndarray, numpy_dtype
from std.memory import memcpy, Pointer
from .shapes import Shape

# MNIST
comptime MNIST_MEAN = 0.1307
comptime MNIST_STD = 0.3081

# Fashion-MNIST
comptime FASHION_MNIST_MEAN = 0.2860
comptime FASHION_MNIST_STD = 0.3530

# CIFAR-10 (per-channel)
comptime CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
comptime CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ImageNet (per-channel)
comptime IMAGENET_MEAN = (0.485, 0.456, 0.406)
comptime IMAGENET_STD = (0.229, 0.224, 0.225)


@fieldwise_init
struct Batch[feature_dtype: DType, label_dtype: DType](
    ImplicitlyCopyable & Movable
):
    var features: Tensor[Self.feature_dtype]
    var labels: Tensor[Self.label_dtype]
    var batch_size: Int

    def __init__(
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
    comptime _feature_dtype: DType
    comptime _label_dtype: DType

    def __len__(self) -> Int:
        ...

    def get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self._feature_dtype], ImmutAnyOrigin]:
        """Get raw pointer to feature data."""
        ...

    def get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self._label_dtype], ImmutAnyOrigin]:
        """Get raw pointer to label data."""
        ...

    def get_feature_shape(self) -> Shape:
        """Get the shape of a single feature sample (excluding batch dimension).
        """
        ...

    def get_label_shape(self) -> Shape:
        """Get the shape of a single label (excluding batch dimension)."""
        ...

    def get_features_per_sample(self) -> Int:
        """Total number of elements per feature sample."""
        ...

    def get_labels_per_sample(self) -> Int:
        """Total number of elements per label."""
        ...

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[Scalar[Self._feature_dtype]] = None,
        normalize_std: Optional[Scalar[Self._feature_dtype]] = None,
    ) -> DataLoader[Self, origin_of(self)]:
        ...

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self._feature_dtype], Tensor[Self._label_dtype]]:
        ...


# ==============================================================================
# UPDATED DATALOADER - Preserves multi-dimensional shapes
# ==============================================================================


@fieldwise_init
struct DataLoader[DatasetSource: Dataset, origin: ImmutOrigin](
    ImplicitlyCopyable & Movable & Sized & Iterator
):
    """Zero-copy batched data loading that preserves tensor shapes."""

    var dataset: Pointer[Self.DatasetSource, Self.origin]
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
    var _batch: Batch[
        Self.DatasetSource._feature_dtype, Self.DatasetSource._label_dtype
    ]
    var _last_batch: Optional[
        Batch[
            Self.DatasetSource._feature_dtype, Self.DatasetSource._label_dtype
        ]
    ]
    var _last_batch_size: Int

    # Optional normalization (mean/std applied after batch fill)
    var _normalize_mean: Optional[Scalar[Self.DatasetSource._feature_dtype]]
    var _normalize_std: Optional[Scalar[Self.DatasetSource._feature_dtype]]

    def __init__(out self, *, copy: Self):
        self.dataset = copy.dataset
        self.batch_size = copy.batch_size
        self.shuffle_data = copy.shuffle_data
        self.drop_last = copy.drop_last
        self._current_idx = copy._current_idx
        self._indices = copy._indices.copy()
        self._num_batches = copy._num_batches
        self._feature_shape = copy._feature_shape
        self._label_shape = copy._label_shape
        self._features_per_sample = copy._features_per_sample
        self._labels_per_sample = copy._labels_per_sample
        self._batch = copy._batch
        self._last_batch = copy._last_batch
        self._last_batch_size = copy._last_batch_size
        self._normalize_mean = copy._normalize_mean
        self._normalize_std = copy._normalize_std

    def __init__(
        out self,
        dataset: Pointer[Self.DatasetSource, Self.origin],
        batch_size: Int,
        shuffle: Bool = False,
        drop_last: Bool = False,
        normalize_mean: Optional[
            Scalar[Self.DatasetSource._feature_dtype]
        ] = None,
        normalize_std: Optional[
            Scalar[Self.DatasetSource._feature_dtype]
        ] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.drop_last = drop_last
        self._current_idx = 0
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std

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
        var batch_features = Tensor[Self.DatasetSource._feature_dtype].zeros(
            Shape(batch_feature_dims)
        )
        var batch_labels = Tensor[Self.DatasetSource._label_dtype].zeros(
            Shape(batch_label_dims)
        )

        # Share batch buffers so copies stay cheap (refcount bump, not deep alloc)
        batch_features.buffer.buffer.shared()
        batch_labels.buffer.buffer.shared()

        self._batch = Batch[
            Self.DatasetSource._feature_dtype, Self.DatasetSource._label_dtype
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

                var last_features = Tensor[
                    Self.DatasetSource._feature_dtype
                ].zeros(Shape(last_feature_dims))
                var last_labels = Tensor[Self.DatasetSource._label_dtype].zeros(
                    Shape(last_label_dims)
                )

                # Share last batch buffers too
                last_features.buffer.buffer.shared()
                last_labels.buffer.buffer.shared()

                self._last_batch = Batch[
                    Self.DatasetSource._feature_dtype,
                    Self.DatasetSource._label_dtype,
                ](last_features^, last_labels^)
            else:
                self._last_batch_size = 0
                self._last_batch = None
        else:
            self._last_batch_size = 0
            self._last_batch = None

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[
        Tensor[Self.DatasetSource._feature_dtype],
        Tensor[Self.DatasetSource._label_dtype],
    ]:
        return self.dataset[].sample(idx)

    def __iter__(mut self) -> ref[self] Self.IteratorType[origin_of(self)]:
        self._current_idx = 0
        if self.shuffle_data:
            reshuffle(self._indices)
        return self

    comptime Element = Batch[
        Self.DatasetSource._feature_dtype, Self.DatasetSource._label_dtype
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len = len(self)
        return (iter_len, {iter_len})

    def __next__(
        mut self,
    ) raises StopIteration -> ref[
        self._batch, self._last_batch.value()
    ] Self.Element:
        """Get next batch with proper shape preservation."""
        if not self.__has_next__():
            raise StopIteration()
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

    def _fill_batch(
        self,
        batch: Batch[
            Self.DatasetSource._feature_dtype, Self.DatasetSource._label_dtype
        ],
        start_idx: Int,
        actual_batch_size: Int,
        ref dataset_ref: Self.DatasetSource,
    ):
        """Fill batch with data, preserving multi-dimensional structure."""
        var dataset_features_ptr = dataset_ref.get_features_ptr()
        var dataset_labels_ptr = dataset_ref.get_labels_ptr()
        var batch_features_ptr = (
            batch.features.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var batch_labels_ptr = (
            batch.labels.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        var total_feature_elements = (
            actual_batch_size * self._features_per_sample
        )
        var total_label_elements = actual_batch_size * self._labels_per_sample

        # OPTIMIZATION: Bulk copy if not shuffled
        if not self.shuffle_data:
            var first_sample_idx = self._indices[start_idx]

            # Copy features in bulk
            var src_features_offset = (
                first_sample_idx * self._features_per_sample
            )
            memcpy(
                dest=batch_features_ptr,
                src=dataset_features_ptr + src_features_offset,
                count=total_feature_elements,
            )

            # Copy labels in bulk
            var src_labels_offset = first_sample_idx * self._labels_per_sample
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

        # Apply normalization after fill (SIMD)
        if self._normalize_mean and self._normalize_std:
            var mean = self._normalize_mean.value()
            var inv_std = (
                Scalar[Self.DatasetSource._feature_dtype](1)
                / self._normalize_std.value()
            )

            comptime simd_width = simd_width_of[
                Scalar[Self.DatasetSource._feature_dtype]
            ]()
            var i = 0
            for i in range(
                0, total_feature_elements - simd_width + 1, simd_width
            ):
                var vec = batch_features_ptr.load[width=simd_width](i)
                vec = (vec - mean) * inv_std
                batch_features_ptr.store[width=simd_width](i, vec)
            for i in range(
                total_feature_elements - simd_width + 1,
                total_feature_elements,
            ):
                batch_features_ptr[i] = (batch_features_ptr[i] - mean) * inv_std

    def __has_next__(self) -> Bool:
        if self.drop_last:
            return (self._current_idx + self.batch_size) <= len(self._indices)
        else:
            return self._current_idx < len(self._indices)

    def __len__(self) -> Int:
        return self._num_batches

    def reset(mut self):
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

    comptime _feature_dtype = Self.feature_dtype
    comptime _label_dtype = Self.label_dtype

    var _features: Tensor[Self.feature_dtype]
    var _labels: Tensor[Self.label_dtype]
    var _size: Int
    var _feature_shape: Shape  # Shape without batch dimension
    var _label_shape: Shape  # Shape without batch dimension
    var _features_per_sample: Int
    var _labels_per_sample: Int

    def __init__(
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

    def __init__(
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

    def __len__(self) -> Int:
        return self._size

    def get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.feature_dtype], ImmutAnyOrigin]:
        return self._features.data_ptr().as_immutable()

    def get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.label_dtype], ImmutAnyOrigin]:
        return self._labels.data_ptr().as_immutable()

    def get_feature_shape(self) -> Shape:
        return self._feature_shape

    def get_label_shape(self) -> Shape:
        return self._label_shape

    def get_features_per_sample(self) -> Int:
        return self._features_per_sample

    def get_labels_per_sample(self) -> Int:
        return self._labels_per_sample

    def __getitem__(
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
        var sample_feature_ptr = (
            sample_feature.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var sample_label_ptr = (
            sample_label.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

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

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        if idx:
            return self.__getitem__(idx.value())
        else:
            return self.__getitem__(Int(random_si64(0, self._size - 1)))

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[Scalar[Self._feature_dtype]] = None,
        normalize_std: Optional[Scalar[Self._feature_dtype]] = None,
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(
            Pointer(to=self),
            batch_size,
            shuffle,
            drop_last,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )


# """
# Updated TensorDataset that properly handles multi-dimensional tensors.
# Works with both the old API (for simple 2D data) and new API (for any dimensions).
# """


@fieldwise_init
struct TensorDataset[feature_dtype: DType, label_dtype: DType = feature_dtype](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset from tensors. References existing data (no copy)."""

    comptime _feature_dtype = Self.feature_dtype
    comptime _label_dtype = Self.label_dtype

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

    def __init__(
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

    def __len__(self) -> Int:
        return self._size

    def get_features_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.feature_dtype], ImmutAnyOrigin]:
        return self._features.data_ptr().as_immutable()

    def get_labels_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.label_dtype], ImmutAnyOrigin]:
        return self._labels.data_ptr().as_immutable()

    # New API methods (required by updated Dataset trait)
    def get_feature_shape(self) -> Shape:
        """Get the shape of a single feature sample (excluding batch dimension).
        """
        return self._feature_shape

    def get_label_shape(self) -> Shape:
        """Get the shape of a single label (excluding batch dimension)."""
        return self._label_shape

    def get_features_per_sample(self) -> Int:
        """Total number of elements per feature sample."""
        return self._features_per_sample

    def get_labels_per_sample(self) -> Int:
        """Total number of elements per label."""
        return self._labels_per_sample

    # Legacy API methods (for backward compatibility)
    def get_feature_dim(self) -> Int:
        """Legacy: Returns total feature elements (same as get_features_per_sample).
        """
        return self._feature_dim

    def get_label_dim(self) -> Int:
        """Legacy: Returns total label elements."""
        return self._label_dim

    def is_labels_scalar(self) -> Bool:
        """Legacy: True if labels are scalar (rank 1)."""
        return self._labels_scalar

    def __getitem__(
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
        var sample_feature_ptr = (
            sample_feature.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var sample_label_ptr = (
            sample_label.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

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

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self.feature_dtype], Tensor[Self.label_dtype]]:
        if idx:
            return self.__getitem__(idx.value())
        else:
            return self.__getitem__(Int(random_si64(0, self._size - 1)))

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[Scalar[Self._feature_dtype]] = None,
        normalize_std: Optional[Scalar[Self._feature_dtype]] = None,
    ) -> DataLoader[Self, origin_of(self)]:
        return DataLoader(
            Pointer(to=self),
            batch_size,
            shuffle,
            drop_last,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
