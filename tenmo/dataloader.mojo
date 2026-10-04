from .tensor import Tensor
from .gpu.device import CPU, Device, GPU
from .kernels.gather_kernel import GatherKernel
from .shared.intarray import IntArray
from .shared.panic import panic
from std.random import shuffle as reshuffle, random_si64
from std.python import PythonObject
from .numpy_interop import from_ndarray
from std.memory import unsafe_memcpy, Pointer
from .shared.shapes import Shape
from std.sys import has_accelerator, simd_width_of

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
struct Batch[sample_dtype: DType, label_dtype: DType](ImplicitlyCopyable):
    var features: Tensor[Self.sample_dtype]
    var labels: Tensor[Self.label_dtype]
    var batch_size: Int

    def __init__(
        out self,
        features: Tensor[Self.sample_dtype],
        labels: Tensor[Self.label_dtype],
    ):
        self.features = features
        self.labels = labels
        self.batch_size = features.shape()[0]

trait Dataset(Sized & Copyable):
    comptime _sample_dtype: DType
    comptime _label_dtype: DType

    def __len__(self) -> Int:
        ...

    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self._sample_dtype], ImmutAnyOrigin]:
        """Get raw pointer to feature data."""
        ...

    def get_labels_ptr(
        ref self,
    ) -> Pointer[Scalar[Self._label_dtype], ImmutAnyOrigin]:
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

    def device(ref self) -> Device:
        """Device the bulk data lives on. Defaults to CPU.

        Overridden by conformers that hold device-resident tensors.
        Loaders allocate their batch buffers here and dispatch their
        gather here.
        """
        return CPU().into()

    def get_features(ref self) -> Tensor[Self._sample_dtype]:
        """Borrow the bulk feature tensor (aliases storage; zero-copy).

        Default panics: only tensor-backed datasets (whose bulk data can
        live on a device) implement this. List-backed datasets
        (`LLMDataset`, `RandomSlidingWindowDataset`) are always host-side
        and are served by the flat-pointer contract instead.
        """
        panic(
            "Dataset.get_features: no tensor-backed bulk storage; the"
            " flat host pointer contract applies."
        )
        return Tensor[Self._sample_dtype].zeros(Shape(0))

    def get_labels(ref self) -> Tensor[Self._label_dtype]:
        """Borrow the bulk label tensor (aliases storage; zero-copy)."""
        panic(
            "Dataset.get_labels: no tensor-backed bulk storage; the flat"
            " host pointer contract applies."
        )
        return Tensor[Self._label_dtype].zeros(Shape(0))

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[Scalar[Self._sample_dtype]] = None,
        normalize_std: Optional[Scalar[Self._sample_dtype]] = None,
    ) -> NativeLoader[Self, origin_of(self)]:
        ...

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self._sample_dtype], Tensor[Self._label_dtype]]:
        ...


@fieldwise_init
struct NativeLoader[DatasetSource: Dataset, origin: ImmOrigin](
    ImplicitlyCopyable & Sized & Iterator
):
    """Zero-copy batched data loading over a `Dataset` trait source.
    (Mojo-native; not directly Python-bound).

    On a device-resident source, sequential batches are view slices that
    alias the dataset (like `DataLoader`); shuffled batches fill the
    persistent device buffers in place. The loader is single-mode
    (`shuffle` is fixed at construction), so the two never mix.
    """

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
        Self.DatasetSource._sample_dtype, Self.DatasetSource._label_dtype
    ]
    var _last_batch: Optional[
        Batch[
            Self.DatasetSource._sample_dtype, Self.DatasetSource._label_dtype
        ]
    ]
    var _last_batch_size: Int

    # Optional normalization (mean/std applied after batch fill)
    var _normalize_mean: Optional[Scalar[Self.DatasetSource._sample_dtype]]
    var _normalize_std: Optional[Scalar[Self.DatasetSource._sample_dtype]]

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
        shuffle: Bool = True,
        drop_last: Bool = False,
        normalize_mean: Optional[
            Scalar[Self.DatasetSource._sample_dtype]
        ] = None,
        normalize_std: Optional[
            Scalar[Self.DatasetSource._sample_dtype]
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

        # Allocate full-size batch on the dataset's device
        var batch_features = Tensor[Self.DatasetSource._sample_dtype].zeros(
            Shape(batch_feature_dims), device=dataset_ref.device()
        )
        var batch_labels = Tensor[Self.DatasetSource._label_dtype].zeros(
            Shape(batch_label_dims), device=dataset_ref.device()
        )

        self._batch = Batch[
            Self.DatasetSource._sample_dtype, Self.DatasetSource._label_dtype
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
                    Self.DatasetSource._sample_dtype
                ].zeros(
                    Shape(last_feature_dims), device=dataset_ref.device()
                )
                var last_labels = Tensor[Self.DatasetSource._label_dtype].zeros(
                    Shape(last_label_dims), device=dataset_ref.device()
                )

                self._last_batch = Batch[
                    Self.DatasetSource._sample_dtype,
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
        Tensor[Self.DatasetSource._sample_dtype],
        Tensor[Self.DatasetSource._label_dtype],
    ]:
        return self.dataset[].sample(idx)

    def __iter__(mut self) -> ref[self] Self.IteratorType[origin_of(self)]:
        self._current_idx = 0
        if self.shuffle_data:
            reshuffle(self._indices)
        return self

    comptime Element = Batch[
        Self.DatasetSource._sample_dtype, Self.DatasetSource._label_dtype
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
        var on_gpu = dataset_ref.device().is_gpu()

        # Choose appropriate batch. __next__ raises StopIteration only, so
        # device errors (OOM, transfer failure) panic with context instead
        # of widening the iterator protocol. The device branch is also
        # comptime-guarded: it instantiates GPU kernels, which have no
        # target architecture on a CPU-only build.
        if is_last_batch and self._last_batch:
            ref current_batch = self._last_batch.value()
            comptime if has_accelerator():
                if on_gpu:
                    try:
                        self._next_batch_device(
                            True, start_idx, actual_batch_size, dataset_ref
                        )
                    except e:
                        panic(
                            "NativeLoader: device batch failed: ", String(e)
                        )
                    self._current_idx = end_idx
                    return current_batch
            self._fill_batch(
                current_batch, start_idx, actual_batch_size, dataset_ref
            )
            self._current_idx = end_idx
            return current_batch
        else:
            ref current_batch = self._batch
            comptime if has_accelerator():
                if on_gpu:
                    try:
                        self._next_batch_device(
                            False, start_idx, actual_batch_size, dataset_ref
                        )
                    except e:
                        panic(
                            "NativeLoader: device batch failed: ", String(e)
                        )
                    self._current_idx = end_idx
                    return current_batch
            self._fill_batch(
                current_batch, start_idx, actual_batch_size, dataset_ref
            )
            self._current_idx = end_idx
            return current_batch

    def _next_batch_device(
        mut self,
        is_last_batch: Bool,
        start_idx: Int,
        actual_batch_size: Int,
        ref dataset_ref: Self.DatasetSource,
    ) raises:
        """Device-resident batch: sequential binds views, shuffled gathers.

        Sequential batches become view slices aliasing the dataset (the
        `DataLoader` shape); shuffled batches gather rows in place into
        the persistent device buffers. No host transfers either way.
        """
        var feats = dataset_ref.get_features()
        var labs = dataset_ref.get_labels()
        var end_idx = start_idx + actual_batch_size

        if not self.shuffle_data:
            var fx = feats.slice(
                start=start_idx, end=end_idx, step=1, axis=0
            )
            var ly = labs.slice(
                start=start_idx, end=end_idx, step=1, axis=0
            )
            if self._normalize_mean and self._normalize_std:
                var mean = self._normalize_mean.value()
                var inv_std = Scalar[Self.DatasetSource._sample_dtype](1) / (
                    self._normalize_std.value()
                )
                fx = ((fx - mean) * inv_std)
            if is_last_batch and self._last_batch:
                ref last_batch = self._last_batch.value()
                last_batch.features = fx
                last_batch.labels = ly
            else:
                self._batch.features = fx
                self._batch.labels = ly
        else:
            if is_last_batch and self._last_batch:
                self._fill_batch_device(
                    self._last_batch.value(),
                    start_idx,
                    actual_batch_size,
                    dataset_ref,
                )
            else:
                self._fill_batch_device(
                    self._batch, start_idx, actual_batch_size, dataset_ref
                )
            self._normalize_device_batch(is_last_batch)

    def _fill_batch_device(
        self,
        batch: Batch[
            Self.DatasetSource._sample_dtype, Self.DatasetSource._label_dtype
        ],
        start_idx: Int,
        actual_batch_size: Int,
        ref dataset_ref: Self.DatasetSource,
    ) raises:
        """Gather rows `_indices[start_idx:start_idx+bs]` in place on device.

        Fills the preallocated device batch buffers; the only host→device
        traffic is the batch's index list. Bulk data stays resident.
        """
        var feats = dataset_ref.get_features()
        var labs = dataset_ref.get_labels()
        var idx = IntArray.with_capacity(actual_batch_size)
        for k in range(actual_batch_size):
            idx.append(self._indices[start_idx + k])
        GatherKernel[Self.DatasetSource._sample_dtype].gather_rows_2d_into(
            feats.buffer.layout(),
            feats.buffer.device_state.value(),
            idx,
            batch.features.buffer.layout(),
            batch.features.buffer.device_state.value(),
        )
        GatherKernel[Self.DatasetSource._label_dtype].gather_rows_2d_into(
            labs.buffer.layout(),
            labs.buffer.device_state.value(),
            idx,
            batch.labels.buffer.layout(),
            batch.labels.buffer.device_state.value(),
        )

    def _normalize_device_batch(mut self, is_last_batch: Bool) raises:
        """Apply (x - mean) * inv_std to a device batch, in place by rebind.

        Transient until normalization is hoisted out of the loader: two
        elementwise launches, zero transfers. No-op unless configured.
        """
        if self._normalize_mean and self._normalize_std:
            var mean = self._normalize_mean.value()
            var inv_std = Scalar[Self.DatasetSource._sample_dtype](1) / (
                self._normalize_std.value()
            )
            if is_last_batch and self._last_batch:
                ref last_batch = self._last_batch.value()
                last_batch.features = (
                    (last_batch.features - mean) * inv_std
                )
            else:
                self._batch.features = (
                    (self._batch.features - mean) * inv_std
                )

    def _fill_batch(
        self,
        batch: Batch[
            Self.DatasetSource._sample_dtype, Self.DatasetSource._label_dtype
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

        # Bulk copy if not shuffled
        if not self.shuffle_data:
            var first_sample_idx = self._indices[start_idx]

            # Copy features in bulk
            var src_features_offset = (
                first_sample_idx * self._features_per_sample
            )
            unsafe_memcpy(
                dest=batch_features_ptr,
                src=dataset_features_ptr.unsafe_offset(src_features_offset),
                count=total_feature_elements,
            )

            # Copy labels in bulk
            var src_labels_offset = first_sample_idx * self._labels_per_sample
            unsafe_memcpy(
                dest=batch_labels_ptr,
                src=dataset_labels_ptr.unsafe_offset(src_labels_offset),
                count=total_label_elements,
            )
        else:
            # Row-by-row copy for shuffled data
            for i in range(actual_batch_size):
                var sample_idx = self._indices[start_idx + i]

                # Copy feature sample
                var src_offset = sample_idx * self._features_per_sample
                var dst_offset = i * self._features_per_sample
                unsafe_memcpy(
                    dest=batch_features_ptr.unsafe_offset(dst_offset),
                    src=dataset_features_ptr.unsafe_offset(src_offset),
                    count=self._features_per_sample,
                )

                # Copy label
                var src_label_offset = sample_idx * self._labels_per_sample
                var dst_label_offset = i * self._labels_per_sample
                unsafe_memcpy(
                    dest=batch_labels_ptr.unsafe_offset(dst_label_offset),
                    src=dataset_labels_ptr.unsafe_offset(src_label_offset),
                    count=self._labels_per_sample,
                )

        # Apply normalization after fill (SIMD)
        if self._normalize_mean and self._normalize_std:
            var mean = self._normalize_mean.value()
            var inv_std = (
                Scalar[Self.DatasetSource._sample_dtype](1)
                / self._normalize_std.value()
            )

            comptime simd_width = simd_width_of[
                Scalar[Self.DatasetSource._sample_dtype]
            ]()
            # Remainder starts where vector coverage ends: the last vector
            # starts at or below total - simd, so starting the scalar tail
            # at total - simd + 1 would re-process (and double-normalize)
            # the tail whenever total is a multiple of simd_width.
            var vec_end = (
                total_feature_elements // simd_width
            ) * simd_width
            for i in range(0, vec_end, simd_width):
                var vec = batch_features_ptr.unsafe_load[width=simd_width](i)
                vec = (vec - mean) * inv_std
                batch_features_ptr.unsafe_store[width=simd_width](i, vec)
            for i in range(vec_end, total_feature_elements):
                batch_features_ptr[unsafe_offset=i] = (
                    batch_features_ptr[unsafe_offset=i] - mean
                ) * inv_std

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



@fieldwise_init
struct NumpyDataset[sample_dtype: DType, label_dtype: DType = sample_dtype](
    ImplicitlyCopyable & Sized & Dataset
):
    """Dataset that preserves original tensor shapes."""

    comptime _sample_dtype = Self.sample_dtype
    comptime _label_dtype = Self.label_dtype

    var _features: Tensor[Self.sample_dtype]
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
        var features = from_ndarray[Self.sample_dtype](
            features_numpy, requires_grad=False, copy=copy
        )
        var labels = from_ndarray[Self.label_dtype](
            labels_numpy, requires_grad=False, copy=copy
        )
        self = Self(features, labels)

    def __init__(
        out self,
        features: Tensor[Self.sample_dtype],
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

    def device(ref self) -> Device:
        return self._features.device()

    def to_gpu(
        ref self, gpu: Optional[GPU] = None, sync: Bool = True
    ) raises -> Self:
        """Return a dataset whose bulk data is resident on `gpu`.

        The returned dataset owns its tensors; the receiver is unchanged.
        """
        var target = gpu.or_else(GPU())
        var f = self._features.to_gpu(target, sync=sync)
        var y = self._labels.to_gpu(target, sync=sync)
        return Self(f, y)

    def to_cpu(ref self, sync: Bool = True) raises -> Self:
        """Return a dataset whose bulk data is resident on the host."""
        return Self(
            self._features.to_cpu(sync=sync),
            self._labels.to_cpu(sync=sync),
        )

    def get_features(ref self) -> Tensor[Self.sample_dtype]:
        return self._features

    def get_labels(ref self) -> Tensor[Self.label_dtype]:
        return self._labels

    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.sample_dtype], ImmutAnyOrigin]:
        if self.device().is_gpu():
            panic(
                "NumpyDataset.get_features_ptr: bulk data is device-resident;"
                " the flat host pointer contract does not apply. Use the"
                " loader's tensor-level path or .to_cpu() first."
            )
        return self._features.data_ptr().as_imm()

    def get_labels_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.label_dtype], ImmutAnyOrigin]:
        if self.device().is_gpu():
            panic(
                "NumpyDataset.get_labels_ptr: bulk data is device-resident;"
                " the flat host pointer contract does not apply. Use the"
                " loader's tensor-level path or .to_cpu() first."
            )
        return self._labels.data_ptr().as_imm()

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
    ) -> Tuple[Tensor[Self.sample_dtype], Tensor[Self.label_dtype]]:
        """Get single sample - not used by DataLoader."""
        if idx < 0 or idx >= self._size:
            panic("NumpyDataset: index out of bounds")

        # Create tensors with proper shape, on the dataset's device
        var sample_feature = Tensor[Self.sample_dtype].zeros(
            self._feature_shape, device=self._features.device()
        )
        var sample_label = Tensor[Self.label_dtype].zeros(
            self._label_shape, device=self._features.device()
        )

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
        unsafe_memcpy(
            dest=sample_feature_ptr,
            src=dataset_features_ptr.unsafe_offset(src_feature_offset),
            count=self._features_per_sample,
        )

        var src_label_offset = idx * self._labels_per_sample
        unsafe_memcpy(
            dest=sample_label_ptr,
            src=dataset_labels_ptr.unsafe_offset(src_label_offset),
            count=self._labels_per_sample,
        )

        return (sample_feature^, sample_label^)

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self.sample_dtype], Tensor[Self.label_dtype]]:
        if idx:
            return self.__getitem__(idx.value())
        else:
            return self.__getitem__(Int(random_si64(0, Int64(self._size - 1))))

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
struct TensorDataset[sample_dtype: DType, label_dtype: DType = sample_dtype](
    ImplicitlyCopyable & Sized & Dataset
):
    """Dataset from tensors. References existing data (no copy)."""

    comptime _sample_dtype = Self.sample_dtype
    comptime _label_dtype = Self.label_dtype

    var _features: Tensor[Self.sample_dtype]
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
        features: Tensor[Self.sample_dtype],
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

    def device(ref self) -> Device:
        return self._features.device()

    def to_gpu(
        ref self, gpu: Optional[GPU] = None, sync: Bool = True
    ) raises -> Self:
        """Return a dataset whose bulk data is resident on `gpu`.

        The returned dataset owns its tensors; the receiver is unchanged.
        """
        var target = gpu.or_else(GPU())
        var f = self._features.to_gpu(target, sync=sync)
        var y = self._labels.to_gpu(target, sync=sync)
        return Self(f, y)

    def to_cpu(ref self, sync: Bool = True) raises -> Self:
        """Return a dataset whose bulk data is resident on the host."""
        return Self(
            self._features.to_cpu(sync=sync),
            self._labels.to_cpu(sync=sync),
        )

    def get_features(ref self) -> Tensor[Self.sample_dtype]:
        return self._features

    def get_labels(ref self) -> Tensor[Self.label_dtype]:
        return self._labels

    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.sample_dtype], ImmutAnyOrigin]:
        if self.device().is_gpu():
            panic(
                "TensorDataset.get_features_ptr: bulk data is device-resident;"
                " the flat host pointer contract does not apply. Use the"
                " loader's tensor-level path or .to_cpu() first."
            )
        return self._features.data_ptr().as_imm()

    def get_labels_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.label_dtype], ImmutAnyOrigin]:
        if self.device().is_gpu():
            panic(
                "TensorDataset.get_labels_ptr: bulk data is device-resident;"
                " the flat host pointer contract does not apply. Use the"
                " loader's tensor-level path or .to_cpu() first."
            )
        return self._labels.data_ptr().as_imm()

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
    ) -> Tuple[Tensor[Self.sample_dtype], Tensor[Self.label_dtype]]:
        """Get single sample - preserves original shape."""
        if idx < 0 or idx >= self._size:
            panic("TensorDataset: index out of bounds")

        # Create tensors with proper shape, on the dataset's device
        var sample_feature = Tensor[Self.sample_dtype].zeros(
            self._feature_shape, device=self._features.device()
        )
        var sample_label = Tensor[Self.label_dtype].zeros(
            self._label_shape, device=self._features.device()
        )

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
        unsafe_memcpy(
            dest=sample_feature_ptr,
            src=dataset_features_ptr.unsafe_offset(src_feature_offset),
            count=self._features_per_sample,
        )

        # Copy label data
        var src_label_offset = idx * self._labels_per_sample
        unsafe_memcpy(
            dest=sample_label_ptr,
            src=dataset_labels_ptr.unsafe_offset(src_label_offset),
            count=self._labels_per_sample,
        )

        return (sample_feature^, sample_label^)

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self.sample_dtype], Tensor[Self.label_dtype]]:
        if idx:
            return self.__getitem__(idx.value())
        else:
            return self.__getitem__(Int(random_si64(0, Int64(self._size - 1))))

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
struct DataLoader[sample_dtype: DType, label_dtype: DType](
    Writable & Sized & ImplicitlyCopyable & Iterator
):
    """Dtype-generic batched data loader over in-memory Tensors.

    Args:
        features: `(N, *feat)` contiguous tensor of sample features.
        labels: `(N, *lab)` tensor of sample labels (any int/float dtype).
        batch_size: Number of samples per batch.
        shuffle: If True, yields randomly permuted batches.
        drop_last: If True, drops the final partial batch.

    DataLoader — dtype-generic, tensor-native, std-Iterator-conforming
    The concrete tensor-native batching engine behind the Python binding
    `tenmo.DataLoader`. Owns its data as `Tensor[sample_dtype]` /
    `Tensor[label_dtype]` and produces `Batch[sample_dtype, label_dtype]` from
    tensor-native ops:
      * sequential (eval):  zero-copy view slices of the source tensors
      * shuffled (train):   row-gather into a persistent preallocated buffer
    `DataLoader` conforms to Mojo's std `Iterator` protocol exactly like the
    trait-generic `NativeLoader` above: `__iter__` / `__has_next__` /
    `__next__` (raising `StopIteration` at epoch end), `bounds()`,
    `reset()`, `__len__`. Use it directly in a `for batch in loader:` loop;
    exhaustion raises `StopIteration`.
    Data movement budget:
      * construction: at most one owned copy (only when the source is a tracked
        or non-contiguous view); the numpy wrap itself is one copy
      * per epoch:    zero bytes sequential; batch_bytes shuffled
      * per batch:    zero allocations (persistent buffers reused)
    The shuffle permutation is re-permuted per epoch via `std.random.shuffle`
    (the same source as the trait-generic `NativeLoader`), so both engines
    share one shuffling behavior. Runs are not reproducible, so no seed
    parameter is offered.
    Contract: a shuffled batch aliases the loader's reused buffer and is valid
    ONLY until the next `__next__()` call. Sequential batches view the
    immutable source tensors and stay valid. Batches are read-only training
    inputs — never mutate them. `__next__()` yields a reference: copy-init it
    (`var b = loader.__next__()`) if you need to hold a batch past the next
    call.
    """

    var features: Tensor[Self.sample_dtype]
    var labels: Tensor[Self.label_dtype]
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var _num_samples: Int
    var _indices: List[Int]
    var _current_idx: Int
    var _features_per_sample: Int
    var _labels_per_sample: Int
    var _batch: Batch[Self.sample_dtype, Self.label_dtype]
    var _last_batch: Optional[
        Batch[Self.sample_dtype, Self.label_dtype]
    ]
    var _last_batch_size: Int
    var _buffers_owned: Bool
    var _num_batches: Int

    def __init__(out self, *, copy: Self):
        self.features = copy.features
        self.labels = copy.labels
        self.batch_size = copy.batch_size
        self.shuffle = copy.shuffle
        self.drop_last = copy.drop_last
        self._num_samples = copy._num_samples
        self._indices = copy._indices.copy()
        self._current_idx = copy._current_idx
        self._features_per_sample = copy._features_per_sample
        self._labels_per_sample = copy._labels_per_sample
        self._batch = copy._batch
        self._last_batch = copy._last_batch
        self._last_batch_size = copy._last_batch_size
        self._buffers_owned = copy._buffers_owned
        self._num_batches = copy._num_batches

    def __init__(
        out self,
        features: Tensor[Self.sample_dtype],
        labels: Tensor[Self.label_dtype],
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
    ):
        if batch_size < 1:
            panic(
                "DataLoader: batch_size must be >= 1, got "
                + String(batch_size)
            )

        # Normalize to an owned, contiguous, non-tracking source. Batching
        # relies on flat row offsets; views/autograd sources pay one copy here.
        var src_f = features
        if src_f.requires_grad or not src_f.is_contiguous():
            src_f = src_f.contiguous(requires_grad=False)
        var src_l = labels
        if src_l.requires_grad or not src_l.is_contiguous():
            src_l = src_l.contiguous(requires_grad=False)

        self.features = src_f^
        self.labels = src_l^
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._current_idx = 0

        var fshape = self.features.shape()
        var lshape = self.labels.shape()
        var n_f = fshape.dims[0]
        var n_l = lshape.dims[0]
        if n_f != n_l:
            panic(
                "DataLoader: features and labels must have the same"
                " number of samples, got "
                + String(n_f)
                + " vs "
                + String(n_l)
            )
        self._num_samples = n_f
        self._features_per_sample = fshape.product() // n_f
        self._labels_per_sample = lshape.product() // n_l

        var n = self._num_samples
        self._num_batches = (
            n // batch_size if drop_last else (n + batch_size - 1) // batch_size
        )

        # Identity index list, re-shuffled per epoch (mirrors NativeLoader).
        self._indices = List[Int](capacity=n)
        for i in range(n):
            self._indices.append(i)

        # Shuffle immediately so a fresh loader is already shuffled (the
        # Python facade iterates without calling __iter__/reset when the
        # loader is not exhausted).
        if shuffle:
            reshuffle(self._indices)

        # Preallocate full-size + remainder buffers for the shuffled path. Inlined
        # here (definite-init cannot observe helper writes before all fields
        # are assigned); set_shuffle(True) re-runs the same construction via
        # _make_buffers.
        var fshape0 = self.features.shape()
        var lshape0 = self.labels.shape()
        var bfeat = List[Int](capacity=fshape0.rank())
        bfeat.append(self.batch_size)
        for i in range(1, fshape0.rank()):
            bfeat.append(fshape0.dims[i])
        var blabel = List[Int](capacity=lshape0.rank())
        blabel.append(self.batch_size)
        for i in range(1, lshape0.rank()):
            blabel.append(lshape0.dims[i])
        self._batch = Batch[Self.sample_dtype, Self.label_dtype](
            Tensor[Self.sample_dtype].zeros(Shape(bfeat)),
            Tensor[Self.label_dtype].zeros(Shape(blabel)),
        )
        self._last_batch_size = 0
        self._last_batch = None
        if not self.drop_last:
            var remainder = self._num_samples % self.batch_size
            if remainder != 0:
                self._last_batch_size = remainder
                var lfeat = List[Int](capacity=fshape0.rank())
                lfeat.append(remainder)
                for i in range(1, fshape0.rank()):
                    lfeat.append(fshape0.dims[i])
                var llabel = List[Int](capacity=lshape0.rank())
                llabel.append(remainder)
                for i in range(1, lshape0.rank()):
                    llabel.append(lshape0.dims[i])
                self._last_batch = Batch[
                    Self.sample_dtype, Self.label_dtype
                ](
                    Tensor[Self.sample_dtype].zeros(Shape(lfeat)),
                    Tensor[Self.label_dtype].zeros(Shape(llabel)),
                )
        self._buffers_owned = True

    def _make_buffers(
        self,
    ) -> Tuple[
        Batch[Self.sample_dtype, Self.label_dtype],
        Optional[Batch[Self.sample_dtype, Self.label_dtype]],
        Int,
    ]:
        """Build persistent full-size and remainder batch buffers.

        Returns:
            The (full-batch, optional remainder-batch, remainder-size) triple
            shaped from the current source tensors. Sequential (eval) epochs
            replace these buffers with zero-copy views, so `set_shuffle(True)`
            re-builds them via this helper.
        """
        var fshape = self.features.shape()
        var lshape = self.labels.shape()

        var bfeat = List[Int](capacity=fshape.rank())
        bfeat.append(self.batch_size)
        for i in range(1, fshape.rank()):
            bfeat.append(fshape.dims[i])
        var blabel = List[Int](capacity=lshape.rank())
        blabel.append(self.batch_size)
        for i in range(1, lshape.rank()):
            blabel.append(lshape.dims[i])
        var full_batch = Batch[
            Self.sample_dtype, Self.label_dtype
        ](
            Tensor[Self.sample_dtype].zeros(Shape(bfeat)),
            Tensor[Self.label_dtype].zeros(Shape(blabel)),
        )

        var last_batch: Optional[
            Batch[Self.sample_dtype, Self.label_dtype]
        ] = None
        var last_batch_size = 0
        # A remainder batch buffer exists only when a partial last batch can
        # occur (drop_last=False and num_samples % batch_size != 0).
        if not self.drop_last:
            var remainder = self._num_samples % self.batch_size
            if remainder != 0:
                last_batch_size = remainder
                var lfeat = List[Int](capacity=fshape.rank())
                lfeat.append(remainder)
                for i in range(1, fshape.rank()):
                    lfeat.append(fshape.dims[i])
                var llabel = List[Int](capacity=lshape.rank())
                llabel.append(remainder)
                for i in range(1, lshape.rank()):
                    llabel.append(lshape.dims[i])
                last_batch = Batch[
                    Self.sample_dtype, Self.label_dtype
                ](
                    Tensor[Self.sample_dtype].zeros(Shape(lfeat)),
                    Tensor[Self.label_dtype].zeros(Shape(llabel)),
                )

        return (full_batch, last_batch, last_batch_size)

    def __len__(self) -> Int:
        return self._num_batches

    def num_samples(self) -> Int:
        return self._num_samples

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "DataLoader(num_samples="
            + String(self.num_samples())
            + ", batch_size="
            + String(self.batch_size)
            + ", shuffle="
            + String(self.shuffle)
            + ", drop_last="
            + String(self.drop_last)
            + ")",
        )

    @no_inline
    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write(
            "DataLoader[feature="
            + String(Self.sample_dtype)
            + ", label="
            + String(Self.label_dtype)
            + "](num_samples="
            + String(self.num_samples())
            + ", batch_size="
            + String(self.batch_size)
            + ", shuffle="
            + String(self.shuffle)
            + ")",
        )

    def __iter__(mut self) -> ref[self] Self.IteratorType[origin_of(self)]:
        self._current_idx = 0
        if self.shuffle:
            reshuffle(self._indices)
        return self

    comptime Element = Batch[Self.sample_dtype, Self.label_dtype]
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
        """Get the next batch; raises `StopIteration` when the epoch ends."""
        if not self.__has_next__():
            raise StopIteration()
        var start = self._current_idx
        var end = min(start + self.batch_size, self._num_samples)
        var bs = end - start
        var is_last = bs < self.batch_size
        self._current_idx = end

        if not self.shuffle:
            # Sequential (eval): zero-copy view slices of the source tensors.
            var fx = self.features.slice(start=start, end=end, step=1, axis=0)
            var ly = self.labels.slice(start=start, end=end, step=1, axis=0)
            self._buffers_owned = False
            if is_last and self._last_batch:
                ref last_batch = self._last_batch.value()
                last_batch.features = fx
                last_batch.labels = ly
                return last_batch
            else:
                self._batch.features = fx
                self._batch.labels = ly
                return self._batch
        else:
            # Shuffled (train): row-gather into the persistent buffer.
            if is_last and self._last_batch:
                ref current_batch = self._last_batch.value()
                self._fill_batch(current_batch, start, bs)
                return current_batch
            else:
                ref current_batch = self._batch
                self._fill_batch(current_batch, start, bs)
                return current_batch

    def _fill_batch(
        self,
        batch: Batch[Self.sample_dtype, Self.label_dtype],
        start: Int,
        bs: Int,
    ):
        """Gather rows `_indices[start:start+bs]` into a batch buffer."""
        var batch_features_ptr = (
            batch.features.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUnsafeAnyOrigin]()
        )
        var batch_labels_ptr = (
            batch.labels.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUnsafeAnyOrigin]()
        )
        var src_feat_ptr = self.features.data_ptr().as_imm()
        var src_label_ptr = self.labels.data_ptr().as_imm()
        for i in range(bs):
            var idx = self._indices[start + i]
            unsafe_memcpy(
                dest=batch_features_ptr.unsafe_offset(
                    i * self._features_per_sample
                ),
                src=src_feat_ptr.unsafe_offset(
                    idx * self._features_per_sample
                ),
                count=self._features_per_sample,
            )
            unsafe_memcpy(
                dest=batch_labels_ptr.unsafe_offset(
                    i * self._labels_per_sample
                ),
                src=src_label_ptr.unsafe_offset(
                    idx * self._labels_per_sample
                ),
                count=self._labels_per_sample,
            )

    def __has_next__(self) -> Bool:
        if self.drop_last:
            return (self._current_idx + self.batch_size) <= self._num_samples
        else:
            return self._current_idx < self._num_samples

    def reset(mut self):
        """Start a new epoch: re-shuffle the permutation if shuffling."""
        self._current_idx = 0
        if self.shuffle:
            reshuffle(self._indices)

    def set_shuffle(mut self, shuffle: Bool):
        """Switch between train (shuffle) and eval (sequential) modes.

        Enabling shuffle re-allocates the persistent buffers if sequential
        views replaced them; the index list is re-shuffled on the next
        __iter__()/reset() call."""
        if shuffle and not self._buffers_owned:
            var buffers = self._make_buffers()
            var (batch0, last_batch0, last_batch_size0) = buffers
            self._batch = batch0
            self._last_batch = last_batch0
            self._last_batch_size = last_batch_size0
            self._buffers_owned = True
        self.shuffle = shuffle


def _window_count(n: Int, seq_length: Int, stride: Int) -> Int:
    """Number of full `seq_length` windows in a stream of `n` IDs.

    Stride-1 gives `n - seq_length` (legacy `RandomSlidingWindowDataset`);
    general stride gives the ceiling
    `(n - seq_length + stride - 1) // stride` (legacy `LLMDataset`).
    Streams holding no full window yield 0, never a panic.

    Token-stream windowing core.
    Tokenizer-free by design: these types consume token-ID streams (a flat 1-D
    `Tensor` or `List[Scalar]`), never text. Tokenization (mbpe
    `BPETokenizer().encode`) happens at the call site; IDs cross via
    `Tensor.from_list[DType.int64]`. This keeps `dataloader.mojo` free of
    any edge into `tenmo.nlp` (import-DAG cycle).
    The legacy `LLMDataset` / `RandomSlidingWindowDataset`
    (`tenmo/nlp/dataset.mojo`) pre-materialize every window into flat lists
    (O(N·T) memory) to satisfy the `Dataset` trait's pointer contract. The
    core below gathers windows lazily from the owned flat stream (O(N) +
    B·T batch buffers) and unifies both legacy numbering schemes:
    stride=1 matches `RandomSlidingWindowDataset`; general stride matches
    `LLMDataset`'s ceiling formula.
    """
    if n <= seq_length:
        return 0
    return (n - seq_length + stride - 1) // stride


struct SlidingWindowDataset[dtype: DType = DType.int64](Sized):
    """Lazy sliding-window dataset over a flat 1-D token-ID stream.

    Owns one contiguous, non-tracking `(N,)` tensor; windows are gathered
    on demand (by `sample` / `WindowLoader`), never pre-materialized.
    Deliberately NOT a `Dataset`-trait conformer — the trait's flat-pointer
    contract is what forces the legacy O(N·T) layout.
    """

    var tokens: Tensor[Self.dtype]
    var seq_length: Int
    var stride: Int
    var _num_samples: Int

    def __init__(
        out self,
        tokens: Tensor[Self.dtype],
        seq_length: Int,
        stride: Int = 1,
    ):
        if seq_length < 1:
            panic(
                "SlidingWindowDataset: seq_length must be >= 1, got "
                + String(seq_length)
            )
        if stride < 1:
            panic(
                "SlidingWindowDataset: stride must be >= 1, got "
                + String(stride)
            )
        if tokens.shape().rank() != 1:
            panic(
                "SlidingWindowDataset: tokens must be a flat 1-D stream, got"
                " rank "
                + String(tokens.shape().rank())
            )
        # Same ownership normalization as DataLoader: batching relies on flat
        # offsets, so views/autograd sources pay one copy here.
        var src = tokens
        if src.requires_grad or not src.is_contiguous():
            src = src.contiguous(requires_grad=False)
        self.tokens = src^
        self.seq_length = seq_length
        self.stride = stride
        self._num_samples = _window_count(
            len(self.tokens), seq_length, stride
        )

    def __init__(
        out self,
        values: List[Scalar[Self.dtype]],
        seq_length: Int,
        stride: Int = 1,
    ):
        if seq_length < 1:
            panic(
                "SlidingWindowDataset: seq_length must be >= 1, got "
                + String(seq_length)
            )
        if stride < 1:
            panic(
                "SlidingWindowDataset: stride must be >= 1, got "
                + String(stride)
            )
        # A 0-length 1-D tensor is unrepresentable (Shape dims must be >= 1),
        # so an empty stream is a caller bug — fail loudly instead of building
        # a corrupt 1-sample scalar.
        if len(values) == 0:
            panic(
                "SlidingWindowDataset: empty token stream (no IDs to window)"
            )
        # from_list is fresh owned, contiguous, non-tracking storage.
        var stream = Tensor[Self.dtype].from_list[Self.dtype](values)
        self.tokens = stream^
        self.seq_length = seq_length
        self.stride = stride
        self._num_samples = _window_count(
            len(self.tokens), seq_length, stride
        )

    def __len__(self) -> Int:
        return self._num_samples

    def num_samples(self) -> Int:
        return self._num_samples

    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        random_offsets: Bool = False,
    ) -> WindowLoader[Self.dtype]:
        """Build a `WindowLoader` over this dataset's stream.

        The loader aliases (never mutates) the stream storage; the dataset
        stays usable afterwards.
        """
        return WindowLoader(
            self.tokens,
            self.seq_length,
            self.stride,
            batch_size,
            shuffle,
            drop_last,
            random_offsets,
        )

    def sample(
        ref self,
        idx: Optional[Int] = None,
    ) raises -> Tuple[Tensor[Self.dtype], Tensor[Self.dtype]]:
        """One window pair: input `stream[off:off+T]`, target shifted by one.

        A random window is drawn when `idx` is None (one-shot parity with
        the legacy datasets).
        """
        if self._num_samples == 0:
            panic(
                "SlidingWindowDataset.sample: no windows (stream shorter"
                " than seq_length)"
            )
        var index = idx.or_else(
            Int(random_si64(0, Int64(self._num_samples - 1)))
        )
        if index < 0 or index >= self._num_samples:
            panic(
                "SlidingWindowDataset.sample: index out of range, got "
                + String(index)
            )
        var off = index * self.stride
        var features = Tensor[Self.dtype].zeros(Shape(self.seq_length))
        var labels = Tensor[Self.dtype].zeros(Shape(self.seq_length))
        var src = self.tokens.data_ptr().as_imm()
        var dst_feat = (
            features.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUnsafeAnyOrigin]()
        )
        var dst_label = (
            labels.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUnsafeAnyOrigin]()
        )
        unsafe_memcpy(
            dest=dst_feat,
            src=src.unsafe_offset(off),
            count=self.seq_length,
        )
        unsafe_memcpy(
            dest=dst_label,
            src=src.unsafe_offset(off + 1),
            count=self.seq_length,
        )
        return features^, labels^


struct WindowLoader[dtype: DType = DType.int64](
    Writable & Sized & ImplicitlyCopyable & Iterator
):
    """Batched iterator over a flat 1-D token-ID stream.

    Two modes in one struct (mirrors `DataLoader`'s protocol field for
    field):

    - enumerate (`random_offsets=False`): window starts `i * stride`
      (default 1), per-epoch `reshuffle`'d when `shuffle` — crisp epoch,
      the rescued legacy semantics.
    - random-offset (`random_offsets=True`): each batch draws fresh starts
      in `[0, N - seq_length - 1]`; nominal epoch length is
      `ceil((N - seq_length) / batch_size)` and stride is ignored — the
      pretraining path (one epoch is not a crisp concept).

    Yields `Batch[dtype, dtype]` pairs of `(B, seq_length)` with the
    shift-by-one relationship (input `[0..T-1]`, target `[1..T]`).
    Both paths gather with `unsafe_memcpy` into persistent buffers —
    `DataLoader`'s zero-copy slice views do not apply (windows overlap /
    stride > 1) — so buffers are always owned and `set_shuffle` never
    rebuilds them.
    """

    var tokens: Tensor[Self.dtype]
    var seq_length: Int
    var stride: Int
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var random_offsets: Bool
    var _num_samples: Int
    var _num_valid_offsets: Int
    var _offsets: List[Int]
    var _current_idx: Int
    var _batch: Batch[Self.dtype, Self.dtype]
    var _last_batch: Optional[Batch[Self.dtype, Self.dtype]]
    var _last_batch_size: Int
    var _buffers_owned: Bool
    var _num_batches: Int

    def __init__(out self, *, copy: Self):
        self.tokens = copy.tokens
        self.seq_length = copy.seq_length
        self.stride = copy.stride
        self.batch_size = copy.batch_size
        self.shuffle = copy.shuffle
        self.drop_last = copy.drop_last
        self.random_offsets = copy.random_offsets
        self._num_samples = copy._num_samples
        self._num_valid_offsets = copy._num_valid_offsets
        self._offsets = copy._offsets.copy()
        self._current_idx = copy._current_idx
        self._batch = copy._batch
        self._last_batch = copy._last_batch
        self._last_batch_size = copy._last_batch_size
        self._buffers_owned = copy._buffers_owned
        self._num_batches = copy._num_batches

    def __init__(
        out self,
        tokens: Tensor[Self.dtype],
        seq_length: Int,
        stride: Int,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        random_offsets: Bool = False,
    ):
        if seq_length < 1:
            panic(
                "WindowLoader: seq_length must be >= 1, got "
                + String(seq_length)
            )
        if stride < 1:
            panic(
                "WindowLoader: stride must be >= 1, got " + String(stride)
            )
        if batch_size < 1:
            panic(
                "WindowLoader: batch_size must be >= 1, got "
                + String(batch_size)
            )
        if tokens.shape().rank() != 1:
            panic(
                "WindowLoader: tokens must be a flat 1-D stream, got rank "
                + String(tokens.shape().rank())
            )

        var src = tokens
        if src.requires_grad or not src.is_contiguous():
            src = src.contiguous(requires_grad=False)

        self.tokens = src^
        self.seq_length = seq_length
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_offsets = random_offsets
        self._current_idx = 0

        var n = len(self.tokens)
        self._num_samples = _window_count(n, seq_length, stride)
        # Stride-independent valid start pool for random-offset mode.
        self._num_valid_offsets = n - seq_length

        self._offsets = List[Int](capacity=self._num_samples)
        for i in range(self._num_samples):
            self._offsets.append(i * stride)
        # Shuffle immediately so a fresh loader is already shuffled (parity
        # with DataLoader).
        if shuffle:
            reshuffle(self._offsets)

        var total = (
            self._num_valid_offsets if random_offsets else self._num_samples
        )
        self._num_batches = 0
        if total > 0:
            self._num_batches = (
                total // batch_size
                if drop_last
                else (total + batch_size - 1) // batch_size
            )

        # Persistent gather-owned full-size + remainder buffers, shaped
        # (B, seq_length) for both rows. Inlined here: definite-init cannot
        # observe helper writes before all fields are assigned (same reason
        # DataLoader inlines its buffer build).
        var bfeat = List[Int](capacity=2)
        bfeat.append(batch_size)
        bfeat.append(seq_length)
        var blabel = List[Int](capacity=2)
        blabel.append(batch_size)
        blabel.append(seq_length)
        self._batch = Batch[Self.dtype, Self.dtype](
            Tensor[Self.dtype].zeros(Shape(bfeat)),
            Tensor[Self.dtype].zeros(Shape(blabel)),
        )
        self._last_batch_size = 0
        self._last_batch = None
        if not self.drop_last:
            var remainder = total % self.batch_size if total > 0 else 0
            if remainder != 0:
                self._last_batch_size = remainder
                var lfeat = List[Int](capacity=2)
                lfeat.append(remainder)
                lfeat.append(seq_length)
                var llabel = List[Int](capacity=2)
                llabel.append(remainder)
                llabel.append(seq_length)
                self._last_batch = Batch[Self.dtype, Self.dtype](
                    Tensor[Self.dtype].zeros(Shape(lfeat)),
                    Tensor[Self.dtype].zeros(Shape(llabel)),
                )
        self._buffers_owned = True

    def _epoch_total(self) -> Int:
        """Units per epoch: valid starts (random mode) or windows (enumerate)."""
        return (
            self._num_valid_offsets
            if self.random_offsets
            else self._num_samples
        )

    def __len__(self) -> Int:
        return self._num_batches

    def num_samples(self) -> Int:
        return self._num_samples

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "WindowLoader(num_samples="
            + String(self.num_samples())
            + ", batch_size="
            + String(self.batch_size)
            + ", shuffle="
            + String(self.shuffle)
            + ", drop_last="
            + String(self.drop_last)
            + ", random_offsets="
            + String(self.random_offsets)
            + ")",
        )

    @no_inline
    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write(
            "WindowLoader[dtype="
            + String(Self.dtype)
            + "](num_samples="
            + String(self.num_samples())
            + ", batch_size="
            + String(self.batch_size)
            + ", shuffle="
            + String(self.shuffle)
            + ", random_offsets="
            + String(self.random_offsets)
            + ")",
        )

    def __iter__(mut self) -> ref[self] Self.IteratorType[origin_of(self)]:
        self._current_idx = 0
        if self.shuffle:
            reshuffle(self._offsets)
        return self

    comptime Element = Batch[Self.dtype, Self.dtype]
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
        """Get the next window batch; raises `StopIteration` at epoch end."""
        if not self.__has_next__():
            raise StopIteration()
        var total = self._epoch_total()
        var start = self._current_idx
        var end = min(start + self.batch_size, total)
        var bs = end - start
        var is_last = bs < self.batch_size
        self._current_idx = end

        if is_last and self._last_batch:
            ref current_batch = self._last_batch.value()
            self._fill_batch(current_batch, start, bs)
            return current_batch
        else:
            ref current_batch = self._batch
            self._fill_batch(current_batch, start, bs)
            return current_batch

    def _fill_batch(
        self,
        batch: Batch[Self.dtype, Self.dtype],
        start: Int,
        bs: Int,
    ):
        """Gather `bs` windows into a batch buffer.

        Enumerate mode reads starts from the (possibly reshuffled) offset
        list; random-offset mode draws fresh starts per batch (`shuffle`
        is irrelevant there).
        """
        if self.random_offsets:
            for i in range(bs):
                var off = Int(
                    random_si64(0, Int64(self._num_valid_offsets - 1))
                )
                self._gather_row(batch, i, off)
        else:
            for i in range(bs):
                self._gather_row(batch, i, self._offsets[start + i])

    def _gather_row(
        self,
        batch: Batch[Self.dtype, Self.dtype],
        row: Int,
        off: Int,
    ):
        """Gather one window: input `stream[off:off+T]`, target shifted by one."""
        var batch_features_ptr = (
            batch.features.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUnsafeAnyOrigin]()
        )
        var batch_labels_ptr = (
            batch.labels.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUnsafeAnyOrigin]()
        )
        var src_ptr = self.tokens.data_ptr().as_imm()
        unsafe_memcpy(
            dest=batch_features_ptr.unsafe_offset(row * self.seq_length),
            src=src_ptr.unsafe_offset(off),
            count=self.seq_length,
        )
        unsafe_memcpy(
            dest=batch_labels_ptr.unsafe_offset(row * self.seq_length),
            src=src_ptr.unsafe_offset(off + 1),
            count=self.seq_length,
        )

    def __has_next__(self) -> Bool:
        var total = self._epoch_total()
        if self.drop_last:
            return (self._current_idx + self.batch_size) <= total
        else:
            return self._current_idx < total

    def reset(mut self):
        """Start a new epoch: re-shuffle the offset permutation if shuffling."""
        self._current_idx = 0
        if self.shuffle:
            reshuffle(self._offsets)

    def set_shuffle(mut self, shuffle: Bool):
        """Switch between train (shuffle) and eval (sequential) modes.

        Unlike `DataLoader.set_shuffle`, no buffer rebuild is ever needed:
        both paths gather into owned buffers (no zero-copy views to replace).
        The offset list is re-shuffled on the next __iter__()/reset() call.
        """
        self.shuffle = shuffle
