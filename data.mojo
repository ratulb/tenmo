from tenmo import Tensor
from common_utils import panic
from random import random_ui64

trait Dataset(Sized & Copyable & Movable):
    alias datatype: DType
    """Interface for datasets."""

    fn __len__(self) -> Int:
        """Total number of samples."""
        ...

    fn __getitem__(
        self, idx: Int
    ) -> Tuple[Tensor[Self.datatype], Tensor[Self.datatype]]:
        """Get single sample (features, label)."""
        ...

    fn features(self) -> Tensor[Self.datatype]:
        ...

    fn labels(self) -> Tensor[Self.datatype]:
        ...


@fieldwise_init
struct TensorDataset[dtype: DType](
    ImplicitlyCopyable & Movable & Sized & Dataset
):
    """Dataset from tensors (like PyTorch TensorDataset)."""

    alias datatype = Self.dtype
    var _features: Tensor[dtype]
    var _labels: Tensor[dtype]
    var _size: Int

    fn __init__(out self, features: Tensor[dtype], labels: Tensor[dtype]):
        """Create dataset from feature and label tensors.

        Args:
            features: Shape (N, feature_dim).
            labels: Shape (N, label_dim) or (N,).
        """
        self._features = features
        self._labels = labels
        self._size = features.shape()[0]

        # Validate same number of samples
        if labels.shape()[0] != self._size:
            panic(
                "TensorDataset: features and labels must have same number of"
                " samples"
            )

    fn features(self) -> Tensor[Self.datatype]:
        return self._features

    fn labels(self) -> Tensor[Self.datatype]:
        return self._labels

    fn __len__(self) -> Int:
        return self._size

    fn __getitem__(self, idx: Int) -> Tuple[Tensor[dtype], Tensor[dtype]]:
        """Get single sample."""
        if idx < 0 or idx >= self._size:
            panic("TensorDataset: index out of bounds")

        # Extract single row for features
        var feature_shape = self._features.shape()
        var feature_dim = feature_shape[1] if feature_shape.rank() > 1 else 1
        var sample_feature = Tensor[dtype].zeros(feature_dim)

        for j in range(feature_dim):
            sample_feature[j] = self._features[idx, j]

        # Extract single label
        var label_shape = self._labels.shape()
        var label_dim = label_shape[1] if label_shape.rank() > 1 else 1
        var sample_label = Tensor[dtype].zeros(label_dim)

        if label_shape.rank() == 1:
            sample_label[0] = self._labels[idx]
        else:
            for j in range(label_dim):
                sample_label[j] = self._labels[idx, j]

        return (sample_feature^, sample_label^)


@fieldwise_init
struct Batch[dtype: DType](ImplicitlyCopyable & Movable):
    """Container for a batch of data."""

    var features: Tensor[dtype]
    var labels: Tensor[dtype]
    var batch_size: Int

    fn __init__(out self, features: Tensor[dtype], labels: Tensor[dtype]):
        self.features = features
        self.labels = labels
        self.batch_size = features.shape()[0]


@fieldwise_init
struct DataLoader[DatasetSource: Dataset, //,  dtype: DType=DType.float32](
    Copyable & Movable & Sized
):
    """Batched data loading with optional shuffling."""

    # var dataset: TensorDataset[dtype]
    var dataset: DatasetSource
    var batch_size: Int
    var shuffle_data: Bool  # Renamed to avoid conflict with shuffle function
    var drop_last: Bool
    var _current_idx: Int
    var _indices: List[Int]
    var _num_batches: Int

    fn __init__(
        out self,
        # dataset: TensorDataset[dtype],
        dataset: DatasetSource,
        batch_size: Int,
        reshuffle: Bool = False,
        drop_last: Bool = False,
    ):
        """Create DataLoader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            reshuffle: Whether to shuffle data each epoch.
            drop_last: Drop last incomplete batch if True.
        """
        self.dataset = dataset.copy()
        self.batch_size = batch_size
        self.shuffle_data = reshuffle
        self.drop_last = drop_last
        self._current_idx = 0

        var total_samples = len(self.dataset)

        # Pre-allocate indices list
        self._indices = List[Int](capacity=UInt(total_samples))

        # Calculate number of batches
        if drop_last:
            self._num_batches = total_samples // batch_size
        else:
            self._num_batches = (total_samples + batch_size - 1) // batch_size

        # Initialize indices
        self._reset_indices()

    fn _reset_indices(mut self):
        """Reset and optionally shuffle indices."""
        self._indices.clear()
        var n = len(self.dataset)

        # Create sequential indices
        for i in range(n):
            self._indices.append(i)

        # Use Mojo's built-in shuffle #TODO
        if self.shuffle_data:
            # shuffle(self._indices) #Strangely this is giving a compilation error!
            self._shuffle_indices()

    fn _shuffle_indices_orig(mut self):
        var n = len(self._indices)
        for i in range(n - 1, 0, -1):
            var j = random_ui64(0, 1) * (i + 1)
            if j > i:
                j = i

            var temp = self._indices[i]
            self._indices[i] = self._indices[j]
            self._indices[j] = temp

    fn _shuffle_indices_new(mut self):
        """Fisher-Yates shuffle algorithm."""
        var n = len(self._indices)
        for i in range(n - 1, 0, -1):
            # random_ui64(0, i+1) gives us a value in [0, i+1)
            var j = Int(random_ui64(0, UInt64(i + 1)))

            var temp = self._indices[i]
            self._indices[i] = self._indices[j]
            self._indices[j] = temp


    fn _shuffle_indices(mut self):
        """Fisher-Yates shuffle algorithm - FIXED VERSION.

        Uses modulo to ensure random index is always in valid range [0, i].
        This prevents index out of bounds errors from random_ui64.
        """
        var n = len(self._indices)
        for i in range(n - 1, 0, -1):
            # Generate random index in range [0, i] inclusive
            # Use modulo to clamp the random value to valid range
            var rand_val = random_ui64(0, UInt64(i + 1))
            var j = Int(rand_val % UInt64(i + 1))

            # Swap indices[i] and indices[j]
            var temp = self._indices[i]
            self._indices[i] = self._indices[j]
            self._indices[j] = temp


    fn __iter__(mut self) -> Self:
        """Start iteration."""
        self._current_idx = 0
        self._reset_indices()
        return self.copy()

    fn __next__(mut self) -> Batch[dtype]:
        var start_idx = self._current_idx
        var end_idx = min(start_idx + self.batch_size, len(self._indices))
        var actual_batch_size = end_idx - start_idx

        # Get feature and label dimensions
        var feature_dim = self.dataset.features().shape()[1]
        var label_shape = self.dataset.labels().shape()
        var label_dim = label_shape[1] if label_shape.rank() > 1 else 1

        # Create batch tensors
        var batch_features = Tensor[dtype].zeros(actual_batch_size, feature_dim)
        var batch_labels = Tensor[dtype].zeros(actual_batch_size, label_dim)

        # Fill batch
        for i in range(actual_batch_size):
            var sample_idx = self._indices[start_idx + i]

            # Copy features
            for j in range(feature_dim):
                batch_features[i, j] = (
                    self.dataset.features()[sample_idx, j]
                ).cast[dtype]()

            # Copy labels
            if label_shape.rank() == 1:
                batch_labels[i, 0] = (self.dataset.labels()[sample_idx]).cast[
                    dtype
                ]()
            else:
                for j in range(label_dim):
                    batch_labels[i, j] = (
                        self.dataset.labels()[sample_idx, j]
                    ).cast[dtype]()

        self._current_idx = end_idx
        return Batch[dtype](batch_features^, batch_labels^)

    fn __has_next__(self) -> Bool:
        """Check if there are more batches."""
        if self.drop_last:
            return (self._current_idx + self.batch_size) <= len(self._indices)
        else:
            return self._current_idx < len(self._indices)

    fn __len__(self) -> Int:
        """Number of batches."""
        return self._num_batches

fn main() raises:
    pass
