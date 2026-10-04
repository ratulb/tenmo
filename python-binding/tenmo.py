"""tenmo — Python bindings for the Tenmo ML framework.

Usage:
    from tenmo import tensor, Sequential, Linear, ReLU, CrossEntropyLoss, SGD

    model = Sequential([Linear(784, 128).into(), ReLU().into(), Linear(128, 10).into()])
    opt = SGD(model, lr=0.01)
    loss = CrossEntropyLoss()
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

import _tenmo

Shape = List[int]


# ── Module-level factory functions ──────────────────────────────────


def tensor(
    data: Union[List, np.ndarray],
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a tensor from a Python list or numpy array.

    For numpy arrays, the original dtype is preserved for all 11 numeric/bool
    types: float16, float32, float64, int8, int16, int32, int64, uint8, uint16,
    uint32, uint64, bool_.
    For Python lists, dtype defaults to float32; pass dtype= to specify.

    Examples:
        tenmo.tensor([1.0, 2.0, 3.0])
        tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        tenmo.tensor(np.array([1.0, 2.0]), requires_grad=True)
        tenmo.tensor([1, 2, 3], dtype="int64")
        tenmo.tensor(np.array([1,2,3], dtype=np.uint8))
    """
    kwargs = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.tensor(data, **kwargs))


def _shape2d(shape) -> Shape:
    """Accept an int or a tuple of ints for shape arguments."""
    if isinstance(shape, int):
        return (shape,)
    return shape


def zeros(
    shape: Shape,
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a tensor of zeros with the given shape.

    dtype defaults to float32; accepts any numpy dtype string
    (e.g. "float64", "int64", "bool").
    """
    kwargs = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.zeros(_shape2d(shape), **kwargs))


def ones(
    shape: Shape,
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a tensor of ones with the given shape (dtype as in zeros)."""
    kwargs = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.ones(_shape2d(shape), **kwargs))


def randn(
    shape: Shape,
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a tensor with random normal values (floating dtypes only)."""
    kwargs = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.randn(_shape2d(shape), **kwargs))


def arange(
    end: Optional[float] = None,
    *,
    start: Optional[float] = None,
    step: Optional[float] = None,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a range tensor (like numpy.arange).

    Examples:
        tenmo.arange(5)              # [0, 1, 2, 3, 4]
        tenmo.arange(2, start=0)     # [0, 1]
        tenmo.arange(10, start=0, step=2)  # [0, 2, 4, 6, 8]
        tenmo.arange(5, dtype="int64")     # int64 tensor [0..4]
    """
    if end is None:
        raise TypeError("arange() requires at least one argument")
    kwargs = {
        "start": start or 0.0,
        "step": step or 1.0,
        "requires_grad": requires_grad,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.arange(end, **kwargs))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication: a @ b."""
    return Tensor(_tenmo.matmul(a._raw, b._raw))


def full(
    shape: Shape,
    value: float,
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a tensor filled with a scalar value."""
    kwargs: dict[str, Any] = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.full(_shape2d(shape), value, **kwargs))


def rand(
    shape: Shape,
    *,
    low: float = 0.0,
    high: float = 1.0,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a tensor with uniform random values in [low, high)."""
    kwargs: dict[str, Any] = {
        "low": low,
        "high": high,
        "requires_grad": requires_grad,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.rand(_shape2d(shape), **kwargs))


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create a 1D tensor with linearly spaced values."""
    kwargs: dict[str, Any] = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.linspace(start, end, steps, **kwargs))


def eye(
    n: int,
    *,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """Create an n×n identity matrix."""
    kwargs: dict[str, Any] = {"requires_grad": requires_grad}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Tensor(_tenmo.eye(n, **kwargs))


def accuracy(
    pred: Tensor, target: Tensor, *, sync: bool = True
) -> float:
    """Classification accuracy (fraction of correct class predictions)."""
    return float(_tenmo.accuracy(pred._raw, target._raw, sync=sync))


def token_accuracy(
    pred: Tensor, target: Tensor, *, sync: bool = True
) -> float:
    """Token-level accuracy (fraction of correctly predicted positions)."""
    return float(_tenmo.token_accuracy(pred._raw, target._raw, sync=sync))


def sequence_accuracy(
    pred: Tensor, target: Tensor, *, sync: bool = True
) -> float:
    """Sequence accuracy (fraction of sequences where ALL positions match)."""
    return float(
        _tenmo.sequence_accuracy(pred._raw, target._raw, sync=sync)
    )


# ── Tensor ──────────────────────────────────────────────────────────

# Dtypes whose native Tensor type carries the autograd + math surface
# (Mojo handlers registered in tenmo_bind.mojo PyInit). Autograd entry
# points below raise a clean TypeError for other dtypes instead of an
# AttributeError off the raw capsule.
_AUTOGRAD_DTYPES = ("float32", "float64")


class Tensor:
    """A multi-dimensional array backed by Tenmo's Mojo engine.

    Supports autograd, shape inspection, math ops, and numpy interop.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw) -> None:
        object.__setattr__(self, "_raw", raw)

    # ── shape inspection ────────────────────────────────────────────

    @property
    def shape(self) -> tuple:
        """Shape of the tensor as a tuple."""
        return self._raw.shape()

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._raw.ndim()

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return self._raw.numels()

    @property
    def requires_grad(self) -> bool:
        """Whether this tensor tracks gradients."""
        return self._raw.requires_grad()

    @property
    def dtype(self):
        """The numpy dtype of this tensor (e.g. np.float32, np.int64)."""
        return self._raw.numpy_dtype()

    def numpy_dtype(self):
        """Return the numpy dtype object for this tensor."""
        return self._raw.numpy_dtype()

    @property
    def itemsize(self) -> int:
        """Byte width of a single element (from the tensor's dtype)."""
        return np.dtype(self.numpy_dtype()).itemsize

    @property
    def nbytes(self) -> int:
        """Total bytes: numel * itemsize."""
        return self.numel * self.itemsize

    def tolist(self) -> list:
        """Convert to a Python list (recursively nested)."""
        return self._raw.tolist()

    def item(self):
        """Extract the single scalar value.

        Returns a Python float for floating tensors, int for integer
        tensors, and bool for bool tensors (no float64 coercion)."""
        return self._raw.item()

    def numpy(self):
        """Convert to a numpy ndarray (requires numpy).

        Single-copy export through the Mojo binding (one memcpy out of
        the tensor buffer, then a no-copy reshape view).
        """
        if np is None:
            raise ImportError("numpy is required for Tensor.numpy()")
        return self._raw.numpy()

    # ── indexing / slicing ───────────────────────────────────────────

    def _normalize_index_key(self, key):
        """Expand an index key into parallel lane lists for the Mojo binding.

        Returns (kinds, starts, stops, steps) where each lane is one of:
            kind 0 = integer index (absolute, pre-checked in bounds)
            kind 1 = slice (normalized via slice.indices(), positive step)
            kind 2 = newaxis (None)
        Raises IndexError for out-of-bounds integers / too many indices,
        NotImplementedError for negative-step and boolean/fancy masks.
        """
        if not isinstance(key, tuple):
            key = (key,)
        lanes = list(key)
        n_ellipsis = sum(1 for k in lanes if isinstance(k, type(Ellipsis)))
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        shp = list(self.shape)
        ndim = len(shp)
        n_explicit = sum(
            1
            for k in lanes
            if isinstance(k, (int, slice)) and not isinstance(k, bool)
        )
        # None consumes no dims; int/slice consume one each; Ellipsis fills
        # whatever remains (clamped to >= 0).
        n_fill = max(0, ndim - n_explicit)
        expanded = []
        for k in lanes:
            if k is Ellipsis:
                expanded.extend([slice(None)] * n_fill)
            else:
                expanded.append(k)
        if n_explicit > ndim:
            raise IndexError(
                f"too many indices for tensor of dimension {ndim}"
            )
        # Pad trailing full slices so lanes match rank.
        dim_count = sum(
            1
            for k in expanded
            if isinstance(k, (int, slice)) and not isinstance(k, bool)
        )
        while dim_count < ndim:
            expanded.append(slice(None))
            dim_count += 1

        kinds, starts, stops, steps = [], [], [], []
        di = 0
        for k in expanded:
            if k is None:
                kinds.append(2)
                starts.append(0)
                stops.append(0)
                steps.append(0)
            elif isinstance(k, bool):
                raise NotImplementedError(
                    "boolean-mask indexing is not supported"
                )
            elif isinstance(k, int):
                dim = shp[di]
                di += 1
                v = k
                if v < 0:
                    v += dim
                if not 0 <= v < dim:
                    raise IndexError(
                        f"index {k} out of bounds for dimension {dim}"
                    )
                kinds.append(0)
                starts.append(v)
                stops.append(0)
                steps.append(0)
            elif isinstance(k, slice):
                dim = shp[di]
                di += 1
                st = k.step if k.step is not None else 1
                if st == 0:
                    raise ValueError("slice step cannot be zero")
                if st < 0:
                    raise NotImplementedError(
                        "negative-step slices (reversal) are not supported"
                    )
                start, stop, _ = k.indices(dim)
                if start >= stop:
                    raise NotImplementedError(
                        "empty slices are not supported (the core cannot "
                        "represent zero-length dimensions)"
                    )
                kinds.append(1)
                starts.append(start)
                stops.append(stop)
                steps.append(st)
            elif isinstance(k, list) or (
                np is not None and isinstance(k, np.ndarray)
            ):
                raise NotImplementedError(
                    "fancy/boolean/sequence indexing (numpy arrays and "
                    "lists as index lanes) is not supported"
                )
            else:
                raise TypeError(
                    f"invalid index lane type: {type(k).__name__}"
                )
        return kinds, starts, stops, steps

    def _native_index(self, key, requires_grad: bool) -> "Tensor":
        kinds, starts, stops, steps = self._normalize_index_key(key)
        # track bit only meaningful for float32; int64 ignores it.
        raw = self._raw.index(kinds, starts, stops, steps, int(requires_grad))
        return Tensor(raw)

    def __getitem__(self, key) -> "Tensor":
        dt_name = self._dtype_name()
        if dt_name not in ("float32", "float64", "int64"):
            # Native indexing registered only for float32/float64/int64
            # (compile memory); fall back to numpy semantics for others.
            if np is None:
                raise ImportError("numpy required for indexing on this dtype")
            try:
                return Tensor.from_numpy(self.numpy()[key])
            except NotImplementedError:
                raise
            except IndexError:
                raise
            except (TypeError, ValueError):
                raise
        try:
            return self._native_index(key, requires_grad=self.requires_grad)
        except NotImplementedError:
            raise
        except IndexError:
            raise

    def __setitem__(self, key, value) -> None:
        dt_name = self._dtype_name()
        if dt_name not in ("float32", "float64", "int64"):
            if np is None:
                raise ImportError("numpy required for indexing on this dtype")
            arr = self.numpy()
            arr[key] = (
                value.numpy() if isinstance(value, Tensor) else value
            )
            object.__setattr__(self, "_raw", Tensor.from_numpy(arr)._raw)
            return
        kinds, starts, stops, steps = self._normalize_index_key(key)
        if isinstance(value, Tensor):
            self._raw.setitem_tensor(kinds, starts, stops, steps, value._raw)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self._raw.setitem_scalar(kinds, starts, stops, steps, value)
        else:
            raise NotImplementedError(
                "fancy/boolean/sequence index assignment is not supported"
            )

    # ── repr ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Tensor({self.tolist()}, shape={self.shape}, "
            f"dtype={np.dtype(self.numpy_dtype()).name})"
        )

    def __str__(self) -> str:
        return repr(self)

    # ── autograd ────────────────────────────────────────────────────

    def _require_autograd(self, op: str) -> None:
        dt = self._dtype_name()
        if dt not in _AUTOGRAD_DTYPES:
            raise TypeError(
                f"{op} requires a tensor with autograd support, got dtype "
                f"{dt!r}. Supported dtypes: "
                + ", ".join(_AUTOGRAD_DTYPES)
                + ". Convert via .to_dtype() where available."
            )

    def requires_grad_(self, val: bool = True) -> Tensor:
        """Enable or disable gradient tracking (in-place). Returns self."""
        self._require_autograd("requires_grad_")
        self._raw.requires_grad_(val)
        return self

    def backward(self) -> None:
        """Run backpropagation from this tensor."""
        self._require_autograd("backward")
        self._raw.backward()

    @property
    def grad(self) -> Optional[Tensor]:
        """The gradient tensor (None if no gradient available)."""
        self._require_autograd("grad")
        raw = self._raw.grad()
        if raw is None:
            return None
        return Tensor(raw)

    def zero_grad(self) -> None:
        """Zero out the gradient."""
        self._require_autograd("zero_grad")
        self._raw.zero_grad()

    # ── math ops ────────────────────────────────────────────────────

    def sum(
        self,
        axes: Optional[List[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axes is None:
            axes = []
        return Tensor(self._raw.sum(axes, keepdims))

    def mean(
        self,
        axes: Optional[List[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axes is None:
            axes = []
        return Tensor(self._raw.mean(axes, keepdims))

    def max(self, axes: Optional[List[int]] = None) -> Tensor:
        if axes is None:
            axes = []
        return Tensor(self._raw.max(axes))

    def min(self, axes: Optional[List[int]] = None) -> Tensor:
        if axes is None:
            axes = []
        return Tensor(self._raw.min(axes))

    def argmax(self, axis: int = 0):
        """Return indices of maximum values along axis.

        Returns a Python int (scalar) or list of ints — not a Tensor,
        since argmax indices are integers.
        """
        return self._raw.argmax(axis)

    def argmin(self, axis: int = 0):
        """Return indices of minimum values along axis.

        Returns a Python int (scalar) or list of ints — not a Tensor,
        since argmin indices are integers.
        """
        return self._raw.argmin(axis)

    def softmax(self, axes: Optional[List[int]] = None) -> Tensor:
        if axes is None:
            axes = []
        return Tensor(self._raw.softmax(axes))

    def flatten(
        self, start_dim: int = 0, end_dim: int = -1
    ) -> Tensor:
        return Tensor(self._raw.flatten(start_dim, end_dim))

    def ravel(self) -> Tensor:
        """Flatten into a 1-D tensor (alias of flatten)."""
        return Tensor(self._raw.flatten())

    def swapaxes(self, axis0: int, axis1: int) -> Tensor:
        """Swap two axes and return the re-permuted tensor."""
        ndim = self.ndim
        axes = list(range(ndim))
        axes[axis0], axes[axis1] = axes[axis1], axes[axis0]
        return Tensor(self._raw.permute(axes))

    def reshape(self, shape: Shape) -> Tensor:
        return Tensor(self._raw.reshape(shape))

    def matmul(self, other: Tensor) -> Tensor:
        return Tensor(self._raw.matmul(other._raw))

    def clip(self, min_val: float, max_val: float) -> Tensor:
        return Tensor(self._raw.clip(min_val, max_val))

    def abs(self) -> Tensor:
        return Tensor(self._raw.abs())

    # ── Unary math / activation ops ─────────────────────────────────

    def exp(self) -> Tensor:
        return Tensor(self._raw.exp())

    def log(self) -> Tensor:
        return Tensor(self._raw.log())

    def sqrt(self) -> Tensor:
        return Tensor(self._raw.sqrt())

    def tanh(self) -> Tensor:
        return Tensor(self._raw.tanh())

    def sigmoid(self) -> Tensor:
        return Tensor(self._raw.sigmoid())

    def relu(self) -> Tensor:
        return Tensor(self._raw.relu())

    def reciprocal(self) -> Tensor:
        return Tensor(self._raw.reciprocal())

    # ── Reduction ops ───────────────────────────────────────────────

    def product(
        self,
        axes: Optional[List[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        ax = axes if axes is not None else []
        return Tensor(self._raw.product(ax, keepdims))

    def variance(
        self,
        axis: int = -100,
        keepdims: bool = False,
        unbiased: bool = True,
    ) -> Tensor:
        return Tensor(self._raw.variance(axis, keepdims, unbiased))

    def std(
        self,
        axis: int = -100,
        keepdims: bool = False,
        unbiased: bool = True,
    ) -> Tensor:
        return Tensor(self._raw.std(axis, keepdims, unbiased))

    def norm(self, p: float = 2.0) -> Tensor:
        return Tensor(self._raw.norm(p))

    # ── Shape / view ops ───────────────────────────────────────────

    def transpose(self, axes: Optional[List[int]] = None) -> Tensor:
        ax = axes if axes is not None else []
        return Tensor(self._raw.transpose(ax))

    def permute(self, axes: List[int]) -> Tensor:
        return Tensor(self._raw.permute(axes))

    def squeeze(self, axes: Optional[List[int]] = None) -> Tensor:
        ax = axes if axes is not None else []
        return Tensor(self._raw.squeeze(ax))

    def unsqueeze(self, axes) -> Tensor:
        if isinstance(axes, int):
            axes = [axes]
        return Tensor(self._raw.unsqueeze(axes))

    def expand(self, target_shape: List[int]) -> Tensor:
        return Tensor(self._raw.expand(target_shape))

    def contiguous(self, owned: bool = True) -> Tensor:
        """Contiguous tensor with the same data.

        owned=True (default) always materializes an independent copy, even
        when already contiguous. owned=False allows returning an alias
        (no copy) when the source is already contiguous+shared.
        """
        return Tensor(self._raw.contiguous(owned))

    def masked_fill(self, mask: Tensor, value: float) -> Tensor:
        if hasattr(mask._raw, "where"):
            return Tensor(self._raw.masked_fill(mask._raw, value))
        import numpy as _np
        mask_bool = tensor(_np.asarray(mask.numpy(), dtype=_np.bool_))
        return Tensor(self._raw.masked_fill(mask_bool._raw, value))

    def detach(self) -> Tensor:
        """Return a new tensor detached from the computation graph."""
        return Tensor(self._raw.detach())

    @property
    def device(self) -> str:
        """Return the device string ('cpu' or 'cuda:0')."""
        return str(self._raw.device())

    def is_contiguous(self) -> bool:
        """Check if the tensor memory layout is contiguous."""
        return bool(self._raw.is_contiguous())

    def is_leaf(self) -> bool:
        """Check if tensor is a leaf in the computation graph."""
        return bool(self._raw.is_leaf())

    def to_dtype(self, dtype: str) -> Tensor:
        """Convert tensor to the specified dtype string."""
        try:
            return Tensor(self._raw.to_dtype(dtype))
        except AttributeError:
            import numpy as _np
            np_dtype = {
                "float16": _np.float16, "float32": _np.float32,
                "float64": _np.float64, "int8": _np.int8,
                "int16": _np.int16, "int32": _np.int32,
                "int64": _np.int64, "uint8": _np.uint8,
                "uint16": _np.uint16, "uint32": _np.uint32,
                "uint64": _np.uint64, "bool": _np.bool_,
            }.get(dtype)
            if np_dtype is None:
                raise ValueError(f"unsupported dtype: {dtype}")
            return tensor(self.numpy().astype(np_dtype))

    def float(self) -> Tensor:
        """Alias for to_dtype('float32')."""
        return self.to_dtype("float32")

    def int64(self) -> Tensor:
        """Alias for to_dtype('int64')."""
        return self.to_dtype("int64")

    def zeros_like(self) -> Tensor:
        """Create a zero tensor with the same shape as self."""
        return Tensor(self._raw.zeros_like())

    def ones_like(self) -> Tensor:
        """Create a ones tensor with the same shape as self."""
        return Tensor(self._raw.ones_like())

    def triu(self, diagonal: int = 0) -> Tensor:
        """Upper triangular part of the tensor."""
        return Tensor(self._raw.triu(diagonal))

    def tril(self, diagonal: int = 0) -> Tensor:
        """Lower triangular part of the tensor."""
        return Tensor(self._raw.tril(diagonal))

    def cumsum(self, axis: int = 0) -> Tensor:
        """Cumulative sum along an axis."""
        return Tensor(self._raw.cumsum(axis))

    def dot(self, other: Tensor) -> Tensor:
        """Dot product of two tensors."""
        return Tensor(self._raw.dot(other._raw))

    def outer(self, other: Tensor) -> Tensor:
        """Outer product of two 1-D tensors."""
        return Tensor(self._raw.outer(other._raw))

    def gather(self, indices: list[int], axis: int = 0) -> Tensor:
        """Gather slices along axis at indices."""
        return Tensor(self._raw.gather(indices, axis))

    def mse(self, target: Tensor) -> Tensor:
        """Mean squared error loss against target."""
        return Tensor(self._raw.mse(target._raw))

    def bce(self, target: Tensor) -> Tensor:
        """Binary cross-entropy loss against target."""
        return Tensor(self._raw.bce(target._raw))

    def bce_logits(self, target: Tensor) -> Tensor:
        """BCE loss with logits (sigmoid applied internally)."""
        return Tensor(self._raw.bce_logits(target._raw))

    def __len__(self) -> int:
        """Length of the first dimension."""
        return self.shape[0]

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiply (a @ b)."""
        return self.matmul(other)

    def __invert__(self) -> Tensor:
        """Bitwise/logical NOT (~t)."""
        inv_np = ~self.numpy().astype(bool)
        return Tensor(_tenmo.tensor(inv_np.astype(float).tolist()))

    def allclose(self, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if two tensors are element-wise close."""
        diff = np.abs(self.numpy() - other.numpy())
        allowed = atol + rtol * np.abs(other.numpy())
        return bool(np.all(diff <= allowed))

    def seed_grad(self, value: float = 1.0) -> None:
        """Initialize gradient with a constant value."""
        self._require_autograd("seed_grad")
        self._raw.seed_grad(value)

    # ── copy / buffer protocol (wrapper-only, Python 3.12+ __buffer__) ──

    def __copy__(self):
        """Shallow copy = independent data copy (no shared buffer)."""
        return self.from_numpy(self.numpy())

    def __deepcopy__(self, memo):
        """Deep copy = same as copy (tensors have no graph to clone)."""
        return self.from_numpy(self.numpy())

    def __buffer__(self, flags):
        """Expose the data buffer to memoryview/bytes (Python 3.12+).

        PEP 688 passes flags=0; the buffer is always a fresh C-contiguous
        copy of the data. The binding never shares memory with Python, so
        reads are exact and writes to the buffer do not alias the tensor."""
        return memoryview(np.ascontiguousarray(self.numpy()))

    # ── operator machinery (guards before raw calls) ───────────────

    __hash__ = None  # unhashable, like PyTorch (== returns a BoolTensor, not bool)

    @staticmethod
    def _broadcastable(sa, sb) -> bool:
        """numpy right-aligned broadcast compatibility check.

        The core panics (process abort) on non-broadcastable shapes, so the
        wrapper must reject them here with a proper ValueError.
        """
        ra, rb = len(sa), len(sb)
        for i in range(1, min(ra, rb) + 1):
            da, db = sa[ra - i], sb[rb - i]
            if da != 1 and db != 1 and da != db:
                return False
        return True

    def _dtype_name(self) -> str:
        return np.dtype(self.numpy_dtype()).name

    def _check_tensor_operand(self, other: "Tensor") -> None:
        if self._dtype_name() != other._dtype_name():
            raise TypeError(
                f"dtype mismatch: {self._dtype_name()} vs "
                f"{other._dtype_name()} (mixed-dtype promotion not supported)"
            )
        if not Tensor._broadcastable(self.shape, other.shape):
            raise ValueError(
                f"shapes {self.shape} and {other.shape} are not "
                f"broadcastable"
            )

    def _check_scalar_operand(self, value) -> None:
        if (
            isinstance(value, (int, float))
            and value < 0
            and self._dtype_name().startswith("uint")
        ):
            raise ValueError(
                f"negative scalar {value} invalid for unsigned dtype "
                f"{self._dtype_name()}"
            )

    def _binary(self, other, tt: str, ts: str):
        """Shared dispatch for binary arithmetic; NotImplemented passthrough."""
        if self._dtype_name() == "bool":
            raise TypeError("arithmetic is not supported for bool tensors")
        if isinstance(other, Tensor):
            self._check_tensor_operand(other)
            return Tensor(getattr(self._raw, tt)(other._raw))
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            return Tensor(getattr(self._raw, ts)(other))
        return NotImplemented

    def _inplace_guard(self, op: str) -> None:
        if self.requires_grad and self._raw.is_leaf():
            raise RuntimeError(
                f"can't {op} a leaf tensor that requires grad "
                f"(use out-of-place ops or detach first)"
            )

    # ── arithmetic dunders ──────────────────────────────────────────

    def __add__(self, other):
        return self._binary(other, "add", "add_scalar")

    def __radd__(self, other):
        if self._dtype_name() == "bool":
            raise TypeError("arithmetic is not supported for bool tensors")
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            return Tensor(self._raw.add_scalar(other))
        return NotImplemented

    def __sub__(self, other):
        return self._binary(other, "sub", "sub_scalar")

    def __rsub__(self, other):
        if self._dtype_name() == "bool":
            raise TypeError("arithmetic is not supported for bool tensors")
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            return Tensor(self._raw.rsub_scalar(other))
        return NotImplemented

    def __mul__(self, other):
        return self._binary(other, "mul", "mul_scalar")

    def __rmul__(self, other):
        if self._dtype_name() == "bool":
            raise TypeError("arithmetic is not supported for bool tensors")
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            return Tensor(self._raw.mul_scalar(other))
        return NotImplemented

    def __truediv__(self, other):
        if self._dtype_name() == "bool":
            raise TypeError("true division is not defined for bool tensors")
        return self._binary(other, "truediv", "truediv_scalar")

    def __rtruediv__(self, other):
        if self._dtype_name() == "bool":
            raise TypeError("true division is not defined for bool tensors")
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            return Tensor(self._raw.rtruediv_scalar(other))
        return NotImplemented

    def __neg__(self):
        dt = self._dtype_name()
        if dt == "bool" or dt.startswith("uint"):
            raise TypeError(f"unary minus undefined for dtype {dt}")
        return Tensor(self._raw.neg())

    def __pow__(self, exponent, mod=None):
        if mod is not None:
            raise TypeError("three-argument pow() not supported")
        if isinstance(exponent, (int, float)) and not isinstance(
            exponent, bool
        ):
            self._check_scalar_operand(exponent)
            return Tensor(self._raw.pow(exponent))
        return NotImplemented

    # ── comparison dunders (elementwise → BoolTensor) ───────────────

    def __eq__(self, other):
        if isinstance(other, Tensor):
            self._check_tensor_operand(other)
            return Tensor(self._raw.eq(other._raw))
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Tensor):
            self._check_tensor_operand(other)
            return Tensor(self._raw.ne(other._raw))
        return NotImplemented

    def _compare(self, other, ts: str, tt: str):
        if isinstance(other, Tensor):
            self._check_tensor_operand(other)
            return Tensor(getattr(self._raw, tt)(other._raw))
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            return Tensor(getattr(self._raw, ts)(other))
        return NotImplemented

    def __lt__(self, other):
        return self._compare(other, "lt", "lt_tensor")

    def __le__(self, other):
        return self._compare(other, "le", "le_tensor")

    def __gt__(self, other):
        return self._compare(other, "gt", "gt_tensor")

    def __ge__(self, other):
        return self._compare(other, "ge", "ge_tensor")

    # ── in-place operators (true buffer mutation) ───────────────────
    # Raw iadd/isub/imul/itruediv (+scalar variants) are registered on
    # float32/int64; they mutate the buffer like the core Tensor dunders.
    # Leaf tensors requiring grad are rejected (core would panic).

    def _inplace(self, other, tt: str, ts: str, verb: str):
        if self._dtype_name() == "bool":
            raise TypeError(
                f"in-place {verb} is not supported for bool tensors"
            )
        self._inplace_guard(verb)
        if isinstance(other, Tensor):
            self._check_tensor_operand(other)
            getattr(self._raw, tt)(other._raw)
        elif isinstance(other, (int, float)) and not isinstance(other, bool):
            self._check_scalar_operand(other)
            getattr(self._raw, ts)(other)
        else:
            return NotImplemented
        return self

    def __iadd__(self, other):
        return self._inplace(other, "iadd", "iadd_scalar", "add to")

    def __isub__(self, other):
        return self._inplace(other, "isub", "isub_scalar", "subtract from")

    def __imul__(self, other):
        return self._inplace(other, "imul", "imul_scalar", "multiply")

    def __itruediv__(self, other):
        return self._inplace(other, "itruediv", "itruediv_scalar", "divide")

    # ── numpy interop ───────────────────────────────────────────────

    @staticmethod
    def from_numpy(arr, *, requires_grad: bool = False):
        """Create a tensor from a numpy array (requires numpy).

        Preserves the original dtype for all 11 numeric/bool types:
        float16, float32, float64, int8, int16, int32, int64, uint8, uint16,
        uint32, uint64, bool_. Unsupported dtypes are cast to float32.
        """
        if np is None:
            raise ImportError("numpy is required for Tensor.from_numpy()")
        return Tensor(_tenmo.tensor(np.ascontiguousarray(arr), requires_grad=requires_grad))


# ── Layer wrappers ──────────────────────────────────────────────────


class _Layer:
    """Base for all layer wrappers. Stores the raw Mojo object."""

    __slots__ = ("_raw",)

    def train(self) -> _Layer:
        """Set the layer to training mode."""
        self._raw.train()
        return self

    def eval(self) -> _Layer:
        """Set the layer to evaluation mode."""
        self._raw.eval()
        return self

    def into(self) -> Module:
        """Wrap this layer into a Module for use in Sequential."""
        return Module(self._raw.into())


class Linear(_Layer):
    """Fully connected layer: out = x @ W^T + bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        bias_zero: bool = True,
        init_method: str = "uniform",
    ) -> None:
        object.__setattr__(
            self,
            "_raw",
            _tenmo.Linear(in_features, out_features, bias=bias, bias_zero=bias_zero, init_method=init_method),
        )

    @property
    def weight(self) -> Tensor:
        """The weight tensor."""
        return Tensor(self._raw.weight())

    @property
    def bias(self) -> Optional[Tensor]:
        """The bias tensor (None if bias=False)."""
        raw = self._raw.bias()
        if raw is None:
            return None
        return Tensor(raw)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]})"


class ReLU(_Layer):
    """ReLU activation: max(0, x)."""

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.ReLU())

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(_Layer):
    """Sigmoid activation: 1 / (1 + exp(-x))."""

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.Sigmoid())

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(_Layer):
    """Tanh activation."""

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.Tanh())

    def __repr__(self) -> str:
        return "Tanh()"


class Flatten(_Layer):
    """Flatten spatial dimensions: (N, C, H, W) -> (N, C*H*W)."""

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.Flatten())

    def __repr__(self) -> str:
        return "Flatten()"


class Module:
    """A type-erased wrapper around a layer, for use in Sequential."""

    __slots__ = ("_raw",)

    def __init__(self, raw: _tenmo._Module) -> None:
        object.__setattr__(self, "_raw", raw)

    def __repr__(self) -> str:
        return "Module(...)"


class Sequential:
    """A sequential container that chains layers.

    Each element should be a Module (call .into() on a layer first).

    Example:
        model = Sequential([
            Linear(784, 128).into(),
            ReLU().into(),
            Linear(128, 10).into(),
        ])
    """

    __slots__ = ("_raw",)

    def __init__(self, layers=None) -> None:
        if layers is None:
            layers = []
        raw_layers = []
        for m in layers:
            if hasattr(m, '_raw'):
                if hasattr(m._raw, 'into'):
                    raw_layers.append(m._raw.into())
                else:
                    raw_layers.append(m._raw)
            elif hasattr(m, 'into'):
                raw_layers.append(m.into()._raw)
            else:
                raw_layers.append(m)
        object.__setattr__(self, "_raw", _tenmo.Sequential(raw_layers))

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return Tensor(self._raw.forward(x._raw))

    def train(self) -> Sequential:
        """Set all layers to training mode."""
        self._raw.train()
        return self

    def eval(self) -> Sequential:
        """Set all layers to evaluation mode."""
        self._raw.eval()
        return self

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        self._raw.zero_grad()

    def num_parameters(self) -> int:
        """Count total learnable parameters."""
        return self._raw.num_parameters()

    def __repr__(self) -> str:
        return f"Sequential(num_params={self.num_parameters()})"


# ── Loss functions ──────────────────────────────────────────────────


class CrossEntropyLoss:
    """Cross-entropy loss for classification.

    Example:
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, target)
        loss.backward()
    """

    __slots__ = ("_raw",)

    def __init__(
        self,
        *,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        training: bool = True,
    ) -> None:
        object.__setattr__(
            self,
            "_raw",
            _tenmo.CrossEntropyLoss(
                reduction=reduction,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
                training=training,
            ),
        )

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the loss.

        Dispatches on the target capsule type: integer targets take the
        class-index path, float targets the probability path. A non-Tensor
        target raises TypeError loudly (never silently float-cast).
        """
        if not isinstance(target, Tensor):
            raise TypeError(
                "CrossEntropyLoss target must be a tenmo Tensor, got "
                f"{type(target).__name__}"
            )
        raw = target._raw
        kind = np.dtype(raw.numpy_dtype()).kind if np is not None else "f"
        if kind == "i":
            return Tensor(self._raw.forward_int(logits._raw, raw))
        if kind == "f":
            return Tensor(self._raw.forward_float(logits._raw, raw))
        raise TypeError(
            "CrossEntropyLoss target dtype must be int64 or float32, got "
            f"{raw.numpy_dtype()}"
        )

    def train(self) -> CrossEntropyLoss:
        """Set to training mode."""
        self._raw.train()
        return self

    def eval(self) -> CrossEntropyLoss:
        """Set to evaluation mode."""
        self._raw.eval()
        return self

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"


class MSELoss:
    """Mean squared error loss (stateless; pred and target share dtype).

    Example:
        loss_fn = MSELoss()
        loss = loss_fn(pred, target)
        loss.backward()
    """

    __slots__ = ("_raw",)

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.MSELoss())

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the loss."""
        return Tensor(self._raw.forward(pred._raw, target._raw))

    def train(self) -> MSELoss:
        """No-op (stateless), for interface parity with BCE losses."""
        return self

    def eval(self) -> MSELoss:
        """No-op (stateless), for interface parity with BCE losses."""
        return self

    def __repr__(self) -> str:
        return "MSELoss()"


class BCELoss:
    """Binary cross-entropy on probabilities (with safe-log epsilon).

    Target values must be in [0, 1] and share the pred dtype.
    """

    __slots__ = ("_raw",)

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.BCELoss())

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the loss."""
        return Tensor(self._raw.forward(pred._raw, target._raw))

    def train(self) -> BCELoss:
        """Set to training mode."""
        self._raw.train()
        return self

    def eval(self) -> BCELoss:
        """Set to evaluation mode."""
        self._raw.eval()
        return self

    def __repr__(self) -> str:
        return "BCELoss()"


class BCEWithLogitsLoss:
    """Binary cross-entropy with logits (numerically stable fuse).

    Takes raw logits; the sigmoid is fused into the loss. Target values
    must be in [0, 1] and share the pred dtype.
    """

    __slots__ = ("_raw",)

    def __init__(self) -> None:
        object.__setattr__(self, "_raw", _tenmo.BCEWithLogitsLoss())

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the loss."""
        return Tensor(self._raw.forward(logits._raw, target._raw))

    def train(self) -> BCEWithLogitsLoss:
        """Set to training mode."""
        self._raw.train()
        return self

    def eval(self) -> BCEWithLogitsLoss:
        """Set to evaluation mode."""
        self._raw.eval()
        return self

    def __repr__(self) -> str:
        return "BCEWithLogitsLoss()"


# ── Optimizer ───────────────────────────────────────────────────────


class SGD:
    """Stochastic gradient descent with momentum.

    The optimizer extracts parameter pointers from the model at construction
    time, so it must be created *after* the model is fully assembled.

    Example:
        model = Sequential([Linear(784, 128).into(), ReLU().into(), Linear(128, 10).into()])
        opt = SGD(model, lr=0.01, momentum=0.9)
    """

    __slots__ = ("_raw",)

    def __init__(
        self,
        model: Sequential,
        *,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_norm: float = 0.0,
        clip_value: float = 0.0,
    ) -> None:
        object.__setattr__(
            self,
            "_raw",
            _tenmo.SGD(
                model._raw,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                clip_norm=clip_norm,
                clip_value=clip_value,
            ),
        )

    def step(self) -> None:
        """Perform a single optimization step."""
        self._raw.step()

    def zero_grad(self) -> None:
        """Zero out gradients for all tracked parameters."""
        self._raw.zero_grad()

    def set_lr(self, lr: float) -> None:
        """Update the learning rate."""
        self._raw.set_lr(lr)

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return float(self._raw.get_lr())

    def state_dict(self) -> dict:
        """Return optimizer state as a dict."""
        return self._raw.state_dict()

    def __repr__(self) -> str:
        return f"SGD(lr={self.get_lr()})"


class AdamW:
    """Adam with decoupled weight decay (the transformer default).

    The optimizer extracts parameter pointers from the model at construction
    time, so it must be created *after* the model is fully assembled.

    Example:
        model = Sequential([Linear(784, 128).into(), ReLU().into(), Linear(128, 10).into()])
        opt = AdamW(model, lr=0.001)
    """

    __slots__ = ("_raw",)

    def __init__(
        self,
        model: Sequential,
        *,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        clip_norm: float = 0.0,
        clip_value: float = 0.0,
    ) -> None:
        object.__setattr__(
            self,
            "_raw",
            _tenmo.AdamW(
                model._raw,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                clip_norm=clip_norm,
                clip_value=clip_value,
            ),
        )

    def step(self) -> None:
        """Perform a single optimization step."""
        self._raw.step()

    def zero_grad(self) -> None:
        """Zero out gradients for all tracked parameters."""
        self._raw.zero_grad()

    def set_lr(self, lr: float) -> None:
        """Update the learning rate."""
        self._raw.set_lr(lr)

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return float(self._raw.get_lr())

    def state_dict(self) -> dict:
        """Return optimizer state as a dict."""
        return self._raw.state_dict()

    def __repr__(self) -> str:
        return f"AdamW(lr={self.get_lr()})"


# ── Epoch-level training/eval (DataLoader runs in Mojo) ──────────


def train_epoch(
    model: Sequential,
    criterion: CrossEntropyLoss,
    optimizer: SGD,
    features,
    labels,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
) -> tuple:
    """Run one training epoch entirely in Mojo (Tenmo DataLoader).
    Returns (avg_loss, accuracy)."""
    kw = dict(batch_size=batch_size, shuffle=shuffle)
    if normalize_mean is not None:
        kw["normalize_mean"] = normalize_mean
    if normalize_std is not None:
        kw["normalize_std"] = normalize_std
    labels_i64 = np.asarray(labels, dtype=np.int64) if np is not None else labels
    return _tenmo.train_epoch(
        model._raw, criterion._raw, optimizer._raw,
        features, labels_i64, **kw,
    )


def eval_epoch(
    model: Sequential,
    criterion: CrossEntropyLoss,
    features,
    labels,
    *,
    batch_size: int = 64,
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
) -> tuple:
    """Run one eval epoch entirely in Mojo (Tenmo DataLoader).
    Returns (avg_loss, accuracy)."""
    kw = dict(batch_size=batch_size)
    if normalize_mean is not None:
        kw["normalize_mean"] = normalize_mean
    if normalize_std is not None:
        kw["normalize_std"] = normalize_std
    labels_i64 = np.asarray(labels, dtype=np.int64) if np is not None else labels
    return _tenmo.eval_epoch(
        model._raw, criterion._raw,
        features, labels_i64, **kw,
    )


# ── Tensor composition functions ────────────────────────────────────


def concat(tensors: list, *, axis: int = 0) -> Tensor:
    """Concatenate tensors along an axis."""
    raws = [t._raw for t in tensors]
    return Tensor(_tenmo.concat(raws, axis=axis))


def stack(tensors: list, *, axis: int = 0) -> Tensor:
    """Stack tensors along a new axis."""
    raws = [t._raw for t in tensors]
    return Tensor(_tenmo.stack(raws, axis=axis))


def where(condition: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise conditional selection."""
    if hasattr(condition._raw, "where"):
        return Tensor(condition._raw.where(a._raw, b._raw))
    import numpy as _np
    cond_np = _np.asarray(condition.numpy(), dtype=_np.bool_)
    return tensor(_np.where(cond_np, a.numpy(), b.numpy()))


# ── DataLoader (tensor-native batches) ─────────────────────────────
#
# A batch is a pair `(features: Tensor, labels: Tensor)` carrying their
# natural source dtypes — the loader never forces float32/int64. Data may be
# any of a Tenmo Tensor, a numpy array, or a nested list; numpy/list values
# are converted to Tenmo Tensors once at construction (a single copy; a
# clean contiguous array passes through with zero extra copies).
#
# v1 registers TWO dtype pairs — (float32, int64) for the class-index
# cross-entropy path and (float32, float32) for MSE/BCE/CE-probability
# targets. The engine DataLoader[sample_dtype, label_dtype]
# is fully generic, but each *registered* pair costs ~1-2GB of compile-time
# memory, so pairs are widened incrementally (add to
# _LOADER_PAIRS and register_data_loader in tenmo_bind.mojo together).
# Unregistered pairs raise NotImplementedError — cast the input to use a
# registered pair.
#
#   * sequential (eval()):  each batch is a zero-copy view slice of the
#     source tensors — no per-batch data movement at all.
#   * shuffled (train()):   rows are gathered into a persistent preallocated
#     buffer, so each epoch moves exactly batch_bytes with zero allocations.
#
# Shuffling uses `std.random.shuffle` (same as the trait-generic Mojo
# NativeLoader); each `reset()` draws a fresh permutation — runs are not
# reproducible, so no seed parameter is offered.
#
# Contract: a shuffled batch aliases the loader's reused buffer and is valid
# ONLY until the next `next()` call. Treat batches as read-only inputs.
# `transform(xb, yb)`, if given, runs in Python on the returned Tensors.

def _dtype_name(obj) -> str:
    """Normalized numpy dtype name of a Tenmo Tensor, ndarray, or list."""
    if isinstance(obj, Tensor):
        return np.dtype(obj.numpy_dtype()).name
    if np is not None and isinstance(obj, np.ndarray):
        return np.dtype(obj.dtype).name
    return np.dtype(np.asarray(obj).dtype).name


def _coerce_input(obj):
    """Ensure the raw binding receives a raw registered capsule (Tenmo
    Tensor), a contiguous numpy array, or a materialized list array."""
    if isinstance(obj, Tensor):
        return obj._raw
    if np is not None and isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    return np.asarray(obj)


_LOADER_PAIRS = {
    ("float32", "int64"): _tenmo.DataLoader,
    ("float32", "float32"): _tenmo.DataLoaderProb,
}


class DataLoader:
    """Iterate batches of `(features, labels)` Tensors from in-memory data.

    Args:
        features: `(N, *feat)` — Tenmo Tensor, numpy array, or list.
        labels: `(N, *lab)` — same value types (any int/float dtype).
        batch_size: Samples per batch.
        shuffle: If True, yield randomly permuted batches.
        drop_last: If True, drop the final partial batch.
        transform: Optional `(xb, yb) -> (xb, yb)` applied per batch.
    """

    def __init__(
        self,
        features,
        labels,
        *,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        transform=None,
    ):
        fd = _dtype_name(features)
        ld = _dtype_name(labels)
        try:
            cls = _LOADER_PAIRS[(fd, ld)]
        except KeyError:
            pairs = ", ".join(
                f"{f} / {l}" for (f, l) in _LOADER_PAIRS
            )
            raise NotImplementedError(
                f"DataLoader: no registered dtype pair for features={fd!r}, "
                f"labels={ld!r}. Registered pairs: {pairs}. The loader "
                "preserves dtypes; cast your data (e.g. "
                "np.asarray(x, dtype=np.float32)) to use a registered pair."
            )
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable or None")
        features = _coerce_input(features)
        labels = _coerce_input(labels)
        self._raw = cls(features, labels, batch_size, shuffle, drop_last)
        self._transform = transform

    def __iter__(self) -> "DataLoader":
        # Auto-restart the epoch when re-iterating after exhaustion, so the
        # canonical loop works without explicit bookkeeping:
        #     for epoch in range(N):
        #         for xb, yb in loader:
        #             ...
        # An explicit reset() is still available to skip ahead mid-epoch.
        if not self._raw.has_next():
            self._raw.reset()
        return self

    def __next__(self):
        if not self._raw.has_next():
            raise StopIteration
        fx, fy = self._raw.next()
        xb, yb = Tensor(fx), Tensor(fy)
        if self._transform is not None:
            xb, yb = self._transform(xb, yb)
        return xb, yb

    def __len__(self) -> int:
        return self._raw.length()

    def reset(self) -> None:
        """Restart the epoch; a new permutation is drawn when shuffling."""
        self._raw.reset()

    def train(self) -> None:
        """Switch to shuffled mode (call reset() to re-permute)."""
        self._raw.set_mode(True)

    def eval(self) -> None:
        """Switch to sequential zero-copy batch mode."""
        self._raw.set_mode(False)

    def to_gpu(self) -> None:
        """Move sources + batch buffers to the GPU (derive-the-device)."""
        self._raw.to_gpu()

    def to_cpu(self) -> None:
        """Move sources + batch buffers back to the CPU."""
        self._raw.to_cpu()

    def device(self) -> str:
        """Return the device string ('cpu' or 'cuda:0')."""
        return str(self._raw.device())
