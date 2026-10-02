"""View-aliasing through the Python boundary (§5 Views & Reshaping).

Stub names match `tests/test_python_bindings.txt` so the §32.18 audit can match
by name. These were previously unobtainable: the binding handlers bound
`var a = ptr[]` (a deep-copy by Tensor's copy ctor) before calling the core
view ops, so every view was a view of a throwaway copy. The no-copy-out sweep
passes the stored pointee by reference, so the core metadata-only view ops
now alias the base tensor's buffer.

Semantics restored (torch/numpy parity for views; documented deviation for contiguous):
- transpose/permute/squeeze/unsqueeze  -> shared metadata views
- expand                                -> stride-0 broadcast view (no guard;
                                             writes broadcast like numpy/torch)
- reshape                               -> view iff the source is contiguous
                                             (offset-aware), else materialize
                                             in correct logical order
- contiguous                            -> ALWAYS materializes an owned copy,
                                             even from an already-contiguous
                                             source. Deliberate core contract
                                             (docstring on `Tensor.contiguous`):
                                             callers rely on isolation (gradbox
                                             segmentation, offset-0 data_buffer
                                             access). Deviates from torch, which
                                             returns self when already
                                             contiguous.
- flatten/ravel                         -> still copies (core `NDBuffer.flatten`
                                             materializes; no change here)

`__getitem__` slicing aliasing is covered in test_indexing.py.
"""
from __future__ import annotations

import numpy as np
import pytest

import tenmo


class TestViewsAndReshaping:
    def test_view_shares_memory_with_original(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = t.transpose([1, 0])
        assert v.shape == (3, 2)
        v[0, 1] = 99.0
        assert t.tolist() == [[1.0, 2.0, 3.0], [99.0, 5.0, 6.0]]
        t[1, 2] = 42.0
        assert v[2, 1].item() == 42.0

    def test_reshape_is_zero_copy_when_contiguous(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = t.reshape((2, 3))
        assert r.shape == (2, 3)
        r[0, 0] = 99.0
        assert t.tolist() == [99.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        t[5] = 42.0
        assert r.tolist() == [[99.0, 2.0, 3.0], [4.0, 5.0, 42.0]]

    def test_reshape_copies_when_input_non_contiguous(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tp = t.transpose([1, 0])
        assert tp.is_contiguous() is False
        r = tp.reshape((6,))
        # Logical order preserved (0,3,1,4,2,5) — a shared view would have
        # re-read the raw buffer as [1,2,3,4,5,6].
        assert r.tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        r[0] = 99.0
        assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert tp.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_dense_offset_slice_reshape_preserves_offset_view(self):
        t = tenmo.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        s = t[2:]
        r = s.reshape((2, 2))
        assert r.tolist() == [[2.0, 3.0], [4.0, 5.0]]
        r[0, 0] = 99.0
        assert t.tolist() == [0.0, 1.0, 99.0, 3.0, 4.0, 5.0]

    def test_contiguous_returns_self_if_already_contiguous(self):
        # Documented deviation from torch: `contiguous()` ALWAYS returns an
        # owned copy (isolation guarantee the core relies on). A shared view
        # from an already-contiguous source is deliberately NOT returned.
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        c = t.contiguous()
        assert c.is_contiguous() is True
        c[0, 0] = 99.0
        assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_contiguous_returns_new_buffer_if_not(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tp = t.transpose([1, 0])
        c = tp.contiguous()
        assert tp.is_contiguous() is False
        assert c.is_contiguous() is True
        c[0, 0] = 99.0
        assert tp.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        assert c.tolist() == [[99.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_contiguous_owned_false_aliases_contiguous_source(self):
        # Opt-in fast path: no copy when already contiguous+shared.
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        c = t.contiguous(owned=False)
        assert c.is_contiguous() is True
        c[0, 0] = 99.0
        assert t.tolist() == [[99.0, 2.0], [3.0, 4.0]]

    def test_contiguous_owned_false_still_copies_strided_source(self):
        # The flag only skips the no-op copy; strided sources materialize.
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tp = t.transpose([1, 0])
        c = tp.contiguous(owned=False)
        assert c.is_contiguous() is True
        c[0, 0] = 99.0
        assert tp.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_expand_broadcasts_size_one_dims_without_copy(self):
        t = tenmo.tensor([[1.0, 2.0]])
        e = t.expand([3, 2])
        assert e.shape == (3, 2)
        assert e.tolist() == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
        # Broadcast write aliases the single backing element (numpy/torch
        # parity — no guard on stride-0 mutation).
        e[1, 0] = 99.0
        assert t.tolist() == [[99.0, 2.0]]
        assert e[2, 0].item() == 99.0

    def test_squeeze_and_unsqueeze_share_memory(self):
        s = tenmo.tensor([[[1.0, 2.0, 3.0]]])
        sq = s.squeeze([0, 1])
        assert sq.shape == (3,)
        sq[1] = 88.0
        assert s.tolist() == [[[1.0, 88.0, 3.0]]]

        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = t.unsqueeze(0)
        assert u.shape == (1, 2, 2)
        u[0, 1, 1] = 7.0
        assert t.tolist() == [[1.0, 2.0], [3.0, 7.0]]


class TestViewAutograd:
    def test_reshape_view_backward_scatters(self):
        x = tenmo.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = x.reshape((2, 2))
        y[0, 0].backward()
        assert x.grad.tolist() == [1.0, 0.0, 0.0, 0.0]