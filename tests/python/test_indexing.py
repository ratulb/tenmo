"""Indexing & slicing tests.

Stub names match `tests/test_python_bindings.txt`. Slices are *shared views* through the Python boundary: the handlers
pass the stored pointee by reference into `View.forward_list` (no `var a = ptr[]`
copy-out), so `t[a:b]` and strided `t[::k]` alias the source. The
documented deviations are all at the very edge of what the core view metadata
can represent:

- Negative-step slices (reversal) are not representable in the core view
  metadata (validators clamp negative steps as if positive). The wrapper
  raises NotImplementedError — asserted here as the documented behavior.
- Boolean-mask and integer-array (fancy) indexing have no core path; the
  wrapper raises NotImplementedError for all numpy-array/list lane types.
- Empty slices (`t[5:2]`) are not representable (Shape forbids 0-length
  dims); the wrapper raises NotImplementedError.

Strided slices are shared views with strides (torch/numpy parity); the spec
name `test_strided_slice_returns_copy_not_view` contradicted that and is
implemented as `test_strided_slice_aliases_source`.

Gradients flow through `ViewBackward` via ancestor ids (not buffer aliasing),
so index/slice-gradient scattering works without touching the shared storage.
"""
from __future__ import annotations

import numpy as np
import pytest

import tenmo


class TestIndexingAndSlicing:
    def test_single_int_index_1d(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        r = t[0]
        assert r.tolist() == 10.0
        r2 = t[2]
        assert r2.tolist() == 30.0

    def test_single_int_index_negative(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        assert t[-1].tolist() == 50.0
        assert t[-5].tolist() == 10.0

    def test_single_int_index_out_of_bounds_raises(self):
        t = tenmo.tensor([10.0, 20.0, 30.0])
        with pytest.raises(IndexError):
            t[3]
        with pytest.raises(IndexError):
            t[-4]

    def test_basic_slice_start_stop(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        assert t[1:4].tolist() == [20.0, 30.0, 40.0]
        assert t[:3].tolist() == [10.0, 20.0, 30.0]
        assert t[2:].tolist() == [30.0, 40.0, 50.0]
        assert t[:].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_basic_slice_with_step(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        assert t[::2].tolist() == [10.0, 30.0, 50.0]
        assert t[1:5:2].tolist() == [20.0, 40.0]

    def test_basic_slice_with_negative_step_reverses(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        # Documented deviation: negative-step (reversal) is not representable
        # in the core view metadata — the wrapper raises NotImplementedError.
        with pytest.raises(NotImplementedError):
            t[::-1]

    def test_slice_beyond_bounds_clamps_like_python(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        assert t[1:100].tolist() == [20.0, 30.0, 40.0, 50.0]
        assert t[-100:2].tolist() == [10.0, 20.0]

    def test_multi_dim_index_tuple(self):
        m = tenmo.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        assert m[1, 2].tolist() == 6.0
        assert m[0, 0].tolist() == 1.0

    def test_multi_dim_mixed_int_and_slice(self):
        m = tenmo.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        assert m[0, 1:].tolist() == [2.0, 3.0]
        assert m[1:, 0].tolist() == [4.0, 7.0]
        assert m[:2, 1:].tolist() == [[2.0, 3.0], [5.0, 6.0]]

    def test_ellipsis_expands_correctly(self):
        m = tenmo.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        assert m[..., 1].tolist() == [2.0, 5.0, 8.0]
        assert m[0, ...].tolist() == [1.0, 2.0, 3.0]
        assert m[..., :2].tolist() == [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]

    def test_newaxis_inserts_dimension(self):
        m = tenmo.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        assert m[None, :, :].shape == (1, 3, 3)
        assert m[:, None, :].shape == (3, 1, 3)
        assert m[:, :, None].shape == (3, 3, 1)
        assert m[None].shape == (1, 3, 3)

    def test_boolean_mask_indexing_selects_correct_elements(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Documented deviation: boolean-mask indexing has no core path.
        with pytest.raises(NotImplementedError):
            t[np.array([True, False, True, False, True])]

    def test_boolean_mask_shape_mismatch_raises(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError):
            t[np.array([True, False])]

    def test_integer_array_fancy_indexing(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(NotImplementedError):
            t[np.array([0, 2])]

    def test_fancy_indexing_with_repeated_indices(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError):
            t[[0, 0, 1]]

    def test_contiguous_slice_returns_zero_copy_view(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v = t[1:4]
        assert v.tolist() == [2.0, 3.0, 4.0]
        v[1] = 99.0
        assert t.tolist() == [1.0, 2.0, 99.0, 4.0, 5.0]

    def test_strided_slice_aliases_source(self):
        # Spec stub name is test_strided_slice_returns_copy_not_view, but a
        # strided slice is a VIEW in torch/numpy — mutation aliases the source.
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v = t[::2]
        assert v.tolist() == [1.0, 3.0, 5.0]
        v[0] = 99.0
        assert t.tolist() == [99.0, 2.0, 3.0, 4.0, 5.0]

    def test_index_assignment_basic(self):
        t = tenmo.tensor([10.0, 20.0, 30.0])
        t[0] = 99.0
        assert t.tolist() == [99.0, 20.0, 30.0]
        m = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        m[0, 1] = 7.0
        assert m.tolist() == [[1.0, 7.0], [3.0, 4.0]]

    def test_index_assignment_via_boolean_mask(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError):
            t[np.array([True, False, True])] = 0.0

    def test_index_assignment_via_slice_broadcasts_scalar(self):
        t = tenmo.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        t[1:4] = 7.0
        assert t.tolist() == [10.0, 7.0, 7.0, 7.0, 50.0]

    def test_view_mutation_reflects_in_base_tensor(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = t[0:, 1:]
        v[0, 0] = 99.0
        assert t.tolist() == [[1.0, 99.0, 3.0], [4.0, 5.0, 6.0]]

    def test_base_tensor_mutation_reflects_in_view(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = t[0:, 1:]
        t[0, 2] = 42.0
        assert v[0, 1].item() == 42.0
        t += 100.0
        assert v.tolist() == [[102.0, 142.0], [105.0, 106.0]]


class TestAutogradGradients:
    def test_indexing_gradient_scatters_to_correct_positions(self):
        x = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = x[1]
        y.backward()
        assert x.grad.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]

    def test_slice_gradient_scatters_to_correct_positions(self):
        x = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = x[1:4]
        y.sum().backward()
        assert x.grad.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]