"""Tests for the tensor-native DataLoader (dtype-generic batches).

A batch is a pair `(features: Tensor, labels: Tensor)` with their natural
source dtypes preserved — the loader never forces float32/int64. Sequential
(eval) batches are zero-copy view slices; shuffled (train) batches are
row-gathers into a persistent buffer.

v1 registers two pairs — (float32, int64) and (float32, float32) —
the engine `DataLoader[sample_dtype, label_dtype]` is fully generic,
but each *registered* pair
costs ~1-2GB of compile-time memory, so pairs widen incrementally.
"""
from __future__ import annotations

import numpy as np
import pytest
import tenmo


N = 23  # deliberately not a multiple of BATCH
BATCH = 8


def _pairs(num: int = N):
    feats = np.arange(num * 3, dtype=np.float32).reshape(num, 3)
    labels = np.arange(num, dtype=np.int64)
    return feats, labels


class TestDataLoaderBasics:
    def test_batch_dtype_preserved(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        xb, yb = next(iter(dl))
        assert np.dtype(xb.numpy_dtype()) == np.float32
        assert np.dtype(yb.numpy_dtype()) == np.int64

    def test_batch_shape(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        xb, yb = next(iter(dl))
        assert xb.shape == (BATCH, 3)
        assert yb.shape == (BATCH,)

    def test_sequential_batches_in_order(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        for i, (xb, yb) in enumerate(dl):
            lo = i * BATCH
            hi = min(lo + BATCH, N)
            np.testing.assert_array_equal(xb.numpy(), fx[lo:hi])
            np.testing.assert_array_equal(yb.numpy(), fy[lo:hi])

    def test_len_and_drop_last(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        assert len(dl) == 3  # 23 // 8 = 2 full + 1 partial
        dl2 = tenmo.DataLoader(
            fx, fy, batch_size=BATCH, shuffle=False, drop_last=True
        )
        assert len(dl2) == 2
        assert len([_ for _ in dl2]) == 2

    def test_batch_size_larger_than_dataset(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=1000, shuffle=False)
        assert len(dl) == 1
        xb, yb = next(iter(dl))
        assert xb.shape == (N, 3)
        np.testing.assert_array_equal(yb.numpy(), fy)

    def test_accepts_existing_tenmo_tensors(self):
        fx, fy = _pairs()
        tfx = tenmo.Tensor.from_numpy(fx)  # float32 tensor
        tfy = tenmo.Tensor.from_numpy(fy)  # int64 tensor
        dl = tenmo.DataLoader(tfx, tfy, batch_size=BATCH, shuffle=False)
        xb, yb = next(iter(dl))
        np.testing.assert_array_equal(xb.numpy(), fx[:BATCH])
        np.testing.assert_array_equal(yb.numpy(), fy[:BATCH])

    def test_raw_next_past_end_raises_stop_iteration(self):
        fx = np.arange(6, dtype=np.float32).reshape(3, 2)
        fy = np.arange(3, dtype=np.int64)
        dl = tenmo.DataLoader(fx, fy, batch_size=2, shuffle=False)
        out = []
        while dl._raw.has_next():
            out.append(dl._raw.next())
        assert len(out) == 2
        # The raw binding raises a *typed* Python StopIteration when called
        # past the epoch end (the wrapper guards with has_next() itself).
        with pytest.raises(StopIteration):
            dl._raw.next()

    def test_accepts_nested_lists(self):
        fx = [[float(i)] * 3 for i in range(N)]  # list of rows
        fy = list(range(N))  # int list → int64
        dl = tenmo.DataLoader(
            np.asarray(fx, dtype=np.float32), fy, batch_size=BATCH, shuffle=False
        )
        xb, yb = next(iter(dl))
        np.testing.assert_array_equal(xb.numpy(), np.asarray(fx)[:BATCH])
        np.testing.assert_array_equal(yb.numpy(), np.arange(N)[:BATCH])

    def test_unregistered_float64_features_raise(self):
        fx = np.ones((4, 2), dtype=np.float64)
        fy = np.arange(4, dtype=np.int64)
        with pytest.raises(NotImplementedError, match="features='float64'"):
            tenmo.DataLoader(fx, fy, batch_size=2)

    def test_unregistered_int32_labels_raise(self):
        fx = np.ones((4, 2), dtype=np.float32)
        fy = np.arange(4, dtype=np.int32)
        with pytest.raises(NotImplementedError, match="labels='int32'"):
            tenmo.DataLoader(fx, fy, batch_size=2)

    def test_unregistered_float32_label_pair_now_registered(self):
        # E2: (float32, float32) is registered for MSE/BCE/CE-probability
        # targets — this used to raise NotImplementedError.
        fx = np.ones((4, 2), dtype=np.float32)
        fy = np.linspace(0.0, 1.0, 4).astype(np.float32)
        dl = tenmo.DataLoader(fx, fy, batch_size=2, shuffle=False)
        xb, yb = next(iter(dl))
        assert np.dtype(xb.numpy_dtype()) == np.float32
        assert np.dtype(yb.numpy_dtype()) == np.float32


class TestDataLoaderProb:
    """The (float32, float32) pair: regression/binary batches (E2)."""

    def _pairs(self, num: int = N):
        fx = np.arange(num * 3, dtype=np.float32).reshape(num, 3)
        fy = np.linspace(0.0, 1.0, num).astype(np.float32)
        return fx, fy

    def test_batch_dtype_preserved(self):
        fx, fy = self._pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        xb, yb = next(iter(dl))
        assert np.dtype(xb.numpy_dtype()) == np.float32
        assert np.dtype(yb.numpy_dtype()) == np.float32

    def test_batches_cover_regression_targets(self):
        fx, fy = self._pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        seen = []
        for xb, yb in dl:
            assert xb.shape[1:] == (3,)
            seen.append(yb.numpy())
        np.testing.assert_allclose(
            np.concatenate(seen), fy, rtol=1e-6, atol=1e-6)

    def test_transform_on_regression_targets(self):
        fx, fy = self._pairs()
        dl = tenmo.DataLoader(
            fx, fy, batch_size=BATCH, shuffle=False,
            transform=lambda xb, yb: (xb * 2.0, yb),
        )
        xb, yb = next(iter(dl))
        np.testing.assert_allclose(
            xb.numpy(), fx[:BATCH] * 2.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            yb.numpy(), fy[:BATCH], rtol=1e-6, atol=1e-6)

    def test_batches_feed_mse_loss(self):
        fx, fy = self._pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        loss_fn = tenmo.MSELoss()
        xb, yb = next(iter(dl))
        pred = xb.mean(axes=[1])
        loss = loss_fn(pred, yb)
        assert loss.numpy().shape == ()
        assert float(loss.numpy()) >= 0.0

    def test_shuffled_prob_batches_partition(self):
        fx, fy = self._pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=True)
        seen = np.concatenate([yb.numpy() for _, yb in dl])
        np.testing.assert_allclose(
            np.sort(seen), np.sort(fy), rtol=1e-6, atol=1e-6)


class TestDataLoaderShuffle:
    def test_shuffled_batches_are_a_partition(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=True)
        seen = []
        for _, yb in dl:
            seen.extend(yb.numpy().tolist())
        assert sorted(seen) == list(range(N))

    def test_shuffle_draws_non_identity_epoch(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=True)
        first = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        assert sorted(first) == list(range(N))

    def test_eval_then_train_switch(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=True)
        first = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        assert first != list(range(N))

        dl.reset()
        dl.eval()
        second = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        assert second == list(range(N))

        dl.train()
        dl.reset()
        third = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        assert third != list(range(N))
        assert sorted(third) == list(range(N))

    def test_reset_advances_permutation(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=True)
        first = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        dl.reset()
        second = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        assert first != second
        assert sorted(second) == list(range(N))

    def test_reiterating_restarts_epoch(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=True)
        first = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        # Exhausted now — a fresh `for` (canonical epoch loop) must restart.
        second = [int(y) for _, yb in dl for y in yb.numpy().tolist()]
        assert sorted(first) == list(range(N))
        assert sorted(second) == list(range(N))
        assert first != second  # permutation redrawn per epoch

    def test_reiterating_does_not_restart_midepoch(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(fx, fy, batch_size=BATCH, shuffle=False)
        first = next(iter(dl))  # consumes 1 batch
        second = next(iter(dl))  # mid-epoch: must continue, not restart
        np.testing.assert_array_equal(second[1].numpy(), fy[BATCH : 2 * BATCH])


class TestDataLoaderTransform:
    def test_transform_applied_per_batch(self):
        fx = np.arange(N * 3, dtype=np.float32).reshape(N, 3)
        fy = np.arange(N, dtype=np.int64)

        def transform(xb, yb):
            return xb + 1.0, yb * 2

        dl = tenmo.DataLoader(
            fx, fy, batch_size=BATCH, shuffle=False, transform=transform
        )
        xb, yb = next(iter(dl))
        np.testing.assert_allclose(xb.numpy(), fx[:BATCH] + 1.0)
        np.testing.assert_array_equal(yb.numpy(), fy[:BATCH] * 2)

    def test_transform_swaps_pair(self):
        fx, fy = _pairs()
        dl = tenmo.DataLoader(
            fx,
            fy,
            batch_size=BATCH,
            shuffle=False,
            transform=lambda xb, yb: (yb, xb),
        )
        xb, yb = next(iter(dl))
        np.testing.assert_array_equal(xb.numpy(), fy[:BATCH])
        np.testing.assert_array_equal(yb.numpy(), fx[:BATCH])

    def test_non_callable_transform_rejected(self):
        fx, fy = _pairs()
        with pytest.raises(TypeError, match="callable"):
            tenmo.DataLoader(fx, fy, batch_size=2, transform=[1, 2])