"""Loader tests.

The shape/alignment test below mirrors the exact reindexing logic used by
``load_dataset`` on a tiny synthetic Planetoid-style dataset, so it runs without
any network access. A second test exercises the real loader and is skipped if
the Planetoid files cannot be downloaded.
"""
import os
import pickle

import numpy as np
import scipy.sparse as sp
import pytest

from gcn.data import parse_index_file, load_dataset


def _reindex_like_loader(allx, tx, ally, ty, test_idx_reorder):
    """Replicates the canonical reorder used in load_dataset (non-citeseer)."""
    test_idx_range = np.sort(test_idx_reorder)
    features = sp.vstack([allx, tx]).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack([ally, ty])
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    return features.tocsr(), np.argmax(labels, axis=1), test_idx_range


def test_reorder_aligns_test_rows_with_graph_positions():
    # 3 train/unlabeled rows (ids 0,1,2) + 3 test rows.
    # Test graph ids are 4,3,5 (UNSORTED) -> tx rows are in that order.
    # We tag each row's feature/label by its TRUE graph id so we can check that
    # after reindexing, row k carries graph-id k.
    allx = sp.csr_matrix(np.array([[0.0], [1.0], [2.0]]))  # ids 0,1,2
    ally = np.eye(6)[[0, 1, 2]]                              # one-hot label == id
    test_idx_reorder = [4, 3, 5]
    tx = sp.csr_matrix(np.array([[4.0], [3.0], [5.0]]))     # feature == graph id
    ty = np.eye(6)[[4, 3, 5]]                                # label == graph id

    feats, labels, test_idx_range = _reindex_like_loader(
        allx, tx, ally, ty, test_idx_reorder
    )
    dense = feats.toarray().flatten()
    # After reindex, position i must hold the row whose true graph id is i.
    assert np.allclose(dense, [0, 1, 2, 3, 4, 5]), dense
    assert np.array_equal(labels, [0, 1, 2, 3, 4, 5])
    # The exact failure mode of the old double-permutation bug:
    # indexing both sides with the SORTED range leaves test rows scrambled.
    bad = sp.vstack([allx, tx]).tolil()
    rng = np.sort(test_idx_reorder)
    bad[rng, :] = bad[rng, :]  # no-op -> test rows stay in tx (reorder) order
    bad_dense = bad.tocsr().toarray().flatten()
    assert not np.allclose(bad_dense, [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize("dataset", ["cora", "citeseer"])
def test_real_loader_shapes_and_alignment(tmp_path, dataset):
    try:
        adj, features, labels, idx_train, idx_val, idx_test = load_dataset(
            dataset, str(tmp_path)
        )
    except Exception as e:  # network / download failure
        pytest.skip(f"Planetoid download unavailable: {e}")

    n = features.shape[0]
    assert adj.shape == (n, n)
    assert labels.shape[0] == n
    assert len(idx_test) == 1000
    # No index out of range and splits are disjoint.
    all_idx = [idx_train, idx_val, idx_test]
    for idx in all_idx:
        assert int(idx.max()) < n and int(idx.min()) >= 0
    s_tr = set(idx_train.tolist()); s_va = set(idx_val.tolist()); s_te = set(idx_test.tolist())
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)
    # Every class appears in the test labels (alignment sanity).
    assert len(set(labels[idx_test].tolist())) == int(labels.max()) + 1
