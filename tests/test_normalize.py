"""Tests for the normalized adjacency operator on small synthetic graphs."""
import numpy as np
import scipy.sparse as sp
import torch

from gcn.data import normalize_adjacency, build_adjacency


def _path_graph(n=5):
    """Simple undirected path graph 0-1-2-...-(n-1) as a dict-of-lists."""
    g = {i: [] for i in range(n)}
    for i in range(n - 1):
        g[i].append(i + 1)
        g[i + 1].append(i)
    return g


def test_normalize_adjacency_is_symmetric():
    adj = build_adjacency(_path_graph(6), 6)
    A = normalize_adjacency(adj).to_dense().cpu().numpy()
    assert np.allclose(A, A.T, atol=1e-6), "normalized adjacency must be symmetric"


def test_normalize_adjacency_diagonal_matches_self_loop():
    # With self-loops added, the diagonal entry of A_hat for node i equals
    # 1 / (deg_i + 1) because A_hat = D~^-1/2 (A+I) D~^-1/2 and (A+I)_ii = 1.
    g = _path_graph(5)
    adj = build_adjacency(g, 5)
    A = normalize_adjacency(adj).to_dense().cpu().numpy()
    # node 0 has 1 neighbor -> tilde-degree 2 -> diag = 1/2
    assert np.isclose(A[0, 0], 0.5, atol=1e-6)
    # interior node 2 has 2 neighbors -> tilde-degree 3 -> diag = 1/3
    assert np.isclose(A[2, 2], 1.0 / 3.0, atol=1e-6)


def test_normalize_adjacency_offdiag_formula():
    # Off-diagonal A_hat[i,j] = 1 / sqrt((deg_i+1)(deg_j+1)) for edge (i,j).
    g = _path_graph(5)
    adj = build_adjacency(g, 5)
    A = normalize_adjacency(adj).to_dense().cpu().numpy()
    # edge (0,1): deg0+1=2, deg1+1=3 -> 1/sqrt(6)
    assert np.isclose(A[0, 1], 1.0 / np.sqrt(6.0), atol=1e-6)


def test_normalize_adjacency_handles_isolated_node():
    # Isolated node: only the self-loop survives; its diagonal must be exactly 1
    # and never NaN/inf (degree -1/2 power guarded).
    g = {0: [1], 1: [0], 2: []}  # node 2 isolated
    adj = build_adjacency(g, 3)
    A = normalize_adjacency(adj).to_dense().cpu().numpy()
    assert np.isfinite(A).all()
    assert np.isclose(A[2, 2], 1.0, atol=1e-6)
