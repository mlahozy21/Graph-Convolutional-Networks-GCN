"""Smoke tests: GCN forward pass works and a train step decreases loss."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from gcn.data import build_adjacency, normalize_adjacency
from gcn.model import GCN
from gcn.utils import set_seed


def _synthetic_problem(n_per_class=15, n_features=8, seed=0):
    """Two well-separated clusters connected as two cliques (easy 2-class task)."""
    set_seed(seed)
    n = 2 * n_per_class
    # features: class 0 centered at -1, class 1 at +1
    feats = np.random.randn(n, n_features).astype(np.float32) * 0.3
    feats[:n_per_class] -= 1.0
    feats[n_per_class:] += 1.0
    labels = np.array([0] * n_per_class + [1] * n_per_class)

    # graph: two intra-class cliques
    g = {i: [] for i in range(n)}
    for group in (range(n_per_class), range(n_per_class, n)):
        nodes = list(group)
        for a in nodes:
            for b in nodes:
                if a != b:
                    g[a].append(b)
    adj = build_adjacency(g, n)
    A = normalize_adjacency(adj)
    X = torch.tensor(feats)
    y = torch.tensor(labels, dtype=torch.long)
    return X, A, y, n


def test_gcn_forward_shape():
    X, A, y, n = _synthetic_problem()
    model = GCN(X.shape[1], 16, 2, dropout=0.0)
    out = model(X, A)
    assert out.shape == (n, 2)
    assert torch.isfinite(out).all()


def test_gcn_train_step_decreases_loss():
    X, A, y, n = _synthetic_problem(seed=1)
    set_seed(1)
    model = GCN(X.shape[1], 16, 2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

    model.train()
    out0 = model(X, A)
    loss0 = F.cross_entropy(out0, y).item()

    for _ in range(40):
        opt.zero_grad()
        out = model(X, A)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(model(X, A), y).item()
        acc = (model(X, A).argmax(1) == y).float().mean().item()

    assert final_loss < loss0, f"loss did not decrease: {loss0} -> {final_loss}"
    assert acc > 0.9, f"GCN failed to separate easy clusters (acc={acc})"
