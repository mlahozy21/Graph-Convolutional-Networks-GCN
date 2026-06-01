# GCN: Semi-Supervised Classification with Graph Convolutional Networks

From-scratch implementation of the GCN model from Kipf & Welling (ICLR 2017) for the GRMDIL course project: paper reproduction, an MLP baseline, and over-smoothing experiments on Cora, Citeseer, and Pubmed.

## Project structure

```
.
├── pyproject.toml          # package metadata and dependencies
├── requirements.txt        # dependencies (alternative to pip install -e .)
├── README.md  LICENSE  .gitignore
├── src/gcn/                # installable library package
│   ├── data.py             # dataset download/loading (Cora, Citeseer, Pubmed)
│   ├── model.py            # GCN layer, GCN, MLP baseline, DeepGCN, ResidualDeepGCN
│   ├── train.py            # training and evaluation pipeline
│   └── utils.py            # reproducibility helpers (set_seed)
├── scripts/                # entry points
│   ├── main.py             # reproduce paper results + MLP baseline comparison
│   └── experiments.py      # over-smoothing experiments (varying depth)
└── docs/
    ├── report.pdf          # project report
    ├── report.tex
    └── references.bib
```

## Installation

```bash
git clone https://github.com/mlahozy21/Graph-Convolutional-Networks-GCN.git
cd Graph-Convolutional-Networks-GCN

python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -e .                   # installs the gcn package and its dependencies
```

`pip install -e .` reads the dependencies from `pyproject.toml`. The Planetoid datasets are downloaded automatically on first run.

## Usage

Run all commands from the repository root.

### Reproduce paper results (GCN + MLP baseline on all datasets)
```bash
python scripts/main.py
```

### Run on a single dataset
```bash
python scripts/main.py --dataset cora
```

### Run over-smoothing experiments
```bash
python scripts/experiments.py
```

## Experiments

1. **Reproduction**: GCN on Cora, Citeseer, Pubmed — compared with Table 2 of the paper.
2. **MLP baseline**: same architecture without graph structure — compared with Table 3, showing the contribution of graph convolutions.
3. **Over-smoothing**: GCN with increasing depth (2, 4, 8, 16 layers) — shows performance degradation with depth, and how residual connections partially mitigate it.

Runs are seeded per repetition (`set_seed(run)`), so results are reproducible while the mean/std still reflect run-to-run variation.

## Results

Test accuracy (%) on the citation-network benchmarks, mean ± std over repeated seeded runs,
next to the figures reported by Kipf & Welling (2017). The from-scratch GCN reproduces the
paper within run-to-run variation, and the gap over the MLP baseline confirms the value of
the graph structure.

| Dataset  | MLP (this work) | **GCN (this work)** | GCN (Kipf 2017) |
|----------|:---------------:|:-------------------:|:---------------:|
| Cora     | 56.9 ± 1.2 | **81.6 ± 0.7** | 81.5 |
| Citeseer | 55.2 ± 1.4 | **71.3 ± 0.5** | 70.3 |
| Pubmed   | 72.1 ± 0.7 | **78.6 ± 0.5** | 79.0 |

The GCN beats the MLP by ~25 pts on Cora but only ~6 pts on Pubmed, consistent with Pubmed's
tf-idf features being already highly informative on their own.

**Over-smoothing.** Stacking more layers sharply degrades accuracy — on Cora, 82.7% (2 layers)
→ 15.8% (16 layers); on Citeseer 71.4% → 20.4%; on Pubmed 78.6% → 38.7%. Residual connections
partially mitigate it (e.g. Pubmed at 8 layers: 65.2% with residuals vs. 39.5% without).


## Reference

Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017. https://arxiv.org/abs/1609.02907

## License

Released under the MIT License — see `LICENSE`.
