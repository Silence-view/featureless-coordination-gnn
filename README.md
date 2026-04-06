# Zero Features, Full Signal: Topology-Driven Coordination Analysis in Meme-Token Markets

COMP0162 Advanced Machine Learning in Finance, UCL 2025/26

How much of the information needed to identify and anticipate coordinated
manipulation is encoded in wallet-token interaction topology, independent of
hand-crafted features? We model Solana's meme-token ecosystem as a heterogeneous
graph from ~65.5M transactions (MemeTrans dataset, 3 weeks Dec 2024) and answer
under three views: **detection** (featureless HeteroGAT, AUC=0.867),
**prediction** (temporal link prediction, MRR=0.490), and **characterisation**
(unsupervised archetype discovery via HDBSCAN).

## Quick start

```bash
pip install -r requirements.txt
python run_experiments.py --step all        # build → train → evaluate
```

Individual steps:

```bash
python run_experiments.py --step build      # construct heterogeneous graphs from parquet
python run_experiments.py --step train      # train FeaturelessHeteroGAT (classification)
python run_experiments.py --step evaluate   # AUC, F1, bootstrap CIs, McNemar
python run_experiments.py --step analyse    # HDBSCAN clustering + t-SNE
python run_experiments.py --step figures    # generate all figures
```

Additional scripts (in `scripts/`):

```bash
python scripts/run_ablation.py              # edge-type ablation study
python scripts/run_multiseed.py             # multi-seed stability analysis
python scripts/run_sensitivity.py           # wallet threshold sensitivity
```

All hyperparameters live in `params.yaml`.

## Project structure

```
Coursework/
├── params.yaml                 # All hyperparameters (model, training, graph, temporal)
├── run_experiments.py          # Main pipeline entry point
├── Makefile                    # make all / build / train / evaluate / clean
├── requirements.txt            # Python dependencies
├── src/
│   ├── data/
│   │   ├── loader.py           # Parquet I/O from BigQuery exports
│   │   ├── features.py         # 116 token features (tabular baselines only)
│   │   ├── wallet_features.py  # 15 wallet behavioural features
│   │   ├── edge_builder.py     # Temporally-decayed co-trade + same-tx edges
│   │   ├── graph.py            # HeteroData assembly (4 edge types)
│   │   └── temporal_edges.py   # Chronological event extraction for link pred
│   ├── models/
│   │   ├── bochner.py          # Bochner time encoding (learnable Fourier features)
│   │   ├── temporal_gat.py     # Temporal HeteroGAT (~314K params, link prediction)
│   │   ├── featureless_gat.py  # Static FeaturelessHeteroGAT (~69K params, classification)
│   │   ├── featured_hetero_gat.py  # FeaturedHeteroGAT (~86K params, feature ablation)
│   │   ├── link_decoder.py     # Link prediction decoder + negative sampling
│   │   └── baselines.py        # LogReg, SVM, RF, GB, MLP tabular baselines
│   ├── train.py                # Classification training loop (early stopping, cosine LR)
│   ├── train_temporal.py       # Temporal link prediction training loop
│   ├── evaluate.py             # AUC, F1, AP, bootstrap CIs, McNemar's test
│   ├── link_eval.py            # MRR, Hits@K for temporal link prediction
│   ├── early_detection.py      # Time-horizon evaluation (1h / 6h / 24h / 3d / 7d)
│   ├── analysis.py             # HDBSCAN clustering + t-SNE embeddings
│   └── visualise.py            # Figure generation (ROC, training curves, t-SNE)
├── scripts/
│   ├── run_ablation.py         # Edge-type ablation experiments
│   ├── run_multiseed.py        # Multi-seed stability (seeds 42, 123, 7)
│   └── run_sensitivity.py      # Wallet threshold sensitivity analysis
├── report/
│   ├── main_v2.tex             # Final 6-page paper (IEEEtran journal)
│   ├── references.bib          # 21 entries, 17 cited
│   ├── video_script.md         # 60-second video script
│   └── figures/
│       ├── architecture.tex    # TikZ architecture diagram
│       ├── architecture.pdf
│       ├── tsne_embeddings.pdf # Wallet embedding t-SNE visualisation
│       └── early_detection.pdf # Signal emergence over time
└── results/                    # Generated at runtime
    ├── graphs/                 # Serialised PyG HeteroData objects
    ├── checkpoints/            # Model weights (.pt)
    ├── predictions/            # Per-node predictions
    ├── analysis/               # Embedding arrays + cluster labels
    ├── figures/                # Generated plots (ROC, ablation, t-SNE, etc.)
    ├── evaluation_results.json
    ├── all_baselines.json
    ├── ablation_results.json
    ├── early_detection_results.json
    ├── featured_gat_results.json
    ├── multiseed_results.json
    └── paper_improvements_results.json
```

## Data

Transaction data queried from Google BigQuery (Solana public dataset), stored as
weekly parquet exports in `../SolEye/data/raw/`. Token labels from MemeTrans:

> Hu et al., "MemeTrans: Meme Token Risk Prediction via Multi-modal Feature
> Fusion on the Solana Blockchain", arXiv:2602.13480, 2026.

Three consecutive weeks for **chronological** train/val/test (no random splits):

| Split | Period | Transactions | Active wallets | Labelled tokens |
|-------|--------|:------------:|:--------------:|:---------------:|
| Train | W50 (Dec 9--15)  | 22.6M | 42,233 | 2,679 |
| Val   | W51 (Dec 16--22) | 22.5M | 46,082 | 2,906 |
| Test  | W52 (Dec 23--29) | 20.4M | 44,661 | 2,538 |

High-risk tokens outnumber low-risk ~3.7:1 across all splits. Only active
wallets (trading >=20 distinct tokens) are retained (~4% of wallets, responsible
for the majority of inter-token coordination).

## Approach

The heterogeneous graph has 4 edge types encoding a coordination spectrum:

- **trade / reverse_trade** — bipartite wallet-token interactions (~1.9M/split)
- **co_trade** — temporally-decayed synchronisation between wallet pairs (~660K)
- **same_tx** — wallets bundled in a single on-chain transaction (~55K)

Three experimental views from one research question:

1. **Detection** — FeaturelessHeteroGAT (sum-then-attention, 69K params, zero features)
2. **Prediction** — Temporal HeteroGAT with Bochner time encoding (314K params)
3. **Characterisation** — HDBSCAN on learned embeddings (25 clusters, 3 archetypes)

## Results

### Coordination detection (classification, test week 52)

| Model | Features | AUC-ROC | F1 | AP |
|-------|:--------:|:-------:|:--:|:--:|
| Logistic Regression | 116 | .841 [.825, .857] | .887 | .955 |
| SVM (RBF) | 116 | .834 [.817, .849] | .805 | .953 |
| Random Forest | 116 | .855 [.840, .870] | .878 | .960 |
| Gradient Boosting | 116 | .849 [.835, .863] | .878 | .959 |
| MLP | 116 | .850 [.835, .865] | .890 | .958 |
| **FeaturelessHeteroGAT** | **0** | **.867** [.852, .882] | **.899** | .960 |
| FeaturedHeteroGAT | 131 | **.916** [.905, .926] | **.918** | **.977** |

Multi-seed stability: AUC = 0.869 +/- 0.002 across seeds {42, 123, 7}.
McNemar's test: featureless GAT vs RF (chi2=7.3, p=0.007), vs LogReg (chi2=13.6, p<0.001).

### Temporal link prediction (1 positive + 19 negatives per query)

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|:---:|:------:|:------:|:-------:|
| Random | .180 | .050 | .150 | .500 |
| EdgeBank | .056 | .005 | .007 | .009 |
| Popularity | .056 | .002 | .005 | .024 |
| Cosine Similarity | .250 | .078 | .268 | .695 |
| **Temporal HeteroGAT** | **.490** | **.296** | **.581** | **.952** |

### Edge-type ablation

| Configuration | AUC | Delta |
|---------------|:---:|:-----:|
| Full graph (4 types) | .867 | -- |
| - co_trade | .868 | +.001 |
| - all W-W edges | .863 | -.004 |
| - same_tx | .820 | **-.047** |

### Early detection (AUC vs observation time)

| Model | 1h | 6h | 24h | 3d | 7d |
|-------|:--:|:--:|:---:|:--:|:--:|
| LogReg (116 feat) | .841 | .841 | .841 | .841 | .841 |
| RF (116 feat) | .855 | .855 | .855 | .855 | .855 |
| FeaturelessHeteroGAT | .500 | .500 | .542 | .611 | .841 |
| % edges available | 0.1% | 1% | 8% | 34% | 95% |

All metrics evaluated out-of-sample on the held-out test week with 95%
bootstrap confidence intervals (n=1,000 resamples).
