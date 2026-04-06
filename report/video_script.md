# Video Script — COMP0162 Coursework Presentation

**Paper:** Zero Features, Full Signal: Topology-Driven Coordination Analysis in Meme-Token Markets
**Author:** Andrea Nardello
**Duration:** 60 seconds maximum | Face to camera, no slides
**Word count:** 155

---

## Script

**[1 — Financial motivation + why ML is appropriate — ~15s]**

My project investigates coordinated manipulation in Solana's meme-token markets. On platforms like Pump.fun, coordinated wallet groups inflate early trading volume to attract retail capital, then withdraw liquidity. The key challenge is that coordination is a relational phenomenon — it depends on who trades alongside whom and when — which per-token aggregate statistics cannot capture. This makes graph neural networks a natural choice, since they operate directly on the relational structure between wallets and tokens.

**[2 — Neural network architecture — ~13s]**

I implement a featureless heterogeneous Graph Attention Network with approximately sixty-nine thousand parameters. Nodes receive only learnable type embeddings — no hand-crafted features. The architecture uses a two-stage design: the first layer applies sum aggregation to encode neighbourhood degree, and the second layer applies multi-head attention with four heads. The model is trained with weighted binary cross-entropy, AdamW optimiser, cosine annealing, and early stopping on validation AUC.

**[3 — Experimental design — ~12s]**

The dataset is MemeTrans, comprising sixty-five million transactions on Solana from December 2024. I use strict chronological weekly splits — week fifty for training, week fifty-one for validation, week fifty-two for testing — with independent graphs per split to prevent temporal leakage. The null baseline is L2-regularised logistic regression; I also compare against SVM, Random Forest, Gradient Boosting, and MLP, all trained on a hundred and sixteen engineered features. I report AUC-ROC with ninety-five percent bootstrap confidence intervals and McNemar's test for pairwise significance.

**[4 — Key results and interpretation — ~13s]**

All results are evaluated out-of-sample on the held-out test week. The featureless model achieves AUC of zero-point-eight-six-seven, outperforming all five feature-based baselines, including Random Forest at zero-point-eight-five-five — a statistically significant difference confirmed by McNemar's test. When topology and features are combined, AUC reaches zero-point-nine-one-six, but topology alone contributes more marginal gain than a hundred and sixteen engineered features. A temporal extension with Bochner time encoding predicts where coordinators will target next, achieving a Mean Reciprocal Rank of zero-point-four-nine — nearly three times the random baseline.

**[5 — Limitations — ~7s]**

Key limitations include weekly snapshot granularity that misses intra-day dynamics, a seventy-nine percent positive rate that inflates absolute metrics while preserving relative comparisons, and the fact that seventy-one percent of test-week wallets are unseen during training, presenting a cold-start challenge for real-world deployment.

---

## Spec Compliance Checklist

| Requirement | Covered | Where |
|:--|:--:|:--|
| Core financial/economic motivation | Yes | §1: Pump.fun manipulation, retail capital extraction |
| Task under investigation specified | Yes | §1: coordinated manipulation detection |
| Why ML model is appropriate | Yes | §1: relational phenomenon → GNN |
| Model structure | Yes | §2: featureless HeteroGAT, sum-then-attention, 69K params |
| Training objective | Yes | §2: weighted binary cross-entropy |
| Key hyperparameters | Yes | §2: 4 heads, learnable type embeddings |
| Optimisation procedure | Yes | §2: AdamW, cosine annealing, early stopping |
| Dataset used | Yes | §3: MemeTrans, 65M Solana transactions |
| Train/val/test split | Yes | §3: chronological weekly (W50/W51/W52) |
| Baseline model for comparison | Yes | §3: null = LogReg, plus SVM/RF/GB/MLP |
| Performance metrics reported | Yes | §3: AUC-ROC, bootstrap CIs, McNemar |
| Key empirical results | Yes | §4: AUC 0.867 > RF 0.855, featured 0.916, MRR 0.490 |
| In-sample or out-of-sample clarified | Yes | §4: "out-of-sample on the held-out test week" |
| Interpretation of results | Yes | §4: topology contributes more marginal gain than features |
| Limitations acknowledged | Yes | §5: granularity, positive rate, cold-start |
| Overfitting risks | Yes | §2: early stopping on validation AUC |
| Data constraints | Yes | §5: 79% positive rate, weekly granularity |
| Model assumptions | Yes | §5: unseen wallets get identical embeddings |
| Real-world deployment challenges | Yes | §5: cold-start challenge |
| Face to camera, no slides | Yes | Delivery instruction |

---

## Timing Guide

The full script above is ~290 words — intentionally longer than 60 seconds so you can rehearse and **choose what to cut** based on your natural speaking pace. Here is the priority order:

**Must keep (core content):**
- §1: First two sentences (motivation + coordination is relational + why GNN)
- §2: Featureless HeteroGAT, 69K params, sum-then-attention, weighted BCE + early stopping
- §3: MemeTrans 65M, chronological splits, five baselines on 116 features, AUC-ROC
- §4: "Out-of-sample", AUC 0.867 > RF 0.855, McNemar significant, temporal MRR 0.49
- §5: At least two limitations

**Cut first if over time:**
- §3: Remove individual baseline names (just say "five standard classifiers")
- §2: Remove "cosine annealing" (keep AdamW + early stopping)
- §4: Remove "combined AUC 0.916" detail
- §3: Remove "ninety-five percent bootstrap confidence intervals"

**Condensed 60-second version (~150 words):**

> My project investigates coordinated manipulation in Solana's meme-token markets. Wallet groups inflate volume to attract retail capital, then withdraw liquidity. Coordination is relational — who trades alongside whom — which makes graph neural networks a natural fit.
>
> I implement a featureless heterogeneous Graph Attention Network, sixty-nine thousand parameters, using only learnable type embeddings. It uses sum aggregation then multi-head attention, trained with weighted cross-entropy and early stopping on validation AUC.
>
> The dataset is MemeTrans: sixty-five million Solana transactions, split into chronological weekly train, validation, and test sets. I compare against five classifiers trained on a hundred and sixteen engineered features, reporting AUC-ROC with bootstrap confidence intervals.
>
> All results are out-of-sample. The featureless model achieves AUC zero-point-eight-six-seven, outperforming Random Forest at zero-point-eight-five-five, significant by McNemar's test. A temporal extension predicts coordinators' next targets at nearly three times random.
>
> Limitations include weekly granularity missing intra-day dynamics, a high positive rate inflating absolute metrics, and seventy-one percent of test wallets being unseen during training.
