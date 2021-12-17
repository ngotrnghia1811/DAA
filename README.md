# DAA

Code for **"Unsupervised Domain Adaptation for Event Detection using Domain-specific Adapters"** (ACL-IJCNLP 2021). \
Authors: Nghia Ngo Trung, Duy Phung, Thien Huu Nguyen (VinAI Research; University of Oregon)

[Paper](https://aclanthology.org/2021.findings-acl.351/)

## Overview

Event detection across domains is difficult because fully fine-tuning BERT on a labeled source domain causes it to overfit to that domain, and naive DANN adversarial training makes performance worse rather than better. DAA fixes BERT and instead adds three lightweight bottleneck adapters per layer — one for the source domain, one for the target, and one joint adapter used for classification. A layer-wise domain-adversarial component aligns the joint adapter's representations, while a Wasserstein distance estimator selects the most transferable source samples to use in that alignment. Orthogonality constraints and a shared MLM head keep the private and joint subspaces disentangled and expressive.

## Method

Each BERT layer $l$ produces hidden state $h_l$. Three adapter stacks transform it independently:

```
h_l  →  A^s_l  (source-private)
h_l  →  A^t_l  (target-private)
h_l  →  A^j_l  (joint, used for event classification)
```

Each adapter is a bottleneck: Linear(d→c) + GELU + Linear(c→d) + skip, with $c \ll d$.

**Total loss:**

$$\mathcal{L} = \mathcal{L}_\text{task} + \lambda_d\mathcal{L}_D + \lambda_w\mathcal{L}_W + \lambda_s\mathcal{L}_S + \lambda_m\mathcal{L}_M$$

| Term | Role |
|---|---|
| $\mathcal{L}_\text{task}$ | Cross-entropy event classification on joint repr |
| $\mathcal{L}_D$ | Layer-wise domain-adversarial (asymmetric DANN, $\beta_l = 2^{3-l}$) on down-projected joint bottlenecks |
| $\mathcal{L}_W$ | WGAN-GP Wasserstein distance on private adapter repr; lowest-scoring source samples are selected for DANN |
| $\mathcal{L}_S$ | Frobenius orthogonality between joint and private adapter outputs |
| $\mathcal{L}_M$ | Shared MLM head on joint + private combined repr |

**Training** uses alternating minimization: run $k$ discriminator updates (fixing adapters), then 1 adapter update using Wasserstein-selected source samples with reversed domain labels.

## Setup

```bash
conda create -n daa python=3.9 && conda activate daa
pip install -r requirements.txt
pip install -e .
```

## Data

Raw corpora need to be converted to a unified JSON-lines format. See [data/README.md](data/README.md) for the exact schema.

```bash
bash scripts/run_preprocessing.sh /path/to/ace05 /path/to/timebank /path/to/litbank
```

**Datasets used in the paper:**

- **ACE-05** (LDC2006T06) — 33 event types across 6 domains: `nw`, `bn`, `bc`, `cts`, `wl`, `un`. UDA setting: `bn+nw` → each out-of-domain target.
- **TimeBank** — binary event identification from news text.
- **LitBank** — binary event identification from 100 literary works (Project Gutenberg).

For each experiment, 20% of the target domain's documents are used as unlabeled training data; the remaining 80% are the test set.

## Training

```bash
# Single experiment
python train.py -c configs/ace05_bn2bc.json

# All paper configurations
bash scripts/run_train.sh
```

Checkpoints, logs, and the config snapshot are written to a timestamped subdirectory under `output/`.

Key hyperparameters (set in the config JSON):

| Parameter | Paper value | Description |
|---|:-:|---|
| `adapter_bottleneck_dim` | 96 | Adapter bottleneck dimension $c$ |
| `src_batch_size` | 90 | Source samples per batch |
| `tgt_batch_size` | 60 | Target samples per batch |
| `data_selection_ratio` | 0.67 | Fraction of source batch used for DANN |
| `alt_min_steps_k` | 5 | Discriminator steps per adapter step |
| `lambda_domain` | — | Weight $\lambda_d$; grid-searched in $[0.01, 5]$ |
| `lambda_wasserstein` | — | Weight $\lambda_w$ |
| `lambda_similarity` | — | Weight $\lambda_s$ |
| `lambda_mlm` | — | Weight $\lambda_m$ |

Loss weights are selected by grid search using `bc` as the development domain. Results are averaged over 5 runs with different random seeds.

## Evaluation

```bash
python evaluate.py \
    -c configs/ace05_bn2bc.json \
    --checkpoint output/<run_dir>/best_model.pt
```

## Results

**ACE-05 event detection** (source: `bn+nw`; F1 on test):

| System | `bc` | `cts` | `wl` | `un` |
|---|:-:|:-:|:-:|:-:|
| BERT | 73.5 | 72.4 | 60.6 | 68.2 |
| BERT+DANN | 70.9 | 51.2 | 58.4 | 63.6 |
| BERT+Adapter | 75.6 | 73.2 | 60.3 | 70.6 |
| **DAA** | **76.9** | **75.6** | **63.1** | **72.3** |

**Event identification** (binary; F1 on out-of-domain test):

| System | TimeBank → LitBank | LitBank → TimeBank |
|---|:-:|:-:|
| BERT | 42.2 | 42.7 |
| BERT+DANN | 44.1 | 49.6 |
| **DAA** | **53.6** | **61.1** |

**Ablation** (ACE-05, components removed one at a time):

| System | `bc` | `cts` | `wl` | `un` |
|---|:-:|:-:|:-:|:-:|
| DAA−D (no LDA) | 75.5 | 73.9 | 61.7 | 70.8 |
| DAA−W (no Wasserstein) | 75.6 | 75.1 | 60.9 | 70.1 |
| DAA−M (no MLM) | 75.4 | 73.3 | 61.2 | 71.0 |
| DAA−S (no orthogonality) | 75.4 | 74.7 | 61.5 | 70.2 |
| **DAA (full)** | **76.9** | **75.6** | **63.1** | **72.3** |

## Project Structure

```
DAA/
├── daa/
│   ├── config.py              # Config class + JSON I/O
│   ├── data.py                # EventDataset / UnlabeledDataset
│   ├── scorer.py              # micro P/R/F1
│   ├── trainer.py             # DAATrainer — alternating minimisation loop
│   ├── util.py                # seed, device, logging, MLM masking
│   ├── vocabs.py              # event-type vocabulary
│   ├── models/
│   │   ├── daa_model.py       # full DAA architecture
│   │   └── baselines.py       # BERT, BERT+DANN, BERT+Adapter
│   └── modules/
│       ├── adapter.py         # BottleneckAdapter, DomainAdapterStack
│       ├── bert.py            # FrozenBERTEncoder
│       ├── domain_adversarial.py   # LayerWiseDomainAdversarial
│       ├── wasserstein.py     # WassersteinEstimator + WGAN-GP + data selection
│       └── losses.py          # task / orthogonality / MLM losses
├── configs/                   # JSON experiment configs
├── data/                      # preprocessed data (see data/README.md)
├── preprocessing/             # corpus → JSON-lines converters
├── scripts/                   # shell scripts for training / eval / preprocessing
├── tests/                     # smoke tests (pytest)
├── train.py                   # training entry-point
└── evaluate.py                # evaluation entry-point
```

## Citation

```bibtex
@inproceedings{ngo-etal-2021-unsupervised,
    title     = "Unsupervised Domain Adaptation for Event Detection using Domain-specific Adapters",
    author    = "Ngo Trung, Nghia  and  Phung, Duy  and  Nguyen, Thien Huu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month     = aug,
    year      = "2021",
    address   = "Online",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2021.findings-acl.351",
    pages     = "3986--3996",
}
```

## License

MIT
