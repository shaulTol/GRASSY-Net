# Cross-Attention Conditioning for Scattering-Guided Molecular Graph Generation

**Course project for Advanced Machine Learning 52025**

This is a minimal fork of [GRASSY-Net](https://github.com/KrishnaswamyLab/GRASSY-Net) (Krishnaswamy Lab), focused on cross-attention conditioning for scattering-guided molecular generation using Graph Diffusion Transformers.

This fork contains only the **DiT component with cross-attention conditioning on scattering moments**, which is the focus of the course project.

## Overview

Geometric scattering transforms produce rich, multiscale molecular fingerprints but are non-invertible. This project replaces the standard AdaLN conditioning in [Graph DiT](https://arxiv.org/abs/2401.13858) with **cross-attention to scattering moment tokens**, allowing graph tokens to attend to structured conditioning signals during generation.

Scattering moments have tensor structure (atom types x wavelet levels x moments). A **dual tokenization** scheme exposes this structure:
- **Atom-type tokens**: one per atom type, capturing cross-scale behavior
- **Level tokens**: one per wavelet level, capturing cross-element patterns

## Setup

```bash
uv venv
uv sync
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## Repository Structure

```
grassy_dit/           # DiT implementation (model, training, sampling)
models/               # ScatteringTransform (fixed wavelet computation)
datasets/             # Data loading utilities
external/moses/       # Validation metrics (validity, FCD, etc.)
utils/                # Config utilities
configs/              # YAML experiment configs (gitignored, see below)
```

## Data Preparation

### 1. Extract scattering moments from a molecular dataset

```bash
python grassy_dit/extract_scattering_fixed.py \
  --dataset datasets/QM9.npy \
  --output grassy_dit/data/qm9 \
  --J 4 --moments 4
```

This produces `scattering_moments.npy` and `molecules.csv` in the output directory.

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `datasets/ZINC12K.npy` | Input `.npy` dataset |
| `--output` | `grassy_dit/data/` | Output directory |
| `--J` | `4` | Number of wavelet scales |
| `--moments` | `4` | Number of statistical moments |

### 2. Split into train/val/test

```bash
python -m grassy_dit.split_datasets \
  --data_dir grassy_dit/data/qm9 \
  --split 0.8 0.1 0.1 \
  --seed 42
```

Creates `train/`, `val/`, `test/` subdirectories, each with `molecules.csv` and `scattering_moments.npy`.

## Training

```bash
python -m grassy_dit.train --config configs/QM9/qm9_baseline.yaml
```

Override any config value from the command line:

```bash
python -m grassy_dit.train --config configs/QM9/qm9_baseline.yaml \
  --override training.epochs=500 model.hidden_size=512
```

### Config Structure

A full config YAML has this structure:

```yaml
dataset:
  name: "qm9"
  data_dir: "path/to/train"
  csv_file: "molecules.csv"
  smiles_col: "smiles"
  scatter_file: "scattering_moments.npy"
  seed: 42
  val_data_dir: "path/to/val"       # optional, enables val-loss checkpointing
  val_csv_file: null                  # null = same filename as csv_file
  val_scatter_file: null              # null = same filename as scatter_file
  val_every_n_epochs: 5

scattering:
  J: 4              # wavelet scales
  num_moments: 4    # statistical moments

model:
  max_node: 29                    # max atoms in dataset
  hidden_size: 1024               # transformer hidden dimension
  num_layer: 12                   # number of transformer blocks
  num_head: 16                    # attention heads
  mlp_ratio: 4.0                  # MLP expansion ratio
  cross_attn_drop: 0.0            # dropout on cross-attention (0.0-1.0)
  cross_attn_bottleneck: null     # bottleneck dim (null = no bottleneck)
  use_fixed_projections: false    # non-learnable orthogonal projections

training:
  epochs: 2000
  batch_size: 64
  learning_rate: 1.0e-4
  l1_lambda: 0.0                  # L1 penalty on cross-attention weights
  conditioning_weight_decay: 0.0  # weight decay on conditioning params only
  weight_decay: 0.0               # global weight decay
  gen_eval_every: 50              # run generation eval every N epochs (optional)

checkpoint:
  save_dir: "./checkpoints/qm9/my_run"
  save_best: true
  save_every_n_epochs: 50
  resume_from: null               # path to resume training

logging:
  wandb:
    enabled: true
    project: "my-project"
    entity: "my-entity"
    name: "run-name"
```

### Regularization Options

Six strategies for controlling cross-attention overfitting:

| Strategy | Config Key | Example | Effect |
|----------|-----------|---------|--------|
| **Cross-attn dropout** | `model.cross_attn_drop` | `0.5` | Drops cross-attention connections during training |
| **L1 regularization** | `training.l1_lambda` | `1e-4` | Penalizes cross-attention weight magnitudes |
| **Bottleneck** | `model.cross_attn_bottleneck` | `32` | Compresses conditioning through low-dim projection |
| **Fixed projections** | `model.use_fixed_projections` | `true` | Non-learnable orthogonal tokenizer projections |
| **Conditioning weight decay** | `training.conditioning_weight_decay` | `0.1` | AdamW weight decay on conditioning params only |
| **Global weight decay** | `training.weight_decay` | `0.01` | AdamW weight decay on all params |

These can be combined freely. Example with dropout + L1:

```yaml
model:
  cross_attn_drop: 0.5
training:
  l1_lambda: 1e-4
```

### Model Sizing Guide

| Dataset | Molecules | `hidden_size` | `num_layer` | `max_node` | Notes |
|---------|-----------|---------------|-------------|------------|-------|
| QM9 | ~130K | 1024 | 12 | 29 | Standard benchmark scale |
| ZINC BBAB | ~13K | 256-384 | 8-12 | 18 | Small dataset, reduce capacity |

## Sampling

### Conditional generation (from scattering moments)

```bash
python -m grassy_dit.sample \
  --checkpoint checkpoints/qm9/best.pt \
  --scattering path/to/scattering_moments.npy \
  --index 0 \
  --num_samples 10 \
  --output generated.txt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Path to trained model `.pt` |
| `--scattering` | required | Path to scattering `.npy` file |
| `--index` | `None` | Index into scattering file (if multiple vectors) |
| `--num_samples` | `10` | Samples per scattering vector |
| `--num_nodes` | `None` | Fixed atom count (`None` = sample from training dist) |
| `--unconditional` | flag | Ignore scattering, generate unconditionally |
| `--output` | `generated.txt` | Output SMILES file |

### Scaffold-constrained generation

Keep a substructure fixed while regenerating the rest:

```bash
python -m grassy_dit.sample \
  --checkpoint model.pt \
  --scattering target.npy \
  --molecule "COc1ccccc1N" \
  --scaffold "c1ccccc1" \
  --num_samples 10
```

### Atom removal and regeneration

Remove specific atoms and let the model fill in:

```bash
python -m grassy_dit.sample \
  --checkpoint model.pt \
  --scattering target.npy \
  --molecule "COc1ccccc1N" \
  --remove-atoms "0,1,2" \
  --num_samples 10
```

### Unconditional generation with metrics

Generate from the prior and compute validity, uniqueness, novelty, FCD:

```bash
python sample_unconstrained.py \
  --checkpoint checkpoints/qm9/best.pt \
  --grassy_checkpoint_dir path/to/grassy_vae_checkpoint/ \
  --training_smiles path/to/train_smiles.txt \
  --n_latent_samples 1000 \
  --output generated_prior.txt
```

## Example Configs

The `configs/` directory is gitignored but contains experiment configs. Key examples:

**Baseline (no regularization):**
```yaml
model:
  hidden_size: 1024
  num_layer: 12
  cross_attn_drop: 0.0
  cross_attn_bottleneck: null
training:
  l1_lambda: 0.0
```

**Dropout sweep (best performing regularization):**
```yaml
model:
  cross_attn_drop: 0.5   # also tried 0.7, 0.9
```

**Bottleneck:**
```yaml
model:
  cross_attn_bottleneck: 32
```

**Combined (all regularization):**
```yaml
model:
  cross_attn_drop: 0.5
  cross_attn_bottleneck: 32
  use_fixed_projections: true
training:
  l1_lambda: 1e-4
```

## Acknowledgments

This project builds on [Graph DiT](https://arxiv.org/abs/2401.13858) and the [GRASSY](https://arxiv.org/abs/2107.08653) scattering framework from the Krishnaswamy Lab.
