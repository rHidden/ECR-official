# Setup

This repo is the official code of *Towards Empathetic Conversational Recommender Systems* (Zhang et al.,
RecSys 2024) from [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR) at commit `26e2342`, plus these setup files:
`requirements.txt`, `setup_env.sh`, `download_data.sh`, `check_env.py` and this guide.
The authors' code (`src_emo/`, `imdb_review_crawl.py`) is unchanged.

## What you need

- Linux with an NVIDIA GPU and driver 455 or newer. The authors used a 24 GB GPU.
- conda (e.g. [Miniforge](https://github.com/conda-forge/miniforge)).
- About 1–2 GB of disk for the environment, plus the data downloads.

GPU compatibility: the authors pinned PyTorch 1.8.1 with CUDA 11.1, which targets GPUs up to compute capability 8.6
(e.g. RTX 30xx, A100, A40, A6000). Newer GPUs (RTX 40xx, L4/L40, H100) may not run it. `check_env.py` runs a
small GPU test that tells you. Upgrading PyTorch is possible, but results may then differ from the paper.

## Steps

```bash
git clone https://github.com/rHidden/ECR-official.git
cd ECR-official
bash setup_env.sh          # conda env "ecr": Python 3.8.13, PyTorch 1.8.1+cu111, pinned packages, backbones
conda activate ecr
bash download_data.sh      # authors' data (emo_data.zip) and checkpoints (ckpt.zip) from Google Drive
python check_env.py        # versions, GPU test, folders
```

Then follow the Quick-Start in [README.md](README.md), starting with `cd src_emo`.

## Notes and assumptions

- **Backbones.** The scripts load DialoGPT and RoBERTa from `src_emo/save/dialogpt/` and `src_emo/save/roberta/`,
  which the authors don't ship or document. `setup_env.sh` fills them with `microsoft/DialoGPT-small` and
  `roberta-base`, the models used by UniCRS, which ECR builds on. If `ckpt.zip` turns out to contain different
  backbones, use those.
- **wandb** is off unless you pass `--use_wandb`.
- **Package versions.** The authors pinned Python, PyTorch, CUDA, transformers, accelerate and PyG. All other
  versions in `requirements.txt` are our choice of releases from the same period.

## Seeds: what the code actually does

| Script | README seed | How it seeds |
|---|---|---|
| `train_pre.py` (pre-training) | 42 | Only with `--use_new_seed` does it call `seed_torch()` (Python, NumPy, torch, cuDNN disabled). Without it (the README command) it calls accelerate's `set_seed()` and sets `cudnn.benchmark = False`, but not `cudnn.deterministic`, so GPU runs may not repeat exactly. |
| `train_rec.py` (recommendation) | 8 | Always `seed_torch()`: fully seeded, cuDNN disabled. |
| `train_emp.py`, `infer_emp.py` (generation) | default 42 | accelerate's `set_seed()` only. |

For seed experiments:

- Pass `--use_new_seed` to `train_pre.py` for deterministic pre-training.
- Run each seed twice to measure how much results move with the same seed.
- Use one GPU model and a single GPU. Batch-wise AUC is computed per evaluation batch, so changing
  `--per_device_eval_batch_size` or the number of GPUs changes it.

## Upstream

```bash
git remote add upstream https://github.com/zxd-octopus/ECR.git   # if not already set
git fetch upstream
```

The upstream repository has no license file. All credit for the ECR code goes to its authors.
