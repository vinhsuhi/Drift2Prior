# Drift2Prior + MeanFlow (branch `meanflow`)

## Goal
One-step (1-NFE) class-conditional ImageNet-256 generation. Original Drift2Prior idea (this repo, `drifting_*.py`):
learn a GMM prior close to the data and a near-identity generator; phase 2 couples each data point with noise from
its own GMM component, so transport is short. New version: the generator is a **MeanFlow** network
(average-velocity u(z, r, t); 1 step: x = e - u(e, 0, 1)), the source is a GMM in SD-VAE latent space, and the
training pair is (x, e ~ N(mu_k(x), diag s_k^2)). Baseline = standard MeanFlow with e ~ N(0, I) at equal compute.
The user lets us modify the core idea freely; decide autonomously.

## Code
- `d2p/dit.py` DiT backbone (t and t-r embeddings, plain attention so `torch.func.jvp` works)
- `d2p/meanflow.py` MeanFlow loss (JVP target, adaptive weighting, logit-normal (t,r), training-time CFG) + sampler
- `d2p/prior.py` `LatentPrior`: modes gauss / kmeans / label; coupling `noise_for(x, y)`; `sample(y)` with p(k|class)
- `scripts/prepare_latents.py` parquet -> SD-VAE (ft-ema) moments, orig + hflip, fp16
- `scripts/fit_prior.py`, `scripts/train.py` (torchrun, manual grad all-reduce), `scripts/sample_fid.py` (FID vs ADM
  VIRTUAL_imagenet256_labeled.npz with torch-fidelity inception-v3-compat), `scripts/toy2d.py` (2D sanity check)
- `slurm/*.sbatch` (source `slurm/common.sh`)

## Paths (Capella, workspace /data/horse/ws/vito813i-d2p, 100 d from Nov27 2026)
data/imagenet256 (HF benjamin-paine/imagenet-1k-256x256 parquet), data/latents, data/fid, env (venv, py3.11),
cache (VAE, inception weights; HF offline on compute nodes), runs/<name>, logs/.

## Rules
- Report raw numbers with the comparison baseline; 1-NFE FID is the headline, 2-NFE secondary.
- Weight decay 0. Keep latest 2 checkpoints + milestone EMA snapshots.
- Watchers: one-shot cron tasks only (no loops/tmux); login node reboots.
- Commit/push when appropriate; stage only intended files.

## Log
- Nov27 2026: project started; data downloaded; code written.
