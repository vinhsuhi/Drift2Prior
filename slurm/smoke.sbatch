#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=14 --mem=120G --time=1:00:00
#SBATCH --job-name=d2p_smoke --output=/data/horse/ws/vito813i-d2p/logs/smoke_%j.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
R=$D2P/runs/smoke_$SLURM_JOB_ID
$D2P/env/bin/torchrun --standalone --nproc_per_node 1 scripts/train.py --out $R --prior $D2P/priors/km1024.pt \
  --model B/2 --bs 64 --steps 300 --w_cfg 2.0 --ckpt_every 300 --snap_every 300 --log_every 50 --joint --lam 1.0 && \
$D2P/env/bin/torchrun --standalone --nproc_per_node 1 scripts/sample_fid.py --ckpt $R/ema_0000300.pt \
  --prior $D2P/priors/km1024.pt --n 500 --steps 1,2 --out $R/fid.json --save_grid
