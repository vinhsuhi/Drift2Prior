#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=14 --mem=120G --time=2:00:00
#SBATCH --job-name=d2p_smoke2 --output=/data/horse/ws/vito813i-d2p/logs/smoke2_%j.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
for cfg in "1.0 1000" "1.0 1" "2.0 1000" "2.0 1"; do
  set -- $cfg
  R=$D2P/runs/smoke2_${SLURM_JOB_ID}_w$1_ts$2
  echo "=== w_cfg $1 tscale $2"
  $D2P/env/bin/torchrun --standalone --nproc_per_node 1 scripts/train.py --out $R --prior $D2P/priors/gauss.pt \
    --model B/2 --bs 64 --steps 2000 --w_cfg $1 --tscale $2 --ckpt_every 2000 --snap_every 2000 --log_every 200 --ema 0.999
done
R=$D2P/runs/smoke2_${SLURM_JOB_ID}_w1.0_ts1000
$D2P/env/bin/torchrun --standalone --nproc_per_node 1 scripts/sample_fid.py --ckpt $R/ema_0002000.pt \
  --prior $D2P/priors/gauss.pt --num 1000 --steps 1,2 --out $R/fid.json --save_grid
