#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:4 --cpus-per-task=56 --mem=200G --time=6:00:00
#SBATCH --job-name=d2p_fid --output=/data/horse/ws/vito813i-d2p/logs/fid_%j.out
# usage: CKPT=... PRIOR=... OUT=....json N=10000 STEPS=1,2 sbatch slurm/fid.sbatch
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
NG=$(nvidia-smi -L | wc -l)
$D2P/env/bin/torchrun --standalone --nproc_per_node $NG scripts/sample_fid.py --ckpt $CKPT --prior $PRIOR --n ${N:-10000} --steps ${STEPS:-1,2} --out $OUT --save_grid
