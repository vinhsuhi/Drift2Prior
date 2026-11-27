#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=6:00:00
#SBATCH --job-name=d2p_toy --output=/data/horse/ws/vito813i-d2p/logs/toy_%j.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
$PY scripts/toy2d.py --out $D2P/runs/toy2d_${SLURM_JOB_ID} --steps ${STEPS:-20000} --Ks 1,8,32 --gammas ${GAMMAS:-1.0}
