#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=12:00:00
#SBATCH --job-name=d2p_theory --output=/data/horse/ws/vito813i-d2p/logs/theory_%j.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
$PY scripts/theory_check.py --out $D2P/runs/theory --ratio_neq ${NEQ:-0.0}
