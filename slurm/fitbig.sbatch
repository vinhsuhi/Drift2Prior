#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=14 --mem=180G --time=6:00:00
#SBATCH --job-name=d2p_fitbig --output=/data/horse/ws/vito813i-d2p/logs/fitbig_%j.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
P=$D2P/priors
$PY scripts/fit_prior.py --mode kmeans --K 4096 --n 600000 --iters 25 --out $P/km4096.pt
$PY scripts/fit_prior.py --mode kmeans --K 16384 --n 1000000 --iters 20 --out $P/km16384.pt
