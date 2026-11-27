#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=14 --mem=180G --time=4:00:00
#SBATCH --job-name=d2p_fit --output=/data/horse/ws/vito813i-d2p/logs/fit_%j.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
P=$D2P/priors; mkdir -p $P
$PY scripts/fit_prior.py --mode gauss --K 1 --out $P/gauss.pt
$PY scripts/fit_prior.py --mode kmeans --K 1024 --gamma 1.0 --out $P/km1024.pt
$PY scripts/fit_prior.py --mode kmeans --K 256 --gamma 1.0 --out $P/km256.pt
$PY scripts/fit_prior.py --mode label --out $P/label.pt
