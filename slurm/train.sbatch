#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:4 --cpus-per-task=56 --mem=400G --time=7-00:00:00
#SBATCH --job-name=d2p_train --output=/data/horse/ws/vito813i-d2p/logs/train_%j.out
# usage: RUN=name PRIOR=path ARGS="..." sbatch slurm/train.sbatch
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
NG=$(nvidia-smi -L | wc -l)
$D2P/env/bin/torchrun --standalone --nproc_per_node $NG scripts/train.py --out $D2P/runs/$RUN --prior $PRIOR $ARGS
