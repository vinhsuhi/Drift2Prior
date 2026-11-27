#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=14 --mem=96G --time=12:00:00
#SBATCH --array=0-7 --job-name=d2p_enc --output=/data/horse/ws/vito813i-d2p/logs/enc_%A_%a.out
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
$PY scripts/prepare_latents.py --src $D2P/data/imagenet256 --out $D2P/data/latents --task $SLURM_ARRAY_TASK_ID --ntasks 8
