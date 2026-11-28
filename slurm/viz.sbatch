#!/bin/bash
#SBATCH --account=p_biomol_gen --partition=capella --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1 --cpus-per-task=14 --mem=120G --time=0:45:00
#SBATCH --job-name=d2p_viz --output=/data/horse/ws/vito813i-d2p/logs/viz_%j.out
# usage: RUNS=dir1,dir2 OUT=dir sbatch slurm/viz.sbatch
source /home/h9/vito813i/Drift2Prior/slurm/common.sh
$PY scripts/visualize.py --runs $RUNS --out $OUT
