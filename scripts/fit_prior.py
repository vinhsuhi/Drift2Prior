"""Fit a source prior on (scaled) latent means and save it.

python scripts/fit_prior.py --mode kmeans --K 1024 --gamma 1.0 --out priors/km1024_g1.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.prior import LatentPrior  # noqa: E402
from scripts.train import SCALE, Latents  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/horse/ws/vito813i-d2p/data/latents")
    ap.add_argument("--mode", required=True, choices=["gauss", "kmeans", "label"])
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--s_min", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = Latents(a.data)
    rng = np.random.default_rng(0)
    idx = rng.choice(data.N, min(a.n, data.N), replace=False)
    mom, y = data.get(idx, rng)
    x = torch.from_numpy(mom[:, :4].astype(np.float32)).cuda().reshape(len(idx), -1) * SCALE
    y = torch.from_numpy(y).cuda()
    K = 1000 if a.mode == "label" else a.K
    p = LatentPrior(K, x.shape[1], 1000).cuda()
    info = p.fit(x, y, a.mode, K, gamma=a.gamma, s_min=a.s_min, iters=a.iters)
    info.update(vars(a))
    info["data_std"] = x.std().item()
    print(json.dumps(info), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dict(K=K, D=x.shape[1], C=1000, state=p.state_dict(), info=info), a.out)


if __name__ == "__main__":
    main()
