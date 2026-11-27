"""2D sanity check: one-step MeanFlow from N(0,I) vs from a k-means GMM prior.

Reports a sliced-Wasserstein distance between 1-step samples and data, and saves a figure.
usage: python scripts/toy2d.py --out DIR [--steps 20000]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.meanflow import meanflow_loss, sample  # noqa: E402
from d2p.prior import LatentPrior  # noqa: E402


def make_data(name, n, rng):
    if name == "checker":
        x = rng.uniform(-2, 2, (n * 3, 2))
        keep = ((np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2) == 0
        return x[keep][:n]
    if name == "pinwheel":
        k = 5
        r = rng.uniform(0.1, 2.0, n)
        th = r * 0.8 * np.pi / 2.0 + rng.integers(0, k, n) * 2 * np.pi / k
        return np.stack([r * np.cos(th), r * np.sin(th)], 1) + rng.normal(0, 0.05, (n, 2))
    if name == "spiral":
        t = rng.uniform(0, 3 * np.pi, n)
        arm = rng.integers(0, 2, n) * np.pi
        r = 0.2 + t / (3 * np.pi) * 1.8
        return np.stack([r * np.cos(t + arm), r * np.sin(t + arm)], 1) + rng.normal(0, 0.03, (n, 2))
    raise ValueError(name)


class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 + 2 * 16, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def emb(self, t):
        f = torch.exp(torch.arange(8, device=t.device) * math.log(2.0))
        a = t[:, None] * f[None] * math.pi
        return torch.cat([a.sin(), a.cos()], 1)

    def forward(self, z, r, t, y):
        return self.f(torch.cat([z, self.emb(t), self.emb(t - r)], 1))


def swd(a, b, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(n, 2, generator=g)
    th = th / th.norm(dim=1, keepdim=True)
    pa, pb = (a @ th.T).sort(0)[0], (b @ th.T).sort(0)[0]
    return (pa - pb).abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--data", default="checker,pinwheel,spiral")
    ap.add_argument("--Ks", default="1,8,32,128")
    ap.add_argument("--gammas", default="1.0")
    ap.add_argument("--joint", default="0,1,2,3", help="0=fixed GMM, 1=pure MF backprop, 2=pure+fixed volume, 3=MF+NLL(lam)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for dname in a.data.split(","):
        rng = np.random.default_rng(0)
        data = torch.tensor(make_data(dname, 50000, rng), dtype=torch.float32, device=dev)
        test = torch.tensor(make_data(dname, 5000, np.random.default_rng(1)), dtype=torch.float32)
        configs = [(int(K), float(g), int(j)) for K in a.Ks.split(",")
                   for g in (a.gammas.split(",") if int(K) > 1 else ["1.0"])
                   for j in (a.joint.split(",") if int(K) > 1 else ["0"])]
        fig, axs = plt.subplots(1, len(configs) + 1, figsize=(3 * (len(configs) + 1), 3))
        axs[0].scatter(test[:, 0], test[:, 1], s=1)
        axs[0].set_title(dname)
        for j, (K, g, jt) in enumerate(configs):
            torch.manual_seed(0)
            prior = LatentPrior(K, 2, 1).to(dev)
            y = torch.zeros(data.shape[0], dtype=torch.long, device=dev)
            info = prior.fit(data, y, "gauss" if K == 1 else "kmeans", K, gamma=g)
            if jt:
                prior.make_learnable(fix_volume=(jt == 2))
            net = MLP().to(dev)
            groups = [dict(params=list(net.parameters()), lr=1e-3)]
            if jt:
                groups.append(dict(params=list(prior.parameters()), lr=1e-3))
            opt = torch.optim.Adam(groups)
            for s in range(a.steps):
                idx = torch.randint(0, data.shape[0], (1024,), device=dev)
                yb = torch.zeros(1024, dtype=torch.long, device=dev)
                loss, _ = meanflow_loss(net, prior, data[idx], yb, 0, drop=0.0, alpha=a.alpha, beta=a.beta, lam=(a.lam if jt == 3 else 0.0))
                opt.zero_grad()
                loss.backward()
                opt.step()
            if jt:
                info = dict(info, final_mean_std=prior.std.mean().item())
            res = {}
            for st in (1, 2):
                xs, _ = sample(net, prior, torch.zeros(5000, dtype=torch.long, device=dev), (2,), steps=st)
                res[st] = swd(xs.cpu(), test)
            xs, _ = sample(net, prior, torch.zeros(5000, dtype=torch.long, device=dev), (2,), steps=1)
            xs = xs.cpu()
            axs[j + 1].scatter(xs[:, 0], xs[:, 1], s=1)
            axs[j + 1].set_title(f"K{K} g{g} {['fixed','pure','fixvol','nll'][jt]} swd1 {res[1]:.3f}", fontsize=8)
            row = f"{dname} K={K} gamma={g} joint={jt} swd_1step={res[1]:.4f} swd_2step={res[2]:.4f} {info}"
            print(row, flush=True)
            rows.append(row)
        for ax in axs:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{dname}.png"), dpi=100)
    open(os.path.join(a.out, "results.txt"), "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
