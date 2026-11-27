"""Empirical check of the collapse analysis (docs/theory.md).

Data x ~ N(0, I_D) (each dim var 1); prior N(mu, diag s^2), K=1, learned jointly with a MeanFlow MLP.
Two-timescale: network lr >> prior lr (theory assumes a near-optimal network).
Runs a grid of (prior-gradient mode, regulariser, init sigma) and records the trajectory of mean s and |mu|.
usage: python scripts/theory_check.py --out DIR
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.meanflow import meanflow_loss  # noqa: E402
from d2p.prior import LatentPrior  # noqa: E402


class MLP(nn.Module):
    def __init__(self, D, h=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(D + 32, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def emb(self, t):
        f = torch.exp(torch.arange(8, device=t.device) * math.log(2.0))
        a = t[:, None] * f[None] * math.pi
        return torch.cat([a.sin(), a.cos()], 1)

    def forward(self, z, r, t, y):
        return self.f(torch.cat([z, self.emb(t), self.emb(t - r)], 1))


def run(D, mode, alpha, beta, lam, s0, mu0, steps, ratio_neq, dev, seed=0, fixvol=False):
    torch.manual_seed(seed)
    prior = LatentPrior(1, D, 1).to(dev)
    prior.std.fill_(s0)
    prior.mu.fill_(mu0)
    prior.make_learnable(fix_volume=fixvol)
    net = MLP(D).to(dev)
    opt = torch.optim.Adam([dict(params=net.parameters(), lr=1e-3), dict(params=prior.parameters(), lr=3e-4)])
    traj = []
    for s in range(steps + 1):
        x = torch.randn(1024, D, device=dev)
        y = torch.zeros(1024, dtype=torch.long, device=dev)
        loss, _ = meanflow_loss(net, prior, x, y, 0, drop=0.0, p=0.0, ratio_neq=ratio_neq,
                                alpha=alpha, beta=beta, lam=lam, prior_grad=mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            prior.log_std.clamp_(-7, 3)
        if s % (steps // 20) == 0:
            traj.append((s, prior.std.mean().item(), prior.mu.abs().mean().item()))
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ratio_neq", type=float, default=0.0, help="0 = pure flow-matching part (what the theory covers)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = []
    for s0 in (0.6, 1.0, 1.4):
        grid += [("sg", 0, 0, 0, s0), ("full", 0, 0, 0, s0), ("sg", 0, 0.2, 0, s0),
                 ("sg", 0, 0, 0.3, s0), ("sg", 0, 0, 1.0, s0), ("sg", 0, 0, 2.0, s0)]
    res = []
    for mode, al, be, la, s0 in grid:
        tr = run(a.D, mode, al, be, la, s0, 0.5, a.steps, a.ratio_neq, dev)
        row = dict(mode=mode, alpha=al, beta=be, lam=la, s0=s0, final_s=tr[-1][1], final_absmu=tr[-1][2], traj=tr)
        print(f"mode={mode:4s} a={al} b={be} lam={la} s0={s0}: s {s0} -> {tr[-1][1]:.3f}  |mu| .5 -> {tr[-1][2]:.3f}", flush=True)
        res.append(row)
    json.dump(res, open(os.path.join(a.out, f"theory_check_neq{a.ratio_neq}.json"), "w"))


if __name__ == "__main__":
    main()
