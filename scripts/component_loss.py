"""Test 1 (motivation for joint learning): is there generator-specific signal per GMM component?

For a trained checkpoint + prior: draw N data latents, compute the per-sample (unweighted) MeanFlow residual
Delta = ||u - u_tgt||^2 / D at random (r,t), and the one-step error ||x_hat - x||^2/D where x_hat = e - u(e,0,1)
for the coupled e. Group by component k and report how much of the per-component loss variation is explained
by what a fit-first prior already knows (component size, within-component variance, distance to global mean).
If per-component loss is large, varies a lot and is NOT explained by variance -> the generator has signal
that likelihood fitting ignores.
usage: python scripts/component_loss.py --ckpt C --prior P --n 50000 --out OUT.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.dit import build  # noqa: E402
from d2p.meanflow import sample_tr  # noqa: E402
from scripts.train import SCALE, Latents, load_prior  # noqa: E402


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res.var() / y.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--data", default="/data/horse/ws/vito813i-d2p/data/latents")
    ap.add_argument("--num", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    d = torch.load(a.ckpt, map_location="cpu")
    model = build(d["args"]["model"], tscale=d["args"].get("tscale", 1.0)).to(dev).eval()
    model.load_state_dict(d["ema"])
    prior = load_prior(a.prior, dev, state=d.get("prior"))
    data = Latents(a.data)
    rng = np.random.default_rng(1)
    idx = rng.choice(data.N, a.num, replace=False)
    ks, dl, one, tt = [], [], [], []
    torch.manual_seed(0)
    for i in range(0, a.num, a.bs):
        mom, y = data.get(idx[i : i + a.bs], rng)
        mom = torch.from_numpy(mom).to(dev).float()
        x = (mom[:, :4] + mom[:, 4:] * torch.randn_like(mom[:, 4:])) * SCALE
        y = torch.from_numpy(y).to(dev)
        with torch.no_grad():
            e, k = prior.noise_for(x, y)
        B = x.shape[0]
        r, t = sample_tr(B, dev)
        sh = (B, 1, 1, 1)
        z = (1 - t.view(sh)) * x + t.view(sh) * e
        v = e - x
        with torch.autocast("cuda", dtype=torch.bfloat16):
            u, dudt = torch.func.jvp(lambda z_, r_, t_: model(z_, r_, t_, y), (z, r, t),
                                     (v, torch.zeros_like(r), torch.ones_like(t)))
            with torch.no_grad():
                xh = e - model(e, torch.zeros_like(t), torch.ones_like(t), y)
        tgt = v - (t - r).view(sh) * dudt
        dl.append(((u.float() - tgt.float()) ** 2).flatten(1).mean(1).detach().cpu())
        one.append(((xh.float() - x) ** 2).flatten(1).mean(1).cpu())
        ks.append(k.cpu())
        tt.append(t.cpu())
    k = torch.cat(ks).numpy()
    dl, one = torch.cat(dl).numpy(), torch.cat(one).numpy()
    K = prior.K
    cnt = np.bincount(k, minlength=K)
    use = cnt >= 20
    mean_dl = np.bincount(k, dl, K)[use] / cnt[use]
    mean_one = np.bincount(k, one, K)[use] / cnt[use]
    var_k = (prior.std.detach().cpu().numpy() ** 2).mean(1)[use]
    mu_n = prior.mu.detach().cpu().numpy()
    dist = np.linalg.norm(mu_n - mu_n.mean(0), axis=1)[use] / np.sqrt(mu_n.shape[1])
    X = np.column_stack([var_k, np.log(cnt[use]), dist])
    # split-half reliability of per-component loss (is the variation real or noise?)
    half = np.arange(len(k)) % 2 == 0
    m1 = np.bincount(k[half], one[half], K)[use] / np.maximum(np.bincount(k[half], minlength=K)[use], 1)
    m2 = np.bincount(k[~half], one[~half], K)[use] / np.maximum(np.bincount(k[~half], minlength=K)[use], 1)
    res = dict(ckpt=a.ckpt, prior=a.prior, n=a.num, components_used=int(use.sum()),
               mf_resid_mean=float(dl.mean()), onestep_err_mean=float(one.mean()),
               onestep_err_comp_cv=float(mean_one.std() / mean_one.mean()),
               onestep_err_comp_split_half_corr=float(np.corrcoef(m1, m2)[0, 1]),
               r2_onestep_by_fitfirst_stats=float(r2(mean_one, X)),
               r2_onestep_by_variance_only=float(r2(mean_one, var_k[:, None])),
               r2_mfresid_by_fitfirst_stats=float(r2(mean_dl, X)),
               onestep_err_over_var_cv=float((mean_one / var_k).std() / (mean_one / var_k).mean()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
