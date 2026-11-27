"""DDP training of MeanFlow on ImageNet-256 SD-VAE latents with a (learned) source prior.

torchrun --nproc_per_node G scripts/train.py --out DIR --prior PRIOR.pt [--model B/2] ...
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.dit import build  # noqa: E402
from d2p.meanflow import meanflow_loss  # noqa: E402
from d2p.prior import LatentPrior  # noqa: E402

SCALE = 0.18215


class Latents:
    def __init__(self, root):
        lats = sorted(glob.glob(os.path.join(root, "train-*_lat.npy")))
        assert lats, root
        self.arrs = [np.load(f, mmap_mode="r") for f in lats]
        self.labs = np.concatenate([np.load(f.replace("_lat", "_lab")).astype(np.int64) for f in lats])
        self.off = np.cumsum([0] + [len(x) for x in self.arrs])
        self.N = int(self.off[-1])

    def get(self, idx, rng):
        idx = np.sort(idx)
        out = np.empty((len(idx), 8, 32, 32), dtype=np.float16)
        flip = rng.integers(0, 2, len(idx))
        f = np.searchsorted(self.off, idx, side="right") - 1
        for i, (fi, ii, fl) in enumerate(zip(f, idx, flip)):
            out[i] = self.arrs[fi][ii - self.off[fi], fl]
        return out, self.labs[idx]


def load_prior(path, device, state=None):
    d = torch.load(path, map_location="cpu")
    p = LatentPrior(d["K"], d["D"], d["C"])
    st = state if state is not None else d["state"]
    if "mu_p" in st:
        p.make_learnable()
    p.load_state_dict(st, strict=False)
    return p.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/data/horse/ws/vito813i-d2p/data/latents")
    ap.add_argument("--prior", required=True)
    ap.add_argument("--model", default="B/2")
    ap.add_argument("--bs", type=int, default=64, help="per GPU")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--w_cfg", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--cfg_tmin", type=float, default=0.0)
    ap.add_argument("--cfg_tmax", type=float, default=1.0)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--ratio_neq", type=float, default=0.25)
    ap.add_argument("--tmu", type=float, default=-0.4)
    ap.add_argument("--tsig", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=10000)
    ap.add_argument("--snap_every", type=int, default=50000, help="milestone EMA snapshots kept forever")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint", action="store_true", help="learn the GMM prior jointly (Drift2Prior)")
    ap.add_argument("--alpha", type=float, default=0.0, help="transport (identity-analogue) weight")
    ap.add_argument("--beta", type=float, default=0.0, help="entropy (log-det) weight")
    ap.add_argument("--prior_lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.0, help="NLL anchor weight")
    ap.add_argument("--fix_volume", action="store_true")
    ap.add_argument("--prior_grad", default="sg", choices=["sg", "full", "none"])
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    torch.manual_seed(a.seed + rank)
    rng = np.random.default_rng(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)

    data = Latents(a.data)
    prior = load_prior(a.prior, dev)
    if a.joint and not prior.is_learnable:
        prior.make_learnable(fix_volume=a.fix_volume)
    model = build(a.model).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    groups = [dict(params=list(model.parameters()), lr=a.lr)]
    if prior.is_learnable:
        groups.append(dict(params=list(prior.parameters()), lr=a.prior_lr))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    step = 0
    ck = sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")))
    if ck:
        d = torch.load(ck[-1], map_location="cpu")
        model.load_state_dict(d["model"])
        ema.load_state_dict(d["ema"])
        opt.load_state_dict(d["opt"])
        if "prior" in d:
            prior.load_state_dict(d["prior"], strict=False)
        step = d["step"]
        rng = np.random.default_rng(a.seed + rank + 1000 * step)
        if rank == 0:
            print(f"resumed {ck[-1]} step {step}", flush=True)
    params = [p for g in groups for p in g["params"]]
    for p in params:
        dist.broadcast(p.data, 0)
    if rank == 0:
        json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
        print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M N={data.N} world={world}", flush=True)
    null = model.num_classes
    t0, hist = time.time(), []
    while step < a.steps:
        idx = rng.integers(0, data.N, a.bs)
        mom, y = data.get(idx, rng)
        mom = torch.from_numpy(mom).to(dev, non_blocking=True).float()
        x = (mom[:, :4] + mom[:, 4:] * torch.randn_like(mom[:, 4:])) * SCALE
        y = torch.from_numpy(y).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, info = meanflow_loss(model, prior, x, y, null, drop=a.drop, w_cfg=a.w_cfg, kappa=a.kappa,
                                       cfg_tmin=a.cfg_tmin, cfg_tmax=a.cfg_tmax, p=a.p,
                                       ratio_neq=a.ratio_neq, tmu=a.tmu, tsig=a.tsig,
                                       alpha=a.alpha, beta=a.beta, lam=a.lam, prior_grad=a.prior_grad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:  # manual grad all-reduce (jvp runs on the raw module, not DDP)
            flat = torch.cat([p.grad.reshape(-1) for p in params])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            o = 0
            for p in params:
                n = p.numel()
                p.grad.copy_(flat[o:o + n].view_as(p.grad))
                o += n
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        if prior.is_learnable:
            with torch.no_grad():
                prior.log_std.clamp_(min=-9.0)
        opt.step()
        step += 1
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        hist.append([info["mse"].item(), info["loss"].item(), gn.item(),
                     info.get("transport", torch.zeros(())).item(), info.get("mean_log_std", torch.zeros(())).item()])
        if step % a.log_every == 0 and rank == 0:
            h = np.array(hist)
            dt = time.time() - t0
            print(f"step {step} mse {h[:,0].mean():.4f} loss {h[:,1].mean():.4f} gnorm {h[:,2].mean():.3f} "
                  f"transport {h[:,3].mean():.4f} logstd {h[:,4].mean():.3f} "
                  f"{a.log_every * a.bs * world / dt:.0f} img/s", flush=True)
            hist, t0 = [], time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            if rank == 0:
                d = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                         args=vars(a), prior=prior.state_dict())
                p = os.path.join(a.out, f"ckpt_{step:07d}.pt")
                torch.save(d, p + ".tmp")
                os.replace(p + ".tmp", p)
                if step % a.snap_every == 0:
                    torch.save(dict(ema=ema.state_dict(), step=step, args=vars(a), prior=prior.state_dict()),
                               os.path.join(a.out, f"ema_{step:07d}.pt"))
                for old in sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")))[:-2]:
                    os.remove(old)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
