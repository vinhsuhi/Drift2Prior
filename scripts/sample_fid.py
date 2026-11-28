"""Generate class-conditional samples with an EMA checkpoint and compute FID vs the ADM ImageNet-256 reference.

torchrun --nproc_per_node G scripts/sample_fid.py --ckpt CKPT --prior PRIOR --n 10000 --steps 1,2
Inception: torch-fidelity 'inception-v3-compat' (the TF-FID port), pool3 2048 features.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.dit import build  # noqa: E402
from d2p.meanflow import sample  # noqa: E402
from scripts.train import SCALE, load_prior  # noqa: E402

REF = "/data/horse/ws/vito813i-d2p/data/fid/VIRTUAL_imagenet256_labeled.npz"


def fid(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--steps", default="1,2")
    ap.add_argument("--bs", type=int, default=125)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_grid", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0, help="sampling-time scale on prior std (fixed priors only)")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    from diffusers.models import AutoencoderKL
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    d = torch.load(a.ckpt, map_location="cpu")
    mname = a.model or d["args"]["model"]
    model = build(mname, tscale=d["args"].get("tscale", 1.0)).to(dev).eval()
    model.load_state_dict(d["ema"])
    prior = load_prior(a.prior, dev, state=d.get("prior"))
    if a.temp != 1.0:
        assert not prior.is_learnable
        prior.std.mul_(a.temp)
    vae = AutoencoderKL.from_pretrained(os.environ.get("D2P_VAE", "/data/horse/ws/vito813i-d2p/cache/sd-vae-ft-ema")).to(dev).eval()
    inc = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(dev).eval()
    ref = np.load(REF)
    mu_r, s_r = ref["mu"], ref["sigma"]
    res = dict(ckpt=a.ckpt, prior=a.prior, n=a.num, step=d.get("step"))
    labels_all = torch.arange(a.num) % 1000
    mine = labels_all[rank::world]
    for st in [int(s) for s in a.steps.split(",")]:
        g = torch.Generator(device=dev).manual_seed(1234 + rank)
        feats = []
        for i in range(0, len(mine), a.bs):
            y = mine[i : i + a.bs].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z, _ = sample(model, prior, y, (4, 32, 32), steps=st, generator=g)
            with torch.no_grad():
                img = vae.decode(z.float() / SCALE).sample
                img = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                if a.save_grid and i == 0 and rank == 0 and a.out:
                    import torchvision
                    torchvision.utils.save_image(img[:64].float() / 255, a.out.replace(".json", f"_s{st}.png"), nrow=8)
                feats.append(inc(img)[0].double().cpu())
        f = torch.cat(feats)
        allf = [None] * world
        dist.all_gather_object(allf, f)
        if rank == 0:
            F = torch.cat(allf).numpy()
            v = fid(F.mean(0), np.cov(F, rowvar=False), mu_r, s_r)
            res[f"fid_{st}step"] = v
            print(f"steps={st} N={len(F)} FID={v:.3f}", flush=True)
    if rank == 0:
        print(json.dumps(res))
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
