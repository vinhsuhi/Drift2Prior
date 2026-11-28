"""Side-by-side sample grids for several runs at the same classes/seeds, plus what the prior noise itself looks like.

For each run: 1-step and 2-step samples (EMA weights). For each GMM prior: decoded prior draws e and decoded
component means. Also real images of the same classes. Writes PNGs to --out.
usage: python scripts/visualize.py --runs DIR1,DIR2 --out OUTDIR
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2p.dit import build  # noqa: E402
from d2p.meanflow import sample  # noqa: E402
from scripts.train import SCALE, Latents, load_prior  # noqa: E402

CLASSES = [1, 88, 207, 279, 323, 417, 980, 33]  # goldfish, macaw, golden retriever, arctic fox, monarch, balloon, volcano, loggerhead
PER = 6


def grid(imgs, cols):
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    n, h, w, _ = imgs.shape
    rows = (n + cols - 1) // cols
    g = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(imgs):
        g[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
    return Image.fromarray(g).resize((cols * 128, rows * 128), Image.LANCZOS)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/data/horse/ws/vito813i-d2p/data/latents")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(os.environ.get("D2P_VAE", "/data/horse/ws/vito813i-d2p/cache/sd-vae-ft-ema")).to(dev).eval()
    dec = lambda z: torch.cat([vae.decode(c.float() / SCALE).sample for c in z.split(24)])  # noqa: E731
    y = torch.tensor(CLASSES, device=dev).repeat_interleave(PER)
    # real images of the same classes
    data = Latents(a.data)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(data.labs == c)[0], PER, replace=False) for c in CLASSES])
    mom, _ = data.get(idx, rng)
    mom = torch.from_numpy(mom[np.argsort(np.argsort(idx))]).to(dev).float()  # get() sorts idx
    grid(dec(mom[:, :4] * SCALE), PER).save(os.path.join(a.out, "real.png"))
    for run in a.runs.split(","):
        name = os.path.basename(run.rstrip("/"))
        ck = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))[-1]
        d = torch.load(ck, map_location="cpu")
        step = d["step"]
        model = build(d["args"]["model"], tscale=d["args"].get("tscale", 1.0)).to(dev).eval()
        model.load_state_dict(d["ema"])
        prior = load_prior(d["args"]["prior"], dev, state=d.get("prior"))
        for steps in (1, 2):
            g = torch.Generator(device=dev).manual_seed(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, _ = sample(model, prior, y, (4, 32, 32), steps=steps, generator=g)
            grid(dec(z), PER).save(os.path.join(a.out, f"{name}_s{step}_{steps}step.png"))
        g = torch.Generator(device=dev).manual_seed(0)
        e, k = prior.sample(y, (4, 32, 32), generator=g)
        grid(dec(e), PER).save(os.path.join(a.out, f"{name}_prior_draw.png"))
        if prior.K > 1:
            grid(dec(prior.mu[k].view(-1, 4, 32, 32)), PER).save(os.path.join(a.out, f"{name}_prior_mean.png"))
        print(name, step, flush=True)


if __name__ == "__main__":
    main()
