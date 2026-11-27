"""Encode ImageNet-256 parquet shards to SD-VAE latent moments (mean, std), with h-flip copies.

Output per parquet file F: OUT/F.npz-like pair  F_lat.npy (N, 2, 8, 32, 32) fp16 [orig, flip] x [mean(4), std(4)]
                                               F_lab.npy (N,) int16
usage: python scripts/prepare_latents.py --src DIR --out DIR --task I --ntasks N
"""
import argparse
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


def center_crop(img, size=256):
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    if s != 1:
        img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    from diffusers.models import AutoencoderKL

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, f"{a.split}-*.parquet")))[a.task :: a.ntasks]
    vae = AutoencoderKL.from_pretrained(os.environ.get("D2P_VAE", "/data/horse/ws/vito813i-d2p/cache/sd-vae-ft-ema")).cuda().eval()
    for f in files:
        base = os.path.join(a.out, os.path.basename(f).replace(".parquet", ""))
        if os.path.exists(base + "_lab.npy"):
            continue
        tab = pq.read_table(f)
        cols = tab.column_names
        icol = "image" if "image" in cols else cols[0]
        lcol = "label" if "label" in cols else cols[1]
        imgs, labs = tab.column(icol).to_pylist(), np.array(tab.column(lcol).to_pylist(), dtype=np.int16)
        N = len(imgs)
        out = np.zeros((N, 2, 8, 32, 32), dtype=np.float16)
        for i in range(0, N, a.bs):
            batch = []
            for d in imgs[i : i + a.bs]:
                b = d["bytes"] if isinstance(d, dict) else d
                arr = np.asarray(center_crop(Image.open(io.BytesIO(b))), dtype=np.uint8)
                batch.append(arr)
            x = torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            with torch.no_grad():
                for j, xx in enumerate((x, x.flip(3))):
                    dist = vae.encode(xx).latent_dist
                    out[i : i + len(batch), j, :4] = dist.mean.float().cpu().numpy()
                    out[i : i + len(batch), j, 4:] = dist.std.float().cpu().numpy()
        np.save(base + "_lat.tmp.npy", out)
        os.replace(base + "_lat.tmp.npy", base + "_lat.npy")
        np.save(base + "_lab.npy", labs)  # written last = completion marker
        print(f"done {f} N={N}", flush=True)


if __name__ == "__main__":
    main()
