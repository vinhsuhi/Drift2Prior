"""DiT backbone for MeanFlow (u(z, r, t | y)).

Adapted from facebookresearch/DiT. Differences:
  * two time conditions: t and (t - r)  (MeanFlow embeds the interval length)
  * plain softmax attention (no SDPA) so torch.func.jvp works through it
"""
import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq=256, tscale=1000.0):
        super().__init__()
        self.freq = freq
        self.tscale = tscale
        self.mlp = nn.Sequential(nn.Linear(freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = (t.float() * self.tscale)[:, None] * f[None]
        emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """num_classes + 1 entries; index num_classes = null (unconditional)."""

    def __init__(self, num_classes, hidden):
        super().__init__()
        self.num_classes = num_classes
        self.table = nn.Embedding(num_classes + 1, hidden)

    def forward(self, y):
        return self.table(y)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (C // h) ** -0.5
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.n1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.n2(x), s2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


def sincos_pos_embed(dim, grid):
    gy, gx = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")

    def emb1d(d, pos):
        om = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2.0))
        out = pos.reshape(-1)[:, None] * om[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    return torch.cat([emb1d(dim // 2, gy), emb1d(dim // 2, gx)], dim=1)


class DiT(nn.Module):
    def __init__(self, input_size=32, patch=2, in_ch=4, dim=768, depth=12, heads=12, num_classes=1000, tscale=1000.0):
        super().__init__()
        self.in_ch, self.patch, self.input_size = in_ch, patch, input_size
        self.num_classes = num_classes
        self.x_embed = nn.Conv2d(in_ch, dim, patch, patch)
        n = (input_size // patch) ** 2
        self.register_buffer("pos", sincos_pos_embed(dim, input_size // patch).unsqueeze(0), persistent=False)
        self.t_emb = TimestepEmbedder(dim, tscale=tscale)
        self.h_emb = TimestepEmbedder(dim, tscale=tscale)  # interval length t - r
        self.y_emb = LabelEmbedder(num_classes, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.final = FinalLayer(dim, patch, in_ch)
        self._init()

    def _init(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embed.bias)
        nn.init.normal_(self.y_emb.table.weight, std=0.02)
        for e in (self.t_emb, self.h_emb):
            nn.init.normal_(e.mlp[0].weight, std=0.02)
            nn.init.normal_(e.mlp[2].weight, std=0.02)
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight)
            nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        c, p = self.in_ch, self.patch
        h = w = self.input_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, z, r, t, y):
        x = self.x_embed(z).flatten(2).transpose(1, 2) + self.pos
        c = self.t_emb(t) + self.h_emb(t - r) + self.y_emb(y)
        for b in self.blocks:
            x = b(x, c)
        return self.unpatchify(self.final(x, c))


CONFIGS = {
    "S/2": dict(dim=384, depth=12, heads=6, patch=2),
    "B/2": dict(dim=768, depth=12, heads=12, patch=2),
    "B/4": dict(dim=768, depth=12, heads=12, patch=4),
    "L/2": dict(dim=1024, depth=24, heads=16, patch=2),
    "XL/2": dict(dim=1152, depth=28, heads=16, patch=2),
}


def build(name, **kw):
    return DiT(**CONFIGS[name], **kw)
