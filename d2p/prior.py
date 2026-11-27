"""Learned source priors for MeanFlow (the Drift2Prior idea in latent space).

The original Drift2Prior learned a GMM prior *in data space* together with a
near-identity generator, then (phase 2) coupled every data point y with noise
drawn from the component y belongs to, so the generator only has to transport
mass a short distance.  Here the generator is a MeanFlow network; the coupling
is the same: x ~ data, k = assign(x), eps ~ N(mu_k, diag(s_k^2)), z_t =
(1-t) x + t eps.  At sampling time k ~ p(k | class) and one network evaluation
maps eps to an image latent.

modes
  gauss   : K=1, mu=0, s=1           (standard MeanFlow baseline)
  kmeans  : K clusters fitted by k-means on data latents;
            s_k = gamma * within-cluster std (per dim) floored at s_min
  label   : one component per class (mu_c = class mean), same scale rule
"""
import torch
import torch.nn as nn


class LatentPrior(nn.Module):
    def __init__(self, K, D, num_classes):
        super().__init__()
        self.register_buffer("mu", torch.zeros(K, D))
        self.register_buffer("std", torch.ones(K, D))
        # p(k | class); row num_classes = unconditional marginal
        self.register_buffer("pk", torch.full((num_classes + 1, K), 1.0 / K))
        self.register_buffer("by_label", torch.tensor(False))
        self.K, self.D, self.C = K, D, num_classes

    # ---------------- fitting ----------------
    @torch.no_grad()
    def assign(self, x, y=None, chunk=4096):
        """x: (B, D) flattened latents -> component index (B,)"""
        if bool(self.by_label):
            return y.clone()
        if self.K == 1:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        mu = self.mu.detach().float()
        m2 = (mu * mu).sum(1)
        out = []
        for i in range(0, x.shape[0], chunk):
            xi = x[i : i + chunk].float()
            d = m2[None] - 2 * xi @ mu.T  # ||x||^2 constant
            out.append(d.argmin(1))
        return torch.cat(out)

    @torch.no_grad()
    def fit(self, x, y, mode, K, gamma=1.0, s_min=0.0, iters=30, seed=0):
        """x: (N, D) float32 on GPU, y: (N,) labels"""
        g = torch.Generator(device=x.device).manual_seed(seed)
        N, D = x.shape
        if mode == "gauss":
            assert K == 1
            self.mu.zero_()
            self.std.fill_(1.0)
            self.pk.fill_(1.0)
            return {}
        if mode == "label":
            self.by_label.fill_(True)
            assert K == self.C
            a = y
        else:
            mu = x[torch.randperm(N, generator=g, device=x.device)[:K]].clone()
            self.mu.copy_(mu)
            for it in range(iters):
                a = self.assign(x)
                s = torch.zeros(K, D, device=x.device).index_add_(0, a, x)
                n = torch.bincount(a, minlength=K).float()
                empty = n == 0
                new = s / n.clamp(min=1)[:, None]
                if empty.any():  # re-seed empty clusters
                    new[empty] = x[torch.randint(0, N, (int(empty.sum()),), generator=g, device=x.device)]
                self.mu.copy_(new)
            a = self.assign(x)
        n = torch.bincount(a, minlength=K).float()
        s = torch.zeros(K, D, device=x.device).index_add_(0, a, x)
        mu = s / n.clamp(min=1)[:, None]
        sq = torch.zeros(K, D, device=x.device).index_add_(0, a, x * x)
        var = (sq / n.clamp(min=1)[:, None] - mu**2).clamp(min=0)
        # components with few points: fall back to global per-dim variance
        gvar = x.var(0)
        few = n < 8
        var[few] = gvar
        std = (gamma * var.sqrt()).clamp(min=s_min)
        self.mu.copy_(mu)
        self.std.copy_(std)
        cnt = torch.zeros(self.C + 1, K, device=x.device)
        cnt.index_put_((y, a), torch.ones_like(y, dtype=torch.float), accumulate=True)
        cnt[self.C] = n
        self.pk.copy_((cnt + 1e-6) / (cnt + 1e-6).sum(1, keepdim=True))
        within = ((x - mu[a]) ** 2).sum(1).mean().item()
        total = ((x - x.mean(0)) ** 2).sum(1).mean().item()
        return dict(within_var=within, total_var=total, frac=within / total, mean_std=std.mean().item(),
                    occupied=int((n > 0).sum()))

    # ---------------- joint learning ----------------
    def make_learnable(self, fix_volume=False):
        """Turn mu / std into trainable parameters (joint learning with the generator).
        fix_volume: keep each component's mean log std (log det / D) at its initial value, so the MeanFlow loss
        can move means and reshape covariances but cannot shrink the noise (collapse-proof by construction)."""
        mu, std = self.mu.clone(), self.std.clone()
        del self._buffers["mu"], self._buffers["std"]
        self.mu_p = nn.Parameter(mu)
        ls = std.clamp(min=1e-4).log()
        self.log_std = nn.Parameter(ls)
        self.register_buffer("vol", ls.mean(1))
        self.register_buffer("fix_volume", torch.tensor(bool(fix_volume)))
        self.learnable = True
        return self

    def eff_log_std(self):
        ls = self._parameters["log_std"]
        if bool(self._buffers["fix_volume"]):
            ls = ls - ls.mean(1, keepdim=True) + self._buffers["vol"][:, None]
        return ls

    @property
    def is_learnable(self):
        return getattr(self, "learnable", False)

    def __getattr__(self, name):
        if name in ("mu", "std"):
            d = self.__dict__.get("_parameters", {})
            if "mu_p" in d:
                return d["mu_p"] if name == "mu" else self.eff_log_std().exp()
        return super().__getattr__(name)

    def entropy_reg(self):
        """mean log std over used components (diag-Gaussian log det / D)."""
        return self.eff_log_std().mean() if self.is_learnable else torch.zeros(())

    def nll(self, x, k):
        """hard-EM negative log-likelihood of data under its assigned component, per dim (KL(p_data||q) anchor)."""
        B = x.shape[0]
        xf = x.reshape(B, -1).detach()
        ls = self.eff_log_std()[k]
        return (ls + 0.5 * ((xf - self.mu[k]) / ls.exp()) ** 2).mean()

    @torch.no_grad()
    def update_pk(self, k, y, momentum=0.999):
        """online p(k | class) from current assignments (EMA of counts)."""
        cnt = torch.zeros_like(self.pk)
        cnt.index_put_((y, k), torch.ones_like(y, dtype=cnt.dtype), accumulate=True)
        cnt[self.C].index_add_(0, k, torch.ones_like(k, dtype=cnt.dtype))
        if not hasattr(self, "_cnt"):
            self._cnt = self.pk.clone() * 50.0
        self._cnt.mul_(momentum).add_(cnt)
        self.pk.copy_((self._cnt + 1e-6) / (self._cnt + 1e-6).sum(1, keepdim=True))

    # ---------------- sampling ----------------
    def noise_for(self, x, y):
        """coupled noise for training: x (B,C,H,W) latents. Differentiable in mu/std if learnable."""
        B = x.shape[0]
        k = self.assign(x.reshape(B, -1), y)
        e = torch.randn_like(x).reshape(B, -1)
        e = self.mu[k] + self.std[k] * e
        return e.reshape_as(x), k

    def sample(self, y, shape, generator=None):
        """noise for generation given labels y (null label allowed)."""
        B = y.shape[0]
        k = torch.multinomial(self.pk[y], 1, generator=generator).squeeze(1)
        e = torch.randn((B, self.D), device=y.device, generator=generator)
        return (self.mu[k] + self.std[k] * e).detach().reshape(B, *shape), k
