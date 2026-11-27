"""MeanFlow objective (Geng et al. 2025, "Mean Flows for One-step Generative Modeling").

z_t = (1 - t) x + t e,  v = e - x  (e from the (learned) prior, coupled to x)
u_tgt = v - (t - r) * d/dt u(z_t, r, t),   d/dt u = v . d_z u + d_t u   (one JVP)
loss  = sg(w) * ||u - sg(u_tgt)||^2,  w = 1 / (||.||^2 + c)^p   (adaptive weighting)
Training-time CFG: v is replaced by  w_cfg * v + kappa * u(z_t,t,t|y) + (1 - w_cfg - kappa) * u(z_t,t,t|null)
for conditional samples (only when t in [cfg_tmin, cfg_tmax]).
"""
import torch


def sample_tr(B, device, mu=-0.4, sigma=1.0, ratio_neq=0.25):
    a = torch.sigmoid(torch.randn(B, device=device) * sigma + mu)
    b = torch.sigmoid(torch.randn(B, device=device) * sigma + mu)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    eq = torch.rand(B, device=device) >= ratio_neq
    r = torch.where(eq, t, r)
    return r, t


def meanflow_loss(model, prior, x, y, null_label, *, drop=0.1, w_cfg=1.0, kappa=0.0,
                  cfg_tmin=0.0, cfg_tmax=1.0, p=1.0, c=1e-3, ratio_neq=0.25, tmu=-0.4, tsig=1.0,
                  alpha=0.0, beta=0.0, lam=0.0, prior_grad="sg"):
    B = x.shape[0]
    e, k = prior.noise_for(x, y)
    joint = prior.is_learnable
    if not joint or prior_grad == "none":
        e_net = e.detach()
    else:
        e_net = e
    r, t = sample_tr(B, x.device, tmu, tsig, ratio_neq)
    sh = (B,) + (1,) * (x.dim() - 1)
    t4, r4 = t.view(sh), r.view(sh)
    z = (1 - t4) * x + t4 * e_net
    v = e.detach() - x
    dropped = torch.rand(B, device=x.device) < drop
    yc = torch.where(dropped, torch.full_like(y, null_label), y)

    if w_cfg != 1.0 or kappa != 0.0:
        with torch.no_grad():
            u_unc = model(z, t, t, torch.full_like(y, null_label))
            u_con = model(z, t, t, y) if kappa != 0.0 else 0.0
            v_g = w_cfg * v + kappa * u_con + (1 - w_cfg - kappa) * u_unc
        apply = (~dropped) & (t >= cfg_tmin) & (t <= cfg_tmax)
        v = torch.where(apply.view(sh), v_g, v)

    def fn(z_, r_, t_):
        return model(z_, r_, t_, yc)

    u, dudt = torch.func.jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    u_tgt = (v - (t4 - r4) * dudt).detach()
    if joint and prior_grad == "full":  # envelope-style gradient: target also depends on e (not on theta)
        u_tgt = u_tgt + (e_net - e.detach())
    err = (u.float() - u_tgt.float()) ** 2
    l = err.flatten(1).sum(1)
    w = 1.0 / (l.detach() + c) ** p
    loss = (w * l).mean()
    info = dict(mse=l.mean().detach() / x[0].numel(), loss=loss.detach())
    if joint:
        # z (hence u) depends on e = mu_k + s_k * eps: L_MF backprops into the prior (as the drift loss did via G(z)).
        transport = ((x - e) ** 2).flatten(1).mean(1).mean()      # identity-loss analogue (path length)
        ent = prior.entropy_reg()                                  # -log det analogue (anti-collapse)
        nll = prior.nll(x, k) if lam > 0 else torch.zeros((), device=x.device)
        loss = loss + alpha * transport - beta * ent + lam * nll
        prior.update_pk(k, y)
        info.update(transport=transport.detach(), mean_log_std=ent.detach())
    return loss, info


@torch.no_grad()
def sample(model, prior, y, shape, steps=1, generator=None):
    """steps=1: x = e - u(e, 0, 1); steps>1: uniform intervals."""
    e, k = prior.sample(y, shape, generator=generator)
    z = e
    ts = torch.linspace(1.0, 0.0, steps + 1, device=y.device)
    for i in range(steps):
        t = torch.full((y.shape[0],), ts[i].item(), device=y.device)
        r = torch.full((y.shape[0],), ts[i + 1].item(), device=y.device)
        z = z - (t - r).view((-1,) + (1,) * (z.dim() - 1)) * model(z, r, t, y)
    return z, k
