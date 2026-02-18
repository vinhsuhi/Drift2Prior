import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Device & seed
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(0)
# np.random.seed(0)

# ------------------------------------------------------------
# Spiral dataset
# ------------------------------------------------------------
def make_spiral(n_samples, noise=0.01, start=0.0, stop=2 * math.pi):
    t = np.linspace(start, stop, n_samples)
    t += np.random.randn(n_samples) * 0.02

    r = 0.5 + 1.5 * (t - t.min()) / (t.max() - t.min() + 1e-8)
    x = r * np.cos(t) + np.random.randn(n_samples) * noise
    y = r * np.sin(t) + np.random.randn(n_samples) * noise

    return np.stack([x, y], axis=1).astype(np.float32)


def make_multi_spiral(n_samples_per_arm, n_arms=2, noise=0.01):
    points = []
    for i in range(n_arms):
        start = i * (2 * math.pi / n_arms)
        stop = start + 2 * math.pi
        arm = make_spiral(
            n_samples_per_arm,
            noise=noise,
            start=start,
            stop=stop,
        )
        points.append(arm)
    return np.concatenate(points, axis=0)


# ------------------------------------------------------------
# Prior sampling
# ------------------------------------------------------------
def sample_prior(n, z_dim=2, device="cpu", empirical_data=None):
    """
    Sample from prior distribution.

    Args:
        n: number of samples
        z_dim: dimension of latent space
        device: torch device
        empirical_data: if provided, sample from this empirical distribution instead of N(0,I)
    """
    if empirical_data is not None:
        # Sample from empirical distribution
        idx = torch.randint(0, empirical_data.size(0), (n,), device=device)
        return empirical_data[idx]
    else:
        # Sample from standard normal
        return torch.randn(n, z_dim, device=device)


# ------------------------------------------------------------
# Generator network
# ------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z):
        return self.net(z)


# ------------------------------------------------------------
# RBF kernel
# ------------------------------------------------------------
def rbf_kernel(x, y, sigma):
    diff = x[:, None, :] - y[None, :, :]
    dist2 = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist2 / (2 * sigma ** 2))


# ------------------------------------------------------------
# Drifting field (Algorithm 1)
# ------------------------------------------------------------
def drifting_field(x, y_pos, y_neg, sigma, remove_self_repulsion=True):
    k_pos = rbf_kernel(x, y_pos, sigma)
    k_neg = rbf_kernel(x, y_neg, sigma)

    if remove_self_repulsion:
        k_neg.fill_diagonal_(0.0)

    w_pos = k_pos / (k_pos.sum(dim=1, keepdim=True) + 1e-8)
    w_neg = k_neg / (k_neg.sum(dim=1, keepdim=True) + 1e-8)

    drift_pos = (w_pos[:, :, None] * (y_pos[None] - x[:, None])).sum(dim=1)
    drift_neg = (w_neg[:, :, None] * (y_neg[None] - x[:, None])).sum(dim=1)

    return drift_pos - drift_neg


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
@torch.no_grad()
def plot_samples(G, real_data, step, n=3000, z_dim=2, use_empirical_prior=False):
    empirical_data = real_data if use_empirical_prior else None
    z = sample_prior(n, z_dim=z_dim, device=real_data.device, empirical_data=empirical_data)
    x = G(z).cpu()

    y = real_data[
        torch.randint(0, real_data.size(0), (n,))
    ].cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # latent space
    axes[0].scatter(z[:, 0].cpu(), z[:, 1].cpu(), s=3, alpha=0.3)
    axes[0].set_title(f"Latent prior (step {step})")
    axes[0].axis("equal")

    # data space
    axes[1].scatter(y[:, 0], y[:, 1], s=5, alpha=0.3, label="real")
    axes[1].scatter(x[:, 0], x[:, 1], s=5, alpha=0.7, label="generated", zorder=5)
    axes[1].legend()
    axes[1].set_title(f"Generated samples (step {step})")
    axes[1].axis("equal")

    fig.tight_layout()
    fig.savefig(f"step_{step}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train(use_empirical_prior=False):
    """
    Train the generator using drifting field.

    Args:
        use_empirical_prior: if True, use target distribution as prior;
                            if False, use standard normal prior
    """
    # fixed empirical dataset
    real_data = torch.from_numpy(
        make_multi_spiral(10000, n_arms=2, noise=0.001)
    ).to(device)

    z_dim = 2
    G = Generator(z_dim=z_dim).to(device)

    opt = optim.Adam(
        G.parameters(),
        lr=1e-3,
    )

    batch_size = 512
    steps = 200000
    sigma = 0.1

    for step in range(steps):
        # real minibatch
        idx = torch.randint(0, real_data.size(0), (batch_size,))
        y = real_data[idx]

        # latent samples
        empirical_data = real_data if use_empirical_prior else None
        z = sample_prior(batch_size, z_dim=z_dim, device=device, empirical_data=empirical_data)

        # generate samples
        x = G(z) # if the generation of G(z) is close to z, 

        # drifting field
        v = drifting_field(
            x=x,
            y_pos=y,
            y_neg=x.detach(),
            sigma=sigma,
        )

        # stop-gradient target
        x_drift = (x + v).detach()

        # loss
        id_loss = ((x - z) ** 2).sum(dim=1).mean()
        drift_loss = ((x - x_drift) ** 2).sum(dim=1).mean() + id_loss

        opt.zero_grad()
        drift_loss.backward()
        opt.step()

        if step % 500 == 0:
            print(
                f"step {step:5d} | "
                f"drift {drift_loss.item():.6f}"
            )

        if step % 2000 == 0:
            plot_samples(G, real_data, step, n=3000, z_dim=z_dim, use_empirical_prior=use_empirical_prior)

    plot_samples(G, real_data, "final", z_dim=z_dim, use_empirical_prior=use_empirical_prior)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Use standard normal prior
    train(use_empirical_prior=True)

    # Or use target distribution as prior
    # train(use_empirical_prior=True)