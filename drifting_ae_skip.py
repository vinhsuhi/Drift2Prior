import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.patches import Ellipse
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA

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


def make_checkerboard(n, noise=0.05, grid=4):
    """Create a checkerboard pattern dataset."""
    pts = []
    cell = 4.0 / grid
    for i in range(grid):
        for j in range(grid):
            if (i + j) % 2 == 0:
                pts.append(
                    np.random.uniform(
                        [-2 + i * cell, -2 + j * cell],
                        [-2 + (i + 1) * cell, -2 + (j + 1) * cell],
                        (n // (grid**2 // 2), 2),
                    )
                )
    c = np.vstack(pts)
    np.random.shuffle(c)
    return (c[:n] + np.random.randn(n, 2) * noise).astype(np.float32)


def make_pinwheel(n, n_arms=5, noise=0.05):
    """Create a pinwheel pattern dataset."""
    per_arm = n // n_arms
    pts = []
    for k in range(n_arms):
        r = np.linspace(0.1, 2.0, per_arm) + np.random.randn(per_arm) * 0.02
        theta = np.linspace(0, 0.8 * np.pi, per_arm) + k * 2 * np.pi / n_arms
        pts.append(
            np.column_stack(
                [
                    r * np.cos(theta) + np.random.randn(per_arm) * noise,
                    r * np.sin(theta) + np.random.randn(per_arm) * noise,
                ]
            )
        )
    c = np.vstack(pts)
    np.random.shuffle(c)
    return c[:n].astype(np.float32)


def make_rings(n, noise=0.03):
    """Create concentric rings dataset."""
    X, _ = make_circles(n, noise=noise, factor=0.5, random_state=42)
    return (X * 2).astype(np.float32)


def make_swissroll2d(n, noise=0.05):
    """Create a 2D Swiss roll dataset."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    return (
        np.column_stack([t * np.cos(t) / 6, t * np.sin(t) / 6]).astype(np.float32)
        + np.random.randn(n, 2).astype(np.float32) * noise
    )


# ------------------------------------------------------------
# Learnable GMM prior (uniform weights, full covariance)
# ------------------------------------------------------------
class GMMPrior(nn.Module):
    def __init__(self, k=8, z_dim=2, init_data=None):
        super().__init__()
        self.k = k
        self.z_dim = z_dim

        if init_data is not None:
            # Initialize from data statistics
            self._init_from_data(init_data)
        else:
            # Default initialization
            self.means = nn.Parameter(torch.randn(k, z_dim) * 0.5)
            self.chol_factors = nn.Parameter(
                torch.eye(z_dim).unsqueeze(0).repeat(k, 1, 1) * 1.0
            )

    def _init_from_data(self, data):
        """
        Initialize GMM parameters based on dataset statistics.

        Args:
            data: (N, z_dim) tensor of data points
        """
        device = data.device

        # Compute data statistics
        data_std = data.std(dim=0)
        data_min = data.min(dim=0)[0]
        data_max = data.max(dim=0)[0]

        # Initialize means: sample random points from data + small perturbation
        # This ensures components start near actual data
        indices = torch.randperm(data.size(0))[: self.k]
        init_means = data[indices].clone()

        # Add small random perturbation to avoid exact duplicates
        perturbation = torch.randn_like(init_means) * 0.1 * data_std
        init_means = init_means + perturbation

        self.means = nn.Parameter(init_means)

        # Initialize covariance based on data spread
        # Use a fraction of the data variance to start with compact clusters
        init_scale = (data_std.mean() * 0.3).item()  # Start with 30% of data std

        # Initialize Cholesky factors as scaled identity
        init_chol = torch.eye(self.z_dim, device=device) * init_scale
        self.chol_factors = nn.Parameter(init_chol.unsqueeze(0).repeat(self.k, 1, 1))

        print(f"Initialized GMM prior from data:")
        print(f"  Data range: [{data_min.cpu().numpy()}, {data_max.cpu().numpy()}]")
        print(f"  Data std: {data_std.cpu().numpy()}")
        print(f"  Initial cluster scale: {init_scale:.4f}")
        print(
            f"  Mean positions range: [{self.means.min().item():.2f}, {self.means.max().item():.2f}]"
        )

    def covariance(self):
        """Return full covariance matrices for each component via Cholesky decomposition."""
        # Covariance = L @ L^T where L is lower triangular
        L = torch.tril(self.chol_factors)  # Ensure lower triangular
        cov = torch.bmm(L, L.transpose(1, 2))
        return cov

    def sample(self, n):
        # uniform mixture weights
        comp = torch.randint(0, self.k, (n,), device=self.means.device)

        # Standard normal samples
        eps = torch.randn(n, self.z_dim, device=self.means.device)

        # Transform using Cholesky factor: z = mu + L @ eps
        L = torch.tril(self.chol_factors[comp])  # (n, z_dim, z_dim)
        z = self.means[comp] + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return z

    def variance_penalty(self):
        """Penalize the determinant (volume) of covariance matrices."""
        # det(Cov) = det(L @ L^T) = det(L)^2 = (prod of diagonal of L)^2
        L = torch.tril(self.chol_factors)
        diag = torch.diagonal(L, dim1=1, dim2=2)  # (k, z_dim)
        log_det = 2 * torch.log(torch.abs(diag) + 1e-8).sum(dim=1)  # (k,)
        return log_det.mean()

    def compactness_penalty(self):
        """
        Penalize large clusters by penalizing the trace of covariance matrices.
        Trace = sum of eigenvalues = sum of variances along principal axes.
        Smaller trace = more compact clusters.
        """
        covs = self.covariance()  # (k, z_dim, z_dim)
        traces = torch.diagonal(covs, dim1=1, dim2=2).sum(dim=1)  # (k,)
        return traces.mean()

    def frobenius_penalty(self):
        """
        Penalize large clusters using Frobenius norm of covariance matrices.
        ||Σ||_F^2 = trace(Σ^T Σ) = sum of all squared elements.
        More aggressive than trace penalty.
        """
        covs = self.covariance()  # (k, z_dim, z_dim)
        frob_norm_sq = (covs**2).sum(dim=(1, 2))  # (k,)
        return frob_norm_sq.mean()

    def max_eigenvalue_penalty(self):
        """
        Penalize the maximum eigenvalue (largest spread direction).
        Controls the longest axis of the ellipsoid.
        """
        covs = self.covariance()  # (k, z_dim, z_dim)
        eigvals = torch.linalg.eigvalsh(covs)  # (k, z_dim), sorted ascending
        max_eigvals = eigvals[:, -1]  # (k,)
        return max_eigvals.mean()

    def distance_penalty(self):
        """Penalize small pairwise distances between component means."""
        diff = self.means[:, None, :] - self.means[None, :, :]
        dist2 = (diff**2).sum(dim=-1)
        # mask out diagonal (self-distances)
        mask = ~torch.eye(self.k, dtype=torch.bool, device=self.means.device)
        return (1.0 / (dist2[mask] + 1e-4)).sum()

    @torch.no_grad()
    def assign_to_components(self, data):
        """
        Assign each data point to the Gaussian component with highest likelihood.

        Args:
            data: (N, z_dim) tensor of data points

        Returns:
            assignments: (N,) tensor of component indices
        """
        # Compute log probability of each data point under each component
        # log p(x|k) = -0.5 * (x-mu_k)^T Sigma_k^{-1} (x-mu_k) - 0.5 * log|Sigma_k| - d/2 * log(2*pi)

        N = data.size(0)
        log_probs = torch.zeros(N, self.k, device=data.device)

        covs = self.covariance()  # (k, z_dim, z_dim)

        for k in range(self.k):
            diff = data - self.means[k]  # (N, z_dim)

            # Compute inverse and determinant via Cholesky
            cov_k = covs[k]
            try:
                L = torch.linalg.cholesky(
                    cov_k + 1e-6 * torch.eye(self.z_dim, device=data.device)
                )
                log_det = 2 * torch.log(torch.diag(L)).sum()

                # Solve L^T x = diff^T for Mahalanobis distance
                v = torch.linalg.solve_triangular(L, diff.T, upper=False)
                mahal_dist = (v**2).sum(dim=0)

                log_probs[:, k] = -0.5 * (
                    mahal_dist + log_det + self.z_dim * np.log(2 * np.pi)
                )
            except:
                # Fallback to simple Euclidean distance if Cholesky fails
                dist2 = (diff**2).sum(dim=1)
                log_probs[:, k] = -dist2

        # Assign to component with highest log probability
        assignments = torch.argmax(log_probs, dim=1)
        return assignments

    def sample_from_components(self, component_indices):
        """
        Sample from specific Gaussian components.

        Args:
            component_indices: (N,) tensor of component indices

        Returns:
            z: (N, z_dim) samples from the specified components
        """
        n = component_indices.size(0)
        eps = torch.randn(n, self.z_dim, device=self.means.device)

        # Transform using Cholesky factor: z = mu + L @ eps
        L = torch.tril(self.chol_factors[component_indices])  # (n, z_dim, z_dim)
        z = self.means[component_indices] + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return z


# ------------------------------------------------------------
# Generator network
# ------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),  # Output same dimension as input
        )

        self.skip_weight = nn.Parameter(
            torch.tensor(1.0)
        )  # Learnable skip connection weight

        self.linear_projection = nn.Linear(
            z_dim, z_dim
        )  # Linear projection for skip connection

    def forward(self, z):
        # Add skip connection to encourage identity mapping
        return self.net(z) + self.linear_projection(z) * self.skip_weight


# ------------------------------------------------------------
# Kernels
# ------------------------------------------------------------


def kernel(x, y, kernel_type, **kwargs):
    if kernel_type == "rbf":
        return rbf_kernel(x, y, **kwargs)
    elif kernel_type == "l2":
        return l2_kernel(x, y, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def rbf_kernel(x, y, sigma):
    diff = x[:, None, :] - y[None, :, :]
    dist2 = (diff**2).sum(dim=-1)
    return torch.exp(-dist2 / (2 * sigma**2))


def l2_kernel(x, y, sigma):
    diff = x[:, None, :] - y[None, :, :]
    dist = torch.norm(diff, dim=-1)
    return torch.exp(-dist / sigma)


# ------------------------------------------------------------
# Drifting field (Algorithm 1)
# ------------------------------------------------------------
def drifting_field(
    x, y_pos, y_neg, sigma, remove_self_repulsion=True, kernel_type="rbf"
):
    k_pos = kernel(x, y_pos, kernel_type, sigma=sigma)
    k_neg = kernel(x, y_neg, kernel_type, sigma=sigma)

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
def plot_samples(
    G, prior, real_data, step, n=3000, use_pca=False, folder_name="", custom_name=None
):
    z = prior.sample(n)
    x = G(z).cpu()

    y = real_data[torch.randint(0, real_data.size(0), (n,))].cpu()

    means = prior.means.cpu().numpy()
    covs = prior.covariance().cpu().numpy()

    # If high-dimensional, use PCA to project to 2D for visualization
    if use_pca and z.shape[1] > 2:
        pca = PCA(n_components=2)

        # Fit PCA on real data only (provides more meaningful reference frame)
        pca.fit(y.numpy())

        # Project to 2D
        z_2d = pca.transform(z.cpu().numpy())
        x_2d = pca.transform(x.numpy())
        y_2d = pca.transform(y.numpy())
        means_2d = pca.transform(means)

        # Note: Can't easily visualize full covariance in 2D after PCA
        plot_ellipses = False
    else:
        z_2d = z.cpu().numpy()
        x_2d = x.numpy()
        y_2d = y.numpy()
        means_2d = means
        plot_ellipses = True

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # latent space
    axes[0].scatter(z_2d[:, 0], z_2d[:, 1], s=3, alpha=0.3)
    axes[0].scatter(means_2d[:, 0], means_2d[:, 1], s=80, c="red", marker="x")

    if plot_ellipses and z.shape[1] == 2:
        for k in range(prior.k):
            # 1-std ellipse from eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(covs[k])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            w, h = 2 * np.sqrt(eigvals)  # 1-std axes
            axes[0].add_patch(
                Ellipse(
                    means[k],
                    w,
                    h,
                    angle=angle,
                    fill=False,
                    color="red",
                    alpha=0.5,
                )
            )

    # Add dimension info and component count to title
    dim_info = f"dim={prior.z_dim}, k={prior.k}"
    title_suffix = (
        f" (PCA from dim {dim_info})"
        if use_pca and z.shape[1] > 2
        else f" ({dim_info})"
    )

    axes[0].set_title(f"Latent prior (step {step}){title_suffix}")
    axes[0].axis("equal")

    # data space
    axes[1].scatter(y_2d[:, 0], y_2d[:, 1], s=5, alpha=0.3, label="real")
    axes[1].scatter(x_2d[:, 0], x_2d[:, 1], s=5, alpha=0.7, label="generated", zorder=5)
    axes[1].legend()
    axes[1].set_title(f"Generated samples (step {step}){title_suffix}")
    axes[1].axis("equal")

    fig.tight_layout()
    if custom_name is not None:
        fig.savefig(
            os.path.join(folder_name, f"{custom_name}_step_{step}.png"), dpi=150
        )
    else:
        fig.savefig(os.path.join(folder_name, f"step_{step}.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# High-dimensional embedding
# ------------------------------------------------------------
def embed_to_high_dim(data_2d, target_dim=1024, seed=None):
    """
    Embed 2D data into high-dimensional space via random rotation.

    Args:
        data_2d: (N, 2) numpy array
        target_dim: target dimensionality
        seed: random seed for reproducibility

    Returns:
        embedded_data: (N, target_dim) numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    N = data_2d.shape[0]

    # Pad with zeros: (N, 2) -> (N, target_dim)
    data_padded = np.zeros((N, target_dim), dtype=np.float32)
    data_padded[:, :2] = data_2d

    # Generate random orthogonal matrix via QR decomposition
    random_matrix = np.random.randn(target_dim, target_dim).astype(np.float32)
    Q, _ = np.linalg.qr(random_matrix)  # Q is orthogonal

    # Apply rotation
    data_rotated = data_padded @ Q.T

    print(f"Embedded 2D data into {target_dim}D space:")
    print(f"  Original data shape: {data_2d.shape}")
    print(f"  Embedded data shape: {data_rotated.shape}")
    print(f"  Data norm (before): {np.linalg.norm(data_2d, axis=1).mean():.4f}")
    print(f"  Data norm (after): {np.linalg.norm(data_rotated, axis=1).mean():.4f}")
    print(
        f"  Non-zero entries: {(np.abs(data_rotated) > 1e-6).sum() / data_rotated.size * 100:.1f}%"
    )

    return data_rotated


# ------------------------------------------------------------
# Dataset selection
# ------------------------------------------------------------
def get_dataset(dataset_name="spiral", n_samples=10000, **kwargs):
    """
    Get a dataset by name.

    Args:
        dataset_name: one of ["spiral", "checkerboard", "pinwheel", "rings", "swissroll"]
        n_samples: number of samples to generate
        **kwargs: additional arguments for the dataset function

    Returns:
        numpy array of shape (n_samples, 2)
    """
    if dataset_name == "spiral":
        n_arms = kwargs.get("n_arms", 2)
        noise = kwargs.get("noise", 0.001)
        return make_multi_spiral(n_samples // n_arms, n_arms=n_arms, noise=noise)
    elif dataset_name == "checkerboard":
        noise = kwargs.get("noise", 0.05)
        grid = kwargs.get("grid", 4)
        return make_checkerboard(n_samples, noise=noise, grid=grid)
    elif dataset_name == "pinwheel":
        n_arms = kwargs.get("n_arms", 5)
        noise = kwargs.get("noise", 0.05)
        return make_pinwheel(n_samples, n_arms=n_arms, noise=noise)
    elif dataset_name == "rings":
        noise = kwargs.get("noise", 0.03)
        return make_rings(n_samples, noise=noise)
    elif dataset_name == "swissroll":
        noise = kwargs.get("noise", 0.05)
        return make_swissroll2d(n_samples, noise=noise)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train(
    dataset_name="spiral",
    n_components=8,
    var_weight=0.0,
    embed_dim=None,
    embed_seed=42,
    kernel_type="rbf",
    **dataset_kwargs,
):
    """
    Train the generator using drifting field with learnable GMM prior.

    Args:
        dataset_name: name of dataset to use (spiral, checkerboard, pinwheel, rings, swissroll)
        n_components: number of Gaussian components in the mixture
        var_weight: weight for variance penalty (log-determinant)
        embed_dim: if specified, embed 2D data into this dimension via random rotation
        embed_seed: random seed for the embedding rotation
        **dataset_kwargs: additional arguments for dataset generation
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    folder_name = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"plots/{folder_name}"
    os.makedirs(folder_name, exist_ok=True)
    # fixed empirical dataset (2D)
    real_data_2d = get_dataset(dataset_name, n_samples=20000, **dataset_kwargs)

    # Optionally embed into high-dimensional space
    if embed_dim is not None and embed_dim > 2:
        real_data_np = embed_to_high_dim(
            real_data_2d, target_dim=embed_dim, seed=embed_seed
        )
        z_dim = embed_dim
        use_pca = True
    else:
        real_data_np = real_data_2d
        z_dim = 2
        use_pca = False

    real_data = torch.from_numpy(real_data_np).to(device)

    G = Generator(z_dim=z_dim).to(device)

    # Initialize GMM prior with data statistics for better coverage
    prior = GMMPrior(k=n_components, z_dim=z_dim, init_data=real_data).to(device)
    # prior = GMMPrior(k=1, z_dim=z_dim, init_data=real_data).to(device)  # For ablation comparison

    opt = optim.Adam(
        list(G.parameters()) + list(prior.parameters()),
        lr=1e-3,
    )

    batch_size = 512
    steps = 10000
    sigma = 0.1

    # Save txt with params and dataset
    with open(os.path.join(folder_name, "params.txt"), "w") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"n_components: {n_components}\n")
        f.write(f"var_weight: {var_weight}\n")
        f.write(f"embed_dim: {embed_dim}\n")
        f.write(f"embed_seed: {embed_seed}\n")
        f.write(f"kernel_type: {kernel_type}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"steps: {steps}\n")
        f.write(f"sigma: {sigma}\n")

    # phase 3: train the generator with drifting field
    # Now sample z from components corresponding to y
    for step in range(steps):
        # First sample real data
        idx = torch.randint(0, real_data.size(0), (batch_size,))
        y = real_data[idx]

        # Sample z from the Gaussian components assigned to y
        z = prior.sample(batch_size)  # Sample from the entire prior for diversity

        # generate samples
        x = G(z)

        # drifting field
        v = drifting_field(
            x=x,
            y_pos=y,
            y_neg=x.detach(),
            sigma=sigma,
            kernel_type=kernel_type,
        )

        # stop-gradient target
        x_drift = (x + v).detach()
        id_loss = ((x - z) ** 2).sum(dim=1).mean()

        # losses
        drift_loss = ((x - x_drift) ** 2).sum(dim=1).mean()

        # Regularization to prevent component collapse
        # Penalize small determinants (collapsed components)
        var_penalty = prior.variance_penalty()

        # Encourage positive log determinant (non-collapsed components)
        # We want to maximize log(det), so we minimize its negative
        collapse_loss = -var_penalty  # Negative because we want to maximize log det

        loss = drift_loss + id_loss / embed_dim + collapse_loss * var_weight / embed_dim

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(
                f"step {step:5d} | "
                f"drift {drift_loss.item():.6f} | "
                f"id {id_loss.item():.6f} | "
                f"log_det {var_penalty.item():.4f} | "
                f"collapse {collapse_loss.item():.6f}"
            )

        if step % 1000 == 0:
            plot_samples(
                G,
                prior,
                real_data,
                step,
                n=6000,
                use_pca=use_pca,
                folder_name=folder_name,
            )

    plot_samples(
        G, prior, real_data, "final", n=6000, use_pca=use_pca, folder_name=folder_name
    )
    # Phase 4: Conditional Training #
    steps_conditional = 10000
    for step in range(steps_conditional):
        # First sample real data
        idx = torch.randint(0, real_data.size(0), (batch_size,))
        y = real_data[idx]

        # Sample z from the Gaussian components assigned to y
        component_indices = prior.assign_to_components(y)
        z = prior.sample_from_components(component_indices)

        # generate samples
        x = G(z)

        # drifting field
        v = drifting_field(
            x=x,
            y_pos=y,
            y_neg=x.detach(),
            sigma=sigma,
            kernel_type=kernel_type,
        )

        # stop-gradient target
        x_drift = (x + v).detach()
        id_loss = ((x - z) ** 2).sum(dim=1).mean()

        # losses
        drift_loss = ((x - x_drift) ** 2).sum(dim=1).mean()

        # Regularization to prevent component collapse
        var_penalty = prior.variance_penalty()
        collapse_loss = -var_penalty

        loss = drift_loss + id_loss / embed_dim + collapse_loss * var_weight / embed_dim

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(
                f"step {step:5d} | "
                f"drift {drift_loss.item():.6f} | "
                f"id {id_loss.item():.6f} | "
                f"log_det {var_penalty.item():.4f} | "
                f"collapse {collapse_loss.item():.6f}"
            )

        if step % 1000 == 0:
            plot_samples(
                G,
                prior,
                real_data,
                step,
                n=6000,
                use_pca=use_pca,
                folder_name=folder_name,
                custom_name="conditional",
            )

    plot_samples(
        G,
        prior,
        real_data,
        "final_conditional",
        n=6000,
        use_pca=use_pca,
        folder_name=folder_name,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Choose a dataset to train on
    # Options: "spiral", "checkerboard", "pinwheel", "rings", "swissroll"

    # ========== 2D EXPERIMENTS ==========

    # Example 1: Two-armed spiral (2D)
    # train(
    #     dataset_name="spiral",
    #     n_arms=2,
    #     noise=0.001,
    #     n_components=40,
    # )

    # Example 2: Checkerboard (2D)
    # train(
    #     dataset_name="checkerboard",
    #     grid=4,
    #     noise=0.05,
    #     n_components=40,
    # )

    # Example 3: Pinwheel (2D)
    # train(
    #     dataset_name="pinwheel",
    #     n_arms=5,
    #     noise=0.05,
    #     n_components=25,
    # )

    # ========== HIGH-DIMENSIONAL EXPERIMENTS ==========

    # Example 4: Spiral embedded in 64D
    # train(
    #     dataset_name="spiral",
    #     n_arms=2,
    #     noise=0.001,
    #     n_components=40,
    #     embed_dim=64,
    #     embed_seed=42
    # )

    # Example 5: Pinwheel embedded in 512D
    train(
        dataset_name="pinwheel",
        n_arms=5,
        noise=0.05,
        n_components=25,
        var_weight=0.0001,  # Weight for collapse prevention (log det regularization)
        embed_dim=2,
        embed_seed=42,
        kernel_type="rbf",
    )

    # Example 6: Swiss roll embedded in 1024D
    # train(
    #     dataset_name="swissroll",
    #     noise=0.05,
    #     n_components=30,
    #     embed_dim=1024,
    #     embed_seed=42
    # )

    # test_conditional_matching(
    #     dataset_name="spiral",
    #     n_components=25,
    #     embed_dim=200,
    #     embed_seed=42,
    #     kernel_type="l2",
    # )
