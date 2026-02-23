"""
drifting_so2_egnn.py — Drifting field generative model for SO(2) with EGNN.

Data:
    A single canonical configuration of 2 points:
        y = [[0, -1], [0, 1]]
    The dataset is the SO(2) orbit of this configuration: for every rotation
    angle θ, the rotated pair  R(θ) @ [[0,-1],[0,1]]  is a valid sample.
    Because the two points are antipodal on a unit circle, the orbit is a
    1-parameter family parameterised by θ ∈ [0, 2π).

Model: EGNN (Equivariant Graph Neural Network)
    Input:  2 nodes, each with a random 2D position (the prior).
    Output: 2 updated 2D positions (the generated configuration).

    Equivariance:
        G(R @ z_pos) = R @ G(z_pos)   for all R ∈ SO(2)

    This holds because each EGNN layer updates positions as
        x_i ← x_i  +  Σ_j (x_i − x_j) · φ_x(m_ij)
    where m_ij depends only on ||x_i − x_j||² (invariant) and node
    features (scalars, also invariant).  Relative positions (x_i − x_j)
    transform equivariantly, so the whole update is equivariant.

Training: Drifting field (same algorithm as drifting.py).
    Each configuration is flattened to a 4D vector [x0_x, x0_y, x1_x, x1_y]
    and the field is computed in that 4D sample space.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data ──────────────────────────────────────────────────────────────────────

# The single canonical data sample: two antipodal points on the unit circle.
CANONICAL = torch.tensor([[0.0, -1.0], [0.0, 1.0]])   # (2, 2)

# Fixed node-type features: one-hot so EGNN can tell the two nodes apart.
# These are rotation-invariant scalars — they never change.
NODE_FEAT = torch.tensor([[1.0, 0.0], [0.0, 1.0]])    # (2, n_feat=2)

# Inter-point distance of the canonical configuration (used as regulariser).
CANONICAL_DIST = (CANONICAL[0] - CANONICAL[1]).norm().item()   # = 2.0


def sample_orbit(n: int, device: str = "cpu") -> torch.Tensor:
    """
    Sample n configurations from the SO(2) orbit of CANONICAL.

    For each sample, draw θ ~ Uniform[0, 2π) and apply the rotation
        R(θ) = [[cos θ, -sin θ],
                [sin θ,  cos θ]]
    to every point in CANONICAL.

    Returns: (n, 2, 2)  — n configurations of 2 points each.
    """
    theta = torch.rand(n, device=device) * (2 * math.pi)
    cos_t, sin_t = theta.cos(), theta.sin()

    # Build (n, 2, 2) rotation matrices.
    R = torch.stack([
        torch.stack([ cos_t, -sin_t], dim=1),   # row 0
        torch.stack([ sin_t,  cos_t], dim=1),   # row 1
    ], dim=1)                                    # (n, 2, 2)

    # Apply R to each of the 2 canonical points.
    # y[b, node, coord] = sum_e R[b, coord, e] * CANONICAL[node, e]
    canonical = CANONICAL.to(device)
    y = torch.einsum("bce,ne->bnc", R, canonical)   # (n, 2, 2)
    return y


def sample_prior(n: int, device: str = "cpu") -> torch.Tensor:
    """
    Sample n prior configurations: two independent 2D positions from N(0, I),
    zero-centred so the centroid of the 2 nodes is always at the origin.

    This matches the orbit samples, where the two antipodal points always sum
    to (0, 0).

    Returns: (n, 2, 2)  — n pairs of zero-centred 2D positions.
    """
    z = torch.randn(n, 2, 2, device=device)
    z = z - z.mean(dim=1, keepdim=True)   # subtract centroid per configuration
    return z


# ── EGNN building blocks ──────────────────────────────────────────────────────

def _mlp(dims: list, activation: nn.Module = nn.SiLU()) -> nn.Sequential:
    """Helper: build a fully-connected MLP with the given layer widths."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation)
    return nn.Sequential(*layers)


class EGNNLayer(nn.Module):
    """
    One EGNN message-passing layer for a 2-node graph in 2D.

    Notation:  N=2 nodes,  d=2 position dims,  f=n_feat feature dims.

    For each directed edge (i→j):
        m_ij = φ_e(h_i, h_j, ||x_i − x_j||²)          invariant message
        Δx_i += (x_i − x_j) · φ_x(m_ij)               equivariant position update
        agg_i += m_ij                                   feature aggregation

    Node feature update:
        h_i' = φ_h(h_i, agg_i)

    Equivariance holds because:
      - ||x_i − x_j||² is invariant → m_ij is invariant → φ_x(m_ij) is scalar.
      - (x_i − x_j) is equivariant → product with a scalar stays equivariant.
    """

    def __init__(self, n_feat: int, n_hidden: int):
        super().__init__()
        # Edge MLP: (h_i, h_j, d²) → message
        self.phi_e = _mlp([2 * n_feat + 1, n_hidden, n_hidden])
        # Position scalar: message → 1 scalar weight
        self.phi_x = _mlp([n_hidden, n_hidden, 1])
        # Node MLP: (h_i, agg_msg) → h_i'
        self.phi_h = _mlp([n_feat + n_hidden, n_hidden, n_feat])

    def forward(
        self,
        pos: torch.Tensor,   # (B, 2, 2) — positions
        h:   torch.Tensor,   # (B, 2, n_feat) — node features
    ):
        B = pos.shape[0]

        # Relative position and squared distance (only one unique edge for N=2)
        dx    = pos[:, 0] - pos[:, 1]                         # (B, 2)
        dist2 = (dx ** 2).sum(dim=-1, keepdim=True)           # (B, 1)

        # Edge messages for both directions
        m_01 = self.phi_e(torch.cat([h[:, 0], h[:, 1], dist2], dim=-1))  # (B, nh)
        m_10 = self.phi_e(torch.cat([h[:, 1], h[:, 0], dist2], dim=-1))  # (B, nh)

        # Position updates: (x_i − x_j) * scalar
        #   Node 0: neighbour is node 1  →  (x_0 − x_1) *  φ_x(m_01)
        #   Node 1: neighbour is node 0  →  (x_1 − x_0) *  φ_x(m_10)
        dpos0 =  dx * self.phi_x(m_01)   # (B, 2)
        dpos1 = -dx * self.phi_x(m_10)   # (B, 2)

        new_pos = torch.stack([pos[:, 0] + dpos0,
                               pos[:, 1] + dpos1], dim=1)   # (B, 2, 2)

        # Node feature updates
        h0_new = self.phi_h(torch.cat([h[:, 0], m_01], dim=-1))
        h1_new = self.phi_h(torch.cat([h[:, 1], m_10], dim=-1))
        new_h  = torch.stack([h0_new, h1_new], dim=1)       # (B, 2, n_feat)

        return new_pos, new_h


class EGNNGenerator(nn.Module):
    """
    SO(2)-equivariant generator using EGNN.

    Takes a prior configuration of 2 random 2D positions and transforms
    them through several equivariant message-passing layers.

    Node features are fixed type indicators (not part of the prior),
    so they stay invariant under rotation and allow the network to
    distinguish node 0 from node 1.

    Input : pos_prior  (B, 2, 2) — two random 2D positions per sample.
    Output: pos_out    (B, 2, 2) — generated configuration.
    """

    def __init__(self, n_feat: int = 2, n_hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [EGNNLayer(n_feat, n_hidden) for _ in range(n_layers)]
        )
        # Register fixed node features as a buffer (moved with .to(device))
        self.register_buffer("node_feat", NODE_FEAT.clone())   # (2, n_feat)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        B = pos.shape[0]
        # Broadcast fixed features to batch dimension
        h = self.node_feat.unsqueeze(0).expand(B, -1, -1)  # (B, 2, n_feat)
        for layer in self.layers:
            pos, h = layer(pos, h)
        return pos   # (B, 2, 2)


# ── RBF kernel & drifting field (same as original) ───────────────────────────

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    diff  = x[:, None, :] - y[None, :, :]
    dist2 = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist2 / (2 * sigma ** 2))


def drifting_field(
    x:    torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    sigma: float,
    remove_self_repulsion: bool = True,
) -> torch.Tensor:
    """
    Compute the drifting field in the flattened configuration space.

    x, y_pos, y_neg are all (B, D) flat vectors.
    For configurations of 2 points in 2D:  D = 4.
    """
    k_pos = rbf_kernel(x, y_pos, sigma)
    k_neg = rbf_kernel(x, y_neg, sigma)

    if remove_self_repulsion:
        k_neg.fill_diagonal_(0.0)

    w_pos = k_pos / (k_pos.sum(dim=1, keepdim=True) + 1e-8)
    w_neg = k_neg / (k_neg.sum(dim=1, keepdim=True) + 1e-8)

    drift_pos = (w_pos[..., None] * (y_pos[None] - x[:, None])).sum(dim=1)
    drift_neg = (w_neg[..., None] * (y_neg[None] - x[:, None])).sum(dim=1)

    return drift_pos - drift_neg


# ── SO(2) Procrustes alignment ────────────────────────────────────────────────

def so2_align(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Align src to tgt via the optimal SO(2) rotation (closed-form Procrustes).

    For each batch element b, finds
        R_b = argmin_{R ∈ SO(2)} || src[b] @ R.T − tgt[b] ||_F
    and returns src[b] @ R_b.T.

    Derivation: expanding the Frobenius norm, maximising tr(R @ H) over SO(2)
    where H = src.T @ tgt gives the closed-form angle
        θ = atan2(H[0,1] − H[1,0],  H[0,0] + H[1,1])

    R is computed from src.detach() so that gradients flow only through the
    linear application  src → src @ R.T,  not through R itself.

    src, tgt : (B, N, 2) — zero-centred point clouds
    Returns  : (B, N, 2) — src after the per-sample optimal rotation
    """
    # Cross-covariance per sample: H[b] = src[b].T @ tgt[b]  shape (B, 2, 2)
    H = torch.bmm(src.detach().transpose(1, 2), tgt)

    a = H[:, 0, 0] + H[:, 1, 1]               # (B,)
    b = H[:, 0, 1] - H[:, 1, 0]               # (B,)
    norm = (a ** 2 + b ** 2).sqrt().clamp(min=1e-8)
    cos_t, sin_t = a / norm, b / norm

    # Build per-sample SO(2) rotation matrices
    R = torch.stack([
        torch.stack([ cos_t, -sin_t], dim=1),  # row 0
        torch.stack([ sin_t,  cos_t], dim=1),  # row 1
    ], dim=1)                                   # (B, 2, 2)

    # Apply: src_aligned[b] = src[b] @ R[b].T  (row-vector convention)
    return torch.bmm(src, R.mT)


# ── Visualisation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_samples(G: nn.Module, step, n: int = 1000):
    z = sample_prior(n, device=device)
    x = G(z).cpu()                          # (n, 2, 2)
    y = sample_orbit(n, device="cpu")       # (n, 2, 2)

    # Reference: canonical and its orbit (a circle per node)
    theta_ref = torch.linspace(0, 2 * math.pi, 300)
    canonical_cpu = CANONICAL.cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Prior positions
    z_cpu = z.cpu()
    for node, color in enumerate(["tab:blue", "tab:orange"]):
        axes[0].scatter(z_cpu[:, node, 0], z_cpu[:, node, 1],
                        s=3, alpha=0.3, c=color, label=f"node {node}")
    axes[0].set_title(f"Prior positions  (step {step})")
    axes[0].legend(fontsize=7)
    axes[0].axis("equal")

    # Data space: orbits of each canonical point are unit circles
    for node_idx, marker in enumerate(["o", "s"]):
        # Real orbit samples
        axes[1].scatter(y[:, node_idx, 0], y[:, node_idx, 1],
                        s=4, alpha=0.2, label=f"real node {node_idx}")
        # Generated positions
        axes[1].scatter(x[:, node_idx, 0], x[:, node_idx, 1],
                        s=4, alpha=0.6, zorder=5, label=f"gen node {node_idx}")
        # Canonical point
        axes[1].scatter(canonical_cpu[node_idx, 0], canonical_cpu[node_idx, 1],
                        s=120, zorder=10, marker=marker,
                        label=f"canonical {node_idx}")

    # Unit circle for reference
    axes[1].plot(theta_ref.cos(), theta_ref.sin(), "k--", lw=0.8, alpha=0.4)
    axes[1].legend(fontsize=6)
    axes[1].set_title(f"Generated vs real  (step {step})")
    axes[1].axis("equal")

    fig.tight_layout()
    fname = f"egnn_step_{step}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    batch_size: int = 512,
    steps:      int = 20000,
    sigma:      float = 0.3,
    n_hidden:   int = 64,
    n_layers:   int = 3,
    lr:         float = 1e-3,
    reg_weight: float = 0.1,
):
    G   = EGNNGenerator(n_feat=2, n_hidden=n_hidden, n_layers=n_layers).to(device)
    opt = optim.Adam(G.parameters(), lr=lr)

    for step in range(steps):
        # Real configurations: random SO(2) rotations of the canonical pair.
        y = sample_orbit(batch_size, device=device)          # (B, 2, 2)
        # Prior: two random 2D positions per sample.
        z = sample_prior(batch_size, device=device)          # (B, 2, 2)

        # Generated configurations.
        x = G(z)                                             # (B, 2, 2)

        # Align x to y before computing the field: for each sample b, rotate
        # x[b] into the same frame as y[b].  This makes Euclidean distances in
        # the flattened 4D space meaningful (orientation-normalised).
        x_aligned = so2_align(x, y)                         # (B, 2, 2)

        # Flatten configurations to 4D vectors for the drifting field.
        # Use reshape (not view) since einsum/bmm may yield non-contiguous tensors.
        x_flat = x_aligned.reshape(batch_size, 4)
        y_flat = y.reshape(batch_size, 4)

        v = drifting_field(
            x=x_flat,
            y_pos=y_flat,
            y_neg=x_flat.detach(),
            sigma=sigma,
        )
        x_drift = (x_flat + v).detach()                     # (B, 4)

        # Drift loss: follow the field in configuration space.
        drift_loss = ((x_flat - x_drift) ** 2).sum(dim=1).mean()

        # Geometry regularisation: keep the inter-point distance close to the
        # canonical distance (2.0).  This is rotation-invariant, so it does
        # not break equivariance.
        inter_dist = (x[:, 0] - x[:, 1]).norm(dim=-1)       # (B,)
        reg_loss   = ((inter_dist - CANONICAL_DIST) ** 2).mean()

        loss = drift_loss #+ reg_weight * reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(
                f"step {step:5d}  |  "
                f"drift {drift_loss.item():.5f}  |  "
                f"reg {reg_loss.item():.5f}"
            )

        if step % 1000 == 0:
            plot_samples(G, step)

    plot_samples(G, "final")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
