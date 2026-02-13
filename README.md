# Drifting Field Generator with Learnable GMM Prior

A generative model that learns to map samples from a learnable Gaussian Mixture Model (GMM) prior to a target data distribution using **drifting fields** — a kernel-based transport mechanism with attraction to real data and self-repulsion among generated samples.

## How It Works

1. **GMM Prior**: A mixture of `k` full-covariance Gaussians whose means and Cholesky-parameterized covariances are learned jointly with the generator. Means are initialized from data points for better coverage.

2. **Generator**: A small MLP with a learnable linear skip connection (`net(z) + projection(z) * skip_weight`), encouraging the mapping to stay close to identity.

3. **Drifting Field**: At each step, generated samples are nudged via an RBF-kernel-weighted field that attracts them toward real data and repels them from other generated samples. The generator is trained to match these drift targets.

4. **Loss**: `drift_loss + id_loss / d + collapse_loss * var_weight / d`, where `d` is the data dimensionality. The collapse loss (negative log-determinant) prevents GMM components from degenerating.

## Datasets

| Name | Function | Description |
|------|----------|-------------|
| `spiral` | `make_multi_spiral` | Multi-armed Archimedean spiral |
| `checkerboard` | `make_checkerboard` | Grid-based checkerboard pattern |
| `pinwheel` | `make_pinwheel` | Radial pinwheel arms |
| `rings` | `make_rings` | Concentric circles |
| `swissroll` | `make_swissroll2d` | 2D Swiss roll curve |

All datasets are natively 2D but can be embedded into arbitrary dimensions via random orthogonal rotation (`embed_dim` parameter).

## Usage

```bash
python drifting_ae_skip.py
```

Edit the `__main__` block to select a dataset and configure hyperparameters:

```python
train(
    dataset_name="pinwheel",
    n_arms=5,
    noise=0.05,
    n_components=25,
    var_weight=0.0001,
    embed_dim=200,
    embed_seed=42
)
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_name` | Which dataset to use | `"spiral"` |
| `n_components` | Number of GMM components | `8` |
| `var_weight` | Collapse prevention weight (log-det regularization) | `0.0` |
| `embed_dim` | Embed 2D data into this dimensionality | `None` (stay 2D) |
| `embed_seed` | Random seed for the embedding rotation | `42` |

## Output

Training saves `step_<N>.png` visualizations every 1000 steps showing:
- **Left**: Latent space — GMM samples, component means (red crosses), and covariance ellipses (2D only)
- **Right**: Data space — real vs. generated samples

For high-dimensional runs, PCA projects both spaces to 2D for plotting.

## TODO

- [x] **Use L2-norm in RBF kernel**: Now we use the same kernel as the paper.
- [ ] **Phase 2 training — conditional sampling via p(y|z)**: Currently (phase 1) `z` and `y` are sampled independently (`z ~ prior`, `y ~ data`). Add a phase 2 that estimates `p(y|z)` (e.g., by assigning each real data point `y` to its nearest GMM component) and then samples paired `(z, y)` from `p(z) * p(y|z)` so the drifting field targets are coherent with the latent samples.

