# Does the jointly learned prior approach the data, or collapse?

Setting: data x ~ p, prior q_phi = sum_k pi_k N(mu_k, diag s_k^2), hard coupling k = c(x) (nearest mu),
e = mu_k + s_k * eps, z_t = (1-t) x + t e, v = e - x. Network u_theta trained by the MeanFlow loss L_MF.
Joint objective  J(theta, phi) = L_MF(theta, phi) + Reg(phi).

## Assumptions (for the analysis)
A1 two timescales: theta is (near) optimal for the current phi  (u* = E[target | z_t]).
A2 within a cell {x : c(x) = k} the data are Gaussian N(m_k, diag b_k), independent dims; cells do not
   overlap in z_t (true for well separated cells / small t; ignored otherwise).
A3 only the r = t (flow-matching) part of L_MF is treated analytically (75% of samples); the r < t part is
   checked numerically (scripts/theory_check.py --ratio_neq).
A4 no adaptive weighting (p = 0); with p = 1 the MF force is rescaled by 1/(Delta + c) (same signs).
Under A2 everything factorises over cells and dimensions, so it suffices to study one component, one
dimension: data N(m, b), prior N(mu, a), a = s^2.

## 1. The irreducible flow-matching loss (envelope view)
z_t and v are jointly Gaussian, so the optimal-network residual is the conditional variance

    R(a) = E_t Var(v | z_t) = E_t [ a b / ((1-t)^2 b + t^2 a) ]          (exact)

* R does not depend on mu at all.
* dR/da = E_t[ b^2 (1-t)^2 / ((1-t)^2 b + t^2 a)^2 ] > 0  for every a.
Proposition 1 (collapse under the full gradient). If phi follows the envelope gradient
d/dphi min_theta L_MF = dR/dphi, then s -> 0 monotonically: every component collapses to a point (a Dirac
prior; one-step generation can then only produce K distinct outputs). The mean receives no force.
More generally, for any data distribution, s = 0 with an injective coupling makes v a function of z_t, so
L_MF's irreducible part is 0 = its global minimum: collapse is always a global minimiser of L_MF over phi.

## 2. What our implementation actually does (stop-gradient target)
The target sg(v - (t-r) du/dt) is detached, so phi only receives gradient through the network input z_t.
With u* linear in z (Gaussian case), A_t = Cov(v,z)/Var(z) = (t a - (1-t) b)/D_t, D_t = (1-t)^2 b + t^2 a:

    E dL/ds = -2 s b E_t[ t (1-t) (t a - (1-t) b) / D_t^2 ]                 (exact, r = t part)
    E dL/dmu = 0

Proposition 2. The mean is never moved by the MF loss. The variance has exactly one stationary point
s_0 (s_0 = sqrt(b) for symmetric t-sampling; s_0 ~ 1.1 sqrt(b) for MeanFlow's logit-normal(-0.4, 1)), and it is
UNSTABLE: s < s_0 collapses to 0, s > s_0 disperses to infinity (numbers: scratchpad theory.py).
So neither gradient convention makes the prior approach the data by itself: "close to the data" must come
from the regulariser.

## 3. Regularisers: which ones give a stable prior near the data
Write F(s) = E dL_MF/ds (from section 2). Equilibria solve F(s) + Reg'(s) = 0; stable iff the sum is
increasing through 0.

| regulariser (per dim) | equilibrium / stability (logit-normal t, b = 1) |
|---|---|
| none | s_0 = 1.1, unstable (collapse or dispersal) |
| entropy -beta log s (original -log det) | beta < ~0.29: stable s* ~ 0.15-0.25 (far below the data scale) + unstable point; beta > 0.29: no equilibrium, s -> infinity |
| transport alpha E(x-e)^2 - beta log s (original identity + log det) | stable at s*^2 ~ beta/(2 alpha) (+ MF bias); data-independent, must be tuned per scale; mu -> cell mean |
| NLL lam E[log s + (x-mu)^2/(2 s^2)] (= KL(p_data || q_phi) up to a constant, hard EM) | stable s* -> sqrt(b) as lam grows: lam = .3 -> .7, 1 -> .9, 2 -> .95; basin s < 1.7 (lam = 1), 2.35 (lam = 2); mu -> cell mean |

Proposition 3 (NLL anchor). The NLL term's restoring slope at s = sqrt(b) is 2 lam / b, so the equilibrium is
stable iff 2 lam / b > -F'(s*) (about 0.8 / b here), and the remaining bias is

    s* / sqrt(b) ~ 1 - F(sqrt(b)) / (2 lam)         (first order),

with means at the cell means (Lloyd / hard-EM fixed point). Hence with the NLL anchor the prior provably
converges to the moment-matched GMM of the data (the closest GMM in KL(p || q)), and the MF force only
shifts the scales by O(1/lam).

Why the original combination was fragile: the identity loss with a sampled prior contains alpha s^2 (pushes s
down), and -log det pushes s up; their balance point beta/(2 alpha) does not know the data scale, and the MF
force (section 2) destabilises any point near the data scale. The NLL term is exactly "identity loss weighted by
1/s^2 plus log det", the unique such combination whose fixed point is the data's own within-cell variance.

## 4. Consequences for the method
1. Joint learning through the MF loss alone collapses (full gradient) or is unstable (stop-gradient):
   the prior does NOT drift to the target on its own.
2. A data-likelihood anchor (lam >= 1 in these units) makes the prior converge to the data-fitted GMM; the
   MF gradient then acts as a bounded correction of the scales (O(F/lam)).
3. In the Gaussian-cell model the MF loss gives the means no signal. Any useful "joint" effect on the means can only
   come from non-Gaussian cells or cell overlap (beyond A2), so it has to be measured empirically: the
   comparison is joint+NLL vs fixed EM-GMM (same K) vs Gaussian.

## Empirical check
scripts/theory_check.py: D = 8, data N(0, I), K = 1, prior initialised at s0 in {.6, 1, 1.4}, mu0 = .5; network lr
1e-3, prior lr 3e-4. Predictions (ratio_neq = 0, p = 0): sg/no reg collapse from .6 and 1.0 and disperse from 1.4,
|mu| stays .5; full collapses from all; NLL lam = 1 -> s ~ .9, lam = 2 -> ~ .95, |mu| -> 0.
