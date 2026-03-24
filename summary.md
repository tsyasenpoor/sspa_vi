# SSPA-VI: Supervised Sparse Poisson Factorization with Variational Inference

## Model Summary

Coordinate Ascent Variational Inference (CAVI) for supervised Poisson factorization,
combining scHPF-style gene program discovery with Jaakkola-Jordan logistic regression.

### Generative Model

- **Count data**: X_ij ~ Poisson(sum_l theta_il * beta_jl)
- **Cell loadings**: theta_il ~ Gamma(alpha_theta, xi_i), with xi_i ~ Gamma(alpha_xi, lambda_xi)
- **Gene loadings**: beta_jl ~ Gamma(alpha_beta, eta_j), with eta_j ~ Gamma(alpha_eta, lambda_eta)
- **Regression weights**: v_kl ~ Laplace(0, b_v) via scale-mixture representation (Bayesian Lasso)
- **Auxiliary weights**: gamma_k ~ N(0, sigma_gamma^2 I)
- **Labels**: y_ik ~ Bernoulli(sigmoid(theta_i^T v_k + x_aux_i^T gamma_k))

### CAVI Update Order (per iteration)

1. **PHI** (multinomial allocation): log phi_ijl = E[log theta_il] + E[log beta_jl], softmax over l
2. **BETA** (gene loadings): a_beta = alpha_beta + sum_i E[z_ijl], b_beta = E[eta_j] + sum_i E[theta_il]
3. **ETA** (gene capacity): b_eta = lambda_eta + sum_l E[beta_jl]
4. **RESCALE** (identifiability): balance E[theta] and E[beta] to geometric mean per factor
5. **ZETA** (JJ auxiliary): zeta_ik = sqrt(E[A_ik^2]) where A_ik = theta_i^T v_k
6. **THETA** (cell loadings): a_theta = alpha_theta + sum_j E[z_ijl], b_theta from quadratic solve
7. **XI** (cell capacity): b_xi = bp + sum_l E[theta_il]
8. **V** (regression weights, Bayesian Lasso): precision = E[1/s] + 2*sum_i lambda(zeta)*E[theta^2]
9. **GAMMA** (auxiliary weights): standard Gaussian posterior update
10. **ZETA** (re-tighten after v/gamma update)

### Key Features Preserved

- **JAX/GPU acceleration** with automatic NumPy fallback
- **Memory-efficient chunked computation**: O(chunk_size * K) not O(n * K)
- **Sparse phi computation**: O(nnz * K) using COO format
- **Adaptive OOM guards**: chunk sizes shrink on GPU memory exhaustion
- **Class-weighted regression**: balanced weights for imbalanced labels
- **Held-out LL monitoring**: Poisson + regression LL on validation set
- **Early stopping**: on held-out regression LL plateau or ELBO convergence
- **Checkpoint/restore**: best model parameters saved during training
- **Probit-calibrated predictions**: accounts for posterior variance in logits

### Test/Validation Theta Inference

Theta inference for new data uses **fixed** beta, v, gamma learned from training.
Only theta and xi are re-inferred using:
- Poisson likelihood (phi computation with fixed E[log beta])
- Quadratic JJ regularization (2*lambda(zeta) @ E[v^2]) to match training regime
- No label information used (labels only needed for evaluation, not theta inference)

### Prior on V: Bayesian Lasso (Laplace)

The Laplace prior is represented as a Gaussian scale mixture:
- v_kl | s_kl ~ N(0, s_kl)
- s_kl ~ Exp(1 / (2 * b_v^2))

The posterior on s_kl is Inverse-Gaussian, giving:
- E[1/s_kl] = 1/(b_v * omega_kl) + 1/omega_kl^2
- where omega_kl = sqrt(mu_v_kl^2 + tau^2_v_kl)

b_v is auto-scaled as b_v_eff = b_v * sqrt(K) / sqrt(N) following Park & Casella (2008).

### Cleanup Summary (this commit)

Removed:
- Gaussian (normal) prior option for v
- All damping (alpha blending, oscillation detection, step caps)
- Pathway masking modes (masked, pathway_init, combined)
- Excessive safeguards (b_theta flooring, prior precision flooring, zeta capping, v clipping)
- spike-and-slab parameters (pi_v, pi_beta, spike_variance_v, spike_value_beta)
- v_warmup parameter

Enabled:
- Factor rescaling for identifiability (was previously disabled)
- Full E[A^2] computation including E[theta]^2 @ tau^2_v term in zeta update
