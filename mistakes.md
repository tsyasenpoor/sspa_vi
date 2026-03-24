# Mistakes Log: Things to Avoid in Future Runs

Track mistakes and gotchas encountered during development to prevent regression.

## Architectural Mistakes

### 1. Adding damping to CAVI updates
**Problem**: Damping (alpha-blending old/new values) is not a valid coordinate-ascent step.
It breaks ELBO monotonicity and can cause variance collapse when applied to sigma_v_diag.
**Rule**: Pure CAVI updates only. If convergence is unstable, fix the root cause (initialization,
prior strength, identifiability) rather than adding damping.

### 2. Gaussian prior for v with K-scaling
**Problem**: The N(0, sigma_v^2/K) prior required heavy damping and oscillation detection
to prevent divergence. The K-scaling interacted badly with the JJ bound, creating a feedback
loop where v grew -> theta shrank -> zeta shrank -> lambda(zeta) grew -> v grew more.
**Rule**: Use Laplace (Bayesian Lasso) prior. Its adaptive shrinkage naturally handles
varying K and prevents v explosion without damping.

### 3. Disabling factor rescaling
**Problem**: Without rescaling, theta and beta scales can diverge, forcing v to compensate
for scale imbalance. This was masked by adding aggressive damping and clipping to v.
**Rule**: Enable rescaling every iteration per the pseudocode. It's not a CAVI step but
a reparameterization that doesn't change the objective value.

### 4. Clipping rescale factors to [0.5, 2.0]
**Problem**: Clipping s_theta and then setting s_beta = 1/s_theta breaks the theta*beta
invariant. The product is only preserved when both are applied without clipping.
**Rule**: Apply rescaling without clipping. Large rescale factors indicate the model
needs rescaling — clipping defeats the purpose.

## Numerical Safeguards That Backfired

### 5. Flooring b_theta at 10% of b_poisson
**Problem**: This prevented the JJ regression correction from having its proper effect.
When the regression term legitimately wants to increase b_theta (shrink theta), the floor
would override it, creating a train-test distribution shift.
**Rule**: Let the quadratic solve determine b_theta naturally.

### 6. Capping zeta at zeta_max=4.0
**Problem**: The JJ bound is valid for ANY zeta. Capping gives a looser bound and prevents
the bound from tightening where it matters most (cells with strong logit signal).
**Rule**: No cap on zeta. If lambda(zeta) gets too small, that's correct behavior —
the JJ bound becomes loose for extreme logits, which is fine.

### 7. Flooring prior precision at 1.0 for Laplace
**Problem**: This effectively replaced the Laplace prior with a unit-variance Gaussian
for large |v|, defeating the purpose of Bayesian Lasso sparsity.
**Rule**: Use the exact Laplace prior precision. If v grows large, the prior should
indeed become less informative — that's the desired behavior for selected features.

### 8. Damping sigma_v_diag separately from mu_v
**Problem**: Damping sigma with (1-alpha)*old + alpha*new creates a one-way ratchet that
collapses variance over iterations, because the data precision only grows.
**Rule**: Use exact posterior variance from the precision update.

## Complexity Creep

### 9. Adding oscillation detection (sign-flip, period-2 detection)
**Problem**: These were band-aids for the underlying instability caused by damping +
Gaussian prior + no rescaling. Each fix added complexity without addressing root cause.
**Rule**: If the model oscillates, the updates are correct but the parameterization is
unstable. Fix with rescaling and proper priors, not detection heuristics.

### 10. Adding pathway masking modes prematurely
**Problem**: masked/pathway_init/combined modes added significant complexity to every
update function and the ELBO computation. Debugging became much harder.
**Rule**: Get unmasked mode working perfectly first. Only then add masking as a clean
separate layer, ideally with minimal changes to core update functions.

### 11. K-dependent step caps and alpha schedules
**Problem**: Step caps like max_step = min(0.05, 2.5/K) and alpha schedules that scale
with K are ad-hoc fixes that may need re-tuning for every new dataset.
**Rule**: The model should work without K-dependent tuning. Proper priors and rescaling
should handle varying K naturally.

## Process Mistakes

### 12. Testing too many changes simultaneously
**Rule**: Change one thing at a time. Run on validation data before accepting.

### 13. Not monitoring ELBO monotonicity
**Rule**: CAVI should increase ELBO monotonically. Any decrease indicates a bug in the
update equations, not a need for damping.

### 14. Comparing models with different safeguards
**Rule**: When comparing v_prior='normal' vs 'laplace', ensure both use the same
safeguards (or none), otherwise the comparison is confounded.
