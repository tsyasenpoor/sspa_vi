# Critical Fixes for VI Training Issues

## Problem Summary

From your output, I identified 3 core issues:

1. **ELBO Decreases Dramatically**
   - Iteration 11: E[log p(v,gamma)] = -83,937,547 (HUGE penalty)
   - Caused by: E[v] jumping from [0.13, 0.19] → [1.07, 1.63] in one iteration

2. **Predictions Skewed High**
   - Val mean: 0.716, Test mean: 0.653
   - Most predictions above 0.5 threshold

3. **Root Cause**: v updates are too aggressive despite damping

---

## Fix 1: Stronger Prior on V (Most Important)

**Location**: `VariationalInference/vi.py`, line 21

**Current**:
```python
sigma_v: float = 1.0,  # Increased from 0.1 - wider slab prior
```

**Fix**:
```python
sigma_v: float = 0.2,  # Strong regularization to prevent v explosion
```

**Why**: With sigma_v=1.0, the prior penalty for v=1.5 is only (1.5²)/(2*1²) = 1.125. Too weak! With sigma_v=0.2, the same v=1.5 incurs penalty (1.5²)/(2*0.04) = 28.1 - much stronger.

---

## Fix 2: More Conservative V Damping

**Location**: `VariationalInference/vi.py`, line 679

**Current**:
```python
v_damping: float = 0.6,      # Moderate damping for v
```

**Fix**:
```python
v_damping: float = 0.2,      # Very conservative damping for v
```

**Why**: Even with 60% damping, if the raw update wants v=2.0 and old v=0.1, the damped update is 0.6*2.0 + 0.4*0.1 = 1.24 (still huge!). With 20% damping: 0.2*2.0 + 0.8*0.1 = 0.48 (much safer).

---

## Fix 3: Tighter Clipping on V

**Location**: `VariationalInference/vi.py`, line 400

**Current**:
```python
self.mu_v[k] = np.clip(self.mu_v[k], -3, 3)
```

**Fix**:
```python
self.mu_v[k] = np.clip(self.mu_v[k], -1.5, 1.5)
```

**Why**: Limiting v to [-1.5, 1.5] prevents extreme logits while still allowing meaningful classification.

---

## Fix 4: Add Numerical Safeguards in V Update

**Location**: `VariationalInference/vi.py`, line 396-402

**Current**:
```python
try:
    self.Sigma_v[k] = np.linalg.inv(prec)
    self.mu_v[k] = self.Sigma_v[k] @ mean_contrib
    # Clip to prevent extreme values that cause rho_v oscillation
    self.mu_v[k] = np.clip(self.mu_v[k], -3, 3)
except np.linalg.LinAlgError:
    pass
```

**Fix**:
```python
try:
    # Add regularization to prevent ill-conditioning
    prec_reg = prec + 1e-4 * np.eye(self.d)
    self.Sigma_v[k] = np.linalg.inv(prec_reg)

    # Clip variance to prevent explosion
    diag_variance = np.diag(self.Sigma_v[k])
    if np.any(diag_variance > 10.0):
        # Variance too large - use more regularization
        prec_reg = prec + 1e-3 * np.eye(self.d)
        self.Sigma_v[k] = np.linalg.inv(prec_reg)

    self.mu_v[k] = self.Sigma_v[k] @ mean_contrib

    # Strict clipping to prevent ELBO explosion
    self.mu_v[k] = np.clip(self.mu_v[k], -1.5, 1.5)
except np.linalg.LinAlgError:
    # Keep previous values if inversion fails
    pass
```

**Why**: Adds regularization to prevent numerical issues and catches large variances early.

---

## Fix 5: More Selective Spike-and-Slab for V

**Location**: `VariationalInference/vi.py`, line 25

**Current**:
```python
pi_v: float = 0.5,  # Prior probability of v being active (increased from 0.05)
```

**Fix**:
```python
pi_v: float = 0.2,  # Prior probability of v being active - assume most programs irrelevant
```

**Why**: Biases the model toward sparsity in v, reducing the number of active gene programs for classification.

---

## Fix 6: Add Gradient Clipping to Prevent Jumps

**Location**: `VariationalInference/vi.py`, after line 811 (in fit function)

**Current**:
```python
self._update_v(y, X_aux)

# Apply damping
self.mu_v = (damping_factors['v'] * self.mu_v +
            (1 - damping_factors['v']) * mu_v_old)
```

**Fix**:
```python
self._update_v(y, X_aux)

# Clip update magnitude to prevent jumps
v_update = self.mu_v - mu_v_old
max_update = 0.5  # Maximum change per iteration
v_update_clipped = np.clip(v_update, -max_update, max_update)
self.mu_v = mu_v_old + v_update_clipped

# Then apply damping
self.mu_v = (damping_factors['v'] * self.mu_v +
            (1 - damping_factors['v']) * mu_v_old)
```

**Why**: Prevents single-step jumps larger than 0.5, regardless of what the update formula suggests.

---

## Fix 7: Initialize V More Conservatively

**Location**: `VariationalInference/vi.py`, line 86-88

**Current**:
```python
base = self.rng.randn(self.kappa, self.d)
scale_factors = 0.5 + 0.5 * np.arange(self.kappa) / self.kappa
self.mu_v = base * scale_factors[:, np.newaxis]
```

**Fix**:
```python
# Initialize v very small - let data drive it up slowly
self.mu_v = 0.01 * self.rng.randn(self.kappa, self.d)
```

**Why**: Starting with small v values prevents early-iteration explosions.

---

## How to Apply Fixes

**Option 1: Quick Fix (Top Priority)**
Apply fixes 1, 2, and 3 only:
1. sigma_v: 1.0 → 0.2
2. v_damping: 0.6 → 0.2
3. v clipping: [-3, 3] → [-1.5, 1.5]

These three alone should fix the ELBO issues.

**Option 2: Comprehensive Fix**
Apply all 7 fixes for maximum stability.

---

## Expected Results After Fixes

1. **ELBO should increase monotonically**
   - No more huge drops
   - Smooth convergence

2. **V values stay bounded**
   - E[v] should stay in [-1.5, 1.5]
   - Prevents huge prior penalties

3. **Predictions may still be biased**
   - This might be due to data imbalance (54 vs 86 samples)
   - Check with: `pd.Series(y_train).value_counts()`
   - If imbalanced, this is expected behavior

4. **Theta inference will work better**
   - With stable v, predictions will be more reasonable
   - Theta ranges should be similar to training

---

## Diagnostic After Re-training

After applying fixes and re-training, check:

1. **ELBO plot**: Should be monotonically increasing
2. **V statistics**: max(|v|) should be < 2.0
3. **Predictions**: Check if imbalance matches training data imbalance
4. **Theta ranges**: Should be similar across train/val/test

If predictions are still too confident, you may need to:
- Add intercept term (separate from v @ theta)
- Use class weights
- Add temperature scaling to probabilities
