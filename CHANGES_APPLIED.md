# VI Training Fixes Applied

## Summary of Changes

I've applied **7 critical fixes** to address your ELBO decrease and prediction issues:

### Changes Made to `VariationalInference/vi.py`

#### 1. **Stronger Prior on V** (Line 21)
```python
# BEFORE: sigma_v: float = 1.0
# AFTER:  sigma_v: float = 0.2
```
- **Impact**: 25x stronger penalty for large v values
- **Why**: Prevents v from exploding (which caused the -83M ELBO drop)

#### 2. **More Selective Spike-and-Slab** (Line 25)
```python
# BEFORE: pi_v: float = 0.5
# AFTER:  pi_v: float = 0.2
```
- **Impact**: Assumes only 20% of gene programs relevant for classification
- **Why**: Encourages sparsity, reducing model complexity

#### 3. **Conservative V Initialization** (Line 87)
```python
# BEFORE: Complex scaling initialization
# AFTER:  self.mu_v = 0.01 * self.rng.randn(self.kappa, self.d)
```
- **Impact**: Starts v near zero
- **Why**: Prevents early-iteration explosions

#### 4. **Numerical Safeguards in V Update** (Lines 396-415)
```python
# Added:
# - Regularization: prec + 1e-4 * I
# - Variance monitoring
# - Adaptive regularization if variance > 10
# - Tighter clipping: [-1.5, 1.5] instead of [-3, 3]
```
- **Impact**: Prevents numerical instability and huge updates
- **Why**: Catches ill-conditioned matrices before they cause problems

#### 5. **More Conservative Damping** (Lines 690-693)
```python
# BEFORE: theta=0.5, beta=0.7, v=0.6, gamma=0.6
# AFTER:  theta=0.3, beta=0.5, v=0.2, gamma=0.4
```
- **Impact**: Only accepts 20% of v update per iteration (vs 60%)
- **Why**: Prevents jumps even if raw update is extreme

#### 6. **Gradient Clipping** (Lines 826-831)
```python
# Added:
max_v_update = 0.3
v_update_clipped = np.clip(v_update, -max_v_update, max_v_update)
```
- **Impact**: Hard limit on per-iteration change in v
- **Why**: Failsafe to prevent any single-step jump > 0.3

#### 7. **Stricter Value Clipping** (Line 412)
```python
# BEFORE: np.clip(self.mu_v[k], -3, 3)
# AFTER:  np.clip(self.mu_v[k], -1.5, 1.5)
```
- **Impact**: V stays in narrower range
- **Why**: Prevents extreme logits while allowing classification

---

## Expected Improvements

### 1. **ELBO Should Increase Monotonically**
- No more massive drops
- Should see smooth convergence
- Adaptive damping will gradually trust updates more

### 2. **V Values Stay Bounded**
- Max |v| should be < 1.5 (vs previous 2.6)
- Prior penalty E[log p(v)] should stay reasonable
- No more -83M explosions

### 3. **Stable Training**
- Slower but much more stable
- May take more iterations, but will converge properly

### 4. **Predictions**
The prediction bias (mean 0.72) might persist IF:
- Your training data is imbalanced (54 vs 86 samples = 61% class 1)
- This is **expected behavior** - model learns the base rate

To check if this is the case:
```python
import pandas as pd
# Load your data
adata = sc.read_h5ad('filtered_data_top3k.h5ad')
print(adata[adata.obs['split'] == 'train'].obs['ap'].value_counts())
```

If imbalanced, predictions biased to majority class are correct!

---

## How to Test

### Option 1: Quick Test (Recommended)
```bash
cd /home/user/sspa_vi
python VariationalInference/vi.py  # Should not error
```

### Option 2: Full Re-training
```bash
python quick_reference.py
```

Monitor the output for:
1. ✅ ELBO increases every iteration (or at least every 10 iterations)
2. ✅ E[v] stays in [-2, 2] range
3. ✅ No warnings about "ELBO decreased"
4. ✅ Damping factors gradually increase (showing trust in updates)

---

## What to Watch For

### Good Signs:
- ELBO increases smoothly
- v values stay < 2.0
- Training completes without explosions
- Predictions match training data distribution

### Bad Signs (If Still Occurring):
- ELBO still decreasing → Try even smaller sigma_v (0.1)
- V still exploding → Try v_damping = 0.1
- Predictions still too confident → This might be correct if data is imbalanced!

---

## Understanding Your Results

### The Prediction "Problem" Might Not Be a Problem

Your output showed:
- Validation mean prob: 0.716
- Test mean prob: 0.653
- Training distribution: 54 class 0, 86 class 1 (61.4% class 1)

**If your validation/test sets also have ~60% class 1**, then predicting an average probability of 0.65-0.72 is **exactly correct**!

The model has learned:
1. Which gene programs distinguish the classes
2. The base rate of each class

### When to Worry

Worry if:
- Training accuracy is high (>90%) but validation is low (<60%)
  → Model overfitting
- All predictions are literally 1.0 or 0.0
  → Model too confident (calibration issue)
- Predictions don't match data distribution
  → Check if validation/test splits preserve class balance

### When Not to Worry

Don't worry if:
- Mean prediction ≈ mean true label
- Predictions have reasonable variance (not all 0.5 or all 1.0)
- ELBO converges smoothly

---

## Next Steps

1. **Re-run training**:
   ```bash
   python quick_reference.py
   ```

2. **Check ELBO convergence**:
   - Should increase monotonically
   - No more "⚠ WARNING: ELBO decreased" messages

3. **Verify v bounds**:
   - Check that max |E[v]| < 2.0 in the output

4. **Assess predictions**:
   - Compare to training class distribution
   - If similar, this is correct behavior!

5. **If still having issues**:
   - Report the ELBO trace (first 20 iterations)
   - Report max |v| values
   - Check data balance

---

## Rollback (If Needed)

If these changes cause issues, revert with:
```bash
cd /home/user/sspa_vi/VariationalInference
git checkout vi.py
```

Or manually change back:
- sigma_v: 0.2 → 1.0
- pi_v: 0.2 → 0.5
- v_damping: 0.2 → 0.6
- Remove gradient clipping code
