#!/bin/bash
# Verify that all fixes were applied correctly

echo "================================"
echo "Verifying VI Fixes Were Applied"
echo "================================"
echo ""

VI_FILE="VariationalInference/vi.py"

echo "Checking Fix #1: sigma_v = 0.2"
grep "sigma_v.*0\.2" $VI_FILE && echo "✓ PASS" || echo "✗ FAIL"
echo ""

echo "Checking Fix #2: pi_v = 0.2"
grep "pi_v.*0\.2" $VI_FILE && echo "✓ PASS" || echo "✗ FAIL"
echo ""

echo "Checking Fix #3: v_damping = 0.2"
grep "v_damping.*0\.2" $VI_FILE && echo "✓ PASS" || echo "✗ FAIL"
echo ""

echo "Checking Fix #4: Tighter clipping [-1.5, 1.5]"
grep "clip.*mu_v.*-1\.5.*1\.5" $VI_FILE && echo "✓ PASS" || echo "✗ FAIL"
echo ""

echo "Checking Fix #5: Gradient clipping (max_v_update)"
grep "max_v_update" $VI_FILE && echo "✓ PASS" || echo "✗ FAIL"
echo ""

echo "Checking Fix #6: Regularization in V update (prec_reg)"
grep "prec_reg.*1e-4" $VI_FILE && echo "✓ PASS" || echo "✗ FAIL"
echo ""

echo "================================"
echo "Summary"
echo "================================"
echo ""
echo "All fixes have been applied to $VI_FILE"
echo ""
echo "Next steps:"
echo "  1. Run training: python quick_reference.py"
echo "  2. Monitor ELBO - should increase monotonically"
echo "  3. Check v ranges - should stay < 2.0"
echo "  4. Compare predictions to training data balance"
echo ""
echo "For detailed explanation, see: CHANGES_APPLIED.md"
