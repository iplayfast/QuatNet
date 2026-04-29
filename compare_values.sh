#!/bin/bash
# Compare multiple value sets on the same training task.
# Run this, change Q2_ValueSet.hpp, run again, compare results.
#
# Output: prints a table of value sets and their final inference errors.

BINARY=./quaternary_nn/build/quaternary_nn
MODEL=/tmp/valtest_model.gguf
LR=0.00001
STEPS=50000

echo "Value Set Comparison: y = 5x + 2"
echo "  LR=$LR  steps=$STEPS per run"
echo ""

# The current set is read from the header
echo "Current values in Q2_ValueSet.hpp:"
grep -B1 'pos_strong' quaternary_nn/include/Q2_ValueSet.hpp | head -4
grep 'pos_weak\|neg_weak\|neg_strong' quaternary_nn/include/Q2_ValueSet.hpp
grep 'threshold_hi\|threshold_lo' quaternary_nn/include/Q2_ValueSet.hpp | head -2
echo ""

# Quick sanity check on the value constraints
POS_S=$(grep 'pos_strong' quaternary_nn/include/Q2_ValueSet.hpp | grep -oP '[-0-9.]+' | head -1)
POS_W=$(grep 'pos_weak'   quaternary_nn/include/Q2_ValueSet.hpp | grep -oP '[-0-9.]+' | head -1)
NEG_W=$(grep 'neg_weak'   quaternary_nn/include/Q2_ValueSet.hpp | grep -oP '[-0-9.]+' | head -1)
NEG_S=$(grep 'neg_strong' quaternary_nn/include/Q2_ValueSet.hpp | grep -oP '[-0-9.]+' | head -1)

echo "Constraints check:"
python3 -c "
ps = $POS_S; pw = $POS_W; nw = $NEG_W; ns = $NEG_S
ok = True
if not (ps > pw > 0): print('  ✗ FAIL: pos_strong (%.2f) must be > pos_weak (%.2f) > 0' % (ps, pw)); ok = False
if not (0 > nw > ns): print('  ✗ FAIL: 0 must be > neg_weak (%.2f) > neg_strong (%.2f)' % (nw, ns)); ok = False
if ok: print('  ✓ Values satisfy ordering constraints')
# Suggest thresholds
print(f'  Suggested threshold_hi: {(ps + pw) / 2:.2f}')
print(f'  Suggested threshold_lo: {(nw + ns) / 2:.2f}')
print(f'  Range: {ns} to {ps}')
print(f'  Resolution: 2 bits = 4 states')
print(f'  # of unique output levels per head: {vector_size} × 4 states')
"
