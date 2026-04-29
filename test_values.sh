#!/bin/bash
# Test a value set: trains standalone quaternary_nn on a target function,
# then runs inference to measure accuracy.
#
# Usage:  ./test_values.sh [lr] [steps] [model_file]
#
# Edit quaternary_nn/include/Q2_ValueSet.hpp to change values, then run this.

LR=${1:-0.00001}
STEPS=${2:-100000}
MODEL=${3:-test_model.gguf}
BINARY=./quaternary_nn/build/quaternary_nn

# Read current value set from header
echo "─── Active Value Set ───"
grep -A 7 'static constexpr ValueSet' quaternary_nn/include/Q2_ValueSet.hpp | head -8

echo ""
echo "─── Training on y = 5x + 2 ───"
echo "  LR=$LR  steps=$STEPS  model=$MODEL"
echo ""

# Remove old model to force fresh start
rm -f "$MODEL"

# Run training: pipe (x, target) pairs to stdin
# Target function: y = 5x + 2, with slight noise
python3 -c "
import random, sys
for i in range($STEPS):
    x = random.uniform(-1.0, 1.0)
    y = 5.0 * x + 2.0 + random.uniform(-0.02, 0.02)
    sys.stdout.write(f'{x} {y}\n')
" | "$BINARY" train "$MODEL" "$LR" 2>&1 | tee /dev/stderr | grep -c 'GROWTH' | xargs -I{} echo "  → {} growth events"

echo ""
echo "─── Inference Test ───"
# Test at specific points
for x in -1.0 -0.5 0.0 0.5 1.0; do
    expected=$(python3 -c "print(5.0 * $x + 2.0)")
    actual=$("$BINARY" infer "$MODEL" "$x" 2>/dev/null)
    diff=$(python3 -c "print(abs($actual - $expected))")
    printf "  x=%5.1f  predicted=%8.4f  expected=%8.4f  error=%8.4f\n" "$x" "$actual" "$expected" "$diff"
done
