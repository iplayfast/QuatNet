// Q2_ValueSet.hpp — Quaternary value table
// Change these constants to experiment with different 2-bit mappings.
//
// Bit encoding:  00=pos_strong, 01=pos_weak, 10=neg_weak, 11=neg_strong
//
// Constraints:
//   pos_strong > pos_weak > 0 > neg_weak > neg_strong
//   threshold_hi between pos_strong and pos_weak
//   threshold_lo between neg_weak and neg_strong
//   Zero boundary at 0
//
// NOTE: llama.cpp has a matching q2_Q_dequant[] in ggml-quants.c
// that must be updated in sync if you deploy new values there.
//
// Recommended sets:
//   {1.0, 0.5, -0.5, -1.0}  thresh ±0.75  — equal magnitude
//   {1.0, 0.1, -0.1, -1.0}  thresh ±0.55  — broad weak range
//   {1.0, 0.01, -0.01, -1.0} thresh ±0.505 — near-zero weak (original)
//   {1.0, 0.25, -0.25, -1.0} thresh ±0.625 — intermediate

#pragma once

namespace Quaternary {

struct ValueSet {
    float pos_strong;
    float pos_weak;
    float neg_weak;
    float neg_strong;
    float threshold_hi;  // boundary: pos_strong vs pos_weak
    float threshold_lo;  // boundary: neg_weak vs neg_strong
};

// ─── CHANGE THIS TO EXPERIMENT ──────────────────────────────────
// Default: equal magnitude, natural thresholds
static constexpr ValueSet ACTIVE_VALUES = {
    /*pos_strong=*/  1.0f,
    /*pos_weak=*/    0.5f,
    /*neg_weak=*/   -0.5f,
    /*neg_strong=*/ -1.0f,
    /*threshold_hi=*/ 0.75f,   // midpoint of 1.0 and 0.5
    /*threshold_lo=*/-0.75f,   // midpoint of -0.5 and -1.0
};

// Derived: zero separates positive from negative
// w > threshold_hi   → pos_strong  (00)
// w > 0              → pos_weak    (01)
// w > threshold_lo   → neg_weak    (10)
// else               → neg_strong  (11)

} // namespace Quaternary
