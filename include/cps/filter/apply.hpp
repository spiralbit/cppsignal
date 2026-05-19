#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/filter/apply.hpp — apply IIR and FIR filters to a signal
//
// FULLY IMPLEMENTED:
//   sosfilt()  — apply SOS (biquad cascade) with Direct Form II Transposed
//   lfilter()  — apply a transfer-function filter (b, a coefficients)
//
// STUB:
//   filtfilt() — zero-phase forward-backward filtering (TODO)
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace cps {

// ── sosfilt() ─────────────────────────────────────────────────────────────────
//
// Apply a cascade of second-order IIR sections (SOS / biquad filter) to signal x.
//
// Uses Direct Form II Transposed (DFII-T), which is the numerically preferred
// realisation because it minimises the number of delay registers and avoids
// the large intermediate values that occur in Direct Form I.
//
// For each biquad section with coefficients [b0, b1, b2, a0=1, a1, a2]:
//
//   y[n] = b0*x[n] + w1[n-1]
//   w1[n] = b1*x[n] - a1*y[n] + w2[n-1]
//   w2[n] = b2*x[n] - a2*y[n]
//
// The state variables w1, w2 (the "delay line") are initialised to zero
// and carry forward between calls if you process a signal in chunks.
//
// Parameters:
//   sos    — second-order sections from butter(), cheby1(), etc.
//   signal — input samples (read-only view, any contiguous range)
//
// Returns: filtered signal as std::vector<Real>, same length as input.

[[nodiscard]] inline std::vector<Real> sosfilt(const SOS&            sos,
                                               std::span<const Real> signal)
{
    if (sos.empty())
        throw ValueError("sosfilt: SOS is empty");

    std::vector<Real> output(signal.begin(), signal.end());

    // Process each biquad section in series.
    // Each section reads from 'output' and writes back to 'output' in-place,
    // so the cascade is applied without extra allocation.
    for (const auto& row : sos) {
        const double b0 = row[0], b1 = row[1], b2 = row[2];
        // row[3] = a0 (always 1.0, skipped)
        const double a1 = row[4], a2 = row[5];

        // Direct Form II Transposed state
        double w1 = 0.0, w2 = 0.0;

        for (auto& y : output) {
            double x  = y;                        // input to this section
            y         = b0 * x + w1;              // output
            w1        = b1 * x - a1 * y + w2;    // update first state register
            w2        = b2 * x - a2 * y;          // update second state register
        }
    }

    return output;
}


// ── lfilter() ─────────────────────────────────────────────────────────────────
//
// Apply a linear filter defined by numerator (b) and denominator (a) polynomial
// coefficients (transfer-function form). This is equivalent to sosfilt() but
// operates on the raw b/a representation.
//
// WARNING: For high filter orders (n > ~4), the b/a form is numerically
// unstable because small errors in high-order polynomial coefficients cause
// large errors in the frequency response. Prefer sosfilt() with an SOS
// filter whenever possible.
//
// The implementation uses Direct Form II Transposed with a delay line of
// length max(len(b), len(a)) - 1.
//
// Parameters:
//   b      — numerator coefficients  [b0, b1, ..., bM]
//   a      — denominator coefficients [a0=1, a1, ..., aN]
//   signal — input samples
//
// Returns: filtered signal, same length as input.

[[nodiscard]] inline std::vector<Real> lfilter(std::span<const Real> b,
                                               std::span<const Real> a,
                                               std::span<const Real> signal)
{
    if (b.empty() || a.empty())
        throw ValueError("lfilter: b and a must be non-empty");
    if (a[0] == 0.0)
        throw ValueError("lfilter: a[0] must not be zero");

    // Normalise so a[0] = 1
    const double a0 = a[0];
    const std::size_t nb = b.size();
    const std::size_t na = a.size();
    const std::size_t nz = std::max(nb, na) - 1;   // delay line length

    std::vector<Real> output(signal.size());
    std::vector<Real> z(nz, 0.0);      // delay line, zero-initialised

    for (std::size_t n = 0; n < signal.size(); ++n) {
        double x = signal[n];

        // Compute output using the first delay register
        output[n] = (b[0] / a0) * x + (nz > 0 ? z[0] : 0.0);

        // Shift and update the delay line
        for (std::size_t i = 0; i + 1 < nz; ++i)
            z[i] = (i + 1 < nb ? b[i+1] / a0 * x : 0.0)
                 - (i + 1 < na ? a[i+1] / a0 * output[n] : 0.0)
                 + z[i+1];

        if (nz > 0) {
            std::size_t i = nz - 1;
            z[i] = (i + 1 < nb ? b[i+1] / a0 * x : 0.0)
                 - (i + 1 < na ? a[i+1] / a0 * output[n] : 0.0);
        }
    }

    return output;
}


// ── filtfilt() ────────────────────────────────────────────────────────────────
//
// Zero-phase forward-backward filtering.
// The filter is applied twice: once forward, once backward on the result.
// This doubles the effective filter order and eliminates all phase distortion,
// at the cost of being non-causal (cannot be used in real-time).
//
// Particularly useful for offline analysis where phase linearity matters
// (e.g. ECG, audio, vibration analysis).
//
// TODO: implement. Requires:
//   1. Apply sosfilt() forward
//   2. Reverse the result
//   3. Apply sosfilt() again (same coefficients)
//   4. Reverse again
//   Plus edge-condition padding to reduce startup transients.

[[nodiscard]] inline std::vector<Real> filtfilt(const SOS&            /*sos*/,
                                                std::span<const Real> /*signal*/)
{
    throw NotImplemented("filtfilt");
}

} // namespace cps
