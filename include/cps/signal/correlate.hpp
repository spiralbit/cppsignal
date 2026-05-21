#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/signal/correlate.hpp — convolution and cross-correlation
//
// FULLY IMPLEMENTED (direct O(N·M) method):
//   convolve()    — linear convolution
//   correlate()   — cross-correlation (includes auto-correlation when x==y)
//
// TODO: FFT-based fast convolution for large inputs (overlap-add / overlap-save)
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <vector>
#include <span>
#include <algorithm>
#include <string>

namespace cps {

// Output size modes — matching scipy.signal.convolve mode parameter.
enum class ConvolveMode {
    Full,   // output length = len(x) + len(y) - 1  (no data lost)
    Same,   // output length = max(len(x), len(y))   (centred)
    Valid   // output length = max(len(x), len(y)) - min(len(x), len(y)) + 1
            //                (only where signals fully overlap)
};


// ── convolve() ────────────────────────────────────────────────────────────────
//
// Linear convolution of x and y: (x * y)[n] = sum_k x[k] * y[n-k]
//
// For FIR filtering, call lfilter(h, {1.0}, x) which is equivalent.
// convolve() is useful for general-purpose polynomial multiplication
// and correlation.
//
// Time complexity: O(N * M) direct summation.
// For large signals (N, M > ~1000), FFT-based convolution is much faster
// (O(N log N)) — see TODO above.
//
// Parameters:
//   x, y  — input sequences
//   mode  — Full (default), Same, or Valid
//
// Returns: convolved sequence.

[[nodiscard]] inline std::vector<Real> convolve(std::span<const Real> x,
                                               std::span<const Real> y,
                                               ConvolveMode mode = ConvolveMode::Full)
{
    if (x.empty() || y.empty())
        throw ValueError("convolve: inputs must not be empty");

    const std::size_t Nx = x.size();
    const std::size_t Ny = y.size();
    const std::size_t Nfull = Nx + Ny - 1;

    // Compute full convolution
    std::vector<Real> full(Nfull, 0.0);
    for (std::size_t i = 0; i < Nx; ++i)
        for (std::size_t j = 0; j < Ny; ++j)
            full[i + j] += x[i] * y[j];

    // Trim to requested mode
    switch (mode) {
    case ConvolveMode::Same: {
        const std::size_t Nout  = std::max(Nx, Ny);
        const std::size_t start = (Nfull - Nout) / 2;
        return {full.begin() + start, full.begin() + start + Nout};
    }

    case ConvolveMode::Valid: {
        if (Nx < Ny)
            return convolve(y, x, ConvolveMode::Valid);
        const std::size_t Nout  = Nx - Ny + 1;
        const std::size_t start = Ny - 1;
        return {full.begin() + start, full.begin() + start + Nout};
    }

    default:  // ConvolveMode::Full
        return full;
    }
}


// ── correlate() ───────────────────────────────────────────────────────────────
//
// Cross-correlation of x with y.
// (x ⋆ y)[k] = sum_n x[n] * y[n - k]   (correlation, not convolution)
//
// Equivalently: correlate(x, y) = convolve(x, reversed(y))
//
// Auto-correlation: correlate(x, x) gives a symmetric output centred at the
// midpoint, with the zero-lag correlation at index len(x)-1.
//
// For a signal with a repeating pattern of period T, the auto-correlation will
// show a peak at lag T (and its multiples) — useful for pitch detection.
//
// Parameters:
//   x, y  — input sequences
//   mode  — Full, Same, or Valid (same semantics as convolve)
//
// Returns: cross-correlation sequence.

[[nodiscard]] inline std::vector<Real> correlate(std::span<const Real> x,
                                                 std::span<const Real> y,
                                                 ConvolveMode mode = ConvolveMode::Full)
{
    // Correlation is convolution with y time-reversed
    std::vector<Real> y_rev(y.rbegin(), y.rend());
    return convolve(x, y_rev, mode);
}

} // namespace cps
