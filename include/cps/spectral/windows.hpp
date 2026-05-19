#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/spectral/windows.hpp — window functions for spectral analysis
//
// Window functions taper a finite-length signal segment to reduce spectral
// leakage: the discontinuity at the segment edges would otherwise spread energy
// from a pure tone across all frequency bins (the "leakage" effect).
//
// Trade-off: all windows trade main-lobe width (frequency resolution) against
// side-lobe level (ability to distinguish nearby tones of different amplitude).
//
//  Window          | Side-lobe | Main-lobe | Best for
//  ────────────────┼───────────┼───────────┼────────────────────────────────
//  Rectangular     | -13 dB    | narrowest | pure research / rarely used
//  Hann            | -31 dB    | medium    | general spectral analysis
//  Hamming         | -41 dB    | medium    | speech processing
//  Blackman        | -58 dB    | wide      | when sidelobes must be very low
//  Blackman-Harris | -92 dB    | widest    | precision spectral measurement
//  Flat-top        | -93 dB    | very wide | accurate amplitude measurement
//  Kaiser          | variable  | variable  | parameterised — best trade-off
//  Tukey           | variable  | variable  | partial taper (hybrid)
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include <vector>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace cps {

// ── make_window() ─────────────────────────────────────────────────────────────
//
// Compute a window of `n` samples of the specified type.
// Extra parameters (beta for Kaiser, alpha for Tukey) are passed via the
// overloads below.
//
// The returned values are in [0, 1] with the exception of Flat-top which can
// be slightly negative. The window is symmetric (appropriate for filter design
// and spectral analysis, but NOT for overlap-add synthesis — use periodic
// windows for that, which is a future TODO).

[[nodiscard]] inline std::vector<Real> make_window(Window type, std::size_t n,
                                                   Real param = 0.0)
{
    if (n == 0)
        throw ValueError("make_window: n must be > 0");
    if (n == 1)
        return {Real(1.0)};

    std::vector<Real> w(n);
    const double M = static_cast<double>(n - 1);  // symmetric window denominator

    switch (type) {

    case Window::Rectangular:
        // No tapering — every sample has weight 1.
        std::fill(w.begin(), w.end(), 1.0);
        break;

    case Window::Hann:
        // w[n] = 0.5 * (1 - cos(2πn/M))
        // Named after Julius von Hann (often misnamed "Hanning").
        for (std::size_t i = 0; i < n; ++i)
            w[i] = 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * i / M));
        break;

    case Window::Hamming:
        // Coefficients 0.54/0.46 minimise the maximum side-lobe level (-41 dB).
        for (std::size_t i = 0; i < n; ++i)
            w[i] = 0.54 - 0.46 * std::cos(2.0 * std::numbers::pi * i / M);
        break;

    case Window::Blackman:
        // Three-term raised-cosine window.
        // w[n] = 0.42 - 0.50*cos(2πn/M) + 0.08*cos(4πn/M)
        for (std::size_t i = 0; i < n; ++i)
            w[i] = 0.42
                 - 0.50 * std::cos(2.0 * std::numbers::pi * i / M)
                 + 0.08 * std::cos(4.0 * std::numbers::pi * i / M);
        break;

    case Window::BlackmanHarris:
        // Four-term Blackman-Harris: -92 dB sidelobes.
        // Coefficients from Harris (1978).
        for (std::size_t i = 0; i < n; ++i)
            w[i] = 0.35875
                 - 0.48829 * std::cos(2.0 * std::numbers::pi * i / M)
                 + 0.14128 * std::cos(4.0 * std::numbers::pi * i / M)
                 - 0.01168 * std::cos(6.0 * std::numbers::pi * i / M);
        break;

    case Window::FlatTop:
        // Five-term flat-top window: maximally flat amplitude response near
        // the main lobe. Used when measuring the amplitude of tonal components
        // rather than resolving nearby frequencies.
        // Coefficients from Heinzel et al. (2002).
        for (std::size_t i = 0; i < n; ++i)
            w[i] =  0.21557895
                 - 0.41663158 * std::cos(2.0 * std::numbers::pi * i / M)
                 + 0.27726316 * std::cos(4.0 * std::numbers::pi * i / M)
                 - 0.08357895 * std::cos(6.0 * std::numbers::pi * i / M)
                 + 0.00694737 * std::cos(8.0 * std::numbers::pi * i / M);
        break;

    case Window::Kaiser: {
        // Kaiser window parameterised by beta (= param).
        // beta = 0   → Rectangular
        // beta = 5   → similar to Hamming
        // beta = 8.6 → similar to Blackman
        // beta = 14  → very low sidelobes
        //
        // Uses the zeroth-order modified Bessel function I0.
        // Computed via the series: I0(x) = sum_{k=0}^inf [(x/2)^k / k!]^2
        if (param < 0.0)
            throw ValueError("make_window: Kaiser beta must be >= 0");

        auto I0 = [](double x) {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k <= 30; ++k) {       // 30 terms is more than enough
                term *= (x / 2.0) / k;
                sum  += term * term;
            }
            return sum;
        };

        const double I0_beta = I0(param);
        const double half_M  = M / 2.0;
        for (std::size_t i = 0; i < n; ++i) {
            double x = 1.0 - std::pow((i - half_M) / half_M, 2.0);
            w[i] = I0(param * std::sqrt(std::max(x, 0.0))) / I0_beta;
        }
        break;
    }

    case Window::Tukey: {
        // Tukey (cosine-tapered) window. param = alpha ∈ [0, 1].
        // alpha = 0 → Rectangular, alpha = 1 → Hann.
        // The centre fraction (1-alpha) is all-ones; the edges taper with
        // a raised-cosine over alpha/2 of the window on each side.
        const double alpha = (param <= 0.0) ? 0.5 : param;
        const int    taper = static_cast<int>(std::floor(alpha * M / 2.0));
        for (std::size_t i = 0; i < n; ++i) {
            int ii = static_cast<int>(i);
            if (ii <= taper)
                w[i] = 0.5 * (1.0 - std::cos(std::numbers::pi * ii / taper));
            else if (ii >= static_cast<int>(n) - 1 - taper)
                w[i] = 0.5 * (1.0 - std::cos(std::numbers::pi * (static_cast<int>(n) - 1 - ii) / taper));
            else
                w[i] = 1.0;
        }
        break;
    }

    default:
        throw ValueError("make_window: unknown window type");
    }

    return w;
}

// Convenience wrappers — preferred for clarity at call sites:

[[nodiscard]] inline std::vector<Real> hann_window           (std::size_t n) { return make_window(Window::Hann,           n); }
[[nodiscard]] inline std::vector<Real> hamming_window        (std::size_t n) { return make_window(Window::Hamming,        n); }
[[nodiscard]] inline std::vector<Real> blackman_window       (std::size_t n) { return make_window(Window::Blackman,       n); }
[[nodiscard]] inline std::vector<Real> blackman_harris_window(std::size_t n) { return make_window(Window::BlackmanHarris, n); }
[[nodiscard]] inline std::vector<Real> flat_top_window       (std::size_t n) { return make_window(Window::FlatTop,        n); }
[[nodiscard]] inline std::vector<Real> kaiser_window         (std::size_t n, Real beta)        { return make_window(Window::Kaiser, n, beta);  }
[[nodiscard]] inline std::vector<Real> tukey_window          (std::size_t n, Real alpha = 0.5) { return make_window(Window::Tukey,  n, alpha); }

} // namespace cps
