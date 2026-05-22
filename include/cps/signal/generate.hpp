#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/signal/generate.hpp — deterministic and stochastic signal generators
//
// All generators accept a time array t (in seconds) or a count n plus a
// sample rate fs. Functions return std::vector<Real>.
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <vector>
#include <span>
#include <cmath>
#include <numbers>
#include <random>
#include <chrono>
#include <stdexcept>

namespace cps {

// ── linspace() / arange() ────────────────────────────────────────────────────
// Helper: evenly-spaced time array, matching numpy.linspace / numpy.arange.

[[nodiscard]] inline std::vector<Real> linspace(Real start, Real stop, std::size_t n,
                                               bool endpoint = true)
{
    if (n == 0) return {};
    if (n == 1) return {start};
    std::vector<Real> t(n);
    double step = (stop - start) / (endpoint ? (n - 1) : n);
    for (std::size_t i = 0; i < n; ++i)
        t[i] = start + i * step;
    return t;
}

[[nodiscard]] inline std::vector<Real> arange(Real start, Real stop, Real step = 1.0)
{
    if (step == 0.0 || std::isnan(step)) throw ValueError("arange: step must not be zero or NaN");
    double count_d = std::max(0.0, std::ceil((stop - start) / step));
    if (count_d > 1e15)
        throw ValueError("arange: range too large (would exceed 10^15 elements)");
    std::size_t n = static_cast<std::size_t>(count_d);
    std::vector<Real> t(n);
    for (std::size_t i = 0; i < n; ++i)
        t[i] = start + i * step;
    return t;
}


// ── sinusoid() ────────────────────────────────────────────────────────────────
//
// Generate a pure sine wave: y[n] = amplitude * sin(2π * freq * t[n] + phase)
//
// Parameters:
//   t         — time vector in seconds
//   freq      — frequency in Hz
//   amplitude — peak amplitude (default 1.0)
//   phase     — initial phase in radians (default 0.0)

[[nodiscard]] inline std::vector<Real> sinusoid(std::span<const Real> t, Real freq,
                                               Real amplitude = 1.0, Real phase = 0.0)
{
    std::vector<Real> y(t.size());
    const double twopi_f = 2.0 * std::numbers::pi * freq;
    for (std::size_t i = 0; i < t.size(); ++i)
        y[i] = amplitude * std::sin(twopi_f * t[i] + phase);
    return y;
}


// ── chirp() ───────────────────────────────────────────────────────────────────
//
// Generate a frequency-swept sinusoid (chirp). The instantaneous frequency
// increases linearly from f0 at t=0 to f1 at t=t1.
//
// The instantaneous phase at time t is:
//   φ(t) = 2π * [f0*t + (f1-f0)/(2*t1) * t²]
//
// Parameters:
//   t   — time vector in seconds
//   f0  — start frequency in Hz
//   f1  — end frequency in Hz at time t1
//   t1  — reference time (usually the end of the sweep)
//   phi — initial phase offset in degrees (default 0)

[[nodiscard]] inline std::vector<Real> chirp(std::span<const Real> t, Real f0, Real f1,
                                            Real t1, Real phi_deg = 0.0)
{
    if (!(t1 > 0.0))
        throw ValueError("chirp: t1 must be > 0");

    const double phi0   = phi_deg * std::numbers::pi / 180.0;
    const double k      = (f1 - f0) / t1;   // chirp rate (Hz/s)

    std::vector<Real> y(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        double ti = t[i];
        double phase = 2.0 * std::numbers::pi * (f0 * ti + 0.5 * k * ti * ti) + phi0;
        y[i] = std::cos(phase);
    }
    return y;
}


// ── gausspulse() ──────────────────────────────────────────────────────────────
//
// Gaussian-modulated sinusoidal pulse, centred at t=0:
//   y(t) = exp(-α * t²) * cos(2π * fc * t)
//   where α = -(π * fc * bw)² / (4 * ln(10) * bw_db/20)
//
// With default parameters (bw=0.5, bw_db=-6):
//   the envelope is at -6 dB at t = ±bw/(2*fc)
//
// Parameters:
//   t      — time vector in seconds
//   fc     — carrier frequency in Hz
//   bw     — fractional bandwidth at bw_db dB (default 0.5 = 50%)
//   bw_db  — reference level for bw in dB (default -6)

[[nodiscard]] inline std::vector<Real> gausspulse(std::span<const Real> t, Real fc,
                                                  Real bw = 0.5, Real bw_db = -6.0)
{
    if (!(fc    > 0.0)) throw ValueError("gausspulse: fc must be > 0");
    if (!(bw    > 0.0)) throw ValueError("gausspulse: bw must be > 0");
    if (!(bw_db < 0.0)) throw ValueError("gausspulse: bw_db must be < 0 "
                                         "(it is a reference level in dB, e.g. -6)");

    // α such that the Gaussian envelope is at bw_db at half-bandwidth.
    // ref < 1 (e.g. 0.5 for -6 dB), so log(ref) < 0, making alpha < 0 (decaying).
    const double ref  = std::pow(10.0, bw_db / 20.0);    // linear amplitude at bw_db
    const double alpha = (std::numbers::pi * fc * bw) * (std::numbers::pi * fc * bw)
                         / (4.0 * std::log(ref));

    std::vector<Real> y(t.size());
    const double twopi_fc = 2.0 * std::numbers::pi * fc;
    for (std::size_t i = 0; i < t.size(); ++i) {
        double ti = t[i];
        y[i] = std::exp(alpha * ti * ti) * std::cos(twopi_fc * ti);
    }
    return y;
}


// ── unit_impulse() ────────────────────────────────────────────────────────────
//
// Discrete unit impulse (Kronecker delta): 1 at idx, 0 elsewhere.
// Useful for computing impulse responses: filter_output = lfilter(b, a, impulse).

[[nodiscard]] inline std::vector<Real> unit_impulse(std::size_t n, std::size_t idx = 0)
{
    if (idx >= n)
        throw ValueError("unit_impulse: idx must be < n");
    std::vector<Real> y(n, 0.0);
    y[idx] = 1.0;
    return y;
}


// ── square_wave() ─────────────────────────────────────────────────────────────
//
// Square wave oscillating between -1 and +1 at the given frequency.
// duty: fraction of the period spent at +1 (default 0.5 = 50%).

[[nodiscard]] inline std::vector<Real> square_wave(std::span<const Real> t, Real freq,
                                                   Real duty = 0.5)
{
    if (!(duty > 0.0 && duty < 1.0))   // catches NaN, ±inf, out-of-range
        throw ValueError("square_wave: duty must be in (0, 1)");

    std::vector<Real> y(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        double phase = std::fmod(t[i] * freq, 1.0);
        if (phase < 0.0) phase += 1.0;   // fmod is negative for negative t
        y[i] = (phase < duty) ? 1.0 : -1.0;
    }
    return y;
}


// ── sawtooth_wave() ───────────────────────────────────────────────────────────
//
// Sawtooth wave rising from -1 to +1 over each period.
// width=1 → pure sawtooth (rising); width=0 → pure sawtooth (falling).

[[nodiscard]] inline std::vector<Real> sawtooth_wave(std::span<const Real> t, Real freq,
                                                     Real width = 1.0)
{
    if (!(width >= 0.0 && width <= 1.0))   // catches NaN, ±inf, out-of-range
        throw ValueError("sawtooth_wave: width must be in [0, 1] "
                         "(0 = pure falling ramp, 1 = pure rising ramp)");

    std::vector<Real> y(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        double phase = std::fmod(t[i] * freq, 1.0);
        if (phase < 0.0) phase += 1.0;   // fmod is negative for negative t
        if (phase < width)
            y[i] = 2.0 * phase / width - 1.0;
        else
            y[i] = -2.0 * (phase - width) / (1.0 - width) + 1.0;
    }
    return y;
}


// ── white_noise() ─────────────────────────────────────────────────────────────
//
// Gaussian white noise with zero mean and given standard deviation.
// A seed can be provided for reproducible output; pass seed=0 to use a
// time-based seed.

[[nodiscard]] inline std::vector<Real> white_noise(std::size_t n, Real std_dev = 1.0,
                                                   unsigned int seed = 42)
{
    if (!(std_dev > 0.0))
        throw ValueError("white_noise: std_dev must be > 0");
    std::mt19937                     rng(seed == 0
                                             ? static_cast<unsigned>(std::chrono::steady_clock::now()
                                                                          .time_since_epoch().count())
                                             : seed);
    std::normal_distribution<double> dist(0.0, std_dev);

    std::vector<Real> y(n);
    for (auto& v : y) v = dist(rng); // GCOV_EXCL_BR_LINE
    return y;
}

} // namespace cps
