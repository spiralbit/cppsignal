#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/measure/metrics.hpp — signal quality and distortion metrics
//
// FULLY IMPLEMENTED:
//   rms()    — root mean square amplitude
//   snr()    — signal-to-noise ratio in dB
//   thd()    — total harmonic distortion in dB
//   sinad()  — signal-to-noise and distortion ratio in dB
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include "../spectral/fft.hpp"
#include <vector>
#include <span>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace cps {

// ── rms() ─────────────────────────────────────────────────────────────────────
//
// Root Mean Square amplitude: sqrt( (1/N) * sum(x[n]²) )
// Equivalent to the standard deviation for zero-mean signals.
// For a pure sine wave of amplitude A: rms = A / sqrt(2).

[[nodiscard]] inline Real rms(std::span<const Real> x)
{
    if (x.empty()) throw ValueError("rms: signal must not be empty");
    double sum = 0.0;
    for (auto v : x) sum += v * v;
    return std::sqrt(sum / static_cast<double>(x.size()));
}


// ── snr() ─────────────────────────────────────────────────────────────────────
//
// Signal-to-Noise Ratio in dB.
//
// Given a clean signal and its noisy version (or a separate noise estimate),
// computes: SNR = 10 * log10( power(signal) / power(noise) )
//
// Overload 1: SNR from signal + noise (two separate vectors).
//   signal — the reference (noiseless) signal
//   noise  — the noise component only (not signal + noise)
//
// Overload 2: SNR from a spectral peak.
//   Finds the strongest frequency bin (signal) and computes the ratio of its
//   power to the sum of all other bins (noise floor).

[[nodiscard]] inline Real snr(std::span<const Real> signal, std::span<const Real> noise)
{
    if (signal.size() != noise.size())
        throw ValueError("snr: signal and noise must have the same length");

    double sig_power  = 0.0, noise_power = 0.0;
    for (auto v : signal) sig_power   += v * v;
    for (auto v : noise)  noise_power += v * v;

    if (noise_power == 0.0)
        throw NumericalError("snr: noise power is zero");

    return 10.0 * std::log10(sig_power / noise_power);
}

// SNR from a single waveform: assumes the dominant spectral peak is signal,
// everything else is noise. Requires rfft internally.
template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] Real snr(std::span<const Real> x, B backend = {})
{
    if (x.empty()) throw ValueError("snr: signal must not be empty");

    auto spec = rfft(x, backend);

    // Power at each bin
    std::vector<double> power(spec.size());
    for (std::size_t k = 0; k < spec.size(); ++k)
        power[k] = std::norm(spec[k]);

    // Find the peak bin (signal)
    auto it = std::max_element(power.begin(), power.end());
    double sig_power = *it;

    // Sum of all other bins = noise
    double noise_power = 0.0;
    for (std::size_t k = 0; k < power.size(); ++k)
        if (power.begin() + k != it) noise_power += power[k];

    if (noise_power == 0.0)
        throw NumericalError("snr: no noise component detected (pure tone?)");

    return 10.0 * std::log10(sig_power / noise_power);
}


// ── thd() ─────────────────────────────────────────────────────────────────────
//
// Total Harmonic Distortion in dB.
//
// THD = 10 * log10( (V2² + V3² + … + Vn²) / V1² )
//
// where V1 is the fundamental amplitude and V2…Vn are harmonic amplitudes.
// THD measures how much energy a nonlinear system has added at harmonics of
// the input frequency.
//
// Parameters:
//   x          — output waveform of the system under test
//   fundamental — fundamental frequency in Hz
//   fs          — sample rate in Hz
//   n_harmonics — how many harmonics to include (default 5)
//
// Returns THD in dB (negative = harmonics are below fundamental; typical
// values for a good DAC: < -90 dB; cheap op-amp: -60 dB).

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] Real thd(std::span<const Real> x, Real fundamental, Real fs,
                       int n_harmonics = 5, B backend = {})
{
    if (fundamental <= 0.0) throw ValueError("thd: fundamental must be > 0");
    if (fs <= 0.0)           throw ValueError("thd: fs must be > 0");
    if (n_harmonics < 1)     throw ValueError("thd: n_harmonics must be >= 1");

    const std::size_t N = x.size();
    auto spec = rfft(x, backend);

    // Frequency resolution in Hz per bin
    double bin_hz = fs / static_cast<double>(N);

    // Helper: find power at the bin nearest to a given frequency
    auto bin_power = [&](double freq) -> double {
        std::size_t k = static_cast<std::size_t>(std::round(freq / bin_hz));
        k = std::min(k, spec.size() - 1);
        return std::norm(spec[k]);
    };

    double fundamental_power = bin_power(fundamental);
    if (fundamental_power == 0.0)
        throw NumericalError("thd: no signal found at fundamental frequency");

    double harmonic_sum = 0.0;
    for (int h = 2; h <= n_harmonics + 1; ++h) {
        double hf = h * fundamental;
        if (hf >= fs / 2.0) break;   // above Nyquist
        harmonic_sum += bin_power(hf);
    }

    return 10.0 * std::log10(harmonic_sum / fundamental_power);
}


// ── sinad() ───────────────────────────────────────────────────────────────────
//
// Signal-to-Noise And Distortion ratio in dB.
// SINAD = 10 * log10( V1² / (noise + harmonics) )
//       = 10 * log10( fundamental_power / (total_power - fundamental_power) )
//
// Unlike SNR, SINAD includes harmonic distortion products in the denominator,
// making it a more complete measure of a DAC or ADC's performance.

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] Real sinad(std::span<const Real> x, Real fundamental, Real fs, B backend = {})
{
    if (fundamental <= 0.0) throw ValueError("sinad: fundamental must be > 0");
    if (fs <= 0.0)           throw ValueError("sinad: fs must be > 0");

    const std::size_t N = x.size();
    auto spec = rfft(x, backend);

    double bin_hz = fs / static_cast<double>(N);
    std::size_t fund_bin = static_cast<std::size_t>(std::round(fundamental / bin_hz));
    fund_bin = std::min(fund_bin, spec.size() - 1);

    double total_power = 0.0, fund_power = 0.0;
    for (std::size_t k = 0; k < spec.size(); ++k) {
        double p = std::norm(spec[k]);
        total_power += p;
        if (k == fund_bin) fund_power = p;
    }

    double distortion = total_power - fund_power;
    if (distortion <= 0.0)
        throw NumericalError("sinad: no noise or distortion detected");

    return 10.0 * std::log10(fund_power / distortion);
}

} // namespace cps
