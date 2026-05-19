#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/filter/analysis.hpp — analyse a filter's frequency response
//
// FULLY IMPLEMENTED:
//   freqz() — compute H(e^jω) at evenly-spaced frequencies
//
// STUB:
//   group_delay() — dω/dφ(ω), useful for phase-linear filter verification
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>

namespace cps {

// Return type for freqz: parallel arrays of frequency (Hz or normalised)
// and complex frequency response H(e^jω).
struct FreqzResult {
    std::vector<Real>    freqs;  // frequencies at which H was evaluated
    std::vector<Complex> H;      // complex frequency response H(e^jω)
};

// ── freqz() ───────────────────────────────────────────────────────────────────
//
// Compute the frequency response of an SOS filter at `nfreqs` equally-spaced
// frequencies from 0 to just below the Nyquist frequency.
//
// The response is evaluated by substituting z = e^(jω) into the transfer
// function for each biquad section and multiplying them together.
//
// Parameters:
//   sos    — SOS from butter(), cheby1(), etc.
//   nfreqs — number of frequency points (default 512)
//   fs     — sample rate in Hz. If > 0, returned freqs are in Hz.
//             If == 0 (default), returned freqs are normalised [0, 0.5].
//
// Returns FreqzResult with:
//   .freqs — evaluation frequencies
//   .H     — complex H(e^jω) at each frequency
//
// To get magnitude in dB:  20 * log10(|H|)
// To get phase in degrees: atan2(imag(H), real(H)) * 180 / π
//
// Example:
//   auto sos = cps::butter(4, 0.1, cps::FilterType::Lowpass);
//   auto [f, H] = cps::freqz(sos, 1024, 1000.0);
//   for (size_t i = 0; i < f.size(); ++i)
//       std::cout << f[i] << " Hz  " << 20*std::log10(std::abs(H[i])) << " dB\n";

[[nodiscard]] inline FreqzResult freqz(const SOS& sos, std::size_t nfreqs = 512, Real fs = 0.0)
{
    if (sos.empty())
        throw ValueError("freqz: SOS is empty");
    if (nfreqs == 0)
        throw ValueError("freqz: nfreqs must be > 0");

    FreqzResult result;
    result.freqs.resize(nfreqs);
    result.H.resize(nfreqs);

    // Evaluate at ω = 0, Δω, 2Δω, … where Δω = π / nfreqs
    // (from DC up to but not including Nyquist = π rad/sample)
    for (std::size_t k = 0; k < nfreqs; ++k) {
        double omega = std::numbers::pi * double(k) / double(nfreqs);

        // z = e^(jω)
        Complex z(std::cos(omega), std::sin(omega));

        // H(z) = prod over SOS sections of (b0 + b1*z^-1 + b2*z^-2)
        //                                  / (1  + a1*z^-1 + a2*z^-2)
        Complex H(1.0, 0.0);
        Complex zinv  = 1.0 / z;         // z^-1
        Complex zinv2 = zinv * zinv;      // z^-2

        for (const auto& row : sos) {
            Complex num = row[0] + row[1] * zinv + row[2] * zinv2;
            Complex den = row[3] + row[4] * zinv + row[5] * zinv2;
            // row[3] is always 1.0 but we use it explicitly for correctness
            H *= num / den;
        }

        // Convert frequency to Hz if fs was provided
        result.freqs[k] = (fs > 0.0)
            ? omega / std::numbers::pi * (fs / 2.0)   // 0 … fs/2
            : omega / std::numbers::pi * 0.5;          // 0 … 0.5

        result.H[k] = H;
    }

    return result;
}

// ── group_delay() ─────────────────────────────────────────────────────────────
//
// Group delay = -dφ/dω where φ is the unwrapped phase of H(e^jω).
// Flat group delay → linear phase → no waveform distortion.
// Butterworth and Chebyshev filters have non-constant group delay near cutoff.
// Bessel filters are designed specifically for constant group delay.
//
// TODO: implement via the formula:
//   gd(ω) = -Re[ z * H'(z) / H(z) ]  evaluated at z = e^jω

[[nodiscard]] inline std::vector<Real> group_delay(const SOS& /*sos*/, std::size_t /*nfreqs*/ = 512)
{
    throw NotImplemented("group_delay");
}

} // namespace cps
