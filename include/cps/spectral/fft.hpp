#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/spectral/fft.hpp — FFT / IFFT / RFFT public API
//
// All functions are templated on a backend (default: PocketFFT) satisfying the
// cps::FFTBackend concept. Swap backends per call-site if needed:
//
//   auto spec = cps::fft(signal);                          // PocketFFT (default)
//   auto spec = cps::fft<cps::backends::FFTW>(signal);    // FFTW (if enabled)
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/concepts.hpp"
#include "../core/result.hpp"
#include "../backends/fft/pocketfft.hpp"
#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace cps {

// ── fft() ─────────────────────────────────────────────────────────────────────
//
// Complex forward Discrete Fourier Transform.
// out[k] = sum_{n=0}^{N-1}  in[n] * exp(-2πi * k * n / N)
//
// Normalisation: none (scipy/numpy convention). The inverse divides by N.
// To get physical amplitudes from a real signal, use rfft() instead and
// divide by N, or use welch() which handles normalisation correctly.
//
// Template param B: any type satisfying cps::FFTBackend.

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] std::vector<Complex> fft(std::span<const Complex> x, B backend = {})
{
    if (x.empty())
        throw ValueError("fft: input must not be empty");

    std::vector<Complex> out(x.size());
    backend.forward(x, out);
    return out;
}

// Convenience overload: real input is promoted to complex before the transform.
// For real-valued signals, rfft() is twice as fast and should be preferred.
template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] std::vector<Complex> fft(std::span<const Real> x, B backend = {})
{
    // Promote real → complex using the range constructor (implicit conversion).
    std::vector<Complex> cx(x.begin(), x.end());
    return fft<B>(cx, backend);
}


// ── ifft() ────────────────────────────────────────────────────────────────────
//
// Complex inverse DFT.
// out[n] = (1/N) * sum_{k=0}^{N-1}  in[k] * exp(+2πi * k * n / N)
//
// Property: ifft(fft(x)) == x  (within floating-point rounding).

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] std::vector<Complex> ifft(std::span<const Complex> x, B backend = {})
{
    if (x.empty())
        throw ValueError("ifft: input must not be empty");

    std::vector<Complex> out(x.size());
    backend.inverse(x, out);
    return out;
}


// ── rfft() ────────────────────────────────────────────────────────────────────
//
// Real-to-complex forward DFT, exploiting conjugate symmetry.
// For a real input of length N, the output has length N/2 + 1 (one-sided).
// The negative-frequency bins are the complex conjugates of the positive ones
// and are omitted — they carry no additional information.
//
// Output: X[0] is DC (real), X[1..N/2-1] are complex, X[N/2] is Nyquist (real
// for even N). The magnitude spectrum is symmetric around Nyquist.
//
// rfft() is the preferred transform for real signals — it uses half the
// memory and runs roughly twice as fast as fft().

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] std::vector<Complex> rfft(std::span<const Real> x, B backend = {})
{
    if (x.empty())
        throw ValueError("rfft: input must not be empty");

    const std::size_t N    = x.size();
    const std::size_t Nout = N / 2 + 1;

    std::vector<Complex> out(Nout);
    backend.rfft(x, out);
    return out;
}


// ── irfft() ───────────────────────────────────────────────────────────────────
//
// Inverse of rfft(): complex-to-real.
// x      — one-sided spectrum from rfft(), length N/2+1
// n      — length of the original real signal (must be provided because
//           N/2+1 is the same for N=8 and N=9 — the length is ambiguous)
//
// Property: irfft(rfft(x), x.size()) == x  (within floating-point rounding).

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] std::vector<Real> irfft(std::span<const Complex> x, std::size_t n, B backend = {})
{
    if (x.empty())
        throw ValueError("irfft: input must not be empty");
    if (n == 0)
        throw ValueError("irfft: n must be > 0");
    if (x.size() != n / 2 + 1)
        throw ValueError("irfft: spectrum size does not match n (expected n/2+1)");

    std::vector<Real> out(n);
    backend.irfft(x, out, n);
    return out;
}


// ── fftfreq() ─────────────────────────────────────────────────────────────────
//
// Return the frequency bin centres for a complex FFT of length n.
// Matches numpy.fft.fftfreq() exactly.
//
// Parameters:
//   n — FFT length
//   d — sample spacing (= 1/fs). Default 1.0 gives normalised frequencies
//       in cycles/sample ∈ [-0.5, +0.5). Pass 1.0/fs for Hz.
//
// The returned array has the same ordering as fft() output:
//   [0, 1, 2, ..., n/2-1, -n/2, ..., -1] / (n*d)
//
// Example: n=8, d=1 → [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]

[[nodiscard]] inline std::vector<Real> fftfreq(std::size_t n, Real d = 1.0)
{
    if (n == 0)
        throw ValueError("fftfreq: n must be > 0");
    if (!(d > 0.0))
        throw ValueError("fftfreq: d must be > 0 (d = 1/fs; use 1.0 for normalised frequencies)");

    std::vector<Real> freq(n);
    const double inv_nd = 1.0 / (static_cast<double>(n) * d);
    const std::size_t half = (n + 1) / 2;   // ceil(n/2)

    // Positive frequencies first: 0, 1, ..., n/2-1
    for (std::size_t k = 0; k < half; ++k)
        freq[k] = static_cast<double>(k) * inv_nd;

    // Negative frequencies: -n/2, ..., -1
    for (std::size_t k = half; k < n; ++k)
        freq[k] = static_cast<double>(k) * inv_nd - 1.0 / d;

    return freq;
}


// ── rfftfreq() ────────────────────────────────────────────────────────────────
//
// Frequency bin centres for rfft() output (one-sided, non-negative only).
// Length is n/2+1. Matches numpy.fft.rfftfreq().
//
// Example: n=8, d=1 → [0, 0.125, 0.25, 0.375, 0.5]

[[nodiscard]] inline std::vector<Real> rfftfreq(std::size_t n, Real d = 1.0)
{
    if (n == 0)
        throw ValueError("rfftfreq: n must be > 0");
    if (!(d > 0.0))
        throw ValueError("rfftfreq: d must be > 0 (d = 1/fs; use 1.0 for normalised frequencies)");

    const std::size_t Nout = n / 2 + 1;
    std::vector<Real> freq(Nout);
    const double inv_nd = 1.0 / (static_cast<double>(n) * d);
    for (std::size_t k = 0; k < Nout; ++k)
        freq[k] = static_cast<double>(k) * inv_nd;
    return freq;
}

} // namespace cps
