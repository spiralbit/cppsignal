#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/spectral/psd.hpp — Power Spectral Density estimation
//
// FULLY IMPLEMENTED:
//   welch()      — Welch's averaged periodogram method
//
// STUB:
//   periodogram() — single-shot PSD estimate (no averaging)
//   csd()         — cross-spectral density between two signals
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include "../backends/fft/pocketfft.hpp"
#include "windows.hpp"
#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace cps {

// Return type for PSD functions: parallel frequency and power arrays.
struct PSDResult {
    std::vector<Real> freqs;  // frequency bins (Hz if fs provided, else normalised)
    std::vector<Real> psd;    // power spectral density at each bin
};

// ── welch() ───────────────────────────────────────────────────────────────────
//
// Estimate the Power Spectral Density using Welch's method.
//
// Welch's method averages periodograms from overlapping windowed segments of
// the signal. Averaging reduces variance (noisiness) at the cost of frequency
// resolution (the bin width = fs / nperseg).
//
// Algorithm:
//   1. Split the signal into overlapping segments of length nperseg.
//   2. Apply a window function to each segment.
//   3. Compute the magnitude-squared FFT of each windowed segment.
//   4. Average the resulting periodograms.
//   5. Normalise by the window's mean squared energy.
//
// Parameters:
//   signal  — input time-domain samples
//   fs      — sample rate in Hz (used for frequency axis and normalisation)
//   opts    — WelchOptions: nperseg, noverlap, window, onesided
//             noverlap=0 → defaults to nperseg/2 (50% overlap)
//
// Returns PSDResult with:
//   .freqs  — frequency bins in Hz
//   .psd    — PSD in units of signal² / Hz  (one-sided by default)
//             Integrate over frequency to recover signal power: ∫psd·df = σ²
//
// Example:
//   auto [f, psd] = cps::welch(signal, 1000.0);

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] PSDResult welch(std::span<const Real> signal, Real fs,
                              WelchOptions opts = {}, B backend = {})
{
    if (signal.empty())
        throw ValueError("welch: signal must not be empty");
    if (fs <= 0.0)
        throw ValueError("welch: fs must be > 0");

    const std::size_t N = signal.size();
    if (opts.nperseg > N)
        throw ValueError("welch: signal is shorter than nperseg");
    const std::size_t nperseg = opts.nperseg;
    // Default overlap: 50% (standard for Welch's method)
    const std::size_t noverlap = (opts.noverlap == 0)
                                     ? nperseg / 2
                                     : std::min(opts.noverlap, nperseg - 1);
    const std::size_t step     = nperseg - noverlap;
    const std::size_t nfft     = nperseg;          // no zero-padding for now
    const std::size_t nfft_half = nfft / 2 + 1;   // rfft output length

    // ── Build window ─────────────────────────────────────────────────────────
    std::vector<Real> win = make_window(opts.window, nperseg);

    // Window normalisation factor: S2 = sum(w²) / nperseg
    // This ensures the PSD is in units of V²/Hz (variance-normalised).
    double S2 = 0.0;
    for (double w : win) S2 += w * w;
    S2 /= static_cast<double>(nperseg);

    // ── Accumulate periodograms ───────────────────────────────────────────────
    std::vector<double> psd_accum(nfft_half, 0.0);
    std::size_t n_segments = 0;

    for (std::size_t start = 0; start + nperseg <= N; start += step) {
        // Extract and window the segment
        std::vector<Real> seg(nperseg);
        for (std::size_t i = 0; i < nperseg; ++i)
            seg[i] = signal[start + i] * win[i];

        // rfft of windowed segment
        std::vector<Complex> spec(nfft_half);
        backend.rfft(seg, spec);

        // Accumulate |X[k]|²
        for (std::size_t k = 0; k < nfft_half; ++k)
            psd_accum[k] += std::norm(spec[k]);   // std::norm = |z|²

        ++n_segments;
    }

    // ── Average and normalise ─────────────────────────────────────────────────
    // PSD[k] = mean_over_segs( |X_seg[k]|² ) / (fs * sum(w²) * nperseg)
    //
    // scale absorbs: segment averaging, window energy (S2 = sum(w²)/nperseg),
    // and the DFT normalisation factor 1/nperseg. Result is in V²/Hz.
    const double scale = 1.0 / (fs * S2 * static_cast<double>(n_segments)
                                     * static_cast<double>(nperseg));

    if (opts.onesided) {
        // One-sided: nfft_half bins, DC to Nyquist.
        // Interior bins (k ≠ 0 and k ≠ Nyquist) are doubled to fold in the
        // energy from the negative-frequency mirror.
        std::vector<Real> psd(nfft_half);
        for (std::size_t k = 0; k < nfft_half; ++k) {
            psd[k] = psd_accum[k] * scale;
            if (k > 0 && k < nfft_half - 1)
                psd[k] *= 2.0;
        }
        std::vector<Real> freqs(nfft_half);
        for (std::size_t k = 0; k < nfft_half; ++k)
            freqs[k] = static_cast<double>(k) * fs / static_cast<double>(nfft);
        return {std::move(freqs), std::move(psd)};
    } else {
        // Two-sided: nfft bins in fftfreq ordering (positive then negative).
        // Each half-spectrum bin is mirrored; DC and (for even nfft) Nyquist
        // appear only once.
        std::vector<Real> psd(nfft, 0.0);
        psd[0] = psd_accum[0] * scale;
        for (std::size_t k = 1; k < nfft_half; ++k) {
            double v = psd_accum[k] * scale;
            psd[k] = v;
            if (k != nfft - k)          // avoid writing Nyquist twice (even nfft)
                psd[nfft - k] = v;
        }
        std::vector<Real> freqs(nfft);
        for (std::size_t k = 0; k < nfft_half; ++k)
            freqs[k] = static_cast<double>(k) * fs / static_cast<double>(nfft);
        for (std::size_t k = nfft_half; k < nfft; ++k)
            freqs[k] = (static_cast<double>(k) - static_cast<double>(nfft))
                       * fs / static_cast<double>(nfft);
        return {std::move(freqs), std::move(psd)};
    }
}


// ── periodogram() ─────────────────────────────────────────────────────────────
//
// Simple single-shot PSD estimate (no segment averaging).
// Higher variance than welch() — only useful for short, stationary signals.
// TODO: implement (straightforward — just Welch with n_segments=1).

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] PSDResult periodogram(std::span<const Real> /*signal*/, Real /*fs*/,
                                    Window /*win*/ = Window::Hann, B /*backend*/ = {})
{
    throw NotImplemented("periodogram");
}


// ── csd() ─────────────────────────────────────────────────────────────────────
//
// Cross-Spectral Density between signals x and y.
// CSD[k] = E[ X*[k] · Y[k] ]  where X,Y are FFTs of x,y.
// The real part gives the cospectrum; imaginary part gives the quadrature spectrum.
// Used to compute coherence and transfer functions between signals.
// TODO: implement (Welch's method applied to the cross-product).

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] PSDResult csd(std::span<const Real> /*x*/, std::span<const Real> /*y*/,
                            Real /*fs*/, WelchOptions /*opts*/ = {}, B /*backend*/ = {})
{
    throw NotImplemented("csd");
}

} // namespace cps
