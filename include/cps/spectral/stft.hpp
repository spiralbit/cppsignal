#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/spectral/stft.hpp — Short-Time Fourier Transform
//
// STATUS: PARTIALLY IMPLEMENTED
//   stft()        — implemented (returns time-frequency complex matrix)
//   spectrogram() — thin wrapper: |stft|²
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

namespace cps {

// ── STFTResult ────────────────────────────────────────────────────────────────
// The STFT produces a 2D time-frequency matrix.
//
//  freqs[f]       — frequency of bin f in Hz (or normalised if fs==0)
//  times[t]       — centre time of frame t in seconds (or samples if fs==0)
//  Zxx[f][t]      — complex STFT coefficient at (frequency f, time t)
//
// To get the spectrogram (power, dB):
//   20 * log10(|Zxx[f][t]| + eps)

struct STFTResult {
    std::vector<Real>                          freqs;   // length = nfreqs
    std::vector<Real>                          times;   // length = n_frames
    std::vector<std::vector<Complex>>          Zxx;     // [nfreqs][n_frames]
};

// ── stft() ────────────────────────────────────────────────────────────────────
//
// Compute the Short-Time Fourier Transform of a real signal.
//
// Splits the signal into overlapping windowed frames and computes the FFT
// of each frame. The result is a 2D complex array representing how the
// frequency content of the signal evolves over time.
//
// Parameters:
//   signal — real-valued time-domain samples
//   fs     — sample rate in Hz (0 = normalised output)
//   opts   — STFTOptions: nperseg, noverlap, nfft, window, onesided
//
// Returns STFTResult (freqs, times, Zxx).

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] STFTResult stft(std::span<const Real> signal, Real fs = 0.0,
                              STFTOptions opts = {}, B backend = {})
{
    if (signal.empty())
        throw ValueError("stft: signal must not be empty");

    const std::size_t N        = signal.size();
    const std::size_t nperseg  = std::min(opts.nperseg, N);
    const std::size_t noverlap = (opts.noverlap == 0)
                                     ? nperseg / 2
                                     : std::min(opts.noverlap, nperseg - 1);
    const std::size_t step     = nperseg - noverlap;
    const std::size_t nfft      = (opts.nfft == 0) ? nperseg : opts.nfft;
    const std::size_t nfft_half = nfft / 2 + 1;
    const std::size_t nfreqs    = opts.onesided ? nfft_half : nfft;

    // Build window
    std::vector<Real> win = make_window(opts.window, nperseg);

    // Frames are left-aligned: frame 0 starts at sample 0.
    std::vector<std::size_t> frame_starts;
    for (std::size_t s = 0; s + nperseg <= N; s += step)
        frame_starts.push_back(s);

    const std::size_t n_frames = frame_starts.size();

    // Allocate output: Zxx[freq][time]
    STFTResult result;
    result.Zxx.assign(nfreqs, std::vector<Complex>(n_frames));

    // Frequency axis
    result.freqs.resize(nfreqs);
    if (opts.onesided) {
        for (std::size_t k = 0; k < nfreqs; ++k)
            result.freqs[k] = (fs > 0.0)
                                  ? static_cast<double>(k) * fs / static_cast<double>(nfft)
                                  : static_cast<double>(k) / static_cast<double>(nfft);
    } else {
        // fftfreq ordering: positive bins first, then negative
        for (std::size_t k = 0; k < nfft_half; ++k)
            result.freqs[k] = (fs > 0.0)
                                  ? static_cast<double>(k) * fs / static_cast<double>(nfft)
                                  : static_cast<double>(k) / static_cast<double>(nfft);
        for (std::size_t k = nfft_half; k < nfft; ++k)
            result.freqs[k] = (fs > 0.0)
                                  ? (static_cast<double>(k) - static_cast<double>(nfft)) * fs / static_cast<double>(nfft)
                                  : (static_cast<double>(k) - static_cast<double>(nfft)) / static_cast<double>(nfft);
    }

    // Time axis (centre of each frame)
    result.times.resize(n_frames);
    for (std::size_t t = 0; t < n_frames; ++t) {
        double centre_sample = static_cast<double>(frame_starts[t]) + static_cast<double>(nperseg) / 2.0;
        result.times[t] = (fs > 0.0) ? centre_sample / fs : centre_sample;
    }

    // Per-frame FFT
    std::vector<Real>    seg(nfft, 0.0);        // zero-padded frame buffer
    std::vector<Complex> spec(nfft_half);        // rfft always gives nfft_half bins

    for (std::size_t t = 0; t < n_frames; ++t) {
        std::size_t start = frame_starts[t];

        // Window and copy into zero-padded buffer
        std::fill(seg.begin(), seg.end(), 0.0);
        for (std::size_t i = 0; i < nperseg; ++i)
            seg[i] = signal[start + i] * win[i];

        backend.rfft(seg, spec);

        // Copy positive-frequency bins (always present)
        for (std::size_t k = 0; k < nfft_half; ++k)
            result.Zxx[k][t] = spec[k];

        if (!opts.onesided) {
            // Reconstruct negative-frequency bins via conjugate symmetry:
            // X[N-k] = conj(X[k]) for a real input signal.
            for (std::size_t k = nfft_half; k < nfft; ++k)
                result.Zxx[k][t] = std::conj(spec[nfft - k]);
        }
    }

    return result;
}


// ── spectrogram() ─────────────────────────────────────────────────────────────
//
// Convenience wrapper: returns |STFT|² (power spectrogram) as a 2D real matrix.
// Equivalent to: |stft().Zxx|²

struct SpectrogramResult {
    std::vector<Real>              freqs;
    std::vector<Real>              times;
    std::vector<std::vector<Real>> power;   // power[freq][time] = |Zxx|²
};

template<FFTBackend B = backends::PocketFFT>
[[nodiscard]] SpectrogramResult spectrogram(std::span<const Real> signal, Real fs = 0.0,
                                            STFTOptions opts = {}, B backend = {})
{
    auto s = stft(signal, fs, opts, backend);

    SpectrogramResult result;
    result.freqs = std::move(s.freqs);
    result.times = std::move(s.times);

    const std::size_t nf = s.Zxx.size();
    const std::size_t nt = nf > 0 ? s.Zxx[0].size() : 0;

    result.power.assign(nf, std::vector<Real>(nt));
    for (std::size_t f = 0; f < nf; ++f)
        for (std::size_t t = 0; t < nt; ++t)
            result.power[f][t] = std::norm(s.Zxx[f][t]);   // |z|²

    return result;
}

} // namespace cps
