#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/core/types.hpp — fundamental data types shared across the library
//
// All public API functions accept and return these types. Keeping them in one
// place makes it easy to swap the underlying scalar type or storage later.
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <complex>
#include <array>
#include <span>
#include <string>
#include <optional>
#include <cstddef>
#include <limits>

namespace cps {

// ── Scalar aliases ───────────────────────────────────────────────────────────

using Real    = double;                  // primary real scalar
using Complex = std::complex<double>;    // primary complex scalar

// ── Filter type ──────────────────────────────────────────────────────────────
// Passed to butter(), cheby1() etc. to select the frequency band.

enum class FilterType {
    Lowpass,   // pass frequencies below the cutoff
    Highpass,  // pass frequencies above the cutoff
    Bandpass,  // pass a band between two cutoffs [Wn_low, Wn_high]
    Bandstop   // reject a band (notch filter)
};

// ── Window type ──────────────────────────────────────────────────────────────
// Used by spectral functions (STFT, Welch) and FIR filter design (firwin).

enum class Window {
    Rectangular,   // no windowing (boxcar)
    Hann,          // good general-purpose window, -18 dB/oct sidelobe roll-off
    Hamming,       // similar to Hann, slightly higher sidelobes but better mainlobe
    Blackman,      // very low sidelobes (-58 dB), wider mainlobe
    BlackmanHarris,// even lower sidelobes (-92 dB)
    Kaiser,        // parameterised: trade mainlobe width against sidelobe level
    FlatTop,       // maximally flat amplitude response — for accurate peak measurement
    Tukey          // tapered cosine, transition between rectangular and Hann
};

// ── Filter representations ───────────────────────────────────────────────────

// Transfer function form: H(z) = B(z)/A(z)
// b[0] + b[1]*z^-1 + ... over a[0] + a[1]*z^-1 + ...
// a[0] is always 1.0 (monic denominator).
struct FilterCoeffs {
    std::vector<Real> b;   // numerator polynomial coefficients
    std::vector<Real> a;   // denominator polynomial coefficients
};

// Second-Order Sections (biquad cascade) — numerically preferred for high orders.
// Each row is one biquad: [b0, b1, b2, a0, a1, a2] where a0 is always 1.0.
// The cascade is: H(z) = prod_k (b0_k + b1_k*z^-1 + b2_k*z^-2)
//                              / (1   + a1_k*z^-1 + a2_k*z^-2)
// We store a0 explicitly so indexing is consistent, but always set to 1.0.
using SOSRow = std::array<Real, 6>;   // [b0, b1, b2, a0, a1, a2]
using SOS    = std::vector<SOSRow>;

// Zero-Pole-Gain form: H(z) = gain * prod(z - zeros) / prod(z - poles)
// Used internally during filter design before converting to SOS.
struct ZPK {
    std::vector<Complex> zeros;
    std::vector<Complex> poles;
    Real                 gain = 1.0;
};

// ── Filter options ───────────────────────────────────────────────────────────
// Passed to all filter design functions. Use C++20 designated initialisers:
//   cps::butter(4, 100.0, cps::FilterType::Lowpass, {.fs = 1000.0})

struct FilterOptions {
    // Sample rate in Hz. When provided, Wn is interpreted in Hz.
    // When left as 0.0, Wn is a normalised frequency in [0, 1]
    // where 1.0 = Nyquist (= fs/2). This matches scipy.signal convention.
    Real fs = 0.0;

    // Set true to design a continuous-time (analog) filter in the s-domain.
    // Most users want the default (false = digital IIR filter).
    bool analog = false;
};

// ── Peak detection options ───────────────────────────────────────────────────
// Passed to find_peaks(). All constraints are optional — omit any you don't need.

struct PeakOptions {
    std::optional<Real> height;      // minimum peak height
    std::optional<Real> prominence;  // minimum peak prominence above surroundings
    std::optional<int>  distance;    // minimum samples between peaks
    std::optional<Real> width;       // minimum peak width (samples, at half-prominence)
    std::optional<Real> threshold;   // minimum difference from neighbouring samples
};

// ── Welch PSD options ────────────────────────────────────────────────────────

struct WelchOptions {
    std::size_t                  nperseg  = 256;           // samples per FFT segment
    std::optional<std::size_t>   noverlap = std::nullopt;  // nullopt = nperseg/2 (50%)
    Window                       window   = Window::Hann;  // window applied to each segment
    // true  = one-sided (DC to Nyquist, nfft/2+1 bins)
    // false = two-sided (full fftfreq ordering, nfft bins)
    bool                         onesided = true;
};

// ── STFT options ─────────────────────────────────────────────────────────────

struct STFTOptions {
    std::size_t                  nperseg  = 256;
    std::optional<std::size_t>   noverlap = std::nullopt;  // nullopt = nperseg/2 (50%)
    std::size_t                  nfft     = 0;             // 0 = nperseg (zero-pad if > nperseg)
    Window                       window   = Window::Hann;
    // true  = one-sided (DC to Nyquist, nfft/2+1 bins)
    // false = two-sided (full fftfreq ordering, nfft bins)
    bool                         onesided = true;
};

} // namespace cps
