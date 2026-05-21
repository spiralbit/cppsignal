#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/filter/design.hpp — IIR and FIR filter design
//
// FULLY IMPLEMENTED:  butter()  — Butterworth IIR (LP and HP)
//                     firwin()  — Windowed-sinc FIR
//
// STUB (TODO):        cheby1(), cheby2(), ellip(), bessel()
//                     Bandpass / bandstop variants of butter()
//
// All IIR functions return SOS (second-order sections), which is numerically
// superior to transfer-function (b, a) form for high filter orders.
// Use cps::sosfilt() or cps::filtfilt() to apply the returned SOS.
//
// ALGORITHM OVERVIEW — Butterworth IIR design
// ────────────────────────────────────────────
// 1. Compute the analog prototype: n poles equally spaced on the unit circle
//    in the left half of the s-plane (stable by construction).
//
// 2. Prewarp the digital cutoff frequency Wn to an analog cutoff Ωa using
//    the bilinear transform frequency correspondence:
//      Ωa = 2 * tan(π * Wn / 2)      (with sample rate normalised to 1)
//
// 3. LP→LP: scale the prototype poles by Ωa so the -3 dB point lands at Wn.
//
// 4. LP→HP: additional s → Ωa/s substitution for highpass designs.
//
// 5. Bilinear transform: map each analog pole p to a digital pole
//      z = (2 + p) / (2 - p)
//    All implicit analog zeros at s=∞ map to z = -1.
//
// 6. Convert ZPK → SOS for numerical stability.
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <numbers>    // std::numbers::pi (C++20)
#include <span>
#include <stdexcept>
#include <numeric>

namespace cps {

// ═════════════════════════════════════════════════════════════════════════════
// INTERNAL HELPERS  (detail:: namespace — not part of the public API)
// ═════════════════════════════════════════════════════════════════════════════

namespace detail {

// Analog Butterworth prototype poles for a normalised (Ωc = 1) lowpass filter.
// Returns only the upper-half-plane poles (positive imaginary part) plus any
// real pole when order is odd. Lower-half conjugates are implicit in the biquad
// denominator and are never stored explicitly.
//
// Pole angles: θ_k = π(2k + n - 1) / (2n), k = 1…n
// (Proakis & Manolakis, §8.3.1). All Re(θ_k) < 0 → stable.
inline std::vector<Complex> butter_analog_poles(int n) {
    std::vector<Complex> poles;
    poles.reserve(n);
    for (int k = 1; k <= n; ++k) {
        double angle = std::numbers::pi * double(2*k + n - 1) / double(2*n);
        poles.emplace_back(std::cos(angle), std::sin(angle));
    }
    return poles;
}

// Prewarp a normalised digital frequency Wn ∈ (0,1) to its analog equivalent.
// Wn = 1 corresponds to the Nyquist frequency (fs/2).
// Formula: Ωa = 2 * tan(π * Wn / 2)
// This ensures the bilinear transform maps the digital -3 dB point exactly to
// Wn (rather than only approximately, which would happen without prewarping).
inline double prewarp(double Wn) {
    return 2.0 * std::tan(std::numbers::pi * Wn / 2.0);
}

// Apply the bilinear transform to a single analog pole or zero.
// Maps s-domain point to z-domain: z = (2 + s) / (2 - s)
// This is the unit-sample-rate version (fs = 1). Analog poles scaled by
// prewarp() before calling this will land at the correct digital frequency.
inline Complex bilinear(Complex s) {
    return (2.0 + s) / (2.0 - s);
}

// Convert a (zeros, poles, gain) triple to SOS rows.
//
// The cascade is built by pairing complex conjugate pole pairs into biquads.
// Any unpaired real pole (odd filter order) becomes a first-order section
// padded to biquad form with b2=0, a2=0.
//
// Zero pairing: complex conjugate zero pairs are matched to complex pole
// biquads first (keeping all coefficients real). Real zeros fill remaining
// slots; z = -1 is used as a default if the zeros list runs out.
//
// The full gain is placed on the first section.
//
// zeros: digital zeros  (for Butterworth LP all are at z = -1; HP at z = +1)
// poles: digital poles  (complex conjugate pairs + possibly one real)
// gain:  overall scalar multiplier
inline SOS zpk2sos(std::vector<Complex> zeros,
                   std::vector<Complex> poles,
                   double               gain)
{
    SOS sections;

    // Tolerance for deciding whether imaginary part is numerically zero
    constexpr double REAL_TOL = 1e-10;

    // ── Separate real and complex poles ──────────────────────────────────────
    std::vector<double>  real_poles;
    std::vector<Complex> cmplx_poles_upper;
    for (auto& p : poles) {
        if (std::abs(p.imag()) < REAL_TOL)
            real_poles.push_back(p.real());
        else if (p.imag() > 0)
            cmplx_poles_upper.push_back(p);  // lower-half conjugates are implicit in biquad
    }

    // ── Separate real and complex zeros ──────────────────────────────────────
    std::vector<Complex> real_zeros, cmplx_zeros_upper;
    for (auto& z : zeros) {
        if (std::abs(z.imag()) < REAL_TOL)
            real_zeros.push_back(z.real());
        else if (z.imag() > 0)
            cmplx_zeros_upper.push_back(z);  // lower-half conjugates are reconstructed below
    }

    // Sort complex poles so the one closest to the unit circle (most sensitive
    // to coefficient quantisation) is processed last — keeps early sections
    // better-conditioned.
    std::sort(cmplx_poles_upper.begin(), cmplx_poles_upper.end(),
              [](const Complex& a, const Complex& b) {
                  return std::abs(std::abs(a) - 1.0) > std::abs(std::abs(b) - 1.0);
              });

    bool gain_applied = false;

    // ── Zero-retrieval helpers ────────────────────────────────────────────────
    // For biquads (complex pole pairs): prefer a complex conjugate zero pair so
    // all numerator coefficients stay real. Fall back to two real zeros.
    auto next_biquad_zeros = [&]() -> std::pair<Complex, Complex> {
        if (!cmplx_zeros_upper.empty()) {
            Complex z = cmplx_zeros_upper.back();
            cmplx_zeros_upper.pop_back();
            return {z, std::conj(z)};   // (z, z*) → real b coefficients guaranteed
        }
        // Pop two real zeros; default to z = -1 if the list is exhausted.
        auto pop = [&]() -> Complex {
            if (!real_zeros.empty()) { Complex z = real_zeros.back(); real_zeros.pop_back(); return z; }
            return Complex(-1.0, 0.0);
        };
        return {pop(), pop()};
    };

    // For first-order sections (lone real pole): one zero.
    auto next_single_zero = [&]() -> Complex {
        if (!real_zeros.empty()) { Complex z = real_zeros.back(); real_zeros.pop_back(); return z; }
        return Complex(-1.0, 0.0);
    };

    // ── Build biquad for each complex conjugate pole pair ────────────────────
    // Denominator from (z - p)(z - p*) = z² - 2·Re(p)·z + |p|²
    // Numerator from   (z - z1)(z - z2):
    //   b0=1, b1=-(z1+z2).real(), b2=(z1*z2).real()  ← always real when z2=z1*
    for (auto& p : cmplx_poles_upper) {
        auto [z1, z2] = next_biquad_zeros();

        double b0 =  1.0;
        double b1 = -std::real(z1 + z2);
        double b2 =  std::real(z1 * z2);

        double a1 = -2.0 * p.real();
        double a2 =  std::norm(p);   // |p|²

        double g = gain_applied ? 1.0 : gain;
        gain_applied = true;

        sections.push_back({ g*b0, g*b1, g*b2, 1.0, a1, a2 });
    }

    // ── Handle real pole pairs ────────────────────────────────────────────────
    for (std::size_t i = 0; i + 1 < real_poles.size(); i += 2) {
        double p1 = real_poles[i];
        double p2 = real_poles[i+1];

        Complex z1 = next_single_zero();
        Complex z2 = next_single_zero();

        double b0 =  1.0;
        double b1 = -std::real(z1 + z2);
        double b2 =  std::real(z1 * z2);
        double a1 = -(p1 + p2);
        double a2 =   p1 * p2;

        double g = gain_applied ? 1.0 : gain; // GCOV_EXCL_BR_LINE
        gain_applied = true;

        sections.push_back({ g*b0, g*b1, g*b2, 1.0, a1, a2 });
    }

    // ── Handle lone real pole (odd filter order) ──────────────────────────────
    // Padded to biquad form: b2=0, a2=0.
    if (real_poles.size() % 2 == 1) {
        double p   = real_poles.back();
        Complex z1 = next_single_zero();

        double b0 = 1.0;
        double b1 = -z1.real();
        double a1 = -p;

        double g = gain_applied ? 1.0 : gain;
        gain_applied = true;

        sections.push_back({ g*b0, g*b1, 0.0, 1.0, a1, 0.0 });
    }

    if (!cmplx_zeros_upper.empty())
        throw NumericalError("zpk2sos: unmatched complex zero pairs remain after all "
                             "sections — more zeros than poles in the filter design");

    return sections;
}

// Validate and normalise the Wn parameter.
// Returns Wn as a fraction of Nyquist in (0, 1).
// Throws ValueError if Wn or fs is out of range.
inline double normalise_wn(double Wn, const FilterOptions& opts) {
    if (!(opts.fs >= 0.0))   // catches NaN and negative fs
        throw ValueError("FilterOptions::fs must be >= 0 "
                         "(0 means Wn is already a normalised fraction of Nyquist)");

    double Wn_norm = Wn;
    if (opts.fs > 0.0)
        Wn_norm = 2.0 * Wn / opts.fs;   // Hz → fraction of Nyquist

    if (!(Wn_norm > 0.0 && Wn_norm < 1.0))   // catches NaN, ±inf, out-of-range
        throw ValueError("Wn must be in (0, 1) when normalised, "
                         "or in (0, fs/2) when fs is provided");
    return Wn_norm;
} // GCOV_EXCL_LINE

} // namespace detail


// ═════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═════════════════════════════════════════════════════════════════════════════

// ── butter() — Butterworth IIR filter ────────────────────────────────────────
//
// Designs a digital Butterworth filter with maximally flat magnitude response.
// "Maximally flat" means no ripple anywhere in the passband or stopband —
// it's the smoothest possible rolloff for a given order.
//
// Parameters:
//   order  — number of poles (higher order = sharper rolloff)
//   Wn     — cutoff frequency. Normalised ∈ (0,1) if opts.fs==0 (default);
//             in Hz if opts.fs is provided. For LP/HP: scalar.
//             For BP/BS: TODO — pass two cutoffs as {Wn_low, Wn_high}.
//   type   — FilterType::Lowpass or Highpass (Bandpass/Bandstop: TODO)
//   opts   — optional sample rate and analog flag
//
// Returns: SOS (second-order sections), apply with cps::sosfilt().
//
// Example:
//   // 4th-order lowpass at 100 Hz, fs=1000 Hz
//   auto sos = cps::butter(4, 100.0, cps::FilterType::Lowpass, {.fs=1000.0});
//   auto filtered = cps::sosfilt(sos, signal);

[[nodiscard]] inline SOS butter(int order, double Wn, FilterType type,
                                FilterOptions opts = {})
{
    if (order <= 0)
        throw ValueError("filter order must be > 0");

    double Wn_norm = detail::normalise_wn(Wn, opts);

    // ── Step 1: analog prototype poles (normalised Ωc = 1) ───────────────────
    std::vector<Complex> a_poles = detail::butter_analog_poles(order);
    // No finite analog zeros for Butterworth LP prototype
    // (all zeros are at s = ∞, which map to z = -1 after bilinear)

    // ── Step 2: prewarp the digital cutoff → analog cutoff ───────────────────
    double Omega_a = detail::prewarp(Wn_norm);

    // ── Step 3: LP → LP transform: scale poles by analog cutoff ──────────────
    // (or LP → HP: s_lp = Omega_a / s_hp, which inverts the pole locations)
    std::vector<Complex> scaled_poles;
    scaled_poles.reserve(order);

    if (type == FilterType::Lowpass) {
        for (auto& p : a_poles)
            scaled_poles.push_back(p * Omega_a);
    }
    else if (type == FilterType::Highpass) {
        // LP→HP bilinear substitution: the cutoff inverts.
        // Each LP pole at p_lp becomes HP pole at Omega_a / p_lp.
        for (auto& p : a_poles)
            scaled_poles.push_back(Omega_a / p);
    }
    else {
        throw NotImplemented("butter: Bandpass/Bandstop not yet implemented");
    }

    // ── Step 4: bilinear transform: s-domain → z-domain ─────────────────────
    std::vector<Complex> d_poles, d_zeros;
    d_poles.reserve(order);
    d_zeros.reserve(order);

    for (auto& p : scaled_poles)
        d_poles.push_back(detail::bilinear(p));

    if (type == FilterType::Lowpass) {
        // All n analog zeros at s=∞ → digital zeros at z = -1
        for (int i = 0; i < order; ++i)
            d_zeros.emplace_back(-1.0, 0.0);
    }
    else {
        // Highpass: analog zeros at s=0 → digital zeros at z = +1
        for (int i = 0; i < order; ++i)
            d_zeros.emplace_back(1.0, 0.0);
    }

    // ── Step 5: compute gain so that passband has unit gain ───────────────────
    // For LP: we want H(z=1) = 1 (DC gain = 1).
    // For HP: we want H(z=-1) = 1 (Nyquist gain = 1).
    // H(z) = gain * prod(z - d_zeros) / prod(z - d_poles)
    // At z=z0: gain = prod(z0 - d_poles) / prod(z0 - d_zeros)
    Complex z0 = (type == FilterType::Lowpass)
                     ? Complex(1.0, 0.0)    // DC       GCOV_EXCL_BR_LINE
                     : Complex(-1.0, 0.0);  // Nyquist  GCOV_EXCL_BR_LINE

    Complex num = 1.0, den = 1.0;
    for (auto& p : d_poles) num *= (z0 - p); // GCOV_EXCL_BR_LINE
    for (auto& z : d_zeros) den *= (z0 - z); // GCOV_EXCL_BR_LINE

    Complex ratio = num / den;
    // For a well-formed filter the ratio is real and positive. A large imaginary
    // residual indicates numerical instability (e.g. very high order).
    if (std::abs(ratio.imag()) > 1e-6 * std::abs(ratio.real()) + 1e-12) throw NumericalError("butter: gain computation has unexpected imaginary residual — check for numerical instability at this filter order"); // GCOV_EXCL_BR_LINE
    double gain = std::real(ratio);
    if (gain <= 0.0) throw NumericalError("butter: computed gain is non-positive — check for numerical instability at this filter order"); // GCOV_EXCL_BR_LINE

    // ── Step 6: ZPK → SOS ────────────────────────────────────────────────────
    return detail::zpk2sos(d_zeros, d_poles, gain);
}


// ── firwin() — windowed-sinc FIR filter ──────────────────────────────────────
//
// Designs a linear-phase FIR filter using the window method.
// The ideal (brick-wall) frequency response is multiplied by a window function
// to reduce Gibbs ringing, at the cost of a transition band.
//
// Parameters:
//   numtaps — number of FIR coefficients (= filter length). Must be odd for
//             Type I (symmetric, zero-phase at DC and Nyquist). Even lengths
//             give Type II (zero at Nyquist — unsuitable for highpass).
//   cutoff  — normalised cutoff ∈ (0,1) or in Hz if opts.fs is provided.
//   win     — window function (Hann, Hamming, Blackman, Kaiser…).
//             Hann: good general choice (-44 dB sidelobes)
//             Hamming: slightly narrower mainlobe (-53 dB)
//             Blackman: widest transition band, very low sidelobes (-74 dB)
//   type    — Lowpass (default) or Highpass
//   opts    — optional fs
//
// Returns: FIR coefficients as a std::vector<Real>.
//          Apply with cps::lfilter(h, {1.0}, signal).
//
// Example:
//   // 101-tap Hann-windowed lowpass at fc=0.1 (normalised)
//   auto h = cps::firwin(101, 0.1, cps::Window::Hann);

[[nodiscard]] inline std::vector<Real> firwin(int numtaps, double cutoff,
                                              Window win      = Window::Hamming,
                                              FilterType type = FilterType::Lowpass,
                                              FilterOptions opts = {})
{
    if (numtaps <= 0)
        throw ValueError("firwin: numtaps must be > 0");
    if (numtaps % 2 == 0 && type == FilterType::Highpass)
        throw ValueError("firwin: even numtaps with Highpass gives zero gain at Nyquist — use odd numtaps");

    double Wn_norm = detail::normalise_wn(cutoff, opts);

    // The ideal lowpass impulse response (sinc centred at the midpoint)
    const int    M     = numtaps - 1;       // filter order
    const double alpha = M / 2.0;           // midpoint (fractional for even M)
    const double fc    = Wn_norm / 2.0;     // cutoff as fraction of sample rate

    std::vector<Real> h(numtaps);

    for (int n = 0; n < numtaps; ++n) {
        double t = n - alpha;
        if (std::abs(t) < 1e-12)
            h[n] = 2.0 * fc;                      // sinc(0) = 1, scaled by 2*fc
        else
            h[n] = std::sin(2.0 * std::numbers::pi * fc * t)
                   / (std::numbers::pi * t);       // sinc(2*fc*t)
    }

    // Apply the window function to taper the sinc and reduce sidelobes.
    // The window is computed inline rather than calling spectral::window()
    // to keep the filter module self-contained.
    for (int n = 0; n < numtaps; ++n) {
        double w = 1.0;
        switch (win) {
        case Window::Rectangular:
            w = 1.0;
            break;
        case Window::Hann:
            w = 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * n / M));
            break;
        case Window::Hamming:
            w = 0.54 - 0.46 * std::cos(2.0 * std::numbers::pi * n / M);
            break;
        case Window::Blackman:
            w = 0.42
              - 0.50 * std::cos(2.0 * std::numbers::pi * n / M)
              + 0.08 * std::cos(4.0 * std::numbers::pi * n / M);
            break;
        case Window::BlackmanHarris:
            w = 0.35875
              - 0.48829 * std::cos(2.0 * std::numbers::pi * n / M)
              + 0.14128 * std::cos(4.0 * std::numbers::pi * n / M)
              - 0.01168 * std::cos(6.0 * std::numbers::pi * n / M);
            break;
        case Window::FlatTop:
            // Five-term flat-top (Heinzel et al. 2002). Rarely used for FIR
            // design due to its very wide transition band.
            w =  0.21557895
               - 0.41663158 * std::cos(2.0 * std::numbers::pi * n / M)
               + 0.27726316 * std::cos(4.0 * std::numbers::pi * n / M)
               - 0.08357895 * std::cos(6.0 * std::numbers::pi * n / M)
               + 0.00694737 * std::cos(8.0 * std::numbers::pi * n / M);
            break;
        default:
            // Kaiser and Tukey require extra parameters (beta/alpha) not accepted
            // by this function. Compute the window with make_window() and multiply
            // it by the sinc coefficients manually.
            throw NotImplemented("firwin: Kaiser and Tukey windows require additional "
                                 "parameters — use make_window() directly");
        }
        h[n] *= w;
    }

    // Normalise so the filter has exactly unity gain at the intended frequency.
    // A finite windowed sinc does not sum to exactly 1 — the truncation and
    // tapering introduce a small error (~0.1–0.3% for typical lengths).
    // scipy.signal.firwin divides by the DC sum before returning; we do the same.
    {
        double s = 0.0;
        for (auto v : h) s += v;
        if (std::abs(s) < 1e-12)
            throw NumericalError("firwin: normalisation sum is near zero — "
                                 "cutoff is too close to 0 or Nyquist");
        for (auto& v : h) v /= s;
    }

    // For highpass: spectral inversion of the (now-normalised) lowpass prototype.
    // Negate all coefficients then add a unit impulse at the centre.
    // This gives HP = delta - LP, whose frequency response is 1 - LP(f).
    if (type == FilterType::Highpass) {
        for (int n = 0; n < numtaps; ++n)
            h[n] = -h[n];
        h[M / 2] += 1.0;   // only valid for odd numtaps (guaranteed above)
    }

    return h;
}


// ── cheby1() ─────────────────────────────────────────────────────────────────
// Chebyshev Type I: equiripple in the passband, monotone stopband.
// rp: maximum passband ripple in dB.
// TODO: implement full design.
[[nodiscard]] inline SOS cheby1(int /*order*/, double /*rp*/, double /*Wn*/,
                                FilterType /*type*/, FilterOptions /*opts*/ = {})
{
    throw NotImplemented("cheby1");
}

// ── cheby2() ─────────────────────────────────────────────────────────────────
// Chebyshev Type II: monotone passband, equiripple stopband.
// rs: minimum stopband attenuation in dB.
// TODO: implement full design.
[[nodiscard]] inline SOS cheby2(int /*order*/, double /*rs*/, double /*Wn*/,
                                FilterType /*type*/, FilterOptions /*opts*/ = {})
{
    throw NotImplemented("cheby2");
}

// ── ellip() ──────────────────────────────────────────────────────────────────
// Elliptic (Cauer): equiripple in both bands. Sharpest rolloff for a given
// order, at the cost of phase non-linearity.
// rp: passband ripple (dB), rs: stopband attenuation (dB).
// TODO: implement (requires Jacobi elliptic functions).
[[nodiscard]] inline SOS ellip(int /*order*/, double /*rp*/, double /*rs*/, double /*Wn*/,
                               FilterType /*type*/, FilterOptions /*opts*/ = {})
{
    throw NotImplemented("ellip");
}

// ── bessel() ─────────────────────────────────────────────────────────────────
// Bessel/Thomson: maximally flat group delay. Ideal for pulse shaping where
// waveform integrity matters more than frequency selectivity.
// TODO: implement (requires Bessel polynomial roots).
[[nodiscard]] inline SOS bessel(int /*order*/, double /*Wn*/,
                                FilterType /*type*/, FilterOptions /*opts*/ = {})
{
    throw NotImplemented("bessel");
}

} // namespace cps
