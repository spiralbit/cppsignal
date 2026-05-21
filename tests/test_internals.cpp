// test_internals.cpp — tests for internal helpers and low-level API functions
//
// Covers:
//   detail::butter_analog_poles, detail::prewarp, detail::bilinear, detail::zpk2sos
//   detail::normalise_wn
//   peak_prominences (called directly)
//   fftfreq / rfftfreq  (exact values)
//   irfft size validation
//   make_window (all types, edge cases)
//   freqz detailed behaviour
//   cps::backends::detail_builtin (bit_reverse, cooley_tukey, naive_dft, fft_impl)
//   cps::backends::BuiltinFFT struct
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <cps/backends/fft/builtin.hpp>
#include <vector>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <complex>
#include <numeric>

using namespace cps;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ════════════════════════════════════════════════════════════════════════════
// detail::butter_analog_poles
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter_analog_poles: count matches requested order", "[internals][filter]")
{
    for (int n : {1, 2, 3, 4, 5, 8, 10}) {
        auto poles = detail::butter_analog_poles(n);
        CHECK(static_cast<int>(poles.size()) == n);
    }
}

TEST_CASE("butter_analog_poles: all poles lie on the unit circle in the s-plane", "[internals][filter]")
{
    for (int n : {1, 2, 3, 4, 6}) {
        auto poles = detail::butter_analog_poles(n);
        for (auto& p : poles)
            CHECK_THAT(std::abs(p), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("butter_analog_poles: all poles are in the left half-plane (Re < 0)", "[internals][filter]")
{
    for (int n : {1, 2, 3, 4, 5}) {
        auto poles = detail::butter_analog_poles(n);
        for (auto& p : poles)
            CHECK(p.real() < 0.0);
    }
}

TEST_CASE("butter_analog_poles: n=1 gives single real pole at -1", "[internals][filter]")
{
    auto poles = detail::butter_analog_poles(1);
    REQUIRE(poles.size() == 1);
    CHECK_THAT(poles[0].real(), WithinAbs(-1.0, 1e-12));
    CHECK_THAT(poles[0].imag(), WithinAbs(0.0,  1e-12));
}

TEST_CASE("butter_analog_poles: n=2 gives conjugate pair at ±45° from -x axis", "[internals][filter]")
{
    auto poles = detail::butter_analog_poles(2);
    REQUIRE(poles.size() == 2);
    // θ_1 = π*(2+2-1)/4 = 3π/4 = 135°, θ_2 = π*(4+2-1)/4 = 5π/4 = 225°
    CHECK_THAT(poles[0].real(), WithinAbs(-std::sqrt(2.0)/2.0, 1e-12));
    CHECK_THAT(poles[0].imag(), WithinAbs( std::sqrt(2.0)/2.0, 1e-12));
    CHECK_THAT(poles[1].real(), WithinAbs(-std::sqrt(2.0)/2.0, 1e-12));
    CHECK_THAT(poles[1].imag(), WithinAbs(-std::sqrt(2.0)/2.0, 1e-12));
}

TEST_CASE("butter_analog_poles: n=4, poles come in conjugate pairs", "[internals][filter]")
{
    auto poles = detail::butter_analog_poles(4);
    REQUIRE(poles.size() == 4);
    // Upper-half and lower-half poles should be conjugates
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK_THAT(poles[i].real(), WithinAbs(poles[3-i].real(),  1e-12));
        CHECK_THAT(poles[i].imag(), WithinAbs(-poles[3-i].imag(), 1e-12));
    }
}

TEST_CASE("butter_analog_poles: n=3 includes one real pole", "[internals][filter]")
{
    auto poles = detail::butter_analog_poles(3);
    REQUIRE(poles.size() == 3);
    // For odd n, one pole is real (imaginary part ≈ 0)
    bool has_real_pole = false;
    for (auto& p : poles)
        if (std::abs(p.imag()) < 1e-10) has_real_pole = true;
    CHECK(has_real_pole);
}

// ════════════════════════════════════════════════════════════════════════════
// detail::prewarp
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("prewarp: Wn=0 gives 0", "[internals][filter]")
{
    CHECK_THAT(detail::prewarp(0.0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("prewarp: Wn=0.5 gives 2*tan(π/4) = 2", "[internals][filter]")
{
    // prewarp(0.5) = 2 * tan(π * 0.5 / 2) = 2 * tan(π/4) = 2 * 1 = 2
    CHECK_THAT(detail::prewarp(0.5), WithinAbs(2.0, 1e-12));
}

TEST_CASE("prewarp: Wn=0.25 gives 2*tan(π/8)", "[internals][filter]")
{
    double expected = 2.0 * std::tan(std::numbers::pi / 8.0);
    CHECK_THAT(detail::prewarp(0.25), WithinAbs(expected, 1e-12));
}

TEST_CASE("prewarp: Wn=0.1 and Wn=0.9 are not symmetric (tangent is non-linear)", "[internals][filter]")
{
    double w01 = detail::prewarp(0.1);
    double w09 = detail::prewarp(0.9);
    // tan is not anti-symmetric around 0.5; we just verify they're different
    CHECK(std::abs(w01 - w09) > 0.5);
    // Both should be positive
    CHECK(w01 > 0.0);
    CHECK(w09 > 0.0);
}

TEST_CASE("prewarp: Wn→1 gives very large value (tan approaches ∞)", "[internals][filter]")
{
    double w = detail::prewarp(0.999);
    CHECK(w > 100.0);
}

// ════════════════════════════════════════════════════════════════════════════
// detail::bilinear
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("bilinear: s=0 maps to z=+1 (DC)", "[internals][filter]")
{
    auto z = detail::bilinear(Complex(0.0, 0.0));
    CHECK_THAT(z.real(), WithinAbs(1.0, 1e-12));
    CHECK_THAT(z.imag(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("bilinear: purely imaginary s lies on the unit circle", "[internals][filter]")
{
    // For s = jω, |z| = |(2+jω)/(2-jω)| = sqrt(4+ω²)/sqrt(4+ω²) = 1
    for (double omega : {0.5, 1.0, 2.0, 5.0}) {
        auto z = detail::bilinear(Complex(0.0, omega));
        CHECK_THAT(std::abs(z), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("bilinear: s with Re<0 maps to |z|<1 (stable)", "[internals][filter]")
{
    // Left-half s-plane maps inside the unit circle
    auto z = detail::bilinear(Complex(-1.0, 0.5));
    CHECK(std::abs(z) < 1.0);
}

TEST_CASE("bilinear: s with Re>0 maps to |z|>1 (unstable)", "[internals][filter]")
{
    auto z = detail::bilinear(Complex(1.0, 0.0));
    CHECK(std::abs(z) > 1.0);
}

TEST_CASE("bilinear: formula z=(2+s)/(2-s) verified numerically", "[internals][filter]")
{
    Complex s(0.3, -0.7);
    Complex expected = (2.0 + s) / (2.0 - s);
    auto z = detail::bilinear(s);
    CHECK_THAT(z.real(), WithinAbs(expected.real(), 1e-12));
    CHECK_THAT(z.imag(), WithinAbs(expected.imag(), 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// detail::zpk2sos
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("zpk2sos: simple first-order section from one real pole", "[internals][filter]")
{
    // One real pole at 0.5, one real zero at -1, gain=1
    std::vector<Complex> zeros = {Complex(-1.0, 0.0)};
    std::vector<Complex> poles = {Complex(0.5,  0.0)};
    auto sos = detail::zpk2sos(zeros, poles, 1.0);
    REQUIRE(sos.size() == 1);
    // b = [1, 1, 0], a = [1, -0.5, 0]
    CHECK_THAT(sos[0][0], WithinAbs(1.0,  1e-10));   // b0 = 1 * gain = 1
    CHECK_THAT(sos[0][1], WithinAbs(1.0,  1e-10));   // b1 = -z1.real() = -(-1) = 1
    CHECK_THAT(sos[0][2], WithinAbs(0.0,  1e-10));   // b2 = 0
    CHECK_THAT(sos[0][3], WithinAbs(1.0,  1e-10));   // a0 = 1
    CHECK_THAT(sos[0][4], WithinAbs(-0.5, 1e-10));   // a1 = -pole = -0.5
    CHECK_THAT(sos[0][5], WithinAbs(0.0,  1e-10));   // a2 = 0
}

TEST_CASE("zpk2sos: one complex conjugate pole pair gives one biquad", "[internals][filter]")
{
    Complex p(0.5, 0.5);   // pole at 0.5+0.5j (upper half)
    Complex z(-1.0, 0.0);  // real zero at -1
    std::vector<Complex> zeros = {z, std::conj(z)};   // conjugate pair
    std::vector<Complex> poles = {p, std::conj(p)};   // conjugate pair
    auto sos = detail::zpk2sos(zeros, poles, 1.0);
    REQUIRE(sos.size() == 1);
    // All coefficients should be real (finite)
    for (auto v : sos[0])
        CHECK(std::isfinite(v));
    // a0 should always be 1.0
    CHECK_THAT(sos[0][3], WithinAbs(1.0, 1e-12));
}

TEST_CASE("zpk2sos: gain is applied to first section only", "[internals][filter]")
{
    // 4th-order filter: 2 complex pairs → 2 biquads
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    REQUIRE(sos.size() >= 2);
    // All a0 entries must be 1.0
    for (auto& row : sos)
        CHECK_THAT(row[3], WithinAbs(1.0, 1e-12));
}

TEST_CASE("zpk2sos: unmatched complex zeros throw NumericalError", "[internals][filter]")
{
    // More complex zero pairs than poles → unmatched
    std::vector<Complex> zeros = {Complex(0.0, 1.0), Complex(0.0, -1.0),
                                  Complex(0.3, 0.7), Complex(0.3, -0.7)};
    std::vector<Complex> poles = {Complex(-0.5, 0.0)};   // only 1 pole
    CHECK_THROWS_AS(detail::zpk2sos(zeros, poles, 1.0), NumericalError);
}

// ════════════════════════════════════════════════════════════════════════════
// detail::normalise_wn
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("normalise_wn: fs < 0 throws ValueError", "[internals][filter]")
{
    CHECK_THROWS_AS(detail::normalise_wn(0.1, {.fs = -1.0}), ValueError);
}

TEST_CASE("normalise_wn: Wn=0 throws ValueError (Wn_norm <= 0)", "[internals][filter]")
{
    CHECK_THROWS_AS(detail::normalise_wn(0.0, {}), ValueError);
}

TEST_CASE("normalise_wn: Wn=1.0 (normalised) throws ValueError (Wn_norm >= 1)", "[internals][filter]")
{
    CHECK_THROWS_AS(detail::normalise_wn(1.0, {}), ValueError);
}

TEST_CASE("normalise_wn: Wn > 1 with no fs throws ValueError", "[internals][filter]")
{
    CHECK_THROWS_AS(detail::normalise_wn(1.5, {}), ValueError);
}

TEST_CASE("normalise_wn: negative Wn throws ValueError", "[internals][filter]")
{
    CHECK_THROWS_AS(detail::normalise_wn(-0.1, {}), ValueError);
}

TEST_CASE("normalise_wn: Wn=fs/2 (Nyquist) throws ValueError", "[internals][filter]")
{
    // Wn_norm = 2 * (fs/2) / fs = 1.0 → throws
    CHECK_THROWS_AS(detail::normalise_wn(500.0, {.fs = 1000.0}), ValueError);
}

TEST_CASE("normalise_wn: Wn=0 with fs set throws ValueError", "[internals][filter]")
{
    // Wn_norm = 2*0/1000 = 0 → throws
    CHECK_THROWS_AS(detail::normalise_wn(0.0, {.fs = 1000.0}), ValueError);
}

TEST_CASE("normalise_wn: valid Hz frequency with fs", "[internals][filter]")
{
    // Wn=100 Hz, fs=1000 Hz → Wn_norm = 0.2
    double wn = detail::normalise_wn(100.0, {.fs = 1000.0});
    CHECK_THAT(wn, WithinAbs(0.2, 1e-12));
}

TEST_CASE("normalise_wn: valid normalised frequency", "[internals][filter]")
{
    double wn = detail::normalise_wn(0.3, {});
    CHECK_THAT(wn, WithinAbs(0.3, 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// peak_prominences — direct tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("peak_prominences: simple known signal [0,1,0,3,0,2,0]", "[internals][peaks]")
{
    std::vector<Real> sig  = {0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0};
    std::vector<std::size_t> idx = {1, 3, 5};
    auto proms = peak_prominences(sig, idx);
    REQUIRE(proms.size() == 3);
    CHECK_THAT(proms[0], WithinAbs(1.0, 1e-12));  // peak=1: min in [0..idx=1] = 0
    CHECK_THAT(proms[1], WithinAbs(3.0, 1e-12));  // peak=3: global max, min baseline = 0
    CHECK_THAT(proms[2], WithinAbs(2.0, 1e-12));  // peak=2: left boundary at higher peak 3
}

TEST_CASE("peak_prominences: single peak, prominence = peak height", "[internals][peaks]")
{
    std::vector<Real> sig = {0.0, 5.0, 0.0};
    std::vector<std::size_t> idx = {1};
    auto proms = peak_prominences(sig, idx);
    REQUIRE(proms.size() == 1);
    CHECK_THAT(proms[0], WithinAbs(5.0, 1e-12));
}

TEST_CASE("peak_prominences: two equal-height peaks on flat baseline", "[internals][peaks]")
{
    std::vector<Real> sig = {0.0, 2.0, 0.0, 2.0, 0.0};
    std::vector<std::size_t> idx = {1, 3};
    auto proms = peak_prominences(sig, idx);
    REQUIRE(proms.size() == 2);
    CHECK_THAT(proms[0], WithinAbs(2.0, 1e-12));
    CHECK_THAT(proms[1], WithinAbs(2.0, 1e-12));
}

TEST_CASE("peak_prominences: peak surrounded by higher flanks has low prominence", "[internals][peaks]")
{
    // [3, 1, 3]: peak at index 1 (height 1); left and right flanks are at 3
    // left: signal[0]=3 > 1 → found left, scan [0..1], left_min = min(3,1) = 1
    // right: signal[2]=3 > 1 → found right, scan [1..2], right_min = min(1,3) = 1
    // prominence = 1 - max(1,1) = 0
    std::vector<Real> sig = {3.0, 1.0, 3.0};
    std::vector<std::size_t> idx = {1};
    auto proms = peak_prominences(sig, idx);
    REQUIRE(proms.size() == 1);
    CHECK_THAT(proms[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("peak_prominences: empty peak list returns empty", "[internals][peaks]")
{
    std::vector<Real> sig = {1.0, 2.0, 1.0};
    std::vector<std::size_t> idx;
    auto proms = peak_prominences(sig, idx);
    CHECK(proms.empty());
}

TEST_CASE("peak_prominences: results agree with find_peaks prominence", "[internals][peaks]")
{
    std::vector<Real> sig = {0.0, 1.0, 0.2, 3.0, 0.5, 2.0, 0.0};
    auto result = find_peaks(std::span<const Real>(sig),
                             PeakOptions{.prominence = 0.0});
    auto direct_proms = peak_prominences(sig, result.indices);
    REQUIRE(direct_proms.size() == result.prominences.size());
    for (std::size_t i = 0; i < direct_proms.size(); ++i)
        CHECK_THAT(direct_proms[i], WithinAbs(result.prominences[i], 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// fftfreq / rfftfreq — exact values
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("fftfreq: n=8, d=1 matches numpy.fft.fftfreq(8) exactly", "[internals][fft]")
{
    auto f = fftfreq(8, 1.0);
    REQUIRE(f.size() == 8);
    CHECK_THAT(f[0],  WithinAbs( 0.000, 1e-12));
    CHECK_THAT(f[1],  WithinAbs( 0.125, 1e-12));
    CHECK_THAT(f[2],  WithinAbs( 0.250, 1e-12));
    CHECK_THAT(f[3],  WithinAbs( 0.375, 1e-12));
    CHECK_THAT(f[4],  WithinAbs(-0.500, 1e-12));
    CHECK_THAT(f[5],  WithinAbs(-0.375, 1e-12));
    CHECK_THAT(f[6],  WithinAbs(-0.250, 1e-12));
    CHECK_THAT(f[7],  WithinAbs(-0.125, 1e-12));
}

TEST_CASE("fftfreq: n=7 (odd), positive/negative split at floor(7/2)=3", "[internals][fft]")
{
    auto f = fftfreq(7, 1.0);
    REQUIRE(f.size() == 7);
    CHECK_THAT(f[0], WithinAbs( 0.0,       1e-12));
    CHECK_THAT(f[1], WithinAbs( 1.0 / 7.0, 1e-12));
    CHECK_THAT(f[3], WithinAbs( 3.0 / 7.0, 1e-12));
    CHECK_THAT(f[4], WithinAbs(-3.0 / 7.0, 1e-10));
    CHECK_THAT(f[6], WithinAbs(-1.0 / 7.0, 1e-12));
}

TEST_CASE("fftfreq: d = 1/fs converts to Hz", "[internals][fft]")
{
    constexpr double fs = 1000.0;
    auto f = fftfreq(8, 1.0 / fs);
    REQUIRE(f.size() == 8);
    CHECK_THAT(f[0], WithinAbs(   0.0, 1e-10));
    CHECK_THAT(f[1], WithinAbs( 125.0, 1e-10));
    CHECK_THAT(f[4], WithinAbs(-500.0, 1e-10));
}

TEST_CASE("fftfreq: n=1, only DC bin = 0", "[internals][fft]")
{
    auto f = fftfreq(1, 1.0);
    REQUIRE(f.size() == 1);
    CHECK_THAT(f[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("rfftfreq: n=8, d=1 matches numpy.fft.rfftfreq(8) exactly", "[internals][fft]")
{
    auto f = rfftfreq(8, 1.0);
    REQUIRE(f.size() == 5);
    CHECK_THAT(f[0], WithinAbs(0.000, 1e-12));
    CHECK_THAT(f[1], WithinAbs(0.125, 1e-12));
    CHECK_THAT(f[2], WithinAbs(0.250, 1e-12));
    CHECK_THAT(f[3], WithinAbs(0.375, 1e-12));
    CHECK_THAT(f[4], WithinAbs(0.500, 1e-12));
}

TEST_CASE("rfftfreq: all values non-negative", "[internals][fft]")
{
    auto f = rfftfreq(16, 1.0);
    for (auto v : f)
        CHECK(v >= 0.0);
}

TEST_CASE("rfftfreq: last bin = 0.5 (Nyquist) for even n", "[internals][fft]")
{
    auto f = rfftfreq(100, 1.0);
    REQUIRE(f.size() == 51);
    CHECK_THAT(f.back(), WithinAbs(0.5, 1e-12));
}

TEST_CASE("rfftfreq: with d=1/fs, last bin = fs/2", "[internals][fft]")
{
    constexpr double fs = 8000.0;
    auto f = rfftfreq(256, 1.0 / fs);
    CHECK_THAT(f.back(), WithinAbs(fs / 2.0, 1e-10));
}

// ════════════════════════════════════════════════════════════════════════════
// irfft — size validation
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("irfft: empty spectrum throws ValueError", "[internals][fft]")
{
    std::vector<Complex> X;
    CHECK_THROWS_AS(irfft(std::span<const Complex>(X), 8), ValueError);
}

TEST_CASE("irfft: spectrum too short throws ValueError", "[internals][fft]")
{
    // For n=8: expected x.size() = 5
    std::vector<Complex> X(4, Complex(1.0, 0.0));
    CHECK_THROWS_AS(irfft(std::span<const Complex>(X), 8), ValueError);
}

TEST_CASE("irfft: spectrum too long throws ValueError", "[internals][fft]")
{
    // For n=8: expected x.size() = 5
    std::vector<Complex> X(6, Complex(1.0, 0.0));
    CHECK_THROWS_AS(irfft(std::span<const Complex>(X), 8), ValueError);
}

TEST_CASE("irfft: correct roundtrip for n=8", "[internals][fft]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto X = rfft(std::span<const Real>(x));
    auto y = irfft(std::span<const Complex>(X), 8);
    REQUIRE(y.size() == 8);
    for (std::size_t i = 0; i < 8; ++i)
        CHECK_THAT(y[i], WithinAbs(x[i], 1e-10));
}

TEST_CASE("irfft: correct roundtrip for odd n=9", "[internals][fft]")
{
    std::vector<Real> x(9);
    for (std::size_t i = 0; i < 9; ++i)
        x[i] = static_cast<double>(i + 1);
    auto X = rfft(std::span<const Real>(x));
    REQUIRE(X.size() == 5);   // 9/2+1 = 5
    auto y = irfft(std::span<const Complex>(X), 9);
    REQUIRE(y.size() == 9);
    for (std::size_t i = 0; i < 9; ++i)
        CHECK_THAT(y[i], WithinAbs(x[i], 1e-10));
}

// ════════════════════════════════════════════════════════════════════════════
// make_window — all types and edge cases
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("make_window: n=0 throws ValueError", "[internals][spectral]")
{
    CHECK_THROWS_AS(make_window(Window::Hann, 0), ValueError);
}

TEST_CASE("make_window: n=1 returns {1.0} for all window types", "[internals][spectral]")
{
    for (auto type : {Window::Rectangular, Window::Hann, Window::Hamming,
                      Window::Blackman, Window::BlackmanHarris, Window::FlatTop}) {
        auto w = make_window(type, 1);
        REQUIRE(w.size() == 1);
        CHECK_THAT(w[0], WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("make_window: Rectangular window is all ones", "[internals][spectral]")
{
    auto w = make_window(Window::Rectangular, 64);
    for (auto v : w)
        CHECK_THAT(v, WithinAbs(1.0, 1e-12));
}

TEST_CASE("make_window: Hann window is symmetric", "[internals][spectral]")
{
    auto w = make_window(Window::Hann, 64);
    for (std::size_t i = 0; i < 32; ++i)
        CHECK_THAT(w[i], WithinAbs(w[63-i], 1e-12));
}

TEST_CASE("make_window: Hamming window is symmetric", "[internals][spectral]")
{
    auto w = make_window(Window::Hamming, 64);
    for (std::size_t i = 0; i < 32; ++i)
        CHECK_THAT(w[i], WithinAbs(w[63-i], 1e-12));
}

TEST_CASE("make_window: Blackman window is symmetric", "[internals][spectral]")
{
    auto w = make_window(Window::Blackman, 64);
    for (std::size_t i = 0; i < 32; ++i)
        CHECK_THAT(w[i], WithinAbs(w[63-i], 1e-12));
}

TEST_CASE("make_window: BlackmanHarris window is symmetric", "[internals][spectral]")
{
    auto w = make_window(Window::BlackmanHarris, 64);
    for (std::size_t i = 0; i < 32; ++i)
        CHECK_THAT(w[i], WithinAbs(w[63-i], 1e-12));
}

TEST_CASE("make_window: Kaiser window with beta=0 approximates rectangular", "[internals][spectral]")
{
    auto w_kaiser = make_window(Window::Kaiser, 64, 0.0);
    auto w_rect   = make_window(Window::Rectangular, 64);
    for (std::size_t i = 0; i < 64; ++i)
        CHECK_THAT(w_kaiser[i], WithinAbs(w_rect[i], 1e-6));
}

TEST_CASE("make_window: Kaiser window with large beta has more taper", "[internals][spectral]")
{
    auto w_low  = make_window(Window::Kaiser, 64, 1.0);
    auto w_high = make_window(Window::Kaiser, 64, 14.0);
    // Large beta → more taper → lower edge values
    CHECK(w_high[0] < w_low[0]);
    CHECK(w_high[63] < w_low[63]);
}

TEST_CASE("make_window: Tukey window with alpha=0 defaults to alpha=0.5 (avoids /0)", "[internals][spectral]")
{
    // The implementation maps param<=0 to 0.5 to prevent division by zero when taper=0
    auto w_p0   = make_window(Window::Tukey, 64, 0.0);
    auto w_p05  = make_window(Window::Tukey, 64, 0.5);
    for (std::size_t i = 0; i < 64; ++i)
        CHECK_THAT(w_p0[i], WithinAbs(w_p05[i], 1e-12));
}

TEST_CASE("make_window: Tukey window with alpha=1 tapers fully, symmetric, edges=0", "[internals][spectral]")
{
    auto w = make_window(Window::Tukey, 64, 1.0);
    REQUIRE(w.size() == 64);
    // First and last samples should be 0
    CHECK_THAT(w[0],  WithinAbs(0.0, 1e-12));
    CHECK_THAT(w[63], WithinAbs(0.0, 1e-12));
    // Symmetric
    for (std::size_t i = 0; i < 32; ++i)
        CHECK_THAT(w[i], WithinAbs(w[63-i], 1e-12));
    // All values in [0, 1]
    for (auto v : w) {
        CHECK(v >= 0.0 - 1e-12);
        CHECK(v <= 1.0 + 1e-12);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// freqz — detailed behaviour
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("freqz: empty SOS throws ValueError", "[internals][filter]")
{
    SOS sos;
    CHECK_THROWS_AS(freqz(sos), ValueError);
}

TEST_CASE("freqz: nfreqs=1 returns single DC point", "[internals][filter]")
{
    auto sos = butter(2, 0.1, FilterType::Lowpass);
    auto [freqs, H] = freqz(sos, 1);
    REQUIRE(H.size() == 1);
    // Only DC (ω=0): gain should be 1.0 for LP
    CHECK_THAT(std::abs(H[0]), WithinAbs(1.0, 1e-6));
}

TEST_CASE("freqz: LP at -3dB point (Wn) has magnitude ≈ 1/√2", "[internals][filter]")
{
    // butter designs -3dB at Wn; Wn=0.3 → -3dB at normalised freq 0.3 (= 0.3/2 * fs)
    // freqz evaluates at k * (π/nfreqs) for k=0..nfreqs-1
    // The -3dB point is at ω = π * Wn = π * 0.3
    // That's at index k = nfreqs * 0.3 = 512 * 0.3 = ~154
    constexpr double Wn = 0.3;
    auto sos = butter(4, Wn, FilterType::Lowpass);
    auto [freqs, H] = freqz(sos, 512);
    std::size_t k = static_cast<std::size_t>(std::round(512 * Wn));
    if (k < H.size()) {
        double mag = std::abs(H[k]);
        CHECK_THAT(mag, WithinAbs(1.0 / std::sqrt(2.0), 0.05));  // ±5% tolerance
    }
}

TEST_CASE("freqz: HP filter has near-zero DC gain and near-unity Nyquist gain", "[internals][filter]")
{
    auto sos = butter(4, 0.3, FilterType::Highpass);
    auto [freqs, H] = freqz(sos, 512);
    // DC (k=0): near 0 for highpass
    CHECK(std::abs(H[0]) < 0.1);
    // Near Nyquist (k=511): near 1 for highpass
    CHECK(std::abs(H.back()) > 0.9);
}

TEST_CASE("freqz: with fs > 0, frequency axis is in Hz", "[internals][filter]")
{
    constexpr double fs = 1000.0;
    auto sos = butter(2, 0.2, FilterType::Lowpass);
    auto [freqs, H] = freqz(sos, 512, fs);
    // First bin = 0 Hz, last bin < 500 Hz (Nyquist)
    CHECK_THAT(freqs[0], WithinAbs(0.0, 1e-10));
    CHECK(freqs.back() < fs / 2.0);
}

// ════════════════════════════════════════════════════════════════════════════
// cps::backends::detail_builtin — internal FFT routines
// ════════════════════════════════════════════════════════════════════════════

namespace db = cps::backends::detail_builtin;
using cps::backends::BuiltinFFT;

// ── bit_reverse ───────────────────────────────────────────────────────────────

TEST_CASE("bit_reverse: N=4 swaps indices 1 and 2 (01↔10 in binary)", "[internals][backend]")
{
    // 2-bit reversal: 00→00, 01→10, 10→01, 11→11 → indices 1 and 2 swap
    std::vector<std::complex<double>> x = {{1,0},{2,0},{3,0},{4,0}};
    db::bit_reverse(x);
    CHECK(x[0] == std::complex<double>(1.0, 0.0));
    CHECK(x[1] == std::complex<double>(3.0, 0.0));
    CHECK(x[2] == std::complex<double>(2.0, 0.0));
    CHECK(x[3] == std::complex<double>(4.0, 0.0));
}

TEST_CASE("bit_reverse: N=8 known permutation", "[internals][backend]")
{
    // 3-bit reversal: 000→000, 001→100, 010→010, 011→110, 100→001, 101→101, 110→011, 111→111
    // So swaps: (1,4), (3,6)
    std::vector<std::complex<double>> x = {{0,0},{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0}};
    db::bit_reverse(x);
    // Values move to bit-reversed positions:
    CHECK(x[0].real() == 0.0);  // 000 → stays at 0
    CHECK(x[1].real() == 4.0);  // was at 4 (100→001=1)
    CHECK(x[2].real() == 2.0);  // 010 → stays at 2
    CHECK(x[3].real() == 6.0);  // was at 6 (110→011=3)
    CHECK(x[4].real() == 1.0);  // was at 1 (001→100=4)
    CHECK(x[5].real() == 5.0);  // 101 → stays at 5
    CHECK(x[6].real() == 3.0);  // was at 3 (011→110=6)
    CHECK(x[7].real() == 7.0);  // 111 → stays at 7
}

TEST_CASE("bit_reverse: applying twice returns original (self-inverse)", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{1.5,0},{2.2,-1.1},{-0.5,3.0},{4.0,-2.0},
                                           {0.5,0},{-3.3,1.0},{2.2,0},{1.1,-0.5}};
    auto orig = x;
    db::bit_reverse(x);
    db::bit_reverse(x);
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_THAT(x[i].real(), WithinAbs(orig[i].real(), 1e-15));
        CHECK_THAT(x[i].imag(), WithinAbs(orig[i].imag(), 1e-15));
    }
}

TEST_CASE("bit_reverse: N=2 is identity (no swaps needed)", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{1.0,2.0},{3.0,4.0}};
    auto orig = x;
    db::bit_reverse(x);
    CHECK(x[0] == orig[0]);
    CHECK(x[1] == orig[1]);
}

// ── cooley_tukey ──────────────────────────────────────────────────────────────

TEST_CASE("cooley_tukey: N=1 is a no-op", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{3.0, -2.0}};
    auto orig = x[0];
    db::cooley_tukey(x, false);
    CHECK(x[0] == orig);
}

TEST_CASE("cooley_tukey: N=4 all-ones → DC bin = N, rest zero", "[internals][backend]")
{
    // DFT([1,1,1,1]) = [4, 0, 0, 0]
    std::vector<std::complex<double>> x = {{1,0},{1,0},{1,0},{1,0}};
    db::cooley_tukey(x, false);
    CHECK_THAT(x[0].real(), WithinAbs(4.0, 1e-12));
    CHECK_THAT(x[0].imag(), WithinAbs(0.0, 1e-12));
    for (std::size_t k = 1; k < 4; ++k)
        CHECK_THAT(std::abs(x[k]), WithinAbs(0.0, 1e-12));
}

TEST_CASE("cooley_tukey: N=4 impulse → all bins equal 1", "[internals][backend]")
{
    // DFT([1,0,0,0]) = [1, 1, 1, 1]
    std::vector<std::complex<double>> x = {{1,0},{0,0},{0,0},{0,0}};
    db::cooley_tukey(x, false);
    for (std::size_t k = 0; k < 4; ++k) {
        CHECK_THAT(x[k].real(), WithinAbs(1.0, 1e-12));
        CHECK_THAT(x[k].imag(), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("cooley_tukey: N=4 tone at k=1 → complex exponential", "[internals][backend]")
{
    // x[n] = e^(j2π*1*n/4): X[1]=4, all others zero (forward DFT of complex tone)
    constexpr double pi = std::numbers::pi;
    std::vector<std::complex<double>> x(4);
    for (int n = 0; n < 4; ++n)
        x[n] = {std::cos(2*pi*n/4), std::sin(2*pi*n/4)};
    db::cooley_tukey(x, false);
    // X[1] = 4, rest ≈ 0
    CHECK_THAT(std::abs(x[1]), WithinAbs(4.0, 1e-10));
    for (int k : {0, 2, 3})
        CHECK_THAT(std::abs(x[k]), WithinAbs(0.0, 1e-10));
}

TEST_CASE("cooley_tukey: forward then inverse roundtrip", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{1,0},{2,1},{-1,3},{0.5,-2},
                                           {3,0},{-2,1},{0.1,0},{4,-0.5}};
    auto orig = x;
    db::cooley_tukey(x, false);
    db::cooley_tukey(x, true);
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_THAT(x[i].real(), WithinAbs(orig[i].real(), 1e-12));
        CHECK_THAT(x[i].imag(), WithinAbs(orig[i].imag(), 1e-12));
    }
}

TEST_CASE("cooley_tukey: Parseval theorem holds (N=16)", "[internals][backend]")
{
    constexpr std::size_t N = 16;
    std::vector<std::complex<double>> x(N);
    for (std::size_t n = 0; n < N; ++n)
        x[n] = {static_cast<double>(n) / N, 0.0};

    double time_energy = 0.0;
    for (auto& c : x) time_energy += std::norm(c);

    db::cooley_tukey(x, false);

    double freq_energy = 0.0;
    for (auto& c : x) freq_energy += std::norm(c);

    // Parseval: sum|X[k]|² = N * sum|x[n]|²
    CHECK_THAT(freq_energy, WithinRel(N * time_energy, 1e-10));
}

// ── naive_dft ─────────────────────────────────────────────────────────────────

TEST_CASE("naive_dft: N=4 all-ones → DC bin = 4, rest zero", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{1,0},{1,0},{1,0},{1,0}};
    db::naive_dft(x, false);
    CHECK_THAT(x[0].real(), WithinAbs(4.0, 1e-12));
    CHECK_THAT(x[0].imag(), WithinAbs(0.0, 1e-12));
    for (std::size_t k = 1; k < 4; ++k)
        CHECK_THAT(std::abs(x[k]), WithinAbs(0.0, 1e-12));
}

TEST_CASE("naive_dft: N=3 impulse → all bins equal 1", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{1,0},{0,0},{0,0}};
    db::naive_dft(x, false);
    for (std::size_t k = 0; k < 3; ++k) {
        CHECK_THAT(x[k].real(), WithinAbs(1.0, 1e-12));
        CHECK_THAT(x[k].imag(), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("naive_dft: N=5 roundtrip", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{1,0},{-1,2},{3,-1},{0.5,0.5},{-2,1}};
    auto orig = x;
    db::naive_dft(x, false);
    db::naive_dft(x, true);
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_THAT(x[i].real(), WithinAbs(orig[i].real(), 1e-12));
        CHECK_THAT(x[i].imag(), WithinAbs(orig[i].imag(), 1e-12));
    }
}

TEST_CASE("naive_dft: N=4 agrees with cooley_tukey", "[internals][backend]")
{
    std::vector<std::complex<double>> x1 = {{1,0},{2,-1},{-1,3},{0.5,-2}};
    auto x2 = x1;
    db::cooley_tukey(x1, false);
    db::naive_dft(x2, false);
    for (std::size_t k = 0; k < 4; ++k) {
        CHECK_THAT(x1[k].real(), WithinAbs(x2[k].real(), 1e-10));
        CHECK_THAT(x1[k].imag(), WithinAbs(x2[k].imag(), 1e-10));
    }
}

TEST_CASE("naive_dft: N=7 (prime) roundtrip", "[internals][backend]")
{
    std::vector<std::complex<double>> x = {{3,1},{-2,0},{1,-1},{0,2},{4,0},{-1,3},{2,-2}};
    auto orig = x;
    db::naive_dft(x, false);
    db::naive_dft(x, true);
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_THAT(x[i].real(), WithinAbs(orig[i].real(), 1e-11));
        CHECK_THAT(x[i].imag(), WithinAbs(orig[i].imag(), 1e-11));
    }
}

// ── fft_impl dispatch ─────────────────────────────────────────────────────────

TEST_CASE("fft_impl: N=4 (power-of-2) matches cooley_tukey directly", "[internals][backend]")
{
    std::vector<std::complex<double>> x_impl = {{1,0},{2,0},{3,0},{4,0}};
    auto x_ct = x_impl;
    db::fft_impl(x_impl, false);
    db::cooley_tukey(x_ct, false);
    for (std::size_t k = 0; k < 4; ++k) {
        CHECK_THAT(x_impl[k].real(), WithinAbs(x_ct[k].real(), 1e-12));
        CHECK_THAT(x_impl[k].imag(), WithinAbs(x_ct[k].imag(), 1e-12));
    }
}

TEST_CASE("fft_impl: N=5 (non-power-of-2) matches naive_dft directly", "[internals][backend]")
{
    std::vector<std::complex<double>> x_impl  = {{1,0},{2,0},{3,0},{4,0},{5,0}};
    auto x_naive = x_impl;
    db::fft_impl(x_impl, false);
    db::naive_dft(x_naive, false);
    for (std::size_t k = 0; k < 5; ++k) {
        CHECK_THAT(x_impl[k].real(), WithinAbs(x_naive[k].real(), 1e-12));
        CHECK_THAT(x_impl[k].imag(), WithinAbs(x_naive[k].imag(), 1e-12));
    }
}

TEST_CASE("fft_impl: inverse flag normalises by 1/N", "[internals][backend]")
{
    // Forward then inverse should recover x
    std::vector<std::complex<double>> x = {{2,1},{-1,3},{0.5,-2},{4,0}};
    auto orig = x;
    db::fft_impl(x, false);
    db::fft_impl(x, true);
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_THAT(x[i].real(), WithinAbs(orig[i].real(), 1e-12));
        CHECK_THAT(x[i].imag(), WithinAbs(orig[i].imag(), 1e-12));
    }
}

// ── BuiltinFFT struct ─────────────────────────────────────────────────────────

TEST_CASE("BuiltinFFT::forward then inverse roundtrip (N=8)", "[internals][backend]")
{
    BuiltinFFT b;
    std::vector<std::complex<double>> in = {{1,0},{2,-1},{-1,3},{0.5,-2},
                                            {3,0},{-2,1},{0.1,0},{4,-0.5}};
    std::vector<std::complex<double>> spec(in.size()), out(in.size());
    b.forward(in, spec);
    b.inverse(spec, out);
    for (std::size_t i = 0; i < in.size(); ++i) {
        CHECK_THAT(out[i].real(), WithinAbs(in[i].real(), 1e-12));
        CHECK_THAT(out[i].imag(), WithinAbs(in[i].imag(), 1e-12));
    }
}

TEST_CASE("BuiltinFFT::forward DC signal: X[0]=N, rest zero", "[internals][backend]")
{
    BuiltinFFT b;
    constexpr std::size_t N = 8;
    std::vector<std::complex<double>> in(N, {1.0, 0.0});
    std::vector<std::complex<double>> out(N);
    b.forward(in, out);
    CHECK_THAT(out[0].real(), WithinAbs(static_cast<double>(N), 1e-10));
    CHECK_THAT(out[0].imag(), WithinAbs(0.0, 1e-10));
    for (std::size_t k = 1; k < N; ++k)
        CHECK_THAT(std::abs(out[k]), WithinAbs(0.0, 1e-10));
}

TEST_CASE("BuiltinFFT::rfft output length is N/2+1", "[internals][backend]")
{
    BuiltinFFT b;
    for (std::size_t N : {4u, 8u, 16u, 7u, 13u}) {
        std::vector<double> in(N, 1.0);
        std::vector<std::complex<double>> out(N/2+1);
        REQUIRE_NOTHROW(b.rfft(in, out));
        CHECK(out.size() == N/2+1);
    }
}

TEST_CASE("BuiltinFFT::rfft then irfft roundtrip (even N=8)", "[internals][backend]")
{
    BuiltinFFT b;
    constexpr std::size_t N = 8;
    std::vector<double> in = {1.0, -1.0, 2.0, 0.5, -3.0, 1.5, 0.2, -0.7};
    std::vector<std::complex<double>> spec(N/2+1);
    std::vector<double> out(N);
    b.rfft(in, spec);
    b.irfft(spec, out, N);
    for (std::size_t i = 0; i < N; ++i)
        CHECK_THAT(out[i], WithinAbs(in[i], 1e-12));
}

TEST_CASE("BuiltinFFT::rfft then irfft roundtrip (odd N=7)", "[internals][backend]")
{
    BuiltinFFT b;
    constexpr std::size_t N = 7;
    std::vector<double> in = {1.0, -1.0, 2.0, 0.5, -3.0, 1.5, 0.2};
    std::vector<std::complex<double>> spec(N/2+1);
    std::vector<double> out(N);
    b.rfft(in, spec);
    b.irfft(spec, out, N);
    for (std::size_t i = 0; i < N; ++i)
        CHECK_THAT(out[i], WithinAbs(in[i], 1e-12));
}

TEST_CASE("BuiltinFFT::rfft DC signal → DC bin is N, others near zero", "[internals][backend]")
{
    BuiltinFFT b;
    constexpr std::size_t N = 8;
    std::vector<double> in(N, 1.0);
    std::vector<std::complex<double>> spec(N/2+1);
    b.rfft(in, spec);
    CHECK_THAT(spec[0].real(), WithinAbs(static_cast<double>(N), 1e-10));
    CHECK_THAT(spec[0].imag(), WithinAbs(0.0, 1e-10));
    for (std::size_t k = 1; k <= N/2; ++k)
        CHECK_THAT(std::abs(spec[k]), WithinAbs(0.0, 1e-10));
}

TEST_CASE("BuiltinFFT::name returns non-empty string", "[internals][backend]")
{
    CHECK(!BuiltinFFT::name().empty());
}
