// test_filter_extended.cpp — extended filter design and application tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <vector>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <numeric>

using namespace cps;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Helpers ───────────────────────────────────────────────────────────────────

static double sos_mag(const SOS& sos, double omega)
{
    Complex z(std::cos(omega), std::sin(omega));
    Complex zinv  = 1.0 / z;
    Complex zinv2 = zinv * zinv;
    Complex H(1.0, 0.0);
    for (const auto& row : sos) {
        Complex num = row[0] + row[1]*zinv + row[2]*zinv2;
        Complex den = row[3] + row[4]*zinv + row[5]*zinv2;
        H *= num / den;
    }
    return std::abs(H);
}

static double mag_db(const SOS& sos, double omega)
{
    return 20.0 * std::log10(sos_mag(sos, omega) + 1e-15);
}

// ── Butterworth LP: DC gain == 1 for orders 1–8 ──────────────────────────────
TEST_CASE("Butterworth LP DC gain is 1 for orders 1–8", "[butter]")
{
    for (int order = 1; order <= 8; ++order) {
        auto sos = butter(order, 0.25, FilterType::Lowpass);
        CHECK_THAT(sos_mag(sos, 0.0), WithinAbs(1.0, 1e-6));
    }
}

// ── Butterworth LP: Nyquist gain ≈ 0 ─────────────────────────────────────────
TEST_CASE("Butterworth LP Nyquist gain is near zero", "[butter]")
{
    // Higher order → deeper notch at Nyquist; even order 2 should be << 0.1
    for (int order : {2, 4, 6, 8}) {
        auto sos = butter(order, 0.2, FilterType::Lowpass);
        CHECK(sos_mag(sos, std::numbers::pi) < 0.01);
    }
}

// ── Butterworth LP: stopband attenuation grows with order ─────────────────────
TEST_CASE("Butterworth LP stopband attenuation increases with order", "[butter]")
{
    const double Wn          = 0.2;
    const double omega_stop  = std::numbers::pi * Wn * 3.0;  // 3× cutoff

    double prev_db = 0.0;
    for (int order : {1, 2, 3, 4}) {
        auto sos = butter(order, Wn, FilterType::Lowpass);
        double db = mag_db(sos, omega_stop);
        CHECK(db < prev_db);     // each order adds more attenuation
        prev_db = db;
    }
}

// ── Butterworth HP: DC gain ≈ 0, Nyquist gain = 1 ────────────────────────────
TEST_CASE("Butterworth HP has zero DC gain and unity Nyquist gain", "[butter]")
{
    for (int order : {1, 2, 4}) {
        auto sos = butter(order, 0.3, FilterType::Highpass);
        CHECK(sos_mag(sos, 0.0)                   < 1e-3);
        CHECK_THAT(sos_mag(sos, std::numbers::pi), WithinAbs(1.0, 1e-6));
    }
}

// ── Butterworth HP: monotone increasing magnitude in passband ─────────────────
TEST_CASE("Butterworth HP magnitude is monotone increasing", "[butter]")
{
    auto sos = butter(4, 0.3, FilterType::Highpass);
    // Sample 5 points from just below cutoff to Nyquist; each should be larger
    double prev = 0.0;
    for (double f : {0.35, 0.45, 0.60, 0.75, 1.0}) {
        double m = sos_mag(sos, std::numbers::pi * f);
        CHECK(m > prev);
        prev = m;
    }
}

// ── butter with fs parameter: same response as normalised ────────────────────
TEST_CASE("butter with fs=1000 matches normalised Wn", "[butter]")
{
    // butter(4, 0.2) and butter(4, 100.0, {.fs=1000}) should be identical
    auto sos_norm = butter(4, 0.2, FilterType::Lowpass);
    auto sos_fs   = butter(4, 100.0, FilterType::Lowpass, {.fs = 1000.0});

    // Compare responses at a few frequencies
    for (double w : {0.0, 0.3, 0.5, 1.0, std::numbers::pi}) {
        CHECK_THAT(sos_mag(sos_fs, w), WithinAbs(sos_mag(sos_norm, w), 1e-6));
    }
}

// ── firwin LP: DC gain == 1 for multiple window types ────────────────────────
TEST_CASE("firwin LP DC gain is 1 for Hann, Hamming, Blackman, Rectangular", "[firwin]")
{
    for (auto win : {Window::Hann, Window::Hamming,
                     Window::Blackman, Window::Rectangular}) {
        auto h = firwin(51, 0.3, win, FilterType::Lowpass);
        double dc = 0.0;
        for (auto v : h) dc += v;
        CHECK_THAT(dc, WithinAbs(1.0, 1e-6));
    }
}

// ── firwin LP: Nyquist gain ≈ 0 ──────────────────────────────────────────────
TEST_CASE("firwin LP Nyquist gain is near zero", "[firwin]")
{
    auto h = firwin(101, 0.2, Window::Hamming, FilterType::Lowpass);
    // Evaluate at Nyquist: sum h[n]*(-1)^n
    double nyquist_gain = 0.0;
    for (std::size_t n = 0; n < h.size(); ++n)
        nyquist_gain += h[n] * (n % 2 == 0 ? 1.0 : -1.0);
    CHECK(std::abs(nyquist_gain) < 0.01);
}

// ── firwin HP: DC gain ≈ 0, Nyquist gain = 1 ─────────────────────────────────
TEST_CASE("firwin HP has near-zero DC gain and unity Nyquist gain", "[firwin]")
{
    // Odd numtaps required for HP
    auto h = firwin(51, 0.3, Window::Hamming, FilterType::Highpass);

    double dc_gain = 0.0;
    double ny_gain = 0.0;
    for (std::size_t n = 0; n < h.size(); ++n) {
        dc_gain += h[n];
        ny_gain += h[n] * (n % 2 == 0 ? 1.0 : -1.0);
    }
    CHECK(std::abs(dc_gain) < 0.01);
    // Finite FIR stop-band residual limits Nyquist gain to within ~1% of 1.0
    CHECK_THAT(std::abs(ny_gain), WithinAbs(1.0, 0.01));
}

// ── firwin: coefficient count matches numtaps ─────────────────────────────────
TEST_CASE("firwin returns exactly numtaps coefficients", "[firwin]")
{
    for (int n : {11, 51, 101, 201}) {
        auto h = firwin(n, 0.2, Window::Hann, FilterType::Lowpass);
        CHECK(static_cast<int>(h.size()) == n);
    }
}

// ── firwin: even numtaps + HP throws ─────────────────────────────────────────
TEST_CASE("firwin throws for even numtaps with Highpass", "[firwin][error]")
{
    CHECK_THROWS_AS(firwin(50, 0.3, Window::Hamming, FilterType::Highpass),
                    ValueError);
}

// ── firwin with fs parameter ──────────────────────────────────────────────────
TEST_CASE("firwin LP with fs parameter has unity DC gain", "[firwin]")
{
    auto h = firwin(51, 100.0, Window::Hamming, FilterType::Lowpass,
                    {.fs = 1000.0});
    double dc = 0.0;
    for (auto v : h) dc += v;
    CHECK_THAT(dc, WithinAbs(1.0, 1e-6));
}

// ── lfilter: moving average (4-tap) ──────────────────────────────────────────
TEST_CASE("lfilter 4-tap moving average smooths a step input", "[lfilter]")
{
    // A step from 0 to 1 should ramp up in 4 samples then hold at 1
    std::vector<Real> x(20, 1.0);
    std::vector<Real> b = {0.25, 0.25, 0.25, 0.25};
    std::vector<Real> a = {1.0};

    auto y = lfilter(b, a, x);
    REQUIRE(y.size() == 20);

    // After 4 samples the MA of all-ones is 1
    for (std::size_t i = 4; i < 20; ++i)
        CHECK_THAT(y[i], WithinAbs(1.0, 1e-12));

    // Before that it ramps
    CHECK_THAT(y[0], WithinAbs(0.25, 1e-12));
    CHECK_THAT(y[1], WithinAbs(0.50, 1e-12));
    CHECK_THAT(y[2], WithinAbs(0.75, 1e-12));
    CHECK_THAT(y[3], WithinAbs(1.00, 1e-12));
}

// ── lfilter: first-order IIR leaky integrator ────────────────────────────────
TEST_CASE("lfilter first-order IIR: impulse response decays geometrically", "[lfilter]")
{
    // H(z) = 1 / (1 - 0.5*z^-1)  → impulse response: y[n] = 0.5^n
    std::vector<Real> b = {1.0};
    std::vector<Real> a = {1.0, -0.5};    // a[1] = -0.5 (so -a[1]*y[n-1])
    auto impulse = unit_impulse(16);

    auto y = lfilter(b, a, impulse);
    REQUIRE(y.size() == 16);
    for (std::size_t n = 0; n < 16; ++n)
        CHECK_THAT(y[n], WithinAbs(std::pow(0.5, static_cast<double>(n)), 1e-10));
}

// ── sosfilt: impulse response of order-1 Butterworth LP ──────────────────────
TEST_CASE("sosfilt impulse response is causal and decays to zero", "[sosfilt]")
{
    auto sos = butter(2, 0.1, FilterType::Lowpass);
    auto imp = unit_impulse(256);
    auto y   = sosfilt(sos, imp);

    REQUIRE(y.size() == 256);
    // First sample must be nonzero (causal response)
    CHECK(std::abs(y[0]) > 1e-6);
    // By sample 200, the LP response to an impulse should have decayed
    double tail = 0.0;
    for (std::size_t i = 200; i < 256; ++i) tail = std::max(tail, std::abs(y[i]));
    CHECK(tail < 1e-6);
}

// ── sosfilt: HP filter passes Nyquist-frequency input ────────────────────────
TEST_CASE("sosfilt HP filter passes Nyquist and blocks DC", "[sosfilt]")
{
    auto sos = butter(4, 0.2, FilterType::Highpass);

    // DC signal (all ones)
    std::vector<Real> dc(512, 1.0);
    auto dc_out = sosfilt(sos, dc);
    double max_dc = 0.0;
    for (std::size_t i = 100; i < 512; ++i)
        max_dc = std::max(max_dc, std::abs(dc_out[i]));
    CHECK(max_dc < 0.001);

    // Nyquist signal (alternating ±1)
    std::vector<Real> ny(512);
    for (std::size_t i = 0; i < 512; ++i) ny[i] = (i % 2 == 0) ? 1.0 : -1.0;
    auto ny_out = sosfilt(sos, ny);
    double max_ny = 0.0;
    for (std::size_t i = 100; i < 512; ++i)
        max_ny = std::max(max_ny, std::abs(ny_out[i]));
    CHECK(max_ny > 0.5);
}

// ── freqz: frequency axis spans [0, fs/2] and magnitude is monotone LP ───────
TEST_CASE("freqz LP response is monotone decreasing", "[freqz]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    auto [f, H] = freqz(sos, 128, 1000.0);

    REQUIRE(f.size() == 128);
    CHECK_THAT(f[0],   WithinAbs(0.0, 1e-9));
    // freqz evaluates at ω=π*k/nfreqs, so f[127] = 127/128 * 500 = 496.09 Hz
    CHECK_THAT(f[127], WithinAbs(127.0 / 128.0 * 500.0, 1.0));

    // Magnitude should be monotone decreasing for a Butterworth LP
    for (std::size_t k = 1; k < H.size(); ++k)
        CHECK(std::abs(H[k]) <= std::abs(H[k-1]) + 1e-6);
}

// ── freqz: HP response is monotone increasing in passband ─────────────────────
TEST_CASE("freqz HP response is monotone increasing above cutoff", "[freqz]")
{
    auto sos = butter(4, 0.3, FilterType::Highpass);
    auto [f, H] = freqz(sos, 128, 1000.0);

    // Above the cutoff (~300 Hz = index ~77 for fs=1000, nfreqs=128)
    // magnitude should be increasing
    for (std::size_t k = 80; k < H.size() - 1; ++k)
        CHECK(std::abs(H[k+1]) >= std::abs(H[k]) - 1e-6);
}

// ── firwin: additional window types ──────────────────────────────────────────

TEST_CASE("firwin throws for numtaps <= 0", "[firwin][error]")
{
    CHECK_THROWS_AS(firwin(0,  0.3, Window::Hamming, FilterType::Lowpass), ValueError);
    CHECK_THROWS_AS(firwin(-1, 0.3, Window::Hamming, FilterType::Lowpass), ValueError);
}

TEST_CASE("firwin BlackmanHarris window has unity DC gain", "[firwin]")
{
    auto h = firwin(51, 0.3, Window::BlackmanHarris, FilterType::Lowpass);
    double dc = 0.0;
    for (auto v : h) dc += v;
    CHECK_THAT(dc, WithinAbs(1.0, 1e-6));
}

TEST_CASE("firwin FlatTop window has unity DC gain", "[firwin]")
{
    auto h = firwin(51, 0.3, Window::FlatTop, FilterType::Lowpass);
    double dc = 0.0;
    for (auto v : h) dc += v;
    CHECK_THAT(dc, WithinAbs(1.0, 1e-6));
}

// ── freqz: error paths ────────────────────────────────────────────────────────

TEST_CASE("freqz throws for empty SOS", "[freqz][error]")
{
    SOS empty;
    CHECK_THROWS_AS(freqz(empty, 128, 1000.0), ValueError);
}

TEST_CASE("freqz throws for nfreqs=0", "[freqz][error]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    CHECK_THROWS_AS(freqz(sos, 0, 1000.0), ValueError);
}

// ── sosfilt / lfilter: error paths ───────────────────────────────────────────

TEST_CASE("sosfilt throws for empty SOS", "[sosfilt][error]")
{
    SOS empty;
    std::vector<Real> x(16, 1.0);
    CHECK_THROWS_AS(sosfilt(empty, x), ValueError);
}

TEST_CASE("lfilter throws for empty coefficients or zero a[0]", "[lfilter][error]")
{
    std::vector<Real> sig    = {1.0, 2.0, 3.0};
    std::vector<Real> b      = {1.0};
    std::vector<Real> a      = {1.0};
    std::vector<Real> empty;
    std::vector<Real> a_zero = {0.0};

    CHECK_THROWS_AS(lfilter(empty,  a,      sig), ValueError);   // b empty
    CHECK_THROWS_AS(lfilter(b,      empty,  sig), ValueError);   // a empty
    CHECK_THROWS_AS(lfilter(b,      a_zero, sig), ValueError);   // a[0] == 0
}

TEST_CASE("lfilter b={1} a={1}: pass-through exercises nz=0 path", "[lfilter]")
{
    // nz = max(|b|, |a|) - 1 = max(1,1) - 1 = 0
    // Exercises (nz > 0 ? z[0] : 0.0) false branch and if(nz>0) false branch.
    std::vector<Real> x = {1.0, 2.0, 3.0, -1.0};
    std::vector<Real> b = {1.0};
    std::vector<Real> a = {1.0};
    auto y = lfilter(b, a, x);
    REQUIRE(y.size() == 4);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_THAT(y[i], WithinAbs(x[i], 1e-12));
}

// ── zpk2sos: exercise complex-zero and real-pole-pair branches ────────────────

TEST_CASE("zpk2sos handles complex zeros — frequency response is finite and non-trivial", "[filter_design]")
{
    // Complex conjugate zero pair — exercises the cmplx_zeros_upper branch.
    // Before the fix, cmplx_zeros_upper was populated but never consumed,
    // causing the numerator to silently default to (z+1)² instead of the
    // intended zeros at 0.5±0.5j.
    std::vector<Complex> zeros = {{0.5,  0.5}, {0.5, -0.5}};
    std::vector<Complex> poles = {{-0.5, 0.5}, {-0.5, -0.5}};
    auto sos = detail::zpk2sos(zeros, poles, 1.0);
    REQUIRE(sos.size() == 1);

    // Frequency response must be finite and non-negative at all frequencies.
    for (double omega : {0.0, 0.3, std::numbers::pi / 2.0, 2.0, std::numbers::pi}) {
        double mag = sos_mag(sos, omega);
        CHECK(std::isfinite(mag));
        CHECK(mag >= 0.0);
    }

    // DC gain must be nonzero (the zeros at 0.5±0.5j are not at z=1).
    CHECK(sos_mag(sos, 0.0) > 1e-6);
}

TEST_CASE("zpk2sos handles paired real poles", "[filter_design]")
{
    // Two real poles — exercises the real-pole-pairs loop
    std::vector<Complex> zeros = {{-1.0, 0.0}, {-1.0, 0.0}};
    std::vector<Complex> poles = {{-0.5, 0.0}, {-0.8, 0.0}};
    auto sos = detail::zpk2sos(zeros, poles, 1.0);
    REQUIRE(sos.size() == 1);
}

// ── High-order Butterworth: numerical stability check ─────────────────────────
// Order-12 requires 6 biquad sections. This exercises zpk2sos's full pairing
// logic and verifies the gain accumulation doesn't blow up numerically.

TEST_CASE("Butterworth order=12 LP has unity DC gain", "[butter]")
{
    auto sos = butter(12, 0.2, FilterType::Lowpass);
    CHECK_THAT(sos_mag(sos, 0.0), WithinAbs(1.0, 1e-5));
    // Stopband should still attenuate heavily
    CHECK(sos_mag(sos, std::numbers::pi) < 1e-6);
}

// ── normalise_wn: negative fs should throw ────────────────────────────────────

TEST_CASE("butter throws ValueError for negative fs", "[butter][error]")
{
    CHECK_THROWS_AS(butter(4, 100.0, FilterType::Lowpass, {.fs = -1.0}), ValueError);
}

// ── freqz with fs=0: normalised frequency axis ────────────────────────────────
// When fs=0 (the default), freqz returns frequencies in [0, 0.5] (fraction of
// sample rate) instead of Hz. This branch is distinct from the fs>0 path.

TEST_CASE("freqz with fs=0 returns normalised frequency axis in [0, 0.5]", "[freqz]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    auto [f, H] = freqz(sos, 256);   // fs omitted → default 0.0 → normalised

    REQUIRE(f.size() == 256);
    CHECK_THAT(f.front(), WithinAbs(0.0,  1e-9));
    // Last point: ω = π*(255/256) → normalised freq = 255/512 ≈ 0.498
    CHECK_THAT(f.back(),  WithinAbs(255.0 / 512.0, 1e-6));
    // DC gain still ~1
    CHECK_THAT(std::abs(H[0]), WithinAbs(1.0, 1e-6));
}

// ── zpk2sos edge cases ────────────────────────────────────────────────────────

TEST_CASE("zpk2sos uses default z=-1 when real zeros exhausted (biquad)", "[filter_design]")
{
    // Complex pole pair with no zeros at all → next_biquad_zeros pops twice from
    // an empty real_zeros list, returning Complex(-1,0) both times (line 153).
    std::vector<Complex> zeros = {};
    std::vector<Complex> poles = {Complex(-0.5, 0.3), Complex(-0.5, -0.3)};
    auto sos = detail::zpk2sos(zeros, poles, 1.0);
    REQUIRE(sos.size() == 1);
    for (double omega : {0.0, 1.0, std::numbers::pi}) {
        CHECK(std::isfinite(sos_mag(sos, omega)));
    }
}

TEST_CASE("zpk2sos uses default z=-1 when real zeros exhausted (lone real pole)", "[filter_design]")
{
    // Single real pole with no zeros → next_single_zero returns Complex(-1,0)
    // from an empty real_zeros list (line 161).
    std::vector<Complex> zeros = {};
    std::vector<Complex> poles = {Complex(-0.5, 0.0)};
    auto sos = detail::zpk2sos(zeros, poles, 1.0);
    REQUIRE(sos.size() == 1);
    for (double omega : {0.0, 1.0, std::numbers::pi}) {
        CHECK(std::isfinite(sos_mag(sos, omega)));
    }
}

TEST_CASE("zpk2sos throws when complex zeros outnumber complex poles", "[filter_design]")
{
    // 2 complex zero pairs vs 1 complex pole pair → 1 zero pair unmatched → throw.
    std::vector<Complex> zeros = {
        Complex(0.3, 0.1), Complex(0.3, -0.1),
        Complex(0.4, 0.2), Complex(0.4, -0.2)
    };
    std::vector<Complex> poles = {Complex(-0.5, 0.3), Complex(-0.5, -0.3)};
    CHECK_THROWS_AS(detail::zpk2sos(zeros, poles, 1.0), NumericalError);
}

// ── firwin: normalisation sum near zero throws ────────────────────────────────
// When the cutoff frequency is extremely small, the windowed sinc sums to
// nearly zero after tapering, and the normalisation step throws.

TEST_CASE("firwin throws NumericalError when cutoff is too close to zero", "[firwin][error]")
{
    // Wn=1e-13 → fc = 5e-14; windowed sinc sum ≈ 1e-13 << 1e-12 → throw.
    CHECK_THROWS_AS(firwin(3, 1e-13, Window::Hamming), NumericalError);
}

// ── firwin: even numtaps + Lowpass does not throw ─────────────────────────────
// Covers design.hpp:384 false sub-branch: numtaps%2==0 is true but type!=Highpass,
// so the short-circuit && is true on left but false on right → no throw.
TEST_CASE("firwin even numtaps with Lowpass does not throw", "[firwin]")
{
    auto h = firwin(50, 0.3, Window::Hamming, FilterType::Lowpass);
    REQUIRE(h.size() == 50);
    double dc = 0.0;
    for (auto v : h) dc += v;
    CHECK_THAT(dc, WithinAbs(1.0, 1e-6));
}

// ── lfilter with na > nb: false branch in delay-line update ──────────────────
// b={1.0} (nb=1), a={1.0,-0.5,0.25} (na=3), nz=2.
// In the inner loop at i=0: i+1=1 >= nb=1 → (i+1 < nb ? ... : 0.0) takes false.
// This covers the else-0.0 path in apply.hpp that is skipped when nb >= na.
TEST_CASE("lfilter na>nb correctly implements second-order IIR", "[lfilter]")
{
    std::vector<Real> b = {1.0};
    std::vector<Real> a = {1.0, -0.5, 0.25};
    auto impulse = unit_impulse(16);

    auto y = lfilter(b, a, impulse);
    REQUIRE(y.size() == 16);
    // y[n] = x[n] + 0.5*y[n-1] - 0.25*y[n-2]
    CHECK_THAT(y[0], WithinAbs(1.0, 1e-12));   // h[0] = 1
    CHECK_THAT(y[1], WithinAbs(0.5, 1e-12));   // h[1] = 0.5
    // Response should decay over time
    CHECK(std::abs(y[15]) < std::abs(y[0]));
}
