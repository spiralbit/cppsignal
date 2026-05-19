// test_filter_design.cpp — unit tests for filter design and application
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <vector>
#include <cmath>
#include <numbers>

using namespace cps;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Evaluate the magnitude response of an SOS filter at a normalised frequency w ∈ [0, π]
static double sos_magnitude(const SOS& sos, double omega)
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

// ── Butterworth LP: DC gain == 1 ─────────────────────────────────────────────
TEST_CASE("Butterworth LP DC gain is 1.0", "[butter]")
{
    for (int order : {1, 2, 3, 4, 6, 8}) {
        auto sos = butter(order, 0.2, FilterType::Lowpass);
        double gain_dc = sos_magnitude(sos, 0.0);   // ω=0 is DC
        CHECK_THAT(gain_dc, WithinAbs(1.0, 1e-6));
    }
}

// ── Butterworth LP: -3 dB at cutoff ──────────────────────────────────────────
TEST_CASE("Butterworth LP is -3 dB at Wn", "[butter]")
{
    const double Wn = 0.3;                      // normalised cutoff
    for (int order : {2, 4, 6}) {
        auto sos = butter(order, Wn, FilterType::Lowpass);
        double omega_c = std::numbers::pi * Wn; // digital cutoff in rad/sample
        double mag     = sos_magnitude(sos, omega_c);
        double mag_db  = 20.0 * std::log10(mag);
        // Butterworth is exactly -3.01 dB at the cutoff (within numerical error)
        CHECK_THAT(mag_db, WithinAbs(-3.01, 0.05));
    }
}

// ── Butterworth HP: Nyquist gain == 1 ────────────────────────────────────────
TEST_CASE("Butterworth HP Nyquist gain is 1.0", "[butter]")
{
    for (int order : {1, 2, 4}) {
        auto sos = butter(order, 0.3, FilterType::Highpass);
        double gain_ny = sos_magnitude(sos, std::numbers::pi); // ω=π is Nyquist
        CHECK_THAT(gain_ny, WithinAbs(1.0, 1e-6));
    }
}

// ── Butterworth LP: deep stopband attenuation ─────────────────────────────────
// At twice the normalised cutoff frequency, a 4th-order Butterworth should
// attenuate by at least ~24 dB (approximately -20*n dB/decade roll-off).
TEST_CASE("Butterworth LP order 4 attenuates stopband", "[butter]")
{
    const double Wn   = 0.2;
    auto sos           = butter(4, Wn, FilterType::Lowpass);
    double omega_stop  = std::numbers::pi * Wn * 2.0;   // 2× cutoff
    double mag_db      = 20.0 * std::log10(sos_magnitude(sos, omega_stop));
    CHECK(mag_db < -24.0);   // should be well below -24 dB
}

// ── freqz: length and frequency axis ─────────────────────────────────────────
TEST_CASE("freqz returns correct length and DC gain", "[freqz]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    auto [f, H] = freqz(sos, 256, 1000.0);

    REQUIRE(f.size() == 256);
    REQUIRE(H.size() == 256);

    // First point is DC (ω=0)
    CHECK_THAT(f[0], WithinAbs(0.0, 1e-9));

    // DC gain should be ~1.0
    CHECK_THAT(std::abs(H[0]), WithinAbs(1.0, 1e-6));
}

// ── sosfilt: DC passthrough ───────────────────────────────────────────────────
TEST_CASE("sosfilt passes a DC signal unchanged through LP filter", "[sosfilt]")
{
    auto sos = butter(4, 0.3, FilterType::Lowpass);
    std::vector<Real> dc(1000, 1.0);   // constant signal

    auto out = sosfilt(sos, dc);

    // Settle time is a few hundred samples; check last 100 are near 1.0
    REQUIRE(out.size() == dc.size());
    for (std::size_t i = 900; i < 1000; ++i)
        CHECK_THAT(out[i], WithinAbs(1.0, 1e-6));
}

// ── sosfilt: blocks Nyquist with LP ──────────────────────────────────────────
TEST_CASE("sosfilt attenuates Nyquist-frequency input with LP filter", "[sosfilt]")
{
    // Generate alternating ±1 (Nyquist frequency)
    std::vector<Real> nyquist(256);
    for (std::size_t i = 0; i < 256; ++i)
        nyquist[i] = (i % 2 == 0) ? 1.0 : -1.0;

    auto sos = butter(4, 0.2, FilterType::Lowpass);
    auto out = sosfilt(sos, nyquist);

    // After settling, output should be very small (heavily attenuated)
    double max_tail = 0.0;
    for (std::size_t i = 64; i < 256; ++i)
        max_tail = std::max(max_tail, std::abs(out[i]));

    CHECK(max_tail < 0.01);
}

// ── lfilter: identity filter b=[1], a=[1] ────────────────────────────────────
TEST_CASE("lfilter with identity coefficients passes signal unchanged", "[lfilter]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, -1.0, 0.5};
    std::vector<Real> b = {1.0};
    std::vector<Real> a = {1.0};

    auto y = lfilter(b, a, x);
    REQUIRE(y.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK_THAT(y[i], WithinAbs(x[i], 1e-12));
}

// ── firwin: DC gain == 1 ──────────────────────────────────────────────────────
TEST_CASE("firwin LP has unity DC gain", "[firwin]")
{
    auto h = firwin(51, 0.2, Window::Hamming, FilterType::Lowpass);
    REQUIRE(h.size() == 51);

    // DC gain = sum of all coefficients
    double dc_gain = 0.0;
    for (auto v : h) dc_gain += v;
    CHECK_THAT(dc_gain, WithinAbs(1.0, 1e-6));
}

// ── butter: bad arguments throw ───────────────────────────────────────────────
TEST_CASE("butter throws ValueError for bad arguments", "[butter][error]")
{
    CHECK_THROWS_AS(butter(0,   0.2, FilterType::Lowpass), ValueError);  // order 0
    CHECK_THROWS_AS(butter(4,  -0.1, FilterType::Lowpass), ValueError);  // Wn < 0
    CHECK_THROWS_AS(butter(4,   1.5, FilterType::Lowpass), ValueError);  // Wn > 1
    CHECK_THROWS_AS(butter(4,   0.3, FilterType::Bandpass), NotImplemented); // not yet
}

// ── firwin: Kaiser and Tukey throw NotImplemented ────────────────────────────
TEST_CASE("firwin throws NotImplemented for Kaiser and Tukey windows", "[firwin][error]")
{
    CHECK_THROWS_AS(firwin(51, 0.2, Window::Kaiser), NotImplemented);
    CHECK_THROWS_AS(firwin(51, 0.2, Window::Tukey),  NotImplemented);
}

// ── sosfilt vs lfilter consistency ───────────────────────────────────────────
// Converting the same SOS to b/a coefficients and running lfilter must give
// the same output as sosfilt on any test signal.
TEST_CASE("sosfilt and lfilter give identical output for order-2 Butterworth LP", "[sosfilt][lfilter]")
{
    auto sos = butter(2, 0.3, FilterType::Lowpass);
    REQUIRE(sos.size() == 1);   // order-2 → single biquad

    // Extract b and a from the one SOS row
    std::vector<Real> b = {sos[0][0], sos[0][1], sos[0][2]};
    std::vector<Real> a = {sos[0][3], sos[0][4], sos[0][5]};

    // Test signal: decaying sine
    auto t   = linspace(0.0, 1.0, 500, false);
    auto sig = sinusoid(t, 50.0, 1.0);

    auto y_sos = sosfilt(sos, sig);
    auto y_lf  = lfilter(b, a, sig);

    REQUIRE(y_sos.size() == y_lf.size());
    for (std::size_t i = 0; i < y_sos.size(); ++i)
        CHECK_THAT(y_sos[i], WithinAbs(y_lf[i], 1e-12));
}
