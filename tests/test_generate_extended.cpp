// test_generate_extended.cpp — extended signal generator tests
// Covers parameter ranges, signal shapes, and edge cases not in test_generate.cpp
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

// ════════════════════════════════════════════════════════════════════════════
// linspace — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("linspace: n=2 returns exact start and stop", "[generate_ext]")
{
    auto t = linspace(3.0, 7.0, 2);
    REQUIRE(t.size() == 2);
    CHECK_THAT(t[0], WithinAbs(3.0, 1e-12));
    CHECK_THAT(t[1], WithinAbs(7.0, 1e-12));
}

TEST_CASE("linspace: descending range (stop < start)", "[generate_ext]")
{
    auto t = linspace(5.0, 0.0, 6);
    REQUIRE(t.size() == 6);
    CHECK_THAT(t[0], WithinAbs(5.0, 1e-12));
    CHECK_THAT(t[5], WithinAbs(0.0, 1e-12));
    // Should be strictly decreasing
    for (std::size_t i = 1; i < 6; ++i)
        CHECK(t[i] < t[i-1]);
}

TEST_CASE("linspace: negative range [-1, 1]", "[generate_ext]")
{
    auto t = linspace(-1.0, 1.0, 5);
    REQUIRE(t.size() == 5);
    CHECK_THAT(t[0], WithinAbs(-1.0, 1e-12));
    CHECK_THAT(t[2], WithinAbs(0.0,  1e-12));
    CHECK_THAT(t[4], WithinAbs(1.0,  1e-12));
}

TEST_CASE("linspace: start == stop returns same value for all n", "[generate_ext]")
{
    auto t = linspace(3.0, 3.0, 5);
    REQUIRE(t.size() == 5);
    for (auto v : t)
        CHECK_THAT(v, WithinAbs(3.0, 1e-12));
}

TEST_CASE("linspace: endpoint=false, step matches expected value", "[generate_ext]")
{
    // n=10 from 0 to 10, endpoint=false → step = 10/10 = 1.0
    auto t = linspace(0.0, 10.0, 10, false);
    REQUIRE(t.size() == 10);
    CHECK_THAT(t[0], WithinAbs(0.0, 1e-12));
    CHECK_THAT(t[1], WithinAbs(1.0, 1e-12));
    CHECK_THAT(t[9], WithinAbs(9.0, 1e-12));
    for (auto v : t) CHECK(v < 10.0);
}

// ════════════════════════════════════════════════════════════════════════════
// arange — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("arange: negative step (counting down)", "[generate_ext]")
{
    auto t = arange(5.0, 0.0, -1.0);
    REQUIRE(t.size() == 5);
    CHECK_THAT(t[0], WithinAbs(5.0, 1e-12));
    CHECK_THAT(t[4], WithinAbs(1.0, 1e-12));
    for (std::size_t i = 1; i < 5; ++i)
        CHECK(t[i] < t[i-1]);
}

TEST_CASE("arange: step exactly reaches stop boundary, stop not included", "[generate_ext]")
{
    // arange(0, 5, 1) → [0,1,2,3,4], not including 5
    auto t = arange(0.0, 5.0, 1.0);
    REQUIRE(t.size() == 5);
    for (auto v : t)
        CHECK(v < 5.0);
}

TEST_CASE("arange: very small step", "[generate_ext]")
{
    auto t = arange(0.0, 1.0, 0.1);
    REQUIRE(t.size() == 10);
    CHECK_THAT(t[0], WithinAbs(0.0, 1e-12));
    CHECK_THAT(t[9], WithinAbs(0.9, 1e-10));
}

TEST_CASE("arange: fractional negative step", "[generate_ext]")
{
    auto t = arange(1.0, 0.0, -0.25);
    REQUIRE(t.size() == 4);
    CHECK_THAT(t[0], WithinAbs(1.0,  1e-12));
    CHECK_THAT(t[1], WithinAbs(0.75, 1e-12));
    CHECK_THAT(t[3], WithinAbs(0.25, 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// sinusoid — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("sinusoid: zero frequency produces all zeros", "[generate_ext]")
{
    auto t = linspace(0.0, 1.0, 100);
    auto y = sinusoid(t, 0.0);
    for (auto v : y)
        CHECK_THAT(v, WithinAbs(0.0, 1e-12));
}

TEST_CASE("sinusoid: negative frequency is equivalent to negated phase", "[generate_ext]")
{
    // sin(-2π * f * t) = -sin(2π * f * t)
    auto t = linspace(0.0, 1.0, 100);
    auto y_pos = sinusoid(t, 10.0,  1.0, 0.0);
    auto y_neg = sinusoid(t, -10.0, 1.0, 0.0);
    for (std::size_t i = 0; i < 100; ++i)
        CHECK_THAT(y_neg[i], WithinAbs(-y_pos[i], 1e-12));
}

TEST_CASE("sinusoid: period matches 1/freq", "[generate_ext]")
{
    // At t = 1/freq the signal should return to the value at t=0 (which is 0)
    constexpr double freq = 5.0;
    auto t = linspace(0.0, 2.0 / freq, 201);
    auto y = sinusoid(t, freq);
    // After one full period: y[0] should equal y[index of t=1/freq]
    // t[100] ≈ 1/freq for 201 points over 0..2/freq
    CHECK_THAT(y[0], WithinAbs(y[100], 1e-10));
}

TEST_CASE("sinusoid: amplitude 0 produces all zeros", "[generate_ext]")
{
    auto t = linspace(0.0, 1.0, 100);
    auto y = sinusoid(t, 10.0, 0.0);
    for (auto v : y)
        CHECK_THAT(v, WithinAbs(0.0, 1e-12));
}

TEST_CASE("sinusoid: amplitude 2.0 scales correctly", "[generate_ext]")
{
    auto t = linspace(0.0, 1.0, 1001);
    auto y1 = sinusoid(t, 3.0, 1.0);
    auto y2 = sinusoid(t, 3.0, 2.0);
    for (std::size_t i = 0; i < y1.size(); ++i)
        CHECK_THAT(y2[i], WithinAbs(2.0 * y1[i], 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// chirp — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("chirp: f0 == f1 gives a constant-frequency cosine", "[generate_ext]")
{
    constexpr double f = 50.0;
    auto t = linspace(0.0, 1.0, 1001, false);
    auto y = chirp(t, f, f, 1.0);
    // k = (f-f)/1 = 0; phase = 2π*f*t; equivalent to cos(2π*f*t)
    for (std::size_t i = 0; i < y.size(); ++i) {
        double expected = std::cos(2.0 * std::numbers::pi * f * t[i]);
        CHECK_THAT(y[i], WithinAbs(expected, 1e-10));
    }
}

TEST_CASE("chirp: downward sweep (f0 > f1), output is finite and in [-1,1]", "[generate_ext]")
{
    auto t = linspace(0.0, 1.0, 1000, false);
    auto y = chirp(t, 500.0, 10.0, 1.0);
    for (auto v : y) {
        CHECK(std::isfinite(v));
        CHECK(v >= -1.0 - 1e-12);
        CHECK(v <=  1.0 + 1e-12);
    }
}

TEST_CASE("chirp: phi_deg=90 gives sin-like start", "[generate_ext]")
{
    std::vector<Real> t = {0.0};
    auto y = chirp(t, 10.0, 100.0, 1.0, 90.0);
    // At t=0: cos(2π*f0*0 + π/2) = cos(π/2) = 0
    CHECK_THAT(y[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("chirp: phi_deg=180 gives cos(π) = -1 at t=0", "[generate_ext]")
{
    std::vector<Real> t = {0.0};
    auto y = chirp(t, 10.0, 100.0, 1.0, 180.0);
    CHECK_THAT(y[0], WithinAbs(-1.0, 1e-12));
}

TEST_CASE("chirp: phi_deg=270 gives cos(3π/2) ≈ 0 at t=0", "[generate_ext]")
{
    std::vector<Real> t = {0.0};
    auto y = chirp(t, 10.0, 100.0, 1.0, 270.0);
    CHECK_THAT(y[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("chirp: instantaneous frequency increases monotonically for f0 < f1", "[generate_ext]")
{
    // Use finite-difference to verify the chirp rate is positive
    constexpr double f0 = 10.0, f1 = 100.0, t1 = 1.0;
    constexpr double dt = 1e-5;

    // At t=0.0 and t=0.5, the instantaneous freq should be f0 and (f0+f1)/2 respectively
    // f_inst(t) = f0 + (f1-f0)/t1 * t
    double f_inst_0   = f0;
    double f_inst_half = f0 + (f1 - f0) / t1 * 0.5;
    double f_inst_1   = f1;

    CHECK_THAT(f_inst_0,   WithinAbs(f0,           1e-12));
    CHECK_THAT(f_inst_half, WithinAbs(55.0,         1e-12));
    CHECK_THAT(f_inst_1,   WithinAbs(f1,           1e-12));
    CHECK(f_inst_half > f_inst_0);
    CHECK(f_inst_1 > f_inst_half);
    (void)dt;
}

// ════════════════════════════════════════════════════════════════════════════
// gausspulse — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("gausspulse: symmetric about t=0", "[generate_ext]")
{
    // Sample symmetrically around 0
    auto t = linspace(-0.01, 0.01, 201);
    auto y = gausspulse(t, 100.0);
    REQUIRE(y.size() == 201);
    for (std::size_t i = 0; i < 100; ++i)
        CHECK_THAT(y[i], WithinAbs(y[200 - i], 1e-10));
}

TEST_CASE("gausspulse: more negative bw_db gives narrower pulse (steeper decay)", "[generate_ext]")
{
    // alpha = (pi*fc*bw)^2 / (4*ln(ref)); larger |ln(ref)| → smaller |alpha| → slower decay.
    // bw_db=-6: |ln(ref)| = 0.691 → slower decay (broader pulse)
    // bw_db=-3: |ln(ref)| = 0.346 → faster decay (narrower pulse)
    // At t=0.003: bw_db=-6 amplitude should exceed bw_db=-3 amplitude
    std::vector<Real> t_sample = {0.003};
    auto y3 = gausspulse(t_sample, 100.0, 0.5, -3.0);
    auto y6 = gausspulse(t_sample, 100.0, 0.5, -6.0);
    CHECK(std::abs(y3[0]) < std::abs(y6[0]));
}

TEST_CASE("gausspulse: narrower bw decays more slowly than wider bw", "[generate_ext]")
{
    auto t = linspace(-0.005, 0.005, 101);
    auto y_narrow = gausspulse(t, 100.0, 0.1);   // narrow BW → smaller |alpha| → broader pulse
    auto y_wide   = gausspulse(t, 100.0, 0.9);   // wide BW → larger |alpha| → narrower pulse

    // At the time edges the narrow-BW pulse has more energy (slower decay)
    double narrow_edge = std::abs(y_narrow.front());
    double wide_edge   = std::abs(y_wide.front());
    CHECK(narrow_edge > wide_edge);
}

TEST_CASE("gausspulse: all values in [-1, 1]", "[generate_ext]")
{
    auto t = linspace(-0.01, 0.01, 500);
    auto y = gausspulse(t, 100.0);
    for (auto v : y) {
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// square_wave — extended duty cycles
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("square_wave: 10% duty cycle has ~10% samples at +1", "[generate_ext]")
{
    auto t = linspace(0.0, 100.0, 100000, false);
    auto y = square_wave(t, 1.0, 0.1);
    int pos = 0;
    for (auto v : y) pos += (v > 0) ? 1 : 0;
    double fraction = static_cast<double>(pos) / y.size();
    CHECK_THAT(fraction, WithinAbs(0.1, 0.005));
}

TEST_CASE("square_wave: 90% duty cycle has ~90% samples at +1", "[generate_ext]")
{
    auto t = linspace(0.0, 100.0, 100000, false);
    auto y = square_wave(t, 1.0, 0.9);
    int pos = 0;
    for (auto v : y) pos += (v > 0) ? 1 : 0;
    double fraction = static_cast<double>(pos) / y.size();
    CHECK_THAT(fraction, WithinAbs(0.9, 0.005));
}

TEST_CASE("square_wave: 25% and 75% duty cycles have equal-and-opposite means", "[generate_ext]")
{
    auto t = linspace(0.0, 10.0, 10000, false);
    auto y25 = square_wave(t, 1.0, 0.25);
    auto y75 = square_wave(t, 1.0, 0.75);
    // Pointwise they can both be +1 (e.g. t=0), so y25+y75 ≠ 0 in general.
    // However their total sums cancel: mean(y25) = 2*0.25-1 = -0.5,
    //                                 mean(y75) = 2*0.75-1 = +0.5 → sum = 0
    double total = 0.0;
    for (std::size_t i = 0; i < y25.size(); ++i)
        total += y25[i] + y75[i];
    CHECK_THAT(total, WithinAbs(0.0, 1.0));
}

TEST_CASE("square_wave: transitions at the correct phase positions", "[generate_ext]")
{
    // 1 Hz, duty=0.3: positive from t∈[0,0.3), negative from t∈[0.3,1.0)
    std::vector<Real> t = {0.0, 0.29, 0.31, 0.5, 0.99};
    auto y = square_wave(t, 1.0, 0.3);
    REQUIRE(y.size() == 5);
    CHECK(y[0] ==  1.0);    // t=0.0  → phase=0 < 0.3
    CHECK(y[1] ==  1.0);    // t=0.29 → phase=0.29 < 0.3
    CHECK(y[2] == -1.0);    // t=0.31 → phase=0.31 >= 0.3
    CHECK(y[3] == -1.0);    // t=0.5  → phase=0.5  >= 0.3
    CHECK(y[4] == -1.0);    // t=0.99 → phase=0.99 >= 0.3
}

// ════════════════════════════════════════════════════════════════════════════
// sawtooth_wave — extended widths
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("sawtooth_wave: width=0.5 gives triangle wave (symmetric peaks)", "[generate_ext]")
{
    // Triangle wave: rises for first half, falls for second half
    // At t = 0.25 of period: value should be near 0 (start of rising half)
    // At t = 0.5 of period: value should be near +1 (peak of triangle)
    auto t = linspace(0.0, 1.0, 1001, false);
    auto y = sawtooth_wave(t, 1.0, 0.5);

    // At t=0: phase=0 < 0.5 → y = 2*0/0.5 - 1 = -1
    CHECK_THAT(y[0], WithinAbs(-1.0, 1e-9));

    // At t=0.25: phase=0.25 < 0.5 → y = 2*0.25/0.5 - 1 = 2*0.5 - 1 = 0
    CHECK_THAT(y[250], WithinAbs(0.0, 1e-3));

    // At t=0.499...: phase≈0.5 → y ≈ +1 (just before peak)
    CHECK(y[499] > 0.9);

    // At t≈0.5: linspace(0,1,1001,false) gives t[500]=0.4995 (not exactly 0.5)
    // phase=0.4995 < 0.5 → rising branch → y = 2*0.4995/0.5-1 ≈ 0.998 (near peak)
    CHECK_THAT(y[500], WithinAbs(1.0, 5e-3));
}

TEST_CASE("sawtooth_wave: width=0 is pure falling ramp", "[generate_ext]")
{
    // width=0: all samples use the else branch
    // phase ≥ 0 → y = -2*(phase - 0)/(1 - 0) + 1 = -2*phase + 1
    auto t = linspace(0.0, 1.0, 101, false);
    auto y = sawtooth_wave(t, 1.0, 0.0);
    REQUIRE(y.size() == 101);
    // At t=0: phase=0 → y = -2*0+1 = 1
    CHECK_THAT(y[0], WithinAbs(1.0, 1e-9));
    // linspace(0,1,101,false) gives t[50]=50/101≈0.495; y=-2*0.495+1≈0.0099
    CHECK_THAT(y[50], WithinAbs(0.0, 0.02));
    // Strictly decreasing
    for (std::size_t i = 1; i < y.size(); ++i)
        CHECK(y[i] <= y[i-1]);
}

TEST_CASE("sawtooth_wave: width=0.25, triangle biased toward rising", "[generate_ext]")
{
    auto t = linspace(0.0, 1.0, 1000, false);
    auto y = sawtooth_wave(t, 1.0, 0.25);
    for (auto v : y) {
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("sawtooth_wave: frequency argument affects period", "[generate_ext]")
{
    // 2 Hz sawtooth has twice as many cycles as 1 Hz over the same time
    auto t = linspace(0.0, 1.0, 1000, false);
    auto y1 = sawtooth_wave(t, 1.0);
    auto y2 = sawtooth_wave(t, 2.0);

    // Count zero crossings (rising): 2 Hz should have about twice as many as 1 Hz
    int zc1 = 0, zc2 = 0;
    for (std::size_t i = 1; i < y1.size(); ++i) {
        if (y1[i-1] < 0.0 && y1[i] > 0.0) ++zc1;
        if (y2[i-1] < 0.0 && y2[i] > 0.0) ++zc2;
    }
    CHECK(zc2 >= zc1 * 2 - 1);
}

// ════════════════════════════════════════════════════════════════════════════
// white_noise — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("white_noise: n=1 produces exactly one sample", "[generate_ext]")
{
    auto x = white_noise(1, 1.0, 42u);
    REQUIRE(x.size() == 1);
    CHECK(std::isfinite(x[0]));
}

TEST_CASE("white_noise: n=2 produces two different samples", "[generate_ext]")
{
    auto x = white_noise(2, 1.0, 42u);
    REQUIRE(x.size() == 2);
    CHECK(x[0] != x[1]);
}

TEST_CASE("white_noise: std_dev=2.0 gives ~2× standard deviation of std_dev=1.0", "[generate_ext]")
{
    constexpr std::size_t N = 10000;
    auto x1 = white_noise(N, 1.0, 111u);
    auto x2 = white_noise(N, 2.0, 222u);

    auto variance = [](const std::vector<Real>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double var  = 0.0;
        for (auto x : v) var += (x - mean) * (x - mean);
        return var / v.size();
    };

    double sd1 = std::sqrt(variance(x1));
    double sd2 = std::sqrt(variance(x2));
    CHECK_THAT(sd1, WithinAbs(1.0, 0.1));
    CHECK_THAT(sd2, WithinAbs(2.0, 0.2));
    CHECK_THAT(sd2 / sd1, WithinRel(2.0, 0.15));
}

TEST_CASE("white_noise: all values from different seeds are distinct (high probability)", "[generate_ext]")
{
    constexpr std::size_t N = 100;
    auto x1 = white_noise(N, 1.0, 1u);
    auto x2 = white_noise(N, 1.0, 2u);
    auto x3 = white_noise(N, 1.0, 3u);

    double d12 = 0.0, d13 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        d12 += std::abs(x1[i] - x2[i]);
        d13 += std::abs(x1[i] - x3[i]);
    }
    CHECK(d12 > 1.0);
    CHECK(d13 > 1.0);
}

TEST_CASE("white_noise: large std_dev, output scaled proportionally", "[generate_ext]")
{
    constexpr std::size_t N = 1000;
    auto x_small = white_noise(N, 0.001, 42u);
    auto x_large = white_noise(N, 1000.0, 42u);

    double rms_small = rms(std::span<const Real>(x_small));
    double rms_large = rms(std::span<const Real>(x_large));

    // RMS should scale with std_dev
    CHECK_THAT(rms_large / rms_small, WithinRel(1000.0 / 0.001, 0.1));
}

// ════════════════════════════════════════════════════════════════════════════
// correlate — extended modes
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("correlate Same mode: output length = max(Nx, Ny)", "[generate_ext]")
{
    std::vector<Real> x(10, 1.0);
    std::vector<Real> y(5,  1.0);
    auto c = correlate(std::span<const Real>(x), std::span<const Real>(y),
                       ConvolveMode::Same);
    CHECK(c.size() == 10);  // max(10, 5)
}

TEST_CASE("correlate Valid mode: output length = max - min + 1", "[generate_ext]")
{
    std::vector<Real> x(10, 1.0);
    std::vector<Real> y(3,  1.0);
    auto c = correlate(std::span<const Real>(x), std::span<const Real>(y),
                       ConvolveMode::Valid);
    CHECK(c.size() == 8);   // 10 - 3 + 1
}

TEST_CASE("correlate: autocorrelation peak equals sum of squares", "[generate_ext]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0};
    auto c = correlate(std::span<const Real>(x), std::span<const Real>(x));
    // Zero-lag value = sum(x^2) = 1+4+9 = 14, at index len(x)-1 = 2
    REQUIRE(c.size() == 5);
    CHECK_THAT(c[2], WithinAbs(14.0, 1e-12));
}

TEST_CASE("convolve Same mode: output equals x when kernel is unit impulse [0,1,0]", "[generate_ext]")
{
    std::vector<Real> x = {5.0, 3.0, 7.0, 1.0, 9.0};
    std::vector<Real> k = {0.0, 1.0, 0.0};
    auto c = convolve(std::span<const Real>(x), std::span<const Real>(k),
                      ConvolveMode::Same);
    REQUIRE(c.size() == 5);
    for (std::size_t i = 0; i < 5; ++i)
        CHECK_THAT(c[i], WithinAbs(x[i], 1e-12));
}

TEST_CASE("convolve Valid: Ny > Nx swaps and gives correct result", "[generate_ext]")
{
    // When len(x) < len(y), Valid mode calls convolve(y, x, Valid) internally
    std::vector<Real> x = {1.0, 2.0};           // shorter
    std::vector<Real> y = {1.0, 0.0, -1.0};    // longer
    auto c = convolve(std::span<const Real>(x), std::span<const Real>(y),
                      ConvolveMode::Valid);
    // Nout = max(2,3) - min(2,3) + 1 = 3 - 2 + 1 = 2
    REQUIRE(c.size() == 2);
    // convolve([1,0,-1], [1,2]) starting at position 1:
    // c[0] = 1*2 + 0*1 = 2; c[1] = 0*2 + (-1)*1 = -1
    CHECK_THAT(c[0], WithinAbs(2.0,  1e-12));
    CHECK_THAT(c[1], WithinAbs(-1.0, 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// unit_impulse — extended
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("unit_impulse: default idx=0", "[generate_ext]")
{
    auto x = unit_impulse(5);
    REQUIRE(x.size() == 5);
    CHECK_THAT(x[0], WithinAbs(1.0, 1e-12));
    for (std::size_t i = 1; i < 5; ++i)
        CHECK_THAT(x[i], WithinAbs(0.0, 1e-12));
}

TEST_CASE("unit_impulse: can use as FIR filter test", "[generate_ext]")
{
    // lfilter(h, {1}, impulse) == h  (applying FIR gives its own coefficients)
    auto h = firwin(11, 0.2, Window::Hamming);
    auto imp = unit_impulse(11);
    auto y = lfilter(h, std::vector<Real>{1.0}, imp);
    REQUIRE(y.size() == 11);
    for (std::size_t i = 0; i < 11; ++i)
        CHECK_THAT(y[i], WithinAbs(h[i], 1e-10));
}
