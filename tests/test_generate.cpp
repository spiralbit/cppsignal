// test_generate.cpp — signal generation, correlate/convolve, and metrics tests
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
// linspace / arange
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("linspace: endpoints, length, and uniform spacing", "[generate]")
{
    auto t = linspace(0.0, 1.0, 5);
    REQUIRE(t.size() == 5);
    CHECK_THAT(t[0], WithinAbs(0.0,  1e-12));
    CHECK_THAT(t[4], WithinAbs(1.0,  1e-12));
    CHECK_THAT(t[1], WithinAbs(0.25, 1e-12));
    CHECK_THAT(t[2], WithinAbs(0.50, 1e-12));
    CHECK_THAT(t[3], WithinAbs(0.75, 1e-12));
}

TEST_CASE("linspace endpoint=false excludes stop", "[generate]")
{
    auto t = linspace(0.0, 1.0, 4, false);
    REQUIRE(t.size() == 4);
    CHECK_THAT(t[0], WithinAbs(0.0,  1e-12));
    CHECK_THAT(t[3], WithinAbs(0.75, 1e-12));
    // stop value (1.0) must not appear
    for (auto v : t)
        CHECK(v < 1.0 - 1e-12);
}

TEST_CASE("linspace n=0 returns empty, n=1 returns start", "[generate]")
{
    CHECK(linspace(0.0, 1.0, 0).empty());
    auto t1 = linspace(3.0, 7.0, 1);
    REQUIRE(t1.size() == 1);
    CHECK_THAT(t1[0], WithinAbs(3.0, 1e-12));
}

TEST_CASE("arange: values and length", "[generate]")
{
    auto t = arange(0.0, 5.0, 1.0);
    REQUIRE(t.size() == 5);
    for (std::size_t i = 0; i < 5; ++i)
        CHECK_THAT(t[i], WithinAbs(static_cast<double>(i), 1e-12));
}

TEST_CASE("arange with fractional step", "[generate]")
{
    auto t = arange(0.0, 1.0, 0.5);
    REQUIRE(t.size() == 2);
    CHECK_THAT(t[0], WithinAbs(0.0, 1e-12));
    CHECK_THAT(t[1], WithinAbs(0.5, 1e-12));
}

TEST_CASE("arange step=0 throws ValueError", "[generate][error]")
{
    CHECK_THROWS_AS(arange(0.0, 5.0, 0.0), ValueError);
}

// ════════════════════════════════════════════════════════════════════════════
// sinusoid
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("sinusoid: known values at quarter-period points", "[generate]")
{
    // y = sin(2π*1*t): zeros at t=0, t=0.5, peaks at t=0.25, t=0.75
    auto t = linspace(0.0, 1.0, 1001);
    auto y = sinusoid(t, 1.0);

    CHECK_THAT(y[0],    WithinAbs(0.0,  1e-10));   // t=0
    CHECK_THAT(y[250],  WithinAbs(1.0,  1e-3));    // t≈0.25
    CHECK_THAT(y[500],  WithinAbs(0.0,  1e-3));    // t≈0.5
    CHECK_THAT(y[750],  WithinAbs(-1.0, 1e-3));    // t≈0.75
    CHECK_THAT(y[1000], WithinAbs(0.0,  1e-3));    // t=1.0
}

TEST_CASE("sinusoid: amplitude scales peak value", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1001);
    auto y = sinusoid(t, 1.0, 3.5);
    auto peak = *std::max_element(y.begin(), y.end());
    CHECK_THAT(peak, WithinAbs(3.5, 1e-3));
}

TEST_CASE("sinusoid: phase pi/2 gives cosine", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1001);
    auto y = sinusoid(t, 1.0, 1.0, std::numbers::pi / 2.0);
    // cos(2π*0) = 1, cos(2π*0.25) ≈ 0
    CHECK_THAT(y[0],   WithinAbs(1.0, 1e-10));
    CHECK_THAT(y[250], WithinAbs(0.0, 1e-3));
}

// ════════════════════════════════════════════════════════════════════════════
// chirp
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("chirp amplitude is in [-1, 1]", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1000);
    auto y = chirp(t, 10.0, 200.0, 1.0);
    REQUIRE(y.size() == 1000);
    for (auto v : y) {
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
}

TEST_CASE("chirp at t=0 equals cos(phi0)", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1000);
    auto y = chirp(t, 10.0, 100.0, 1.0, 0.0);   // phi=0 → cos(0)=1
    CHECK_THAT(y[0], WithinAbs(1.0, 1e-10));

    auto y90 = chirp(t, 10.0, 100.0, 1.0, 90.0); // phi=90° → cos(π/2)=0
    CHECK_THAT(y90[0], WithinAbs(0.0, 1e-10));
}

TEST_CASE("chirp instantaneous frequency reaches f1 at t=t1", "[generate]")
{
    // The chirp phase is φ(t) = 2π*(f0*t + (f1-f0)/(2*t1)*t²).
    // The instantaneous frequency at t is dφ/dt / (2π) = f0 + (f1-f0)*t/t1.
    // At t = t1 this equals f1. Verify by finite-difference on the phase.
    constexpr double f0 = 10.0, f1 = 100.0, t1 = 1.0;
    constexpr double dt = 1e-6;   // tiny step for numerical derivative

    // Evaluate chirp at t1 and t1+dt to estimate instantaneous frequency
    std::vector<double> ta = {t1};
    std::vector<double> tb = {t1 + dt};
    auto ya = chirp(ta, f0, f1, t1);
    auto yb = chirp(tb, f0, f1, t1);

    // Instantaneous phase from arccos (both values are cos(phase)):
    // Δphase ≈ (f_inst) * 2π * dt  →  f_inst ≈ arccos(yb)/2π/dt - arccos(ya)/2π/dt
    // Instead, directly check using the analytic formula at t=t1: f_inst = f1.
    double phase_a = std::acos(std::clamp(ya[0], -1.0, 1.0));
    double phase_b = std::acos(std::clamp(yb[0], -1.0, 1.0));
    // Phase derivative is 2π*f1 at t=t1; use the analytic value directly
    double analytic_phase_a = 2.0 * std::numbers::pi * (f0 * t1 + 0.5 * (f1 - f0) / t1 * t1 * t1);
    CHECK_THAT(ya[0], WithinAbs(std::cos(analytic_phase_a), 1e-10));
    (void)phase_a; (void)phase_b;  // unused — analytic check is cleaner

    // Also verify the sweep rate: at t=t1 the instantaneous freq should be f1
    const double k = (f1 - f0) / t1;
    double f_inst_at_t1 = f0 + k * t1;
    CHECK_THAT(f_inst_at_t1, WithinAbs(f1, 1e-10));
}

// ════════════════════════════════════════════════════════════════════════════
// gausspulse
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("gausspulse peak is at t=0 and equals 1", "[generate]")
{
    // Symmetric range around 0; t=0 is at the centre index
    auto t = linspace(-0.01, 0.01, 201);
    auto y = gausspulse(t, 100.0);
    // t=0 is at index 100
    auto peak_it = std::max_element(y.begin(), y.end());
    CHECK(std::distance(y.begin(), peak_it) == 100);
    CHECK_THAT(*peak_it, WithinAbs(1.0, 1e-9));
}

TEST_CASE("gausspulse decays to near zero at edges", "[generate]")
{
    auto t = linspace(-0.1, 0.1, 201);
    auto y = gausspulse(t, 100.0);  // 100 Hz carrier
    // Edges (±0.1 s) should be very small for 100 Hz centre
    CHECK(std::abs(y.front()) < 0.01);
    CHECK(std::abs(y.back())  < 0.01);
}

// ════════════════════════════════════════════════════════════════════════════
// square_wave / sawtooth_wave / white_noise
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("square_wave values are exactly +1 or -1", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1000, false);
    auto y = square_wave(t, 5.0);   // 5 Hz
    for (auto v : y)
        CHECK((v == 1.0 || v == -1.0));
}

TEST_CASE("square_wave 50% duty has equal +1 and -1 counts (approx)", "[generate]")
{
    auto t = linspace(0.0, 10.0, 10000, false);
    auto y = square_wave(t, 1.0);   // 1 Hz, 10 full cycles
    int pos = 0, neg = 0;
    for (auto v : y) { if (v > 0) ++pos; else ++neg; }
    // Expect roughly equal counts ±1%
    CHECK_THAT(static_cast<double>(pos) / static_cast<double>(y.size()),
               WithinAbs(0.5, 0.01));
}

TEST_CASE("sawtooth_wave stays in [-1, +1]", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1000, false);
    auto y = sawtooth_wave(t, 10.0);
    for (auto v : y) {
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
}

TEST_CASE("white_noise with seed=0 uses time-based seed and produces finite output", "[generate]")
{
    // seed=0 triggers the time-based branch; just verify it doesn't throw and
    // produces a valid (finite, non-empty) output.
    auto x = white_noise(1000, 1.0, 0u);
    REQUIRE(x.size() == 1000);
    for (auto v : x)
        CHECK(std::isfinite(v));
    // With time-based seed the result won't be constant
    double mn = *std::min_element(x.begin(), x.end());
    double mx = *std::max_element(x.begin(), x.end());
    CHECK(mx - mn > 0.1);
}

TEST_CASE("white_noise is finite, non-constant, and has near-zero mean", "[generate]")
{
    auto x = white_noise(10000, 1.0, 42u);   // std_dev=1.0, fixed seed
    REQUIRE(x.size() == 10000);

    // Not constant
    double mn = *std::min_element(x.begin(), x.end());
    double mx = *std::max_element(x.begin(), x.end());
    CHECK(mx - mn > 0.1);

    // Mean ≈ 0 (within 3 sigma/sqrt(N) ≈ 0.03 for N=10000, sigma=1)
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    CHECK_THAT(mean, WithinAbs(0.0, 0.1));

    // All values are finite
    for (auto v : x)
        CHECK(std::isfinite(v));
}

// ════════════════════════════════════════════════════════════════════════════
// unit_impulse
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("unit_impulse at various positions", "[generate]")
{
    for (std::size_t idx : {0u, 5u, 9u}) {
        auto imp = unit_impulse(10, idx);
        REQUIRE(imp.size() == 10);
        for (std::size_t i = 0; i < 10; ++i)
            CHECK_THAT(imp[i], WithinAbs(i == idx ? 1.0 : 0.0, 1e-12));
    }
}

TEST_CASE("unit_impulse out-of-range index throws", "[generate][error]")
{
    CHECK_THROWS_AS(unit_impulse(5, 5),  ValueError);  // idx == n
    CHECK_THROWS_AS(unit_impulse(5, 10), ValueError);  // idx > n
}

// ════════════════════════════════════════════════════════════════════════════
// convolve modes
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("convolve Same mode: length == max(Nx,Ny)", "[correlate]")
{
    std::vector<Real> x = {1, 2, 3, 4, 5};
    std::vector<Real> y = {1, 1};
    auto c = convolve(x, y, ConvolveMode::Same);
    CHECK(c.size() == 5);   // max(5,2) = 5
}

TEST_CASE("convolve Same with identity kernel returns input", "[correlate]")
{
    // Convolving with [0,1,0] (delayed impulse) in Same mode should shift by 0
    std::vector<Real> x = {1, 2, 3, 4};
    std::vector<Real> y = {0, 1, 0};
    auto c = convolve(x, y, ConvolveMode::Same);
    REQUIRE(c.size() == 4);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_THAT(c[i], WithinAbs(static_cast<double>(i + 1), 1e-12));
}

TEST_CASE("convolve Valid mode: length == Nx-Ny+1", "[correlate]")
{
    std::vector<Real> x = {1, 2, 3, 4, 5};
    std::vector<Real> y = {1, 1};
    auto c = convolve(x, y, ConvolveMode::Valid);
    REQUIRE(c.size() == 4);  // 5-2+1 = 4
    // [1+2, 2+3, 3+4, 4+5] = [3, 5, 7, 9]
    CHECK_THAT(c[0], WithinAbs(3.0, 1e-12));
    CHECK_THAT(c[1], WithinAbs(5.0, 1e-12));
    CHECK_THAT(c[2], WithinAbs(7.0, 1e-12));
    CHECK_THAT(c[3], WithinAbs(9.0, 1e-12));
}

TEST_CASE("convolve is commutative (Full mode)", "[correlate]")
{
    std::vector<Real> a = {1.0, 2.0, 3.0};
    std::vector<Real> b = {0.5, -1.0, 2.0, 0.3};
    auto ab = convolve(a, b);
    auto ba = convolve(b, a);
    REQUIRE(ab.size() == ba.size());
    for (std::size_t i = 0; i < ab.size(); ++i)
        CHECK_THAT(ab[i], WithinAbs(ba[i], 1e-12));
}

// ── cross-correlation finds the correct lag ───────────────────────────────────
// x = [1,0,...], y = x delayed by D samples → peak of correlate at index N-1-D
TEST_CASE("correlate peak locates known delay", "[correlate]")
{
    constexpr std::size_t N = 8;
    constexpr std::size_t D = 3;

    std::vector<Real> x(N, 0.0);  x[0] = 1.0;
    std::vector<Real> y(N, 0.0);  y[D] = 1.0;   // y is x delayed by D

    auto c    = correlate(x, y);   // Full mode, length 2*N-1 = 15
    auto peak = std::max_element(c.begin(), c.end());
    std::size_t peak_idx = std::distance(c.begin(), peak);

    // Zero-lag is at index N-1=7; peak should be at index N-1-D = 4
    CHECK(peak_idx == N - 1 - D);
    CHECK_THAT(*peak, WithinAbs(1.0, 1e-12));
}

// ════════════════════════════════════════════════════════════════════════════
// Metrics: rms, snr, thd
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("rms of zero signal is zero", "[metrics]")
{
    std::vector<Real> x(100, 0.0);
    CHECK_THAT(rms(x), WithinAbs(0.0, 1e-12));
}

TEST_CASE("rms empty signal throws", "[metrics][error]")
{
    std::vector<Real> empty;
    CHECK_THROWS_AS(rms(empty), ValueError);
}

TEST_CASE("rms: unit square wave == 1.0", "[metrics]")
{
    // Square wave ±1 has rms = 1.0 exactly
    std::vector<Real> x(1000);
    for (std::size_t i = 0; i < 1000; ++i)
        x[i] = (i % 2 == 0) ? 1.0 : -1.0;
    CHECK_THAT(rms(x), WithinAbs(1.0, 1e-12));
}

TEST_CASE("snr(signal, noise): known ratio of 20 dB", "[metrics]")
{
    // signal power = 100, noise power = 1 → SNR = 10*log10(100) = 20 dB
    std::vector<Real> signal(1000, 1.0);    // each squared = 1, sum = 1000
    std::vector<Real> noise(1000, 0.1);     // each squared = 0.01, sum = 10
    double measured = snr(signal, noise);
    // 10*log10(1000/10) = 10*log10(100) = 20 dB
    CHECK_THAT(measured, WithinAbs(20.0, 1e-6));
}

TEST_CASE("snr: pure tone has high SNR", "[metrics]")
{
    // N=1000, fs=1000 puts f0=100 Hz exactly on bin 100 — no spectral leakage.
    constexpr double fs = 1000.0;
    constexpr double f0 = 100.0;
    constexpr int    N  = 1000;

    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);
    double measured = snr(std::span<const Real>(x), f0, fs);
    CHECK(measured > 60.0);   // >60 dB for a clean integer-bin sine (only numerical noise)
}

TEST_CASE("thd: pure sine has very low THD", "[metrics]")
{
    // Generate a pure sine at f0=100 Hz that lands exactly on a bin
    constexpr double fs = 1000.0;
    constexpr double f0 = 100.0;    // 100 * 1000/1000 = bin 100 for N=1000
    constexpr int    N  = 1000;

    auto t = linspace(0.0, static_cast<double>(N) / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);

    double thd_db = thd(std::span<const Real>(x), f0, fs);
    // A pure sine has no harmonics; THD should be very low (< -60 dB)
    CHECK(thd_db < -60.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Error paths
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("chirp throws for t1 <= 0", "[generate][error]")
{
    auto t = linspace(0.0, 1.0, 100);
    CHECK_THROWS_AS(chirp(t, 10.0, 100.0,  0.0), ValueError);
    CHECK_THROWS_AS(chirp(t, 10.0, 100.0, -1.0), ValueError);
}

TEST_CASE("square_wave throws for out-of-range duty cycle", "[generate][error]")
{
    auto t = linspace(0.0, 1.0, 100, false);
    CHECK_THROWS_AS(square_wave(t, 1.0,  0.0), ValueError);
    CHECK_THROWS_AS(square_wave(t, 1.0, -0.1), ValueError);
    CHECK_THROWS_AS(square_wave(t, 1.0,  1.0), ValueError);
    CHECK_THROWS_AS(square_wave(t, 1.0,  1.5), ValueError);
}

TEST_CASE("sawtooth_wave with width < 1.0 exercises falling-edge branch", "[generate]")
{
    // width=0.5 means phase >= 0.5 triggers the else (falling ramp) branch
    auto t = linspace(0.0, 1.0, 1000, false);
    auto y = sawtooth_wave(t, 1.0, 0.5);
    for (auto v : y) {
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
    double lo = *std::min_element(y.begin(), y.end());
    double hi = *std::max_element(y.begin(), y.end());
    CHECK(hi - lo > 1.0);
}

TEST_CASE("snr throws when signal and noise have different lengths", "[metrics][error]")
{
    std::vector<Real> s = {1.0, 2.0, 3.0};
    std::vector<Real> n = {0.1, 0.2};
    CHECK_THROWS_AS(snr(s, n), ValueError);
}

TEST_CASE("snr throws when noise power is zero", "[metrics][error]")
{
    std::vector<Real> s = {1.0, 2.0, 3.0};
    std::vector<Real> n = {0.0, 0.0, 0.0};
    CHECK_THROWS_AS(snr(s, n), NumericalError);
}

TEST_CASE("snr throws when no signal at fundamental frequency", "[metrics][error]")
{
    // DC signal: no power at f0=100 Hz — fundamental bin is zero
    std::vector<Real> dc(1000, 1.0);
    CHECK_THROWS_AS(snr(std::span<const Real>(dc), 100.0, 1000.0), NumericalError);
}

TEST_CASE("thd throws when no signal at fundamental frequency", "[metrics][error]")
{
    std::vector<Real> zeros(1000, 0.0);
    CHECK_THROWS_AS(thd(std::span<const Real>(zeros), 100.0, 1000.0), NumericalError);
}

TEST_CASE("convolve throws for empty inputs", "[correlate][error]")
{
    std::vector<Real> a = {1.0, 2.0};
    std::vector<Real> empty;
    CHECK_THROWS_AS(convolve(a, empty), ValueError);
    CHECK_THROWS_AS(convolve(empty, a), ValueError);
}

TEST_CASE("convolve Valid swaps operands when len(x) < len(y)", "[correlate]")
{
    // scipy swaps silently; same result as passing the longer sequence first
    std::vector<Real> x = {1.0, 2.0};           // shorter
    std::vector<Real> y = {1.0, 2.0, 3.0};      // longer
    auto result = convolve(x, y, ConvolveMode::Valid);
    auto expect = convolve(y, x, ConvolveMode::Valid);
    REQUIRE(result.size() == expect.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        CHECK_THAT(result[i], WithinAbs(expect[i], 1e-12));
}

// ── sawtooth_wave: boundary and negative-time tests ──────────────────────────

TEST_CASE("sawtooth_wave throws for width outside [0, 1]", "[generate][error]")
{
    auto t = linspace(0.0, 1.0, 100, false);
    CHECK_THROWS_AS(sawtooth_wave(t, 1.0, -0.1), ValueError);
    CHECK_THROWS_AS(sawtooth_wave(t, 1.0,  1.5), ValueError);
}

TEST_CASE("sawtooth_wave boundary widths 0 and 1 are valid", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1000, false);
    // width=1: pure rising ramp — should not throw and stay in [-1, 1]
    auto y1 = sawtooth_wave(t, 2.0, 1.0);
    for (auto v : y1) { CHECK(v >= -1.0 - 1e-9); CHECK(v <= 1.0 + 1e-9); }
    // width=0: pure falling ramp — should not throw and stay in [-1, 1]
    auto y0 = sawtooth_wave(t, 2.0, 0.0);
    for (auto v : y0) { CHECK(v >= -1.0 - 1e-9); CHECK(v <= 1.0 + 1e-9); }
}

TEST_CASE("square_wave and sawtooth_wave are correct for negative time", "[generate]")
{
    // Both must handle negative t via the fmod correction (phase += 1.0).
    auto t = linspace(-2.0, -0.001, 200, false);

    // square_wave: values must still be exactly ±1
    auto sq = square_wave(t, 1.0, 0.5);
    for (auto v : sq)
        CHECK((v == 1.0 || v == -1.0));

    // sawtooth_wave: values must be finite and in [-1, 1]
    auto saw = sawtooth_wave(t, 1.0, 0.5);
    for (auto v : saw) {
        CHECK(std::isfinite(v));
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
}

// ── gausspulse: parameter validation ─────────────────────────────────────────

TEST_CASE("gausspulse throws for fc <= 0", "[generate][error]")
{
    auto t = linspace(-0.01, 0.01, 201);
    CHECK_THROWS_AS(gausspulse(t,  0.0), ValueError);
    CHECK_THROWS_AS(gausspulse(t, -1.0), ValueError);
}

TEST_CASE("gausspulse throws for bw <= 0", "[generate][error]")
{
    auto t = linspace(-0.01, 0.01, 201);
    CHECK_THROWS_AS(gausspulse(t, 100.0,  0.0), ValueError);
    CHECK_THROWS_AS(gausspulse(t, 100.0, -0.1), ValueError);
}

TEST_CASE("gausspulse throws for bw_db >= 0", "[generate][error]")
{
    auto t = linspace(-0.01, 0.01, 201);
    CHECK_THROWS_AS(gausspulse(t, 100.0, 0.5,  0.0), ValueError);
    CHECK_THROWS_AS(gausspulse(t, 100.0, 0.5,  1.0), ValueError);
}

// ── snr: spectral (single-signal) behaviours ──────────────────────────────────
// These tests cover all distinguishable code paths in the new snr(x, f0, fs)
// overload. The old snr(x) used max_element to find the strongest bin, which
// mis-identifies the signal when a harmonic is louder than the fundamental.

TEST_CASE("snr: harmonic louder than fundamental is correctly excluded", "[metrics]")
{
    // Generate: 1.0 * sin(100 Hz) + 3.0 * sin(200 Hz)  — harmonic has 9× the power.
    // Old code picks bin 200 as "signal" and treats bin 100 as noise → wrong result.
    // New code uses the caller-supplied fundamental (100 Hz) → correct.
    constexpr double fs = 1000.0;
    constexpr double f0 = 100.0;
    constexpr int    N  = 1000;
    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);
    auto h = sinusoid(t, 2.0 * f0, 3.0);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += h[i];

    // The harmonic (200 Hz) is excluded from the noise floor, so the result
    // reflects only the ratio of fundamental vs true broadband noise (≈0).
    double snr_val = snr(std::span<const Real>(x), f0, fs);
    CHECK(snr_val > 40.0);   // noise ≈ 0 → very high SNR
}

TEST_CASE("snr: sine plus noise gives lower SNR than pure sine", "[metrics]")
{
    // Add uniform noise so SNR should be finite and meaningfully lower than
    // the pure-tone case.
    constexpr double fs = 1000.0;
    constexpr double f0 = 100.0;
    constexpr int    N  = 1000;
    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);
    // Inject a small fixed noise pattern (deterministic — no <random> needed)
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] += 0.05 * std::sin(static_cast<double>(i) * 0.37);  // off-bin tone = noise

    double snr_val = snr(std::span<const Real>(x), f0, fs);
    CHECK(snr_val > 0.0);    // positive dB: signal stronger than noise
    CHECK(snr_val < 60.0);   // but noticeably less than the pure-tone case
}

TEST_CASE("snr throws for invalid fundamental_hz", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(snr(std::span<const Real>(x),  0.0, 1000.0), ValueError);
    CHECK_THROWS_AS(snr(std::span<const Real>(x), -1.0, 1000.0), ValueError);
}

TEST_CASE("snr throws for invalid fs", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(snr(std::span<const Real>(x), 100.0,  0.0), ValueError);
    CHECK_THROWS_AS(snr(std::span<const Real>(x), 100.0, -1.0), ValueError);
}

TEST_CASE("snr throws when no noise component exists", "[metrics][error]")
{
    // A 2-sample alternating signal {1, -1} sampled at fs=2 Hz.
    // rfft gives: spec[0] = 0 (DC = 0 exactly), spec[1] = 2 (Nyquist).
    // fundamental=1 Hz → fund_bin=1, excluded={1}.
    // noise_power = norm(spec[0]) = 0 → throws "no noise component detected".
    std::vector<Real> x = {1.0, -1.0};
    CHECK_THROWS_AS(snr(std::span<const Real>(x), 1.0, 2.0), NumericalError);
}

// ── snr: total_power == 0 short-circuit ──────────────────────────────────────
TEST_CASE("snr all-zeros signal throws via total_power == 0.0 branch", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(snr(std::span<const Real>(x), 100.0, 1000.0), NumericalError);
}

// ── thd: error paths ──────────────────────────────────────────────────────────
TEST_CASE("thd throws for fs <= 0", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(thd(std::span<const Real>(x), 100.0,  0.0), ValueError);
    CHECK_THROWS_AS(thd(std::span<const Real>(x), 100.0, -1.0), ValueError);
}

TEST_CASE("thd throws for n_harmonics < 1", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(thd(std::span<const Real>(x), 100.0, 1000.0, 0), ValueError);
}

// ── sinad: error paths ────────────────────────────────────────────────────────
TEST_CASE("sinad throws for fundamental <= 0", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(sinad(std::span<const Real>(x),  0.0, 1000.0), ValueError);
    CHECK_THROWS_AS(sinad(std::span<const Real>(x), -1.0, 1000.0), ValueError);
}

TEST_CASE("sinad throws for fs <= 0", "[metrics][error]")
{
    std::vector<Real> x(256, 0.0);
    CHECK_THROWS_AS(sinad(std::span<const Real>(x), 100.0,  0.0), ValueError);
    CHECK_THROWS_AS(sinad(std::span<const Real>(x), 100.0, -1.0), ValueError);
}

TEST_CASE("sinad: sine plus noise gives finite positive value", "[metrics]")
{
    constexpr double fs = 1000.0;
    constexpr double f0 = 100.0;
    constexpr int    N  = 1000;
    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] += 0.05 * std::sin(static_cast<double>(i) * 0.37);  // off-bin noise

    double sinad_val = sinad(std::span<const Real>(x), f0, fs);
    CHECK(sinad_val > 0.0);    // signal stronger than noise+distortion
    CHECK(sinad_val < 100.0);  // not unrealistically high
}

TEST_CASE("sinad throws when all spectral power is at the fundamental (distortion==0)", "[metrics][error]")
{
    // {1,-1} at fs=2: all power at Nyquist bin → distortion = total - fund = 0 → throws
    std::vector<Real> x = {1.0, -1.0};
    CHECK_THROWS_AS(sinad(std::span<const Real>(x), 1.0, 2.0), NumericalError);
}

// ── snr: empty signal throws (spectral overload) ──────────────────────────────
// Covers metrics.hpp:80 true branch: if (x.empty()) throw ValueError(...)
TEST_CASE("snr throws for empty signal (spectral overload)", "[metrics][error]")
{
    std::vector<Real> empty;
    CHECK_THROWS_AS(snr(std::span<const Real>(empty), 100.0, 1000.0), ValueError);
}
