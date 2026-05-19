// test_fft_extended.cpp — extended FFT / RFFT / fftfreq tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <cmath>
#include <numbers>
#include <vector>
#include <span>
#include <numeric>

using namespace cps;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

constexpr double EPS  = 1e-9;
constexpr double LOOSE = 1e-6;

// ── FFT of all-zeros ──────────────────────────────────────────────────────────
TEST_CASE("FFT of all-zeros is all-zeros", "[fft]")
{
    std::vector<Complex> x(16, {0.0, 0.0});
    auto X = fft(std::span<const Complex>(x));
    REQUIRE(X.size() == 16);
    for (auto& c : X) {
        CHECK_THAT(c.real(), WithinAbs(0.0, EPS));
        CHECK_THAT(c.imag(), WithinAbs(0.0, EPS));
    }
}

// ── FFT of a DC signal (all ones) ─────────────────────────────────────────────
// For x[n]=1, X[0]=N and X[k]=0 for k≠0.
TEST_CASE("FFT of DC signal: X[0]=N, all others zero", "[fft]")
{
    constexpr std::size_t N = 32;
    std::vector<Complex> x(N, {1.0, 0.0});
    auto X = fft(std::span<const Complex>(x));

    REQUIRE(X.size() == N);
    CHECK_THAT(X[0].real(), WithinAbs(static_cast<double>(N), EPS));
    CHECK_THAT(X[0].imag(), WithinAbs(0.0, EPS));
    for (std::size_t k = 1; k < N; ++k) {
        CHECK_THAT(X[k].real(), WithinAbs(0.0, LOOSE));
        CHECK_THAT(X[k].imag(), WithinAbs(0.0, LOOSE));
    }
}

// ── Parseval's theorem ────────────────────────────────────────────────────────
// For the unnormalised forward DFT: sum|X[k]|² = N * sum|x[n]|²
TEST_CASE("Parseval: sum|X[k]|^2 == N * sum|x[n]|^2", "[fft]")
{
    std::vector<Complex> x = {
        {1.0,  0.5}, {-0.3, 1.2}, {0.7, -0.9}, {0.0, 0.4},
        {2.0, -1.0}, {-1.5, 0.3}, {0.8,  0.8}, {-0.2, -0.6}
    };
    const std::size_t N = x.size();
    auto X = fft(std::span<const Complex>(x));

    double time_energy = 0.0;
    for (auto& c : x) time_energy += std::norm(c);
    double freq_energy = 0.0;
    for (auto& C : X) freq_energy += std::norm(C);

    CHECK_THAT(freq_energy, WithinRel(N * time_energy, 1e-9));
}

// ── FFT linearity ─────────────────────────────────────────────────────────────
// fft(a*x + b*y) == a*fft(x) + b*fft(y)
TEST_CASE("FFT is linear", "[fft]")
{
    constexpr std::size_t N = 16;
    constexpr double a = 2.0, b = -0.5;

    std::vector<Complex> x(N), y(N), z(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n] = {std::cos(2.0 * std::numbers::pi * n / N), 0.0};
        y[n] = {std::sin(4.0 * std::numbers::pi * n / N), 0.0};
        z[n] = a * x[n] + b * y[n];
    }

    auto X  = fft(std::span<const Complex>(x));
    auto Y  = fft(std::span<const Complex>(y));
    auto Z  = fft(std::span<const Complex>(z));

    for (std::size_t k = 0; k < N; ++k) {
        Complex expected = a * X[k] + b * Y[k];
        CHECK_THAT(Z[k].real(), WithinAbs(expected.real(), LOOSE));
        CHECK_THAT(Z[k].imag(), WithinAbs(expected.imag(), LOOSE));
    }
}

// ── Conjugate symmetry of FFT for real input ─────────────────────────────────
// For real x: X[k] = conj(X[N-k])
TEST_CASE("FFT of real signal satisfies conjugate symmetry X[k]=conj(X[N-k])", "[fft]")
{
    constexpr std::size_t N = 16;
    std::vector<Complex> x(N);
    for (std::size_t n = 0; n < N; ++n)
        x[n] = {std::sin(2.0 * std::numbers::pi * 3 * n / N) + 0.5, 0.0};

    auto X = fft(std::span<const Complex>(x));
    for (std::size_t k = 1; k < N/2; ++k) {
        CHECK_THAT(X[k].real(),  WithinAbs( X[N-k].real(), EPS));
        CHECK_THAT(X[k].imag(),  WithinAbs(-X[N-k].imag(), EPS));
    }
}

// ── Round-trip for various power-of-2 sizes ──────────────────────────────────
TEST_CASE("ifft(fft(x))==x for N = 4, 16, 64, 256", "[fft]")
{
    for (std::size_t N : {4u, 16u, 64u, 256u}) {
        std::vector<Complex> x(N);
        for (std::size_t n = 0; n < N; ++n)
            x[n] = {static_cast<double>(n % 7) - 3.0,
                    std::cos(2.0 * std::numbers::pi * n / N)};

        auto X  = fft(std::span<const Complex>(x));
        auto x2 = ifft(std::span<const Complex>(X));
        REQUIRE(x2.size() == N);
        for (std::size_t n = 0; n < N; ++n) {
            CHECK_THAT(x2[n].real(), WithinAbs(x[n].real(), EPS));
            CHECK_THAT(x2[n].imag(), WithinAbs(x[n].imag(), EPS));
        }
    }
}

// ── Non-power-of-2 round-trip ─────────────────────────────────────────────────
// The built-in fallback handles arbitrary sizes correctly (albeit slowly).
TEST_CASE("ifft(fft(x))==x for non-power-of-2 sizes N=5,7,13", "[fft]")
{
    for (std::size_t N : {5u, 7u, 13u}) {
        std::vector<Complex> x(N);
        for (std::size_t n = 0; n < N; ++n)
            x[n] = {static_cast<double>(n), -static_cast<double>(n) * 0.5};

        auto X  = fft(std::span<const Complex>(x));
        auto x2 = ifft(std::span<const Complex>(X));
        REQUIRE(x2.size() == N);
        for (std::size_t n = 0; n < N; ++n) {
            CHECK_THAT(x2[n].real(), WithinAbs(x[n].real(), 1e-10));
            CHECK_THAT(x2[n].imag(), WithinAbs(x[n].imag(), 1e-10));
        }
    }
}

// ── RFFT output size = N/2+1 ─────────────────────────────────────────────────
TEST_CASE("rfft output length == N/2+1 for various N", "[fft]")
{
    for (std::size_t N : {8u, 16u, 32u, 64u, 128u}) {
        std::vector<Real> x(N, 1.0);
        auto X = rfft(std::span<const Real>(x));
        CHECK(X.size() == N / 2 + 1);
    }
}

// ── RFFT: DC and Nyquist bins are purely real for a real input ────────────────
TEST_CASE("rfft DC and Nyquist bins have zero imaginary part", "[fft]")
{
    constexpr std::size_t N = 64;
    std::vector<Real> x(N);
    for (std::size_t n = 0; n < N; ++n)
        x[n] = std::cos(2.0 * std::numbers::pi * 4 * n / N);  // k=4 cosine

    auto X = rfft(std::span<const Real>(x));
    CHECK_THAT(X[0].imag(),    WithinAbs(0.0, LOOSE));  // DC
    CHECK_THAT(X[N/2].imag(),  WithinAbs(0.0, LOOSE));  // Nyquist
}

// ── RFFT round-trip for odd N ─────────────────────────────────────────────────
TEST_CASE("irfft(rfft(x), N)==x for odd N", "[fft]")
{
    std::vector<Real> x = {1.0, 3.0, -1.5, 2.5, 0.5};
    const std::size_t N = x.size();
    auto X  = rfft(std::span<const Real>(x));
    auto x2 = irfft(std::span<const Complex>(X), N);
    REQUIRE(x2.size() == N);
    for (std::size_t i = 0; i < N; ++i)
        CHECK_THAT(x2[i], WithinAbs(x[i], 1e-10));
}

// ── fftfreq: sample rate scaling ─────────────────────────────────────────────
TEST_CASE("fftfreq scales correctly with sample spacing d", "[fft]")
{
    // d = 1/fs, so fftfreq(N, 1/fs) returns frequencies in Hz
    auto f = fftfreq(8, 1.0 / 1000.0);  // fs = 1000 Hz
    REQUIRE(f.size() == 8);
    CHECK_THAT(f[0], WithinAbs(   0.0, EPS));
    CHECK_THAT(f[1], WithinAbs( 125.0, EPS));   // 1 * 1000/8
    CHECK_THAT(f[4], WithinAbs(-500.0, EPS));   // Nyquist (negative half)
    CHECK_THAT(f[7], WithinAbs(-125.0, EPS));
}

// ── rfftfreq: last bin is Nyquist ─────────────────────────────────────────────
TEST_CASE("rfftfreq last bin == fs/2", "[fft]")
{
    constexpr double fs = 2000.0;
    auto f = rfftfreq(64, 1.0 / fs);
    REQUIRE(f.size() == 33);
    CHECK_THAT(f[0],  WithinAbs(0.0,    EPS));
    CHECK_THAT(f[32], WithinAbs(fs/2.0, EPS));  // 1000 Hz
}

// ── Error paths ────────────────────────────────────────────────────────────────

TEST_CASE("fft throws for empty input", "[fft][error]")
{
    std::vector<Complex> empty;
    CHECK_THROWS_AS(fft(std::span<const Complex>(empty)), ValueError);
}

TEST_CASE("ifft throws for empty input", "[fft][error]")
{
    std::vector<Complex> empty;
    CHECK_THROWS_AS(ifft(std::span<const Complex>(empty)), ValueError);
}

TEST_CASE("rfft throws for empty input", "[fft][error]")
{
    std::vector<Real> empty;
    CHECK_THROWS_AS(rfft(std::span<const Real>(empty)), ValueError);
}

TEST_CASE("irfft throws for empty input", "[fft][error]")
{
    std::vector<Complex> empty;
    CHECK_THROWS_AS(irfft(std::span<const Complex>(empty), 0), ValueError);
}

TEST_CASE("irfft throws for spectrum size mismatch", "[fft][error]")
{
    // 3 complex samples but n=8 expects n/2+1 = 5
    std::vector<Complex> spec(3);
    CHECK_THROWS_AS(irfft(std::span<const Complex>(spec), 8), ValueError);
}

TEST_CASE("fftfreq throws for n=0", "[fft][error]")
{
    CHECK_THROWS_AS(fftfreq(0), ValueError);
}

TEST_CASE("rfftfreq throws for n=0", "[fft][error]")
{
    CHECK_THROWS_AS(rfftfreq(0), ValueError);
}
