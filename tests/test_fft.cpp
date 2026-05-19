// test_fft.cpp — unit tests for FFT / IFFT / RFFT / windows
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <cmath>
#include <numbers>
#include <vector>

using namespace cps;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Tolerance for floating-point comparisons
constexpr double EPS = 1e-9;
constexpr double LOOSE = 1e-6;

// ── Helper: magnitude at each bin ─────────────────────────────────────────────
static std::vector<double> magnitudes(const std::vector<Complex>& spec)
{
    std::vector<double> m(spec.size());
    for (std::size_t i = 0; i < spec.size(); ++i)
        m[i] = std::abs(spec[i]);
    return m;
}

// ── FFT round-trip ────────────────────────────────────────────────────────────
TEST_CASE("ifft(fft(x)) == x for complex input", "[fft]")
{
    // Random-ish complex signal
    std::vector<Complex> x = {
        {1.0, 0.0}, {2.0, -1.0}, {0.5, 0.5}, {-1.0, 2.0},
        {3.0, 0.0}, {-0.5, -0.5}, {0.0, 1.0}, {1.5, -1.5}
    };

    auto X    = fft(std::span<const Complex>(x));
    auto x_rt = ifft(std::span<const Complex>(X));

    REQUIRE(x_rt.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_THAT(x_rt[i].real(), WithinAbs(x[i].real(), EPS));
        CHECK_THAT(x_rt[i].imag(), WithinAbs(x[i].imag(), EPS));
    }
}

// ── RFFT round-trip ───────────────────────────────────────────────────────────
TEST_CASE("irfft(rfft(x), n) == x for real input", "[fft]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::size_t N = x.size();

    auto X    = rfft(std::span<const Real>(x));
    auto x_rt = irfft(std::span<const Complex>(X), N);

    REQUIRE(x_rt.size() == N);
    for (std::size_t i = 0; i < N; ++i)
        CHECK_THAT(x_rt[i], WithinAbs(x[i], EPS));
}

// ── Known FFT values for a pure tone ─────────────────────────────────────────
// A cosine at bin k has energy only at bin k and N-k.
TEST_CASE("FFT of pure cosine has energy at correct bin", "[fft]")
{
    const std::size_t N    = 64;
    const std::size_t k0   = 4;    // target bin
    std::vector<Real> x(N);
    for (std::size_t n = 0; n < N; ++n)
        x[n] = std::cos(2.0 * std::numbers::pi * k0 * n / N);

    auto X   = rfft(std::span<const Real>(x));
    auto mag = magnitudes(X);

    // Peak should be at bin k0
    std::size_t peak_bin = std::distance(mag.begin(),
                               std::max_element(mag.begin(), mag.end()));
    CHECK(peak_bin == k0);

    // Magnitude at DC and other bins should be nearly zero
    CHECK_THAT(mag[0], WithinAbs(0.0, LOOSE));
}

// ── fftfreq ───────────────────────────────────────────────────────────────────
TEST_CASE("fftfreq returns correct bin centres", "[fft]")
{
    auto f = fftfreq(8, 1.0);
    REQUIRE(f.size() == 8);
    // Expected: [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
    CHECK_THAT(f[0], WithinAbs(0.0,     EPS));
    CHECK_THAT(f[1], WithinAbs(0.125,   EPS));
    CHECK_THAT(f[4], WithinAbs(-0.5,    EPS));
    CHECK_THAT(f[7], WithinAbs(-0.125,  EPS));
}

// ── rfftfreq ──────────────────────────────────────────────────────────────────
TEST_CASE("rfftfreq returns non-negative frequencies only", "[fft]")
{
    auto f = rfftfreq(8, 1.0 / 1000.0);   // sample spacing = 1/1000 → fs=1000 Hz
    REQUIRE(f.size() == 5);               // N/2+1 = 5
    CHECK_THAT(f[0], WithinAbs(0.0,   EPS));
    CHECK_THAT(f[4], WithinAbs(500.0, EPS));   // Nyquist = 500 Hz
}

// ── Window functions ──────────────────────────────────────────────────────────
TEST_CASE("Hann window is symmetric and peaks at centre", "[windows]")
{
    // Use odd N: the exact centre sample is 0.5*(1-cos(pi)) = 1.0 exactly.
    // Even N has no centre sample and the peak is ~0.9994 — not a useful check.
    auto w = hann_window(65);
    REQUIRE(w.size() == 65);

    // Endpoints should be (near) zero
    CHECK_THAT(w.front(), WithinAbs(0.0, EPS));
    CHECK_THAT(w.back(),  WithinAbs(0.0, EPS));

    // Should be symmetric
    for (std::size_t i = 0; i < w.size() / 2; ++i)
        CHECK_THAT(w[i], WithinAbs(w[w.size() - 1 - i], EPS));

    // Centre sample (index 32) should be exactly 1.0
    auto peak = std::max_element(w.begin(), w.end());
    CHECK_THAT(*peak, WithinAbs(1.0, EPS));
}

TEST_CASE("Kaiser window with beta=0 approximates rectangular", "[windows]")
{
    auto w = kaiser_window(64, 0.0);
    for (auto v : w)
        CHECK_THAT(v, WithinAbs(1.0, 1e-6));
}

TEST_CASE("All windows return correct length", "[windows]")
{
    constexpr std::size_t N = 128;
    CHECK(hann_window(N).size()             == N);
    CHECK(hamming_window(N).size()          == N);
    CHECK(blackman_window(N).size()         == N);
    CHECK(blackman_harris_window(N).size()  == N);
    CHECK(flat_top_window(N).size()         == N);
    CHECK(kaiser_window(N, 5.0).size()      == N);
    CHECK(tukey_window(N, 0.5).size()       == N);
}
