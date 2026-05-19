// test_windows.cpp — comprehensive window function tests
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

constexpr double EPS  = 1e-9;
constexpr double LOOSE = 1e-6;

// Helper: check a window is symmetric (w[i] == w[N-1-i])
static void check_symmetric(const std::vector<Real>& w)
{
    for (std::size_t i = 0; i < w.size() / 2; ++i)
        CHECK_THAT(w[i], WithinAbs(w[w.size() - 1 - i], EPS));
}

// Helper: check all window values are within [lo, hi]
static void check_range(const std::vector<Real>& w, double lo, double hi)
{
    for (auto v : w) {
        CHECK(v >= lo - LOOSE);
        CHECK(v <= hi + LOOSE);
    }
}

// ── Rectangular: all ones ─────────────────────────────────────────────────────
TEST_CASE("Rectangular window is all ones", "[windows]")
{
    for (std::size_t N : {1u, 8u, 64u, 128u}) {
        auto w = make_window(Window::Rectangular, N);
        REQUIRE(w.size() == N);
        for (auto v : w)
            CHECK_THAT(v, WithinAbs(1.0, EPS));
    }
}

// ── Hann: endpoints zero, peak 1 (odd N), symmetric ─────────────────────────
TEST_CASE("Hann window endpoints are zero", "[windows]")
{
    auto w = hann_window(64);
    CHECK_THAT(w.front(), WithinAbs(0.0, EPS));
    CHECK_THAT(w.back(),  WithinAbs(0.0, EPS));
}

TEST_CASE("Hann window peak == 1.0 for odd N", "[windows]")
{
    // For odd N, the exact centre sample = 0.5*(1 - cos(pi)) = 1.0
    auto w = hann_window(65);
    auto peak = *std::max_element(w.begin(), w.end());
    CHECK_THAT(peak, WithinAbs(1.0, EPS));
}

TEST_CASE("Hann window is symmetric and non-negative", "[windows]")
{
    auto w = hann_window(64);
    check_symmetric(w);
    check_range(w, 0.0, 1.0);
}

// ── Hamming: endpoints = 0.08, peak = 1 (odd N), symmetric ───────────────────
TEST_CASE("Hamming window endpoints are 0.08", "[windows]")
{
    // w[0] = 0.54 - 0.46*cos(0) = 0.54 - 0.46 = 0.08
    auto w = hamming_window(64);
    CHECK_THAT(w.front(), WithinAbs(0.08, EPS));
    CHECK_THAT(w.back(),  WithinAbs(0.08, EPS));
}

TEST_CASE("Hamming window peak == 1.0 for odd N", "[windows]")
{
    // For odd N, centre = 0.54 - 0.46*cos(pi) = 0.54 + 0.46 = 1.0
    auto w = hamming_window(65);
    auto peak = *std::max_element(w.begin(), w.end());
    CHECK_THAT(peak, WithinAbs(1.0, EPS));
}

TEST_CASE("Hamming window is symmetric and in [0.08, 1]", "[windows]")
{
    auto w = hamming_window(128);
    check_symmetric(w);
    check_range(w, 0.08, 1.0);
}

// ── Blackman: endpoints = 0, symmetric ───────────────────────────────────────
TEST_CASE("Blackman window endpoints are zero", "[windows]")
{
    // w[0] = 0.42 - 0.50 + 0.08 = 0.0
    auto w = blackman_window(64);
    CHECK_THAT(w.front(), WithinAbs(0.0, EPS));
    CHECK_THAT(w.back(),  WithinAbs(0.0, EPS));
}

TEST_CASE("Blackman window peak == 1.0 for odd N", "[windows]")
{
    auto w = blackman_window(65);
    auto peak = *std::max_element(w.begin(), w.end());
    CHECK_THAT(peak, WithinAbs(1.0, EPS));
}

TEST_CASE("Blackman window is symmetric and non-negative", "[windows]")
{
    auto w = blackman_window(128);
    check_symmetric(w);
    check_range(w, 0.0, 1.0);
}

// ── Blackman-Harris: symmetric, nearly zero endpoints ─────────────────────────
TEST_CASE("Blackman-Harris window is symmetric", "[windows]")
{
    auto w = blackman_harris_window(128);
    check_symmetric(w);
}

TEST_CASE("Blackman-Harris window endpoints are near zero", "[windows]")
{
    auto w = blackman_harris_window(64);
    // 0.35875 - 0.48829 + 0.14128 - 0.01168 ≈ 0.00006
    CHECK_THAT(w.front(), WithinAbs(0.0, 1e-4));
    CHECK_THAT(w.back(),  WithinAbs(0.0, 1e-4));
}

// ── Flat-top: symmetric, peak near 1, may have slightly negative values ───────
TEST_CASE("Flat-top window is symmetric", "[windows]")
{
    auto w = flat_top_window(128);
    check_symmetric(w);
}

TEST_CASE("Flat-top window peak is near 1.0", "[windows]")
{
    auto w = flat_top_window(65);
    auto peak = *std::max_element(w.begin(), w.end());
    CHECK_THAT(peak, WithinAbs(1.0, 1e-4));
}

// ── Kaiser: beta=0 ≈ rectangular, larger beta → narrower peak ─────────────────
TEST_CASE("Kaiser beta=0 is all ones (rectangular)", "[windows]")
{
    auto w = kaiser_window(64, 0.0);
    for (auto v : w)
        CHECK_THAT(v, WithinAbs(1.0, LOOSE));
}

TEST_CASE("Kaiser beta=8.6 has small edge values", "[windows]")
{
    // Use odd N so the exact centre sample falls on an integer index (peak = 1.0)
    auto w = kaiser_window(65, 8.6);
    // Edges should be well below 0.1 for large beta
    CHECK(w.front() < 0.1);
    CHECK(w.back()  < 0.1);
    // Centre (index 32, M=64) should be exactly 1.0
    CHECK_THAT(w[32], WithinAbs(1.0, LOOSE));
}

TEST_CASE("Kaiser is symmetric", "[windows]")
{
    auto w = kaiser_window(64, 5.0);
    check_symmetric(w);
    check_range(w, 0.0, 1.0);
}

TEST_CASE("Kaiser larger beta gives smaller edge values", "[windows]")
{
    auto w5  = kaiser_window(64, 5.0);
    auto w14 = kaiser_window(64, 14.0);
    // Edge value decreases as beta increases
    CHECK(w14.front() < w5.front());
}

// ── Tukey: taper region is zero at ends, flat centre is ones ─────────────────
TEST_CASE("Tukey window is symmetric", "[windows]")
{
    auto w = tukey_window(64, 0.5);
    check_symmetric(w);
    check_range(w, 0.0, 1.0);
}

TEST_CASE("Tukey alpha=1 has zero endpoints", "[windows]")
{
    auto w = tukey_window(65, 1.0);
    CHECK_THAT(w.front(), WithinAbs(0.0, EPS));
    CHECK_THAT(w.back(),  WithinAbs(0.0, EPS));
}

TEST_CASE("Tukey central samples are exactly 1.0 for small alpha", "[windows]")
{
    // alpha=0.25 means 12.5% taper on each side; centre 75% should be 1.0
    auto w = tukey_window(100, 0.25);
    // Samples 25..74 (the inner 50%) should all be 1.0
    for (std::size_t i = 25; i < 75; ++i)
        CHECK_THAT(w[i], WithinAbs(1.0, EPS));
}

// ── All windows: correct length for various N ─────────────────────────────────
TEST_CASE("make_window returns the requested length for every type", "[windows]")
{
    constexpr std::size_t N = 100;
    for (auto type : {Window::Rectangular, Window::Hann, Window::Hamming,
                      Window::Blackman, Window::BlackmanHarris, Window::FlatTop}) {
        auto w = make_window(type, N);
        CHECK(w.size() == N);
    }
    CHECK(kaiser_window(N, 5.0).size()  == N);
    CHECK(tukey_window(N, 0.5).size()   == N);
}

// ── N=1 edge case ─────────────────────────────────────────────────────────────
TEST_CASE("All windows handle N=1 gracefully", "[windows]")
{
    for (auto type : {Window::Rectangular, Window::Hann, Window::Hamming,
                      Window::Blackman}) {
        auto w = make_window(type, 1);
        REQUIRE(w.size() == 1);
        // Single-sample window must be 1.0 (pass-through)
        CHECK_THAT(w[0], WithinAbs(1.0, LOOSE));
    }
}

// ── Error paths ────────────────────────────────────────────────────────────────

TEST_CASE("make_window throws for n=0", "[windows][error]")
{
    CHECK_THROWS_AS(make_window(Window::Hann,        0), ValueError);
    CHECK_THROWS_AS(make_window(Window::Rectangular, 0), ValueError);
}

TEST_CASE("make_window Kaiser throws for negative beta", "[windows][error]")
{
    CHECK_THROWS_AS(make_window(Window::Kaiser, 16, -1.0), ValueError);
}

TEST_CASE("make_window throws for unknown window type", "[windows][error]")
{
    // Cast an out-of-range integer to Window to exercise the default throw branch
    auto unknown = static_cast<Window>(999);
    CHECK_THROWS_AS(make_window(unknown, 16), ValueError);
}
