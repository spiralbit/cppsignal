// test_fftw_backend.cpp — FFTW backend correctness + safety tests
//
// Compiled only when CPS_ENABLE_FFTW=ON (CPS_HAS_FFTW is defined).
// Covers:
//   • Basic FFT/IFFT/RFFT/IRFFT round-trips via the FFTW backend
//   • Equivalence with the default PocketFFT backend
//   • Bug 5: static_cast<int>(n) truncation for n > INT_MAX  [BUG]
//   • Bug 6: fftw_malloc / planner null-pointer not checked   [BUG]
//
// Bug 4 (thread-local PlanCache destructor locking a destroyed mutex at
// program shutdown) cannot be expressed as a self-contained unit test.

#ifndef CPS_HAS_FFTW
// If compiled without FFTW, skip everything gracefully.
int main() { return 0; }
#else

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cps/cps.hpp>
#include <cps/backends/fft/fftw.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>
#include <numbers>

using Catch::Approx;
using cps::Real;
using cps::Complex;

static constexpr cps::backends::FFTW kFFFTW{};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<Real> sine_wave(std::size_t N, double freq, double fs)
{
    std::vector<Real> x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * freq * i / fs);
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic round-trip correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FFTW: fft → ifft round-trip recovers input", "[fftw]")
{
    auto x_real = sine_wave(256, 10.0, 1000.0);
    std::vector<Complex> x(x_real.begin(), x_real.end());

    auto X   = cps::fft<cps::backends::FFTW>(std::span<const Complex>(x), kFFFTW);
    auto x2  = cps::ifft<cps::backends::FFTW>(X, kFFFTW);

    REQUIRE(x2.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK(x2[i].real() == Approx(x[i].real()).margin(1e-10));
        CHECK(x2[i].imag() == Approx(x[i].imag()).margin(1e-10));
    }
}

TEST_CASE("FFTW: rfft → irfft round-trip recovers input", "[fftw]")
{
    auto x = sine_wave(256, 25.0, 1000.0);

    auto X  = cps::rfft<cps::backends::FFTW>(x, kFFFTW);
    auto x2 = cps::irfft<cps::backends::FFTW>(X, x.size(), kFFFTW);

    REQUIRE(x2.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK(x2[i] == Approx(x[i]).margin(1e-10));
}

TEST_CASE("FFTW: rfft output size is N/2+1", "[fftw]")
{
    for (std::size_t N : {8u, 16u, 32u, 64u, 100u, 256u}) {
        auto x = sine_wave(N, 10.0, 1000.0);
        auto X = cps::rfft<cps::backends::FFTW>(x, kFFFTW);
        CHECK(X.size() == N / 2 + 1);
    }
}

TEST_CASE("FFTW: DC component of all-ones signal equals N", "[fftw]")
{
    const std::size_t N = 64;
    std::vector<Real> ones(N, 1.0);
    auto X = cps::rfft<cps::backends::FFTW>(ones, kFFFTW);
    CHECK(X[0].real() == Approx(static_cast<double>(N)).margin(1e-10));
    CHECK(X[0].imag() == Approx(0.0).margin(1e-10));
}

TEST_CASE("FFTW: Parseval's theorem holds for rfft", "[fftw]")
{
    auto x = sine_wave(256, 10.0, 1000.0);
    auto X = cps::rfft<cps::backends::FFTW>(x, kFFFTW);
    const std::size_t N = x.size();

    double time_power = 0.0;
    for (auto v : x) time_power += v * v;

    // For one-sided rfft: sum = |X[0]|² + 2*sum(|X[1..N/2-1]|²) + |X[N/2]|²
    double freq_power = std::norm(X[0]);
    for (std::size_t k = 1; k + 1 < X.size(); ++k)
        freq_power += 2.0 * std::norm(X[k]);
    freq_power += std::norm(X.back());
    freq_power /= static_cast<double>(N);

    CHECK(time_power == Approx(freq_power).epsilon(1e-8));
}

// ─────────────────────────────────────────────────────────────────────────────
// Agreement with PocketFFT (default backend)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FFTW: rfft agrees with PocketFFT on a sine wave", "[fftw]")
{
    auto x = sine_wave(128, 7.0, 500.0);

    auto X_pocket = cps::rfft(x);
    auto X_fftw   = cps::rfft<cps::backends::FFTW>(x, kFFFTW);

    REQUIRE(X_pocket.size() == X_fftw.size());
    for (std::size_t k = 0; k < X_pocket.size(); ++k) {
        CHECK(X_fftw[k].real() == Approx(X_pocket[k].real()).margin(1e-9));
        CHECK(X_fftw[k].imag() == Approx(X_pocket[k].imag()).margin(1e-9));
    }
}

TEST_CASE("FFTW: fft agrees with PocketFFT on complex input", "[fftw]")
{
    std::vector<Complex> x(64);
    for (std::size_t i = 0; i < 64; ++i)
        x[i] = {std::cos(0.1 * i), std::sin(0.2 * i)};

    auto X_pocket = cps::fft(std::span<const Complex>(x));
    auto X_fftw   = cps::fft<cps::backends::FFTW>(std::span<const Complex>(x), kFFFTW);

    REQUIRE(X_pocket.size() == X_fftw.size());
    for (std::size_t k = 0; k < X_pocket.size(); ++k) {
        CHECK(X_fftw[k].real() == Approx(X_pocket[k].real()).margin(1e-9));
        CHECK(X_fftw[k].imag() == Approx(X_pocket[k].imag()).margin(1e-9));
    }
}

TEST_CASE("FFTW: plan is reused on repeated calls (no crash, consistent output)", "[fftw]")
{
    auto x = sine_wave(256, 10.0, 1000.0);
    auto X1 = cps::rfft<cps::backends::FFTW>(x, kFFFTW);
    auto X2 = cps::rfft<cps::backends::FFTW>(x, kFFFTW);

    REQUIRE(X1.size() == X2.size());
    for (std::size_t k = 0; k < X1.size(); ++k) {
        CHECK(X1[k].real() == Approx(X2[k].real()).margin(1e-15));
        CHECK(X1[k].imag() == Approx(X2[k].imag()).margin(1e-15));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug 5: static_cast<int>(n) overflow for n > INT_MAX [BUG]
//
// FFTW planners accept int n. For n > INT_MAX the cast wraps to a negative or
// wrong value; FFTW then fails or produces undefined results.
//
// Fix: check n <= INT_MAX before calling the planner and throw ValueError.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 5 (FFTW): n > INT_MAX should throw ValueError [BUG]")
{
    // We cannot actually allocate a 2^31-element buffer, but the size guard
    // must fire before any allocation is attempted.
    constexpr std::size_t kTooBig =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;

    std::vector<Real> dummy(1, 0.0);   // small signal; size check fires before alloc
    // We can't call rfft with a span of 1 element and expect the size guard to
    // fire — the guard must be inside the backend before get_plan.
    // Instead, create a span that reports the huge size without allocating it.
    // Use a pointer + size constructor; dereferencing won't happen before throw.
    const Real* ptr = dummy.data();
    std::span<const Real> big_span(ptr, kTooBig);

    CHECK_THROWS_AS(cps::rfft<cps::backends::FFTW>(big_span, kFFFTW), cps::ValueError);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug 6: fftw_malloc / planner return values not checked [BUG]
//
// If fftw_malloc returns NULL (OOM) or the planner returns NULL (planning
// failure), the current code passes NULL to fftw_execute_dft → crash.
//
// This cannot be triggered deterministically in a unit test without mocking
// fftw_malloc. Documented here as an untestable defect; the fix is to check
// the return values and throw.
//
// Fix: in each planning lambda, check fftw_malloc != NULL and planner != NULL;
// throw ValueError on failure.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 6 (FFTW): planner null-check — documented as untestable [NOTE]")
{
    // Cannot inject OOM in a unit test. This test simply verifies that normal
    // operation (no OOM) continues to work after the defensive checks are added.
    auto x = sine_wave(64, 5.0, 500.0);
    CHECK_NOTHROW(cps::rfft<cps::backends::FFTW>(x, kFFFTW));
}

#endif // CPS_HAS_FFTW
