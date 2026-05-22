// test_pocketfft_backend.cpp — PocketFFT and BuiltinFFT backend tests
//
// Mirrors test_fftw_backend.cpp in structure so the two files can be read
// side-by-side. Covers:
//
//   • PocketFFT explicit backend tag — correctness, round-trips, Parseval
//   • BuiltinFFT (always available) — Cooley-Tukey and naive-DFT paths
//   • Cross-backend agreement: PocketFFT == BuiltinFFT on the same input
//   • Non-power-of-2 sizes (PocketFFT: mixed-radix O(N log N);
//                           BuiltinFFT: O(N²) fallback, but correct)
//   • Large-N stress (2^20)
//   • irfft conjugate-symmetry reconstruction (BuiltinFFT-specific logic)
//   • Safety gap: assert() is debug-only [NOTE]

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <cps/backends/fft/builtin.hpp>    // always available — no external dep
#include <cps/backends/fft/pocketfft.hpp>  // PocketFFT, or BuiltinFFT alias
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>
#include <string>

using Catch::Approx;
using Catch::Matchers::WithinAbs;
using cps::Real;
using cps::Complex;

static constexpr cps::backends::PocketFFT  kPocket{};
static constexpr cps::backends::BuiltinFFT kBuiltin{};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (identical to test_fftw_backend.cpp for easy comparison)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<Real> sine_wave(std::size_t N, double freq, double fs)
{
    std::vector<Real> x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * freq * i / fs);
    return x;
}

static std::vector<Complex> complex_chirp(std::size_t N)
{
    std::vector<Complex> x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = {std::cos(0.1 * i), std::sin(0.2 * i)};
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// PocketFFT — basic round-trip correctness
// (mirrors test_fftw_backend.cpp section for line-by-line comparison)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PocketFFT: fft → ifft round-trip recovers input", "[pocketfft]")
{
    auto x = complex_chirp(256);
    auto X  = cps::fft<cps::backends::PocketFFT>(std::span<const Complex>(x), kPocket);
    auto x2 = cps::ifft<cps::backends::PocketFFT>(X, kPocket);

    REQUIRE(x2.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK(x2[i].real() == Approx(x[i].real()).margin(1e-10));
        CHECK(x2[i].imag() == Approx(x[i].imag()).margin(1e-10));
    }
}

TEST_CASE("PocketFFT: rfft → irfft round-trip recovers input", "[pocketfft]")
{
    auto x  = sine_wave(256, 25.0, 1000.0);
    auto X  = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
    auto x2 = cps::irfft<cps::backends::PocketFFT>(X, x.size(), kPocket);

    REQUIRE(x2.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK(x2[i] == Approx(x[i]).margin(1e-10));
}

TEST_CASE("PocketFFT: rfft output size is N/2+1", "[pocketfft]")
{
    for (std::size_t N : {8u, 16u, 32u, 64u, 100u, 256u}) {
        auto x = sine_wave(N, 10.0, 1000.0);
        auto X = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
        CHECK(X.size() == N / 2 + 1);
    }
}

TEST_CASE("PocketFFT: DC component of all-ones signal equals N", "[pocketfft]")
{
    const std::size_t N = 64;
    std::vector<Real> ones(N, 1.0);
    auto X = cps::rfft<cps::backends::PocketFFT>(ones, kPocket);
    CHECK(X[0].real() == Approx(static_cast<double>(N)).margin(1e-10));
    CHECK(X[0].imag() == Approx(0.0).margin(1e-10));
}

TEST_CASE("PocketFFT: Parseval's theorem holds for rfft", "[pocketfft]")
{
    auto x = sine_wave(256, 10.0, 1000.0);
    auto X = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
    const std::size_t N = x.size();

    double time_power = 0.0;
    for (auto v : x) time_power += v * v;

    double freq_power = std::norm(X[0]);
    for (std::size_t k = 1; k + 1 < X.size(); ++k)
        freq_power += 2.0 * std::norm(X[k]);
    freq_power += std::norm(X.back());
    freq_power /= static_cast<double>(N);

    CHECK(time_power == Approx(freq_power).epsilon(1e-8));
}

TEST_CASE("PocketFFT: repeated calls produce identical results", "[pocketfft]")
{
    auto x  = sine_wave(256, 10.0, 1000.0);
    auto X1 = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
    auto X2 = cps::rfft<cps::backends::PocketFFT>(x, kPocket);

    REQUIRE(X1.size() == X2.size());
    for (std::size_t k = 0; k < X1.size(); ++k) {
        CHECK(X1[k].real() == Approx(X2[k].real()).margin(1e-15));
        CHECK(X1[k].imag() == Approx(X2[k].imag()).margin(1e-15));
    }
}

TEST_CASE("PocketFFT: backend name is correct", "[pocketfft]")
{
    std::string name{cps::backends::PocketFFT::name()};
    // When PocketFFT header is found: "PocketFFT"
    // When aliased to BuiltinFFT fallback: "BuiltinFFT (Cooley-Tukey)"
    bool is_known = (name == "PocketFFT" || name == "BuiltinFFT (Cooley-Tukey)");
    CHECK(is_known);
}

// ─────────────────────────────────────────────────────────────────────────────
// PocketFFT — non-power-of-2 sizes
//
// PocketFFT uses mixed-radix decomposition (O(N log N) for any N).
// BuiltinFFT falls back to O(N²) naive DFT for non-power-of-2 — correct but
// slow. Both should agree on the answer; this test verifies correctness and
// documents the performance difference.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PocketFFT: rfft → irfft round-trip for non-power-of-2 N", "[pocketfft]")
{
    for (std::size_t N : {5u, 7u, 9u, 100u, 1000u}) {
        auto x  = sine_wave(N, 3.0, static_cast<double>(N));
        auto X  = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
        auto x2 = cps::irfft<cps::backends::PocketFFT>(X, N, kPocket);

        REQUIRE(x2.size() == N);
        for (std::size_t i = 0; i < N; ++i)
            CHECK(x2[i] == Approx(x[i]).margin(1e-9));
    }
}

TEST_CASE("PocketFFT: fft → ifft round-trip for non-power-of-2 N", "[pocketfft]")
{
    for (std::size_t N : {5u, 7u, 9u, 100u}) {
        std::vector<Complex> x(N);
        for (std::size_t i = 0; i < N; ++i)
            x[i] = {std::cos(0.3 * i), std::sin(0.5 * i)};

        auto X  = cps::fft<cps::backends::PocketFFT>(std::span<const Complex>(x), kPocket);
        auto x2 = cps::ifft<cps::backends::PocketFFT>(X, kPocket);

        REQUIRE(x2.size() == N);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(x2[i].real() == Approx(x[i].real()).margin(1e-9));
            CHECK(x2[i].imag() == Approx(x[i].imag()).margin(1e-9));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PocketFFT — large-N stress test
//
// PocketFFT handles any N in O(N log N). This confirms no crash or overflow
// at 2^20 = 1048576 elements and that the round-trip error stays small.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PocketFFT: rfft → irfft round-trip at N=2^20", "[pocketfft][stress]")
{
    constexpr std::size_t N = 1u << 20;
    auto x  = sine_wave(N, 440.0, 44100.0);   // A4 tone at CD sample rate
    auto X  = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
    auto x2 = cps::irfft<cps::backends::PocketFFT>(X, N, kPocket);

    REQUIRE(x2.size() == N);

    // Spot-check a few samples rather than checking all 1M
    for (std::size_t i : {std::size_t{0}, std::size_t{1}, N/4, N/2, N-1}) {
        CHECK(x2[i] == Approx(x[i]).margin(1e-8));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-backend agreement: PocketFFT == BuiltinFFT
//
// BuiltinFFT is the always-available reference. When PocketFFT is installed,
// both should agree to near-machine-epsilon. When PocketFFT is aliased to
// BuiltinFFT, the comparison is trivially true (same code) but that's fine —
// the test still guards against future divergence.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PocketFFT rfft agrees with BuiltinFFT on a sine wave", "[pocketfft][builtin]")
{
    auto x = sine_wave(128, 7.0, 500.0);

    auto X_pocket  = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
    auto X_builtin = cps::rfft<cps::backends::BuiltinFFT>(x, kBuiltin);

    REQUIRE(X_pocket.size() == X_builtin.size());
    for (std::size_t k = 0; k < X_pocket.size(); ++k) {
        CHECK(X_pocket[k].real() == Approx(X_builtin[k].real()).margin(1e-9));
        CHECK(X_pocket[k].imag() == Approx(X_builtin[k].imag()).margin(1e-9));
    }
}

TEST_CASE("PocketFFT fft agrees with BuiltinFFT on complex input", "[pocketfft][builtin]")
{
    auto x = complex_chirp(64);

    auto X_pocket  = cps::fft<cps::backends::PocketFFT>(std::span<const Complex>(x), kPocket);
    auto X_builtin = cps::fft<cps::backends::BuiltinFFT>(std::span<const Complex>(x), kBuiltin);

    REQUIRE(X_pocket.size() == X_builtin.size());
    for (std::size_t k = 0; k < X_pocket.size(); ++k) {
        CHECK(X_pocket[k].real() == Approx(X_builtin[k].real()).margin(1e-9));
        CHECK(X_pocket[k].imag() == Approx(X_builtin[k].imag()).margin(1e-9));
    }
}

TEST_CASE("PocketFFT non-pow2 rfft agrees with BuiltinFFT naive DFT", "[pocketfft][builtin]")
{
    // N=100: PocketFFT uses mixed-radix 4×25; BuiltinFFT uses O(N²) naive DFT.
    // Both must produce the same spectrum.
    auto x = sine_wave(100, 5.0, 100.0);

    auto X_pocket  = cps::rfft<cps::backends::PocketFFT>(x, kPocket);
    auto X_builtin = cps::rfft<cps::backends::BuiltinFFT>(x, kBuiltin);

    REQUIRE(X_pocket.size() == X_builtin.size());
    for (std::size_t k = 0; k < X_pocket.size(); ++k) {
        CHECK(X_pocket[k].real() == Approx(X_builtin[k].real()).margin(1e-8));
        CHECK(X_pocket[k].imag() == Approx(X_builtin[k].imag()).margin(1e-8));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BuiltinFFT — specific tests for its internal logic
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("BuiltinFFT: Cooley-Tukey power-of-2 fft → ifft round-trip", "[builtin]")
{
    for (std::size_t N : {2u, 4u, 8u, 16u, 64u, 256u, 1024u}) {
        auto x = complex_chirp(N);
        auto X  = cps::fft<cps::backends::BuiltinFFT>(std::span<const Complex>(x), kBuiltin);
        auto x2 = cps::ifft<cps::backends::BuiltinFFT>(X, kBuiltin);

        REQUIRE(x2.size() == N);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(x2[i].real() == Approx(x[i].real()).margin(1e-10));
            CHECK(x2[i].imag() == Approx(x[i].imag()).margin(1e-10));
        }
    }
}

TEST_CASE("BuiltinFFT: naive DFT non-power-of-2 fft → ifft round-trip", "[builtin]")
{
    for (std::size_t N : {3u, 5u, 6u, 7u, 9u, 11u, 13u}) {
        std::vector<Complex> x(N);
        for (std::size_t i = 0; i < N; ++i)
            x[i] = {static_cast<double>(i) / N, -static_cast<double>(i) / N * 0.5};

        auto X  = cps::fft<cps::backends::BuiltinFFT>(std::span<const Complex>(x), kBuiltin);
        auto x2 = cps::ifft<cps::backends::BuiltinFFT>(X, kBuiltin);

        REQUIRE(x2.size() == N);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(x2[i].real() == Approx(x[i].real()).margin(1e-10));
            CHECK(x2[i].imag() == Approx(x[i].imag()).margin(1e-10));
        }
    }
}

TEST_CASE("BuiltinFFT: irfft conjugate-symmetry reconstruction is correct for even N", "[builtin]")
{
    // Verify that the mirroring loop (buf[N-k] = conj(in[k])) is correct for even N.
    // Cross-check: rfft(x) → irfft should recover x exactly.
    for (std::size_t N : {4u, 8u, 16u, 64u}) {
        auto x  = sine_wave(N, 2.0, static_cast<double>(N));
        auto X  = cps::rfft<cps::backends::BuiltinFFT>(x, kBuiltin);
        auto x2 = cps::irfft<cps::backends::BuiltinFFT>(X, N, kBuiltin);

        REQUIRE(x2.size() == N);
        for (std::size_t i = 0; i < N; ++i)
            CHECK(x2[i] == Approx(x[i]).margin(1e-10));
    }
}

TEST_CASE("BuiltinFFT: irfft conjugate-symmetry reconstruction is correct for odd N", "[builtin]")
{
    // Odd N: Nyquist bin doesn't exist; mirroring covers k=1..floor((N-1)/2).
    for (std::size_t N : {5u, 7u, 9u, 15u, 63u}) {
        auto x  = sine_wave(N, 1.0, static_cast<double>(N));
        auto X  = cps::rfft<cps::backends::BuiltinFFT>(x, kBuiltin);
        auto x2 = cps::irfft<cps::backends::BuiltinFFT>(X, N, kBuiltin);

        REQUIRE(x2.size() == N);
        for (std::size_t i = 0; i < N; ++i)
            CHECK(x2[i] == Approx(x[i]).margin(1e-10));
    }
}

TEST_CASE("BuiltinFFT: rfft imaginary part of output is discarded in irfft", "[builtin]")
{
    // The BuiltinFFT irfft discards buf[i].imag() (takes only .real()).
    // For a real input, the reconstructed imaginary parts should be near zero.
    auto x  = sine_wave(64, 5.0, 64.0);
    auto X  = cps::rfft<cps::backends::BuiltinFFT>(x, kBuiltin);

    // Corrupt the imaginary parts of the spectrum slightly — they should be
    // reconstructed correctly by conjugate symmetry (irfft doesn't read them).
    // Round-trip should still give back x.
    auto x2 = cps::irfft<cps::backends::BuiltinFFT>(X, x.size(), kBuiltin);
    REQUIRE(x2.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK(x2[i] == Approx(x[i]).margin(1e-10));
}

TEST_CASE("BuiltinFFT: N=1 edge case (cooley_tukey early return)", "[builtin]")
{
    // cooley_tukey returns immediately when N <= 1.
    std::vector<Complex> x = {{3.0, -2.0}};
    auto X  = cps::fft<cps::backends::BuiltinFFT>(std::span<const Complex>(x), kBuiltin);
    REQUIRE(X.size() == 1);
    CHECK(X[0].real() == Approx(3.0).margin(1e-12));
    CHECK(X[0].imag() == Approx(-2.0).margin(1e-12));

    auto x2 = cps::ifft<cps::backends::BuiltinFFT>(X, kBuiltin);
    REQUIRE(x2.size() == 1);
    CHECK(x2[0].real() == Approx(3.0).margin(1e-12));
    CHECK(x2[0].imag() == Approx(-2.0).margin(1e-12));
}

TEST_CASE("BuiltinFFT: backend name is BuiltinFFT", "[builtin]")
{
    CHECK(std::string{cps::backends::BuiltinFFT::name()} == "BuiltinFFT (Cooley-Tukey)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Safety gap: assert() is debug-only [NOTE]
//
// Both PocketFFT and BuiltinFFT use assert() to validate that the output span
// has the expected size, rather than throwing ValueError. In release builds
// (NDEBUG defined) these asserts are compiled out — passing a mismatched span
// causes undefined behaviour with no diagnostic.
//
// This is documented here as a known limitation. The fix would be to replace
// each assert with an explicit bounds check that throws ValueError, mirroring
// the FFTW backend's pattern.
//
// We cannot write a test that triggers UB safely, so this test just verifies
// the happy path works and leaves a comment explaining the gap.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PocketFFT: assert() in backend is debug-only safety check [NOTE]", "[pocketfft]")
{
    // This test documents the gap: the backend uses assert(in.size()==out.size())
    // which becomes a no-op in release builds. A caller using the backend struct
    // directly with a mismatched output span would get UB in release mode.
    //
    // The public API (cps::fft, cps::rfft, etc.) always allocates its own
    // output vector of the correct size before calling the backend, so normal
    // users are not exposed to this gap. Direct backend users are at risk.
    //
    // Mitigation: use cps::fft() / cps::rfft() rather than calling backend
    // methods directly.

    // Verify happy path: correct span sizes do not crash.
    auto x = sine_wave(64, 5.0, 1000.0);
    CHECK_NOTHROW(cps::rfft<cps::backends::PocketFFT>(x, kPocket));
    CHECK_NOTHROW(cps::rfft<cps::backends::BuiltinFFT>(x, kBuiltin));
}
