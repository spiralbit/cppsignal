// ─────────────────────────────────────────────────────────────────────────────
// test_validation.cpp — parameter validation gaps across the library
//
// Three categories of test here:
//
//   [BUG]  — These tests FAIL against the current implementation.  They
//             document a gap where NaN (or another out-of-domain value) slips
//             through a guard that uses a comparison operator.  IEEE 754 NaN
//             comparisons always return false, so   (NaN <= 0) || (NaN >= 1)
//             is false, letting NaN reach arithmetic that silently produces
//             garbage or undefined behaviour.
//
//   [OK]   — The validation already works correctly.  These tests PASS and act
//             as regression guards so a future refactor cannot break them.
//
//   [NOTE] — The behaviour is intentional / acceptable but worth documenting.
//
// Files covered:
//   filter/design.hpp  — butter(), firwin()  (via normalise_wn)
//   signal/generate.hpp — arange(), chirp(), gausspulse(), square_wave(),
//                         sawtooth_wave()
//   measure/metrics.hpp — thd(), snr(spectral), sinad()
//   spectral/fft.hpp    — irfft()   (regression for n=0 fix)
//   spectral/stft.hpp   — stft()    (regression for nfft<nperseg fix)
// ─────────────────────────────────────────────────────────────────────────────

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

using Catch::Matchers::WithinAbs;

static const double kNaN = std::numeric_limits<double>::quiet_NaN();
static const double kInf = std::numeric_limits<double>::infinity();

// ═════════════════════════════════════════════════════════════════════════════
// filter/design.hpp — butter()
//
// normalise_wn guards:  (Wn_norm <= 0.0 || Wn_norm >= 1.0)
// NaN fails both comparisons → passes the guard → propagates to NaN SOS.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter: NaN Wn should throw ValueError [BUG]") {
    // NaN <= 0.0 = false, NaN >= 1.0 = false → guard bypassed
    CHECK_THROWS_AS(cps::butter(2, kNaN, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: NaN fs should throw ValueError [BUG]") {
    // Wn_norm = 2*Wn/NaN = NaN; same bypass
    CHECK_THROWS_AS(
        cps::butter(2, 100.0, cps::FilterType::Lowpass, {.fs = kNaN}),
        cps::ValueError);
}

TEST_CASE("butter: +Inf Wn throws ValueError [OK]") {
    // Inf >= 1.0 = true → guard fires correctly
    CHECK_THROWS_AS(cps::butter(2, kInf, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: -Inf Wn throws ValueError [OK]") {
    // -Inf <= 0.0 = true → guard fires correctly
    CHECK_THROWS_AS(cps::butter(2, -kInf, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: +Inf fs throws ValueError [OK]") {
    // Wn_norm = 2*100/Inf = 0.0 <= 0.0 → throws
    CHECK_THROWS_AS(
        cps::butter(2, 100.0, cps::FilterType::Lowpass, {.fs = kInf}),
        cps::ValueError);
}

TEST_CASE("butter: negative fs throws ValueError [OK]") {
    CHECK_THROWS_AS(
        cps::butter(2, 100.0, cps::FilterType::Lowpass, {.fs = -1000.0}),
        cps::ValueError);
}

TEST_CASE("butter: Wn=0 throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::butter(2, 0.0, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: Wn=1 throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::butter(2, 1.0, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: Wn exactly at Nyquist (fs/2) throws ValueError [OK]") {
    // 2*500/1000 = 1.0 → >= 1.0 → throws
    CHECK_THROWS_AS(
        cps::butter(2, 500.0, cps::FilterType::Lowpass, {.fs = 1000.0}),
        cps::ValueError);
}

TEST_CASE("butter: order <= 0 throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::butter(0,  0.3, cps::FilterType::Lowpass), cps::ValueError);
    CHECK_THROWS_AS(cps::butter(-1, 0.3, cps::FilterType::Lowpass), cps::ValueError);
}

// ═════════════════════════════════════════════════════════════════════════════
// filter/design.hpp — firwin()
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("firwin: NaN cutoff should throw ValueError [BUG]") {
    CHECK_THROWS_AS(cps::firwin(11, kNaN), cps::ValueError);
}

TEST_CASE("firwin: NaN fs should throw ValueError [BUG]") {
    CHECK_THROWS_AS(cps::firwin(11, 100.0, cps::Window::Hamming,
                                cps::FilterType::Lowpass, {.fs = kNaN}),
                    cps::ValueError);
}

TEST_CASE("firwin: +Inf cutoff throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::firwin(11, kInf), cps::ValueError);
}

TEST_CASE("firwin: cutoff=0 throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::firwin(11, 0.0), cps::ValueError);
}

TEST_CASE("firwin: +Inf fs throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::firwin(11, 100.0, cps::Window::Hamming,
                                cps::FilterType::Lowpass, {.fs = kInf}),
                    cps::ValueError);
}

TEST_CASE("firwin: negative fs throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::firwin(11, 100.0, cps::Window::Hamming,
                                cps::FilterType::Lowpass, {.fs = -1000.0}),
                    cps::ValueError);
}

TEST_CASE("firwin: numtaps <= 0 throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::firwin(0,  0.3), cps::ValueError);
    CHECK_THROWS_AS(cps::firwin(-1, 0.3), cps::ValueError);
}

TEST_CASE("firwin: even numtaps + Highpass throws ValueError [OK]") {
    CHECK_THROWS_AS(
        cps::firwin(10, 0.3, cps::Window::Hamming, cps::FilterType::Highpass),
        cps::ValueError);
}

// ═════════════════════════════════════════════════════════════════════════════
// signal/generate.hpp — arange()
//
// Guard:  step == 0.0
// NaN == 0.0 = false → guard bypassed → static_cast<size_t>(NaN) = UB.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("arange: NaN step should throw ValueError [BUG]") {
    CHECK_THROWS_AS(cps::arange(0.0, 10.0, kNaN), cps::ValueError);
}

TEST_CASE("arange: step=0 throws ValueError [OK]") {
    CHECK_THROWS_AS(cps::arange(0.0, 10.0, 0.0), cps::ValueError);
}

TEST_CASE("arange: +Inf step returns empty vector [NOTE]") {
    // (stop-start)/Inf = 0 → n=0 → empty, no crash
    auto r = cps::arange(0.0, 10.0, kInf);
    CHECK(r.empty());
}

TEST_CASE("arange: negative step sweeps downward [OK]") {
    auto r = cps::arange(5.0, 0.0, -1.0);
    REQUIRE(r.size() == 5);
    CHECK_THAT(r[0], WithinAbs(5.0, 1e-12));
    CHECK_THAT(r[4], WithinAbs(1.0, 1e-12));
}

// ═════════════════════════════════════════════════════════════════════════════
// signal/generate.hpp — chirp()
//
// Guard:  t1 <= 0.0
// NaN <= 0.0 = false → guard bypassed → k = (f1-f0)/NaN = NaN → NaN output.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("chirp: NaN t1 should throw ValueError [BUG]") {
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    CHECK_THROWS_AS(cps::chirp(t, 10.0, 100.0, kNaN), cps::ValueError);
}

TEST_CASE("chirp: t1=0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    CHECK_THROWS_AS(cps::chirp(t, 10.0, 100.0, 0.0), cps::ValueError);
}

TEST_CASE("chirp: negative t1 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    CHECK_THROWS_AS(cps::chirp(t, 10.0, 100.0, -1.0), cps::ValueError);
}

TEST_CASE("chirp: -Inf t1 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    CHECK_THROWS_AS(cps::chirp(t, 10.0, 100.0, -kInf), cps::ValueError);
}

TEST_CASE("chirp: NaN f0 propagates to NaN output [NOTE]") {
    // f0 is not range-checked — NaN propagates through to output
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    auto y = cps::chirp(t, kNaN, 100.0, 1.0);
    CHECK(std::isnan(y[1]));
}

TEST_CASE("chirp: NaN f1 propagates to NaN output [NOTE]") {
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    auto y = cps::chirp(t, 10.0, kNaN, 1.0);
    CHECK(std::isnan(y[1]));
}

// ═════════════════════════════════════════════════════════════════════════════
// signal/generate.hpp — gausspulse()
//
// Guards:  fc <= 0.0,  bw <= 0.0,  bw_db >= 0.0
// All three use comparison operators that return false for NaN.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("gausspulse: NaN fc should throw ValueError [BUG]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, kNaN), cps::ValueError);
}

TEST_CASE("gausspulse: NaN bw should throw ValueError [BUG]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 100.0, kNaN), cps::ValueError);
}

TEST_CASE("gausspulse: NaN bw_db should throw ValueError [BUG]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 100.0, 0.5, kNaN), cps::ValueError);
}

TEST_CASE("gausspulse: fc=0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 0.0), cps::ValueError);
}

TEST_CASE("gausspulse: negative fc throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, -100.0), cps::ValueError);
}

TEST_CASE("gausspulse: bw=0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 100.0, 0.0), cps::ValueError);
}

TEST_CASE("gausspulse: negative bw throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 100.0, -0.5), cps::ValueError);
}

TEST_CASE("gausspulse: bw_db=0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 100.0, 0.5, 0.0), cps::ValueError);
}

TEST_CASE("gausspulse: positive bw_db throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.0};
    CHECK_THROWS_AS(cps::gausspulse(t, 100.0, 0.5, 3.0), cps::ValueError);
}

// ═════════════════════════════════════════════════════════════════════════════
// signal/generate.hpp — square_wave()
//
// Guard:  duty <= 0.0 || duty >= 1.0
// NaN <= 0.0 = false, NaN >= 1.0 = false → both false → guard bypassed.
// With NaN duty, every sample goes to the `else` branch and returns -1.0
// (phase < NaN is always false).
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("square_wave: NaN duty should throw ValueError [BUG]") {
    std::vector<cps::Real> t = {0.0, 0.25, 0.5, 0.75};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, kNaN), cps::ValueError);
}

TEST_CASE("square_wave: duty=0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, 0.0), cps::ValueError);
}

TEST_CASE("square_wave: duty=1 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, 1.0), cps::ValueError);
}

TEST_CASE("square_wave: duty>1 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, 1.5), cps::ValueError);
}

TEST_CASE("square_wave: duty<0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, -0.1), cps::ValueError);
}

TEST_CASE("square_wave: +Inf duty throws ValueError [OK]") {
    // Inf >= 1.0 = true → guard fires
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, kInf), cps::ValueError);
}

TEST_CASE("square_wave: -Inf duty throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::square_wave(t, 1.0, -kInf), cps::ValueError);
}

TEST_CASE("square_wave: NaN freq propagates to output [NOTE]") {
    // freq is not range-checked; fmod(t*NaN, 1) = NaN → always returns -1.0
    std::vector<cps::Real> t = {0.1, 0.4, 0.6};
    REQUIRE_NOTHROW(cps::square_wave(t, kNaN, 0.5));
}

// ═════════════════════════════════════════════════════════════════════════════
// signal/generate.hpp — sawtooth_wave()
//
// Guard:  width < 0.0 || width > 1.0
// NaN < 0.0 = false, NaN > 1.0 = false → guard bypassed.
// With NaN width the then-branch (phase < NaN) is always false so the else
// branch runs: -2*(phase-NaN)/(1-NaN)+1 = NaN for every sample.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("sawtooth_wave: NaN width should throw ValueError [BUG]") {
    std::vector<cps::Real> t = {0.0, 0.5, 1.0};
    CHECK_THROWS_AS(cps::sawtooth_wave(t, 1.0, kNaN), cps::ValueError);
}

TEST_CASE("sawtooth_wave: width<0 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::sawtooth_wave(t, 1.0, -0.1), cps::ValueError);
}

TEST_CASE("sawtooth_wave: width>1 throws ValueError [OK]") {
    std::vector<cps::Real> t = {0.5};
    CHECK_THROWS_AS(cps::sawtooth_wave(t, 1.0, 1.1), cps::ValueError);
}

TEST_CASE("sawtooth_wave: width=0 is valid (pure falling ramp) [OK]") {
    std::vector<cps::Real> t = {0.0, 0.25, 0.5, 0.75};
    REQUIRE_NOTHROW(cps::sawtooth_wave(t, 1.0, 0.0));
}

TEST_CASE("sawtooth_wave: width=1 is valid (pure rising ramp) [OK]") {
    std::vector<cps::Real> t = {0.0, 0.25, 0.5, 0.75};
    REQUIRE_NOTHROW(cps::sawtooth_wave(t, 1.0, 1.0));
}

TEST_CASE("sawtooth_wave: NaN freq propagates to NaN output [NOTE]") {
    std::vector<cps::Real> t = {0.5};
    REQUIRE_NOTHROW(cps::sawtooth_wave(t, kNaN, 0.5));
}

// ═════════════════════════════════════════════════════════════════════════════
// measure/metrics.hpp — thd()
//
// Guard:  fundamental <= 0.0
// NaN <= 0.0 = false → guard bypassed.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("thd: NaN fundamental should throw ValueError [BUG]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::thd(sig, kNaN, 1000.0), cps::ValueError);
}

TEST_CASE("thd: NaN fs should throw ValueError [BUG]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::thd(sig, 100.0, kNaN), cps::ValueError);
}

TEST_CASE("thd: fundamental <= 0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::thd(sig,  0.0, 1000.0), cps::ValueError);
    CHECK_THROWS_AS(cps::thd(sig, -1.0, 1000.0), cps::ValueError);
}

TEST_CASE("thd: fs <= 0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::thd(sig, 100.0,  0.0), cps::ValueError);
    CHECK_THROWS_AS(cps::thd(sig, 100.0, -1.0), cps::ValueError);
}

TEST_CASE("thd: n_harmonics=0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::thd(sig, 100.0, 1000.0, 0), cps::ValueError);
}

TEST_CASE("thd: returns -Inf when all harmonics exceed Nyquist [NOTE]") {
    // fundamental=400 Hz, fs=1000 Hz → 2nd harmonic = 800 Hz > 500 Hz Nyquist.
    // harmonic_sum=0 → 10*log10(0/fund) = -Inf. Mathematically correct: THD=-∞dB.
    const int N = 1024;
    std::vector<cps::Real> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2 * std::numbers::pi * 400.0 * i / 1000.0);
    double result = cps::thd(sig, 400.0, 1000.0, 1);
    CHECK(std::isinf(result));
    CHECK(result < 0.0);   // -Inf, not +Inf
}

// ═════════════════════════════════════════════════════════════════════════════
// measure/metrics.hpp — snr() spectral overload
//
// Guard:  fundamental <= 0.0, fs <= 0.0
// NaN passes both.  Also: snr lacks the n_harmonics >= 1 validation that
// thd has, creating an inconsistency.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("snr(spectral): NaN fundamental should throw ValueError [BUG]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::snr(sig, kNaN, 1000.0), cps::ValueError);
}

TEST_CASE("snr(spectral): NaN fs should throw ValueError [BUG]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::snr(sig, 100.0, kNaN), cps::ValueError);
}

TEST_CASE("snr(spectral): n_harmonics=0 should throw ValueError [BUG]") {
    // thd() validates n_harmonics >= 1 but snr() does not — inconsistency
    const int N = 256;
    std::vector<cps::Real> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2 * std::numbers::pi * 100.0 * i / 1000.0);
    CHECK_THROWS_AS(cps::snr(sig, 100.0, 1000.0, 0), cps::ValueError);
}

TEST_CASE("snr(spectral): n_harmonics negative should throw ValueError [BUG]") {
    const int N = 256;
    std::vector<cps::Real> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2 * std::numbers::pi * 100.0 * i / 1000.0);
    CHECK_THROWS_AS(cps::snr(sig, 100.0, 1000.0, -1), cps::ValueError);
}

TEST_CASE("snr(spectral): fundamental <= 0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::snr(sig,  0.0, 1000.0), cps::ValueError);
    CHECK_THROWS_AS(cps::snr(sig, -1.0, 1000.0), cps::ValueError);
}

TEST_CASE("snr(spectral): fs <= 0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::snr(sig, 100.0,  0.0), cps::ValueError);
    CHECK_THROWS_AS(cps::snr(sig, 100.0, -1.0), cps::ValueError);
}

TEST_CASE("snr(two-vector): zero noise throws NumericalError [OK]") {
    std::vector<cps::Real> sig   = {1.0, 2.0, 3.0};
    std::vector<cps::Real> noise = {0.0, 0.0, 0.0};
    CHECK_THROWS_AS(cps::snr(sig, noise), cps::NumericalError);
}

TEST_CASE("snr(two-vector): mismatched lengths throws ValueError [OK]") {
    std::vector<cps::Real> a = {1.0, 2.0, 3.0};
    std::vector<cps::Real> b = {1.0, 2.0};
    CHECK_THROWS_AS(cps::snr(a, b), cps::ValueError);
}

// ═════════════════════════════════════════════════════════════════════════════
// measure/metrics.hpp — sinad()
//
// Same NaN guards as thd / snr(spectral).
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("sinad: NaN fundamental should throw ValueError [BUG]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::sinad(sig, kNaN, 1000.0), cps::ValueError);
}

TEST_CASE("sinad: NaN fs should throw ValueError [BUG]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::sinad(sig, 100.0, kNaN), cps::ValueError);
}

TEST_CASE("sinad: fundamental <= 0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::sinad(sig,  0.0, 1000.0), cps::ValueError);
    CHECK_THROWS_AS(cps::sinad(sig, -1.0, 1000.0), cps::ValueError);
}

TEST_CASE("sinad: fs <= 0 throws ValueError [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    CHECK_THROWS_AS(cps::sinad(sig, 100.0,  0.0), cps::ValueError);
    CHECK_THROWS_AS(cps::sinad(sig, 100.0, -1.0), cps::ValueError);
}

// ═════════════════════════════════════════════════════════════════════════════
// spectral/fft.hpp — irfft()
//
// Regression tests for the n=0 fix: the old code accepted n=0 because
// 0/2+1==1 matched a size-1 spectrum, then caused UB inside the backend.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("irfft: n=0 throws ValueError [REGRESSION for n=0 fix]") {
    std::vector<cps::Complex> spec = {{1.0, 0.0}};  // size 1 = 0/2+1
    CHECK_THROWS_AS(cps::irfft(spec, 0), cps::ValueError);
}

TEST_CASE("irfft: empty spectrum throws ValueError [OK]") {
    std::vector<cps::Complex> empty;
    CHECK_THROWS_AS(cps::irfft(empty, 4), cps::ValueError);
}

TEST_CASE("irfft: spectrum size mismatch throws ValueError [OK]") {
    auto spec = cps::rfft(std::vector<cps::Real>(8, 1.0));  // size=5
    CHECK_THROWS_AS(cps::irfft(spec, 12), cps::ValueError); // 12/2+1=7 ≠ 5
}

TEST_CASE("irfft: n=1 (minimal valid case) returns single sample [OK]") {
    // n=1 → 1/2+1=1 → spec must have 1 element
    std::vector<cps::Complex> spec = {{3.0, 0.0}};
    auto out = cps::irfft(spec, 1);
    REQUIRE(out.size() == 1);
    CHECK_THAT(out[0], WithinAbs(3.0, 1e-10));
}

// ═════════════════════════════════════════════════════════════════════════════
// spectral/stft.hpp — stft()
//
// Regression tests for the nfft < nperseg fix: the old code allocated seg
// with nfft elements but wrote nperseg elements, overrunning the buffer.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("stft: nfft < nperseg throws ValueError [REGRESSION for nfft fix]") {
    std::vector<cps::Real> sig(256, 1.0);
    cps::STFTOptions opts;
    opts.nperseg = 64;
    opts.nfft    = 32;   // smaller than nperseg → was UB, now throws
    CHECK_THROWS_AS(cps::stft(sig, 1000.0, opts), cps::ValueError);
}

TEST_CASE("stft: nfft == nperseg is valid (no zero-padding) [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    cps::STFTOptions opts;
    opts.nperseg = 64;
    opts.nfft    = 64;
    REQUIRE_NOTHROW(cps::stft(sig, 1000.0, opts));
}

TEST_CASE("stft: nfft > nperseg is valid (zero-padding) [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    cps::STFTOptions opts;
    opts.nperseg = 64;
    opts.nfft    = 128;
    REQUIRE_NOTHROW(cps::stft(sig, 1000.0, opts));
}

TEST_CASE("stft: nfft=0 (default) uses nperseg — no buffer issue [OK]") {
    std::vector<cps::Real> sig(256, 1.0);
    REQUIRE_NOTHROW(cps::stft(sig, 1000.0));
}

TEST_CASE("stft: empty signal throws ValueError [OK]") {
    std::vector<cps::Real> empty;
    CHECK_THROWS_AS(cps::stft(empty, 1000.0), cps::ValueError);
}

TEST_CASE("stft: signal shorter than nperseg throws ValueError [OK]") {
    std::vector<cps::Real> sig(32, 1.0);
    CHECK_THROWS_AS(cps::stft(sig, 1000.0, {.nperseg = 64}), cps::ValueError);
}
