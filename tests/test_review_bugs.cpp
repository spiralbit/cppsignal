// test_review_bugs.cpp — failing tests for bugs found in code review
//
// Each test is tagged [BUG] and is expected to FAIL until the corresponding
// fix is applied. Tests are numbered to match the review findings.
//
// FFTW bugs (4, 5, 6) require -DCPS_ENABLE_FFTW=ON and are omitted here
// because the FFTW backend is off by default. Bug 4 (mutex/thread teardown)
// cannot be expressed as a unit test at all.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cps/cps.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
#include <string>

using Catch::Approx;
using cps::Real;

// ─────────────────────────────────────────────────────────────────────────────
// Bug 1 — firwin: numtaps=1 with non-rectangular window returns silent NaN
//
// M = numtaps-1 = 0; window formulas compute cos(2π*n/0) = cos(NaN) = NaN.
// The normalisation sum is NaN; abs(NaN) < 1e-12 is false so the guard
// doesn't fire. The function returns {NaN} silently.
//
// Fix: early-exit returning {1.0} when numtaps == 1 (a 1-sample window is
// identically 1.0 for every window type).
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 1: firwin numtaps=1 with Hann window returns valid coefficient [BUG]")
{
    // Currently returns {NaN} silently
    auto h = cps::firwin(1, 0.2, cps::Window::Hann);
    REQUIRE(h.size() == 1);
    CHECK_FALSE(std::isnan(h[0]));
    CHECK(h[0] == Approx(1.0));
}

TEST_CASE("Bug 1: firwin numtaps=1 with Hamming window returns valid coefficient [BUG]")
{
    auto h = cps::firwin(1, 0.3, cps::Window::Hamming);
    REQUIRE(h.size() == 1);
    CHECK_FALSE(std::isnan(h[0]));
    CHECK(h[0] == Approx(1.0));
}

TEST_CASE("Bug 1: firwin numtaps=1 with Blackman window returns valid coefficient [BUG]")
{
    auto h = cps::firwin(1, 0.4, cps::Window::Blackman);
    REQUIRE(h.size() == 1);
    CHECK_FALSE(std::isnan(h[0]));
    CHECK(h[0] == Approx(1.0));
}

TEST_CASE("Bug 1: firwin numtaps=1 with Rectangular window returns 1.0 [OK]")
{
    // Rectangular is the only window that works today: cos(2π*0/0) is computed
    // for n=0 only, but the Rectangular case fills with 1.0 unconditionally.
    auto h = cps::firwin(1, 0.2, cps::Window::Rectangular);
    REQUIRE(h.size() == 1);
    CHECK_FALSE(std::isnan(h[0]));
}


// ─────────────────────────────────────────────────────────────────────────────
// Bug 2 — white_noise: std_dev <= 0 is undefined behaviour
//
// std::normal_distribution requires stddev > 0 (C++ standard [rand.dist.norm]).
// Passing 0 or a negative value is UB — typically a silent corrupt distribution
// in release builds or an assertion abort in debug builds.
//
// Fix: validate std_dev > 0 and throw ValueError.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 2: white_noise std_dev=0 should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::white_noise(10, 0.0), cps::ValueError);
}

TEST_CASE("Bug 2: white_noise std_dev negative should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::white_noise(10, -1.0), cps::ValueError);
}

TEST_CASE("Bug 2: white_noise std_dev NaN should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::white_noise(10, std::numeric_limits<double>::quiet_NaN()), cps::ValueError);
}

TEST_CASE("Bug 2: white_noise with valid std_dev does not throw [OK]")
{
    CHECK_NOTHROW(cps::white_noise(10, 1.0));
    CHECK_NOTHROW(cps::white_noise(10, 0.001));
}


// ─────────────────────────────────────────────────────────────────────────────
// Bug 3 — fftfreq / rfftfreq: d=0 produces all-inf output silently
//
// inv_nd = 1.0 / (n * 0.0) = inf; every frequency bin becomes inf or -inf.
// No validation on d. The caller receives a completely wrong frequency axis
// with no diagnostic.
//
// Fix: validate d > 0 and throw ValueError.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 3: fftfreq d=0 should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::fftfreq(256, 0.0), cps::ValueError);
}

TEST_CASE("Bug 3: fftfreq d=NaN should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::fftfreq(256, std::numeric_limits<double>::quiet_NaN()), cps::ValueError);
}

TEST_CASE("Bug 3: fftfreq d=negative should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::fftfreq(256, -1.0), cps::ValueError);
}

TEST_CASE("Bug 3: rfftfreq d=0 should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::rfftfreq(256, 0.0), cps::ValueError);
}

TEST_CASE("Bug 3: rfftfreq d=NaN should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::rfftfreq(256, std::numeric_limits<double>::quiet_NaN()), cps::ValueError);
}

TEST_CASE("Bug 3: rfftfreq d=negative should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::rfftfreq(256, -1.0), cps::ValueError);
}

TEST_CASE("Bug 3: fftfreq with valid d produces finite bins [OK]")
{
    auto f = cps::fftfreq(8, 1.0 / 1000.0);
    REQUIRE(f.size() == 8);
    for (auto v : f)
        CHECK(std::isfinite(v));
}

TEST_CASE("Bug 3: rfftfreq with valid d produces finite bins [OK]")
{
    auto f = cps::rfftfreq(8, 1.0 / 1000.0);
    REQUIRE(f.size() == 5);
    for (auto v : f)
        CHECK(std::isfinite(v));
}


// ─────────────────────────────────────────────────────────────────────────────
// Bug 7 — Tukey window: alpha > 1 accepted, silently produces wrong window
//
// The guard only catches alpha <= 0; values above 1.0 cause the left and right
// taper regions to overlap, producing a malformed asymmetric window.
// Any STFT/Welch/FIR that uses it is silently corrupted.
//
// Fix: throw ValueError when param > 1.0.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 7: tukey_window alpha=1.5 should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::tukey_window(32, 1.5), cps::ValueError);
}

TEST_CASE("Bug 7: tukey_window alpha=2.0 should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::tukey_window(32, 2.0), cps::ValueError);
}

TEST_CASE("Bug 7: tukey_window alpha=inf should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::tukey_window(32, std::numeric_limits<double>::infinity()), cps::ValueError);
}

TEST_CASE("Bug 7: tukey_window alpha=NaN should throw ValueError [BUG]")
{
    CHECK_THROWS_AS(cps::tukey_window(32, std::numeric_limits<double>::quiet_NaN()), cps::ValueError);
}

TEST_CASE("Bug 7: tukey_window alpha=0.5 (valid) does not throw [OK]")
{
    CHECK_NOTHROW(cps::tukey_window(32, 0.5));
}

TEST_CASE("Bug 7: tukey_window alpha=1.0 (pure Hann) does not throw [OK]")
{
    CHECK_NOTHROW(cps::tukey_window(32, 1.0));
}


// ─────────────────────────────────────────────────────────────────────────────
// Bug 8 — peak_prominences: no bounds check on caller-supplied indices
//
// peak_prominences() accesses signal[idx] without checking idx < signal.size().
// An out-of-bounds idx causes an OOB read (undefined behaviour).
// Internal calls from find_peaks() are always safe, but the public API is not.
//
// Fix: throw ValueError when any index >= signal.size().
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 8: peak_prominences index == size() should throw ValueError [BUG]")
{
    std::vector<cps::Real> sig = {0.0, 1.0, 0.0, 2.0, 0.0};
    std::vector<std::size_t> bad_idx = {sig.size()};
    CHECK_THROWS_AS(cps::peak_prominences(sig, bad_idx), cps::ValueError);
}

TEST_CASE("Bug 8: peak_prominences index >> size() should throw ValueError [BUG]")
{
    std::vector<cps::Real> sig = {0.0, 1.0, 0.0};
    std::vector<std::size_t> bad_idx = {100};
    CHECK_THROWS_AS(cps::peak_prominences(sig, bad_idx), cps::ValueError);
}

TEST_CASE("Bug 8: peak_prominences with valid indices does not throw [OK]")
{
    std::vector<cps::Real> sig = {0.0, 1.0, 0.0, 2.0, 0.0};
    std::vector<std::size_t> good_idx = {1, 3};
    CHECK_NOTHROW(cps::peak_prominences(sig, good_idx));
}


// ─────────────────────────────────────────────────────────────────────────────
// Bug 9 — arange: size_t overflow for enormous ranges
//
// static_cast<size_t>(ceil((stop-start)/step)) is UB when the double value
// exceeds SIZE_MAX (~1.8e19). The subsequent vector allocation either fails
// with bad_alloc (not ValueError) or silently produces an empty/wrong vector.
//
// Fix: check the computed count against a safe threshold and throw ValueError.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Bug 9: arange with enormous range should throw ValueError [BUG]")
{
    // ceil(1e30 / 1.0) = 1e30 which overflows size_t → UB or bad_alloc
    CHECK_THROWS_AS(cps::arange(0.0, 1e30, 1.0), cps::ValueError);
}

TEST_CASE("Bug 9: arange with moderately large range works fine [OK]")
{
    // 1e6 elements is well within range
    auto v = cps::arange(0.0, 1e6, 1.0);
    CHECK(v.size() == 1000000u);
}


// ─────────────────────────────────────────────────────────────────────────────
// Bug 10 — thd / snr / sinad: empty signal error message identifies rfft,
//           not the calling function, and functions lack an explicit empty check
//
// Currently, passing an empty signal throws "rfft: input must not be empty"
// rather than "thd/snr/sinad: signal must not be empty". The error message is
// misleading since the user called thd/snr/sinad, not rfft directly.
//
// Fix: add explicit if (x.empty()) throw ValueError("thd/snr/sinad: signal
//      must not be empty") at the top of each function.
// ─────────────────────────────────────────────────────────────────────────────

static void check_error_mentions(std::function<void()> fn, const std::string& expected_prefix)
{
    try {
        fn();
        FAIL("expected a ValueError to be thrown");
    } catch (const cps::ValueError& e) {
        std::string msg = e.what();
        INFO("exception message: " << msg);
        CHECK(msg.find(expected_prefix) != std::string::npos);
    }
}

TEST_CASE("Bug 10: thd empty signal error message identifies thd [BUG]")
{
    std::vector<cps::Real> empty;
    check_error_mentions(
        [&]{ (void)cps::thd(empty, 100.0, 1000.0); },
        "thd");
}

TEST_CASE("Bug 10: snr(spectral) empty signal error message identifies snr [BUG]")
{
    std::vector<cps::Real> empty;
    check_error_mentions(
        [&]{ (void)cps::snr(empty, 100.0, 1000.0); },
        "snr");
}

TEST_CASE("Bug 10: sinad empty signal error message identifies sinad [BUG]")
{
    std::vector<cps::Real> empty;
    check_error_mentions(
        [&]{ (void)cps::sinad(empty, 100.0, 1000.0); },
        "sinad");
}
