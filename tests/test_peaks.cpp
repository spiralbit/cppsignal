// test_peaks.cpp — unit tests for peak detection and signal generation
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <vector>
#include <cmath>
#include <numbers>

using namespace cps;
using Catch::Matchers::WithinAbs;

// ── find_peaks: simple case ───────────────────────────────────────────────────
TEST_CASE("find_peaks detects obvious local maxima", "[peaks]")
{
    // Three clear peaks at indices 2, 5, 8
    std::vector<Real> x = {0, 1, 3, 1, 0, 4, 0, 1, 2, 1, 0};
    auto result = find_peaks(x);

    REQUIRE(result.indices.size() == 3);
    CHECK(result.indices[0] == 2);
    CHECK(result.indices[1] == 5);
    CHECK(result.indices[2] == 8);
}

// ── find_peaks: height filter ─────────────────────────────────────────────────
TEST_CASE("find_peaks height filter excludes short peaks", "[peaks]")
{
    std::vector<Real> x = {0, 1, 3, 1, 0, 4, 0, 1, 2, 1, 0};
    auto result = find_peaks(x, {.height = 2.5});

    // Only peaks at index 2 (h=3) and index 5 (h=4) exceed 2.5
    REQUIRE(result.indices.size() == 2);
    CHECK(result.indices[0] == 2);
    CHECK(result.indices[1] == 5);
}

// ── find_peaks: distance filter ───────────────────────────────────────────────
TEST_CASE("find_peaks distance filter removes nearby shorter peaks", "[peaks]")
{
    // Two peaks close together at idx 2 (h=3) and idx 4 (h=2)
    std::vector<Real> x = {0, 1, 3, 2, 2.5, 1, 0};
    //                              ^ idx2  ^ idx4
    auto result = find_peaks(x, {.distance = 3});

    // idx4 (h=2.5) is within distance 3 of idx2 (h=3); idx2 is taller → keep idx2
    // Actually idx2=3, idx4=2.5: idx2 is taller, so idx4 removed
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 2);
}

// ── find_peaks: no peaks in monotone signal ───────────────────────────────────
TEST_CASE("find_peaks returns empty for monotone signal", "[peaks]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = find_peaks(x);
    CHECK(result.indices.empty());
}

// ── find_peaks: handles flat plateau ─────────────────────────────────────────
TEST_CASE("find_peaks does not detect plateau as peak (strict inequality)", "[peaks]")
{
    std::vector<Real> x = {0, 1, 1, 1, 0};  // flat top — not a strict local max
    auto result = find_peaks(x);
    CHECK(result.indices.empty());
}

// ── find_peaks: prominence ────────────────────────────────────────────────────
TEST_CASE("find_peaks prominence filter removes shallow peaks", "[peaks]")
{
    // Large peak at idx 5 (height 10), small shoulder at idx 2 (height 3)
    std::vector<Real> x = {0, 1, 3, 2, 4, 10, 3, 1, 0};
    auto result = find_peaks(x, {.prominence = 4.0});

    // The peak at idx 2 has prominence ≈ 2 (3 - 1 valley), excluded
    // The peak at idx 4 (h=4) has prominence ≈ 2 (4-2), also excluded
    // The peak at idx 5 (h=10) has very high prominence, included
    bool has_idx5 = false;
    for (auto i : result.indices)
        if (i == 5) has_idx5 = true;
    CHECK(has_idx5);
}

// ── Signal generation: sinusoid ───────────────────────────────────────────────
TEST_CASE("sinusoid generates correct values", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1001, true);
    auto y = sinusoid(t, 1.0);  // 1 Hz sine

    // At t=0: sin(0)=0, at t=0.25: sin(π/2)=1, at t=0.5: sin(π)=0
    CHECK_THAT(y[0],   WithinAbs(0.0,  1e-10));
    CHECK_THAT(y[250], WithinAbs(1.0,  1e-3));   // ~t=0.25s
    CHECK_THAT(y[500], WithinAbs(0.0,  1e-3));   // ~t=0.5s
}

// ── Signal generation: unit impulse ──────────────────────────────────────────
TEST_CASE("unit_impulse has 1 at idx and 0 elsewhere", "[generate]")
{
    auto imp = unit_impulse(10, 3);
    REQUIRE(imp.size() == 10);
    for (std::size_t i = 0; i < 10; ++i) {
        if (i == 3) CHECK_THAT(imp[i], WithinAbs(1.0, 1e-12));
        else        CHECK_THAT(imp[i], WithinAbs(0.0, 1e-12));
    }
}

// ── Signal generation: chirp ──────────────────────────────────────────────────
TEST_CASE("chirp has correct length and amplitude range", "[generate]")
{
    auto t = linspace(0.0, 1.0, 1000);
    auto y = chirp(t, 10.0, 100.0, 1.0);  // sweep 10→100 Hz over 1 second

    REQUIRE(y.size() == 1000);
    // Values should be in [-1, 1] (cosine sweep)
    for (auto v : y) {
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
}

// ── Correlation: auto-correlation of impulse ──────────────────────────────────
TEST_CASE("auto-correlation of unit impulse is itself", "[correlate]")
{
    std::vector<Real> imp = {0, 0, 1, 0, 0};
    auto ac = correlate(imp, imp);

    // Full auto-correlation of length-5 has length 9
    REQUIRE(ac.size() == 9);

    // Peak should be at the centre (index 4 for length 9)
    CHECK_THAT(ac[4], WithinAbs(1.0, 1e-12));

    // All other values should be zero
    for (std::size_t i = 0; i < ac.size(); ++i)
        if (i != 4) CHECK_THAT(ac[i], WithinAbs(0.0, 1e-12));
}

// ── Convolve: known result ─────────────────────────────────────────────────────
TEST_CASE("convolve([1,1,1], [1,1,1]) == [1,2,3,2,1]", "[correlate]")
{
    std::vector<Real> a = {1, 1, 1};
    std::vector<Real> b = {1, 1, 1};
    auto c = convolve(a, b);

    REQUIRE(c.size() == 5);
    CHECK_THAT(c[0], WithinAbs(1.0, 1e-12));
    CHECK_THAT(c[1], WithinAbs(2.0, 1e-12));
    CHECK_THAT(c[2], WithinAbs(3.0, 1e-12));
    CHECK_THAT(c[3], WithinAbs(2.0, 1e-12));
    CHECK_THAT(c[4], WithinAbs(1.0, 1e-12));
}

// ── RMS: known values ─────────────────────────────────────────────────────────
TEST_CASE("rms of sine wave is amplitude / sqrt(2)", "[metrics]")
{
    const double A = 3.0;
    auto t = linspace(0.0, 10.0, 10000);
    auto y = sinusoid(t, 1.0, A);

    double measured = rms(y);
    double expected = A / std::sqrt(2.0);
    CHECK_THAT(measured, WithinAbs(expected, 0.001));
}

TEST_CASE("rms of constant signal equals the constant", "[metrics]")
{
    std::vector<Real> x(1000, 2.5);
    CHECK_THAT(rms(x), WithinAbs(2.5, 1e-9));
}

// ── find_peaks: threshold filter ─────────────────────────────────────────────

TEST_CASE("find_peaks threshold filter removes peaks with small rise above neighbours", "[peaks]")
{
    // Peak at idx 2 (h=3): rise_left=3-1=2, rise_right=3-2.5=0.5 < threshold=1.0
    // Peak at idx 5 (h=5): rise_left=5-1=4, rise_right=5-0=5 — both pass
    std::vector<Real> x = {0, 1, 3, 2.5, 1, 5, 0};
    auto result = find_peaks(x, {.threshold = 1.0});
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 5);
}

// ── find_peaks: threshold filter short-circuit (left rise fails first) ────────
// Covers the peaks.hpp lambda || path where the first operand is TRUE → short-circuit.
// rise_left = 5-3 = 2 < threshold=3 → first operand true → lambda returns immediately.
TEST_CASE("find_peaks threshold: left-rise failure short-circuits", "[peaks]")
{
    std::vector<Real> x = {4.0, 3.0, 5.0, 0.0};
    auto r = find_peaks(x, {.threshold = 3.0});
    REQUIRE(r.indices.empty());
}

// ── find_peaks: distance filter keeps taller second peak ─────────────────────

TEST_CASE("find_peaks distance filter keeps taller second peak", "[peaks]")
{
    // Peaks at idx 2 (h=2) and idx 4 (h=3), separation=2 < distance=3
    // First peak is shorter → keep[i]=false branch exercises the break path
    std::vector<Real> x = {0, 1, 2, 1, 3, 1, 0};
    auto result = find_peaks(x, {.distance = 3});
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 4);
}

// ── peak_prominences: higher peak to the left exercises inner min loop ────────

TEST_CASE("find_peaks distance filter preserves well-separated peaks", "[peaks]")
{
    // Three peaks at indices 2, 5, 8 — separation 3 is not less than distance=2
    // → else branch (peaks.hpp line 170: break because farther away) exercises
    std::vector<Real> x = {0, 1, 3, 1, 0, 4, 0, 1, 2, 1, 0};
    auto result = find_peaks(x, {.distance = 2});
    REQUIRE(result.indices.size() == 3);
    CHECK(result.indices[0] == 2);
    CHECK(result.indices[1] == 5);
    CHECK(result.indices[2] == 8);
}

TEST_CASE("peak_prominences correct when higher peak exists to the left", "[peaks]")
{
    // Signal: {0,1,5,1,3,1,0} — peak at idx 4 (h=3), higher peak at idx 2 (h=5)
    // Left base = min in signal[2..4] = min(5,1,3) = 1
    // Right base = min in signal[4..end] = min(3,1,0) = 0
    // Prominence = 3 - max(1, 0) = 2
    std::vector<Real> signal = {0, 1, 5, 1, 3, 1, 0};
    std::vector<std::size_t> peaks = {4};
    auto proms = peak_prominences(signal, peaks);
    REQUIRE(proms.size() == 1);
    CHECK_THAT(proms[0], WithinAbs(2.0, 1e-9));
}

// ── find_peaks: boundary interior indices ────────────────────────────────────

TEST_CASE("find_peaks detects peak at index 1 (first interior index)", "[peaks]")
{
    // Peak at index 1 — exercises the lower boundary of the interior scan
    std::vector<Real> x = {0.5, 2.0, 1.0, 0.0};
    auto result = find_peaks(x);
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 1);
}

TEST_CASE("find_peaks detects peak at last interior index (n-2)", "[peaks]")
{
    // Peak at index 2 (= n-2 for n=4) — exercises the upper boundary of the interior scan
    std::vector<Real> x = {0.0, 1.0, 3.0, 0.5};
    auto result = find_peaks(x);
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 2);
}

// ── find_peaks: distance=1 skips the filter block (value() <= 1 branch) ──────

TEST_CASE("find_peaks distance=1 applies no filtering (value<=1 path)", "[peaks]")
{
    // opts.distance.has_value()=true but value()=1 so value()>1 is false →
    // the whole condition is false and the distance-filter block is skipped.
    std::vector<Real> x = {0, 1, 3, 1, 0, 4, 0, 1, 2, 1, 0};
    auto result = find_peaks(x, {.distance = 1});
    REQUIRE(result.indices.size() == 3);   // all peaks kept
}

// ── find_peaks: degenerate small inputs ──────────────────────────────────────
// Signals with fewer than 3 samples cannot have interior maxima.

TEST_CASE("find_peaks returns empty for signal of length 0", "[peaks]")
{
    std::vector<Real> x;
    auto result = find_peaks(x);
    CHECK(result.indices.empty());
}

TEST_CASE("find_peaks returns empty for signal of length 1", "[peaks]")
{
    std::vector<Real> x = {5.0};
    auto result = find_peaks(x);
    CHECK(result.indices.empty());
}

TEST_CASE("find_peaks returns empty for signal of length 2", "[peaks]")
{
    std::vector<Real> x = {1.0, 2.0};
    auto result = find_peaks(x);
    CHECK(result.indices.empty());
}

// ── Bug regression: peak_prominences scan direction ───────────────────────────
// When there is a valley between the nearest higher left peak and a more-distant
// higher peak, the left contour base must be the minimum between the NEAREST
// higher peak and the subject peak — not the leftmost higher peak.
//
// Signal: [2, 5, 1, 4, 2, 3, 1, 7]
//          0  1  2  3  4  5  6  7
// Peak at idx 5 (h=3). Nearest higher left peak: idx 3 (h=4).
// Left contour base = min(signal[3..5]) = min(4,2,3) = 2.
// Right contour base = min(signal[5..7]) = min(3,1,7) = 1.
// Prominence = 3 - max(2,1) = 1.
// (Buggy code scans left-to-right, hits idx 1 first, takes min(signal[1..5])=1,
//  giving wrong prominence = 2.)
TEST_CASE("peak_prominences uses nearest higher peak not leftmost", "[peaks][regression]")
{
    std::vector<Real> signal = {2, 5, 1, 4, 2, 3, 1, 7};
    std::vector<std::size_t> peaks = {5};
    auto proms = peak_prominences(signal, peaks);
    REQUIRE(proms.size() == 1);
    CHECK_THAT(proms[0], WithinAbs(1.0, 1e-9));
}

// ── Bug regression: find_peaks width filter silently ignored ──────────────────
// Passing opts.width should reduce the peak set; currently width is never
// applied so the full set is returned regardless.
TEST_CASE("find_peaks width filter excludes narrow peaks", "[peaks][regression]")
{
    // Three sharp spikes — each rises and falls in exactly 1 sample on each side.
    // Any minimum-width requirement > ~1 should exclude all of them.
    std::vector<Real> x = {0, 0, 1, 3, 1, 0, 1, 5, 1, 0, 1, 2, 1, 0, 0};

    auto all = find_peaks(x);
    REQUIRE(all.indices.size() == 3);   // baseline: 3 peaks without filter

    auto result = find_peaks(x, {.width = 10.0});   // far wider than any peak
    CHECK(result.indices.empty());
}

TEST_CASE("find_peaks width filter passes peak with sufficient width", "[peaks]")
{
    // Triangular peak {0,0,5,0,0} at index 2.
    // prominence = 5, level = 2.5, left_pos = 1.5, right_pos = 2.5 → width = 1.0
    std::vector<Real> x = {0, 0, 5, 0, 0};
    auto r09 = find_peaks(x, {.width = 0.9});  // 1.0 >= 0.9 → peak included
    REQUIRE(r09.indices.size() == 1);
    CHECK(r09.indices[0] == 2);

    auto r11 = find_peaks(x, {.width = 1.1});  // 1.0 < 1.1 → peak excluded
    CHECK(r11.indices.empty());
}
