// test_spectral.cpp — Welch PSD and STFT / spectrogram tests
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
// Welch PSD tests
// ════════════════════════════════════════════════════════════════════════════

// ── Output dimensions ────────────────────────────────────────────────────────
TEST_CASE("welch output length is nperseg/2+1", "[welch]")
{
    std::vector<Real> x(4096, 1.0);
    WelchOptions opts;
    opts.nperseg = 256;
    auto [f, psd] = welch(x, 1000.0, opts);

    CHECK(f.size()   == 256 / 2 + 1);
    CHECK(psd.size() == 256 / 2 + 1);
}

// ── Frequency axis ────────────────────────────────────────────────────────────
TEST_CASE("welch frequency axis spans [0, fs/2]", "[welch]")
{
    std::vector<Real> x(2048, 1.0);
    WelchOptions opts;
    opts.nperseg = 128;
    auto [f, psd] = welch(x, 500.0, opts);

    CHECK_THAT(f.front(), WithinAbs(0.0,   1e-9));
    CHECK_THAT(f.back(),  WithinAbs(250.0, 1e-6));  // fs/2 = 250 Hz
}

// ── Frequency bin spacing matches fs/nperseg ─────────────────────────────────
TEST_CASE("welch frequency resolution matches fs/nperseg", "[welch]")
{
    constexpr double fs = 1000.0;
    constexpr std::size_t nperseg = 256;

    std::vector<Real> x(4096, 1.0);
    WelchOptions opts;
    opts.nperseg = nperseg;
    auto [f, psd] = welch(x, fs, opts);

    double expected_df = fs / static_cast<double>(nperseg);
    CHECK_THAT(f[1] - f[0], WithinAbs(expected_df, 1e-6));
}

// ── Pure tone: peak at the correct bin ───────────────────────────────────────
// f0=125 Hz, nperseg=256, fs=1000 → exact bin 32 (125*256/1000 = 32)
TEST_CASE("welch pure tone peak is at the correct frequency bin", "[welch]")
{
    constexpr double fs   = 1000.0;
    constexpr double f0   = 125.0;
    constexpr int    N    = 4096;

    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);

    WelchOptions opts;
    opts.nperseg  = 256;
    opts.noverlap = 128;
    opts.window   = Window::Hann;
    auto [f, psd] = welch(x, fs, opts);

    // Find the peak bin
    auto peak_it  = std::max_element(psd.begin(), psd.end());
    std::size_t peak_bin = std::distance(psd.begin(), peak_it);

    // Peak should be at or adjacent to f=125 Hz (bin 32)
    CHECK(peak_bin >= 30);
    CHECK(peak_bin <= 34);
    // The peak frequency should be close to f0
    CHECK_THAT(f[peak_bin], WithinAbs(f0, 10.0));
}

// ── All-zeros: PSD should be zero (or throw) ─────────────────────────────────
TEST_CASE("welch PSD of zero signal is all zeros", "[welch]")
{
    std::vector<Real> x(1024, 0.0);
    WelchOptions opts;
    opts.nperseg = 128;
    auto [f, psd] = welch(x, 1000.0, opts);
    for (auto v : psd)
        CHECK_THAT(v, WithinAbs(0.0, 1e-20));
}

// ── Parseval-like property ────────────────────────────────────────────────────
// For a pure tone of amplitude A: signal power = A²/2
// One-sided Welch PSD integrates to approximately the signal power:
//   sum(psd[1..N-2]) * 2*df + psd[0]*df + psd[N-1]*df ≈ A²/2
TEST_CASE("welch PSD integrates to approximate signal power (Parseval)", "[welch]")
{
    constexpr double fs = 1000.0;
    constexpr double A  = 2.0;
    constexpr double f0 = 125.0;
    constexpr int    N  = 16384;   // long signal for accurate averaging

    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, A);

    WelchOptions opts;
    opts.nperseg  = 256;
    opts.noverlap = 128;
    opts.window   = Window::Hann;
    auto [f, psd] = welch(x, fs, opts);

    const double df     = fs / static_cast<double>(opts.nperseg);
    const double signal_power = A * A / 2.0;

    // Trapezoidal integration of one-sided PSD
    double integrated = psd[0] * df / 2.0 + psd.back() * df / 2.0;
    for (std::size_t k = 1; k + 1 < psd.size(); ++k)
        integrated += psd[k] * df;

    // Allow 20% tolerance — Welch is a statistical estimator
    CHECK_THAT(integrated, WithinRel(signal_power, 0.20));
}

// ════════════════════════════════════════════════════════════════════════════
// STFT tests
// ════════════════════════════════════════════════════════════════════════════

// ── Output dimensions ────────────────────────────────────────────────────────
TEST_CASE("stft output dimensions match nperseg and signal length", "[stft]")
{
    constexpr std::size_t N        = 1024;
    constexpr std::size_t nperseg  = 128;
    constexpr std::size_t noverlap = 64;

    std::vector<Real> x(N, 0.0);
    STFTOptions opts;
    opts.nperseg  = nperseg;
    opts.noverlap = noverlap;
    opts.window   = Window::Hann;
    auto sg = stft(x, 1000.0, opts);

    // freqs: one-sided → nperseg/2+1
    CHECK(sg.freqs.size() == nperseg / 2 + 1);
    // Zxx[f] should have one entry per time frame
    REQUIRE(!sg.Zxx.empty());
    CHECK(sg.Zxx.size() == sg.freqs.size());
    CHECK(sg.Zxx[0].size() == sg.times.size());
    // Number of frames: roughly (N - nperseg) / hop + 1
    const std::size_t hop    = nperseg - noverlap;
    const std::size_t nframe = (N - nperseg) / hop + 1;
    CHECK_THAT(static_cast<double>(sg.times.size()),
               WithinAbs(static_cast<double>(nframe), 2.0));
}

// ── Zero signal → zero STFT ──────────────────────────────────────────────────
TEST_CASE("stft of all-zeros signal has zero magnitude everywhere", "[stft]")
{
    std::vector<Real> x(512, 0.0);
    STFTOptions opts;
    opts.nperseg  = 64;
    opts.noverlap = 32;
    auto sg = stft(x, 1000.0, opts);

    for (auto& row : sg.Zxx)
        for (auto& c : row)
            CHECK_THAT(std::abs(c), WithinAbs(0.0, 1e-15));
}

// ── Frequency axis ────────────────────────────────────────────────────────────
TEST_CASE("stft frequency axis starts at 0 and ends at fs/2", "[stft]")
{
    constexpr double fs = 2000.0;
    std::vector<Real> x(1024, 1.0);
    STFTOptions opts;
    opts.nperseg  = 128;
    opts.noverlap = 64;
    auto sg = stft(x, fs, opts);

    CHECK_THAT(sg.freqs.front(), WithinAbs(0.0,    1e-9));
    CHECK_THAT(sg.freqs.back(),  WithinAbs(fs/2.0, 1.0));
}

// ── spectrogram power == |Zxx|² ──────────────────────────────────────────────
TEST_CASE("spectrogram power equals |Zxx|^2 from stft", "[spectrogram]")
{
    constexpr double fs = 1000.0;
    constexpr double f0 = 100.0;

    auto t = linspace(0.0, 1.0, 1000, false);
    auto x = sinusoid(t, f0, 1.0);

    STFTOptions opts;
    opts.nperseg  = 128;
    opts.noverlap = 64;
    opts.window   = Window::Hann;

    auto sg_c = stft(x, fs, opts);
    auto sg_p = spectrogram(x, fs, opts);

    REQUIRE(sg_p.power.size()    == sg_c.Zxx.size());
    REQUIRE(sg_p.freqs.size()    == sg_c.freqs.size());
    REQUIRE(sg_p.times.size()    == sg_c.times.size());

    for (std::size_t f = 0; f < sg_c.Zxx.size(); ++f)
        for (std::size_t t2 = 0; t2 < sg_c.Zxx[f].size(); ++t2)
            CHECK_THAT(sg_p.power[f][t2],
                       WithinAbs(std::norm(sg_c.Zxx[f][t2]), 1e-20));
}

// ── spectrogram: pure tone shows peak in correct frequency row ────────────────
TEST_CASE("spectrogram of pure tone peaks at the correct frequency", "[spectrogram]")
{
    constexpr double fs   = 1000.0;
    constexpr double f0   = 125.0;  // chosen to land near a bin

    auto t = linspace(0.0, 2.0, 2000, false);
    auto x = sinusoid(t, f0, 1.0);

    STFTOptions opts;
    opts.nperseg  = 256;
    opts.noverlap = 128;
    opts.window   = Window::Hann;
    auto sg = spectrogram(x, fs, opts);

    // Find which frequency row has the highest average power
    std::size_t peak_freq_row = 0;
    double max_avg = 0.0;
    for (std::size_t fi = 0; fi < sg.freqs.size(); ++fi) {
        double avg = 0.0;
        for (auto p : sg.power[fi]) avg += p;
        if (avg > max_avg) { max_avg = avg; peak_freq_row = fi; }
    }

    // That row's frequency should be close to f0 = 125 Hz
    CHECK_THAT(sg.freqs[peak_freq_row], WithinAbs(f0, 15.0));
}

// ── Error paths ────────────────────────────────────────────────────────────────

TEST_CASE("welch throws for empty signal", "[welch][error]")
{
    std::vector<Real> empty;
    CHECK_THROWS_AS(welch(empty, 1000.0), ValueError);
}

TEST_CASE("welch throws for fs <= 0", "[welch][error]")
{
    std::vector<Real> x(256, 1.0);
    CHECK_THROWS_AS(welch(x,  0.0), ValueError);
    CHECK_THROWS_AS(welch(x, -1.0), ValueError);
}

TEST_CASE("welch throws when signal is shorter than nperseg", "[welch][error]")
{
    std::vector<Real> x(50, 1.0);
    WelchOptions opts;
    opts.nperseg = 100;
    CHECK_THROWS_AS(welch(x, 1000.0, opts), ValueError);
}

TEST_CASE("stft throws for empty signal", "[stft][error]")
{
    std::vector<Real> empty;
    CHECK_THROWS_AS(stft(empty, 1000.0), ValueError);
}

TEST_CASE("stft throws when signal is shorter than nperseg", "[stft][error]")
{
    std::vector<Real> x(50, 1.0);
    STFTOptions opts;
    opts.nperseg = 100;
    CHECK_THROWS_AS(stft(x, 1000.0, opts), ValueError);
}

// ── welch onesided=false: two-sided spectrum ──────────────────────────────────
// The two-sided PSD should have length nperseg, span negative frequencies, and
// integrate to the same total power as the one-sided estimate.
TEST_CASE("welch onesided=false gives full two-sided spectrum", "[welch]")
{
    constexpr double fs  = 1000.0;
    constexpr double f0  = 125.0;
    constexpr int    N   = 8192;

    auto t = linspace(0.0, N / fs, N, false);
    auto x = sinusoid(t, f0, 1.0);

    WelchOptions opts;
    opts.nperseg  = 256;
    opts.noverlap = 128;
    opts.window   = Window::Hann;

    opts.onesided = false;
    auto [f2, p2] = welch(x, fs, opts);

    // Two-sided output length is nfft (= nperseg = 256)
    CHECK(f2.size() == opts.nperseg);
    CHECK(p2.size() == opts.nperseg);

    // Frequency axis starts at 0, runs positive then negative (fftfreq ordering)
    CHECK_THAT(f2.front(), WithinAbs(0.0, 1e-9));
    // Second half should contain negative frequencies
    bool has_negative = false;
    for (auto fv : f2) if (fv < 0.0) { has_negative = true; break; }
    CHECK(has_negative);

    // Spectrum must be symmetric: p2[k] == p2[N-k] for k=1..N/2-1
    const std::size_t M = opts.nperseg;
    for (std::size_t k = 1; k < M / 2; ++k)
        CHECK_THAT(p2[k], WithinAbs(p2[M - k], 1e-12));

    // Total integrated power ≈ A²/2 = 0.5 for amplitude=1, measured via
    // sum(psd) * df (same result as one-sided after folding)
    const double df = fs / static_cast<double>(M);
    double total_power = 0.0;
    for (auto v : p2) total_power += v * df;
    CHECK_THAT(total_power, WithinRel(0.5, 0.20));
}

// ── stft with default noverlap (nullopt → nperseg/2) ─────────────────────────
TEST_CASE("stft with default noverlap uses nperseg/2", "[stft]")
{
    constexpr std::size_t nperseg = 64;
    std::vector<Real> x(256, 1.0);
    STFTOptions opts;
    opts.nperseg = nperseg;
    // opts.noverlap left as nullopt → should default to nperseg/2 = 32
    auto sg = stft(x, 1000.0, opts);
    CHECK(sg.freqs.size() == nperseg / 2 + 1);
    const std::size_t expected_hop = nperseg / 2;  // noverlap = 32 → hop = 32
    const std::size_t nframe = (x.size() - nperseg) / expected_hop + 1;
    CHECK_THAT(static_cast<double>(sg.times.size()),
               WithinAbs(static_cast<double>(nframe), 2.0));
}

// ── stft with fs=0 (normalised output) ───────────────────────────────────────
TEST_CASE("stft with fs=0 produces normalised frequency axis", "[stft]")
{
    std::vector<Real> x(256, 1.0);
    STFTOptions opts;
    opts.nperseg  = 64;
    opts.noverlap = 32;
    auto sg = stft(x, 0.0, opts);   // fs = 0 → normalised

    // Normalised one-sided freqs: k/nfft for k=0..nfft/2, so range [0, 0.5]
    CHECK_THAT(sg.freqs.front(), WithinAbs(0.0,  1e-9));
    CHECK_THAT(sg.freqs.back(),  WithinAbs(0.5,  1e-9));
    // Time axis: centre sample index (not divided by fs)
    CHECK(sg.times.front() < 100.0);   // sample-domain values, not seconds
}

// ── stft onesided=false with fs=0 (normalised two-sided) ─────────────────────
TEST_CASE("stft onesided=false with fs=0 has normalised two-sided axis", "[stft]")
{
    constexpr std::size_t nperseg = 64;
    std::vector<Real> x(256, 1.0);
    STFTOptions opts;
    opts.nperseg  = nperseg;
    opts.noverlap = nperseg / 2;
    opts.onesided = false;
    auto sg = stft(x, 0.0, opts);   // fs = 0, two-sided

    // Two-sided: nfft rows
    CHECK(sg.freqs.size() == nperseg);
    // First freq is 0, second half contains negative values
    CHECK_THAT(sg.freqs.front(), WithinAbs(0.0, 1e-9));
    bool has_negative = false;
    for (auto fv : sg.freqs) if (fv < 0.0) { has_negative = true; break; }
    CHECK(has_negative);
    // Times are in sample units (not divided by fs)
    CHECK(sg.times.front() < 1000.0);
}

// ── stft onesided=false: two-sided complex spectrum ───────────────────────────
TEST_CASE("stft onesided=false has nfft frequency rows and negative bins", "[stft]")
{
    constexpr std::size_t nperseg = 64;
    std::vector<Real> x(512, 0.0);
    // Put a pure tone at a known bin
    auto t = linspace(0.0, 512.0 / 1000.0, 512, false);
    x = sinusoid(t, 100.0, 1.0);

    STFTOptions opts;
    opts.nperseg  = nperseg;
    opts.noverlap = nperseg / 2;
    opts.onesided = false;
    auto sg = stft(x, 1000.0, opts);

    // Two-sided: nfft rows
    CHECK(sg.freqs.size() == nperseg);
    CHECK(sg.Zxx.size()   == nperseg);

    // Frequency axis contains negative values
    bool has_negative = false;
    for (auto fv : sg.freqs) if (fv < 0.0) { has_negative = true; break; }
    CHECK(has_negative);

    // Conjugate symmetry: |Zxx[k]| == |Zxx[N-k]| for k=1..N/2-1
    const std::size_t nt = sg.Zxx[0].size();
    for (std::size_t t2 = 0; t2 < nt; ++t2) {
        for (std::size_t k = 1; k < nperseg / 2; ++k) {
            double mag_pos = std::abs(sg.Zxx[k][t2]);
            double mag_neg = std::abs(sg.Zxx[nperseg - k][t2]);
            CHECK_THAT(mag_pos, WithinAbs(mag_neg, 1e-10));
        }
    }
}

// ── stft with explicit nfft (zero-padding path) ───────────────────────────────
TEST_CASE("stft with explicit nfft zero-pads frames and gives correct freq axis", "[stft]")
{
    constexpr std::size_t nperseg = 32;
    constexpr std::size_t nfft    = 64;   // opts.nfft != 0 → stft.hpp:68 right branch
    std::vector<Real> x(256, 1.0);
    STFTOptions opts;
    opts.nperseg  = nperseg;
    opts.noverlap = nperseg / 2;
    opts.nfft     = nfft;
    auto sg = stft(x, 1000.0, opts);

    // One-sided output: nfft/2+1 = 33 frequency rows
    CHECK(sg.freqs.size() == nfft / 2 + 1);
    // Frequency axis ends at Nyquist (fs/2 = 500 Hz)
    CHECK_THAT(sg.freqs.back(), WithinAbs(500.0, 1.0));
}
