// test_stress.cpp — stress tests: large data, small data, and repeated-call stability
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

static constexpr std::size_t LARGE_N = 65536;
static constexpr std::size_t HUGE_N  = 131072;

// ════════════════════════════════════════════════════════════════════════════
// FFT — large inputs
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("fft/ifft roundtrip: 65536 complex samples", "[stress][fft]")
{
    std::vector<Complex> x(LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i)
        x[i] = {std::cos(2.0 * std::numbers::pi * 100.0 * i / LARGE_N),
                std::sin(2.0 * std::numbers::pi * 100.0 * i / LARGE_N)};
    auto X = fft(std::span<const Complex>(x));
    auto y = ifft(std::span<const Complex>(X));
    REQUIRE(y.size() == LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i) {
        CHECK_THAT(y[i].real(), WithinAbs(x[i].real(), 1e-8));
        CHECK_THAT(y[i].imag(), WithinAbs(x[i].imag(), 1e-8));
    }
}

TEST_CASE("rfft/irfft roundtrip: 65536 real samples", "[stress][fft]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 440.0);
    auto X = rfft(std::span<const Real>(x));
    REQUIRE(X.size() == LARGE_N / 2 + 1);
    auto y = irfft(std::span<const Complex>(X), LARGE_N);
    REQUIRE(y.size() == LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i)
        CHECK_THAT(y[i], WithinAbs(x[i], 1e-8));
}

TEST_CASE("rfft/irfft roundtrip: 131072 real samples", "[stress][fft]")
{
    std::vector<Real> x(HUGE_N);
    for (std::size_t i = 0; i < HUGE_N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 200.0 * i / HUGE_N);
    auto X = rfft(std::span<const Real>(x));
    auto y = irfft(std::span<const Complex>(X), HUGE_N);
    REQUIRE(y.size() == HUGE_N);
    double max_err = 0.0;
    for (std::size_t i = 0; i < HUGE_N; ++i)
        max_err = std::max(max_err, std::abs(y[i] - x[i]));
    CHECK(max_err < 1e-7);
}

TEST_CASE("fft real overload: 65536 real samples promoted to complex", "[stress][fft]")
{
    std::vector<Real> x(LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i)
        x[i] = std::cos(2.0 * std::numbers::pi * 10.0 * i / LARGE_N);
    auto X = fft(std::span<const Real>(x));
    REQUIRE(X.size() == LARGE_N);
    // DC component should be near zero for a cosine (zero-mean); bin 10 should dominate
    CHECK(std::abs(X[10]) > std::abs(X[0]) * 10.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Signal generators — large inputs
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("sinusoid: 65536 samples, finite values in [-1,1]", "[stress][generate]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto y = sinusoid(t, 440.0);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y) {
        REQUIRE(std::isfinite(v));
        CHECK(v >= -1.0 - 1e-12);
        CHECK(v <=  1.0 + 1e-12);
    }
}

TEST_CASE("chirp: 65536 samples, finite values in [-1,1]", "[stress][generate]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto y = chirp(t, 20.0, 20000.0, 1.0);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y) {
        REQUIRE(std::isfinite(v));
        CHECK(v >= -1.0 - 1e-12);
        CHECK(v <=  1.0 + 1e-12);
    }
}

TEST_CASE("square_wave: 65536 samples, only +1 and -1", "[stress][generate]")
{
    auto t = linspace(0.0, 10.0, LARGE_N, false);
    auto y = square_wave(t, 440.0);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y)
        CHECK((v == 1.0 || v == -1.0));
}

TEST_CASE("sawtooth_wave: 65536 samples, stays in [-1,1]", "[stress][generate]")
{
    auto t = linspace(0.0, 10.0, LARGE_N, false);
    auto y = sawtooth_wave(t, 440.0, 0.7);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y) {
        REQUIRE(std::isfinite(v));
        CHECK(v >= -1.0 - 1e-9);
        CHECK(v <=  1.0 + 1e-9);
    }
}

TEST_CASE("white_noise: 65536 samples, all finite, mean near 0", "[stress][generate]")
{
    auto x = white_noise(LARGE_N, 1.0, 123u);
    REQUIRE(x.size() == LARGE_N);
    for (auto v : x)
        REQUIRE(std::isfinite(v));
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / LARGE_N;
    CHECK_THAT(mean, WithinAbs(0.0, 0.05));
}

TEST_CASE("gausspulse: 65536 samples, all finite, peak near centre", "[stress][generate]")
{
    auto t = linspace(-0.001, 0.001, LARGE_N);
    auto y = gausspulse(t, 10000.0);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y)
        REQUIRE(std::isfinite(v));
    auto peak_it = std::max_element(y.begin(), y.end());
    std::size_t peak_idx = static_cast<std::size_t>(std::distance(y.begin(), peak_it));
    CHECK(peak_idx > LARGE_N / 4);
    CHECK(peak_idx < 3 * LARGE_N / 4);
}

// ════════════════════════════════════════════════════════════════════════════
// Filter operations — large inputs
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("sosfilt: 4th-order LP on 65536 samples produces finite output", "[stress][filter]")
{
    auto sos = butter(4, 0.1, FilterType::Lowpass);
    std::vector<Real> x(LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i)
        x[i] = std::cos(2.0 * std::numbers::pi * 0.01 * static_cast<double>(i));
    auto y = sosfilt(sos, x);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y)
        CHECK(std::isfinite(v));
}

TEST_CASE("sosfilt: 8th-order HP on 65536 samples, low-freq input is attenuated", "[stress][filter]")
{
    // 8th-order HP at 0.3 normalised: a 0.01-normalised sine should be well below passband
    auto sos = butter(8, 0.3, FilterType::Highpass);
    std::vector<Real> x(LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 0.005 * static_cast<double>(i));
    auto y = sosfilt(sos, x);
    REQUIRE(y.size() == LARGE_N);
    // After transient, RMS of output should be much smaller than input
    double in_rms  = rms(std::span<const Real>(x).subspan(LARGE_N / 2));
    double out_rms = rms(std::span<const Real>(y).subspan(LARGE_N / 2));
    CHECK(out_rms < in_rms * 0.01);
}

TEST_CASE("lfilter: 65536 samples with simple IIR, finite output", "[stress][filter]")
{
    std::vector<Real> b = {1.0, 2.0, 1.0};
    std::vector<Real> a = {1.0, -0.9, 0.81};
    std::vector<Real> x(LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 0.05 * static_cast<double>(i));
    auto y = lfilter(b, a, x);
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y)
        CHECK(std::isfinite(v));
}

// ════════════════════════════════════════════════════════════════════════════
// Spectral analysis — large inputs
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("welch: 65536 samples, one-sided PSD has correct length and non-negative values", "[stress][spectral]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 440.0);
    auto [freqs, psd] = welch(std::span<const Real>(x), static_cast<Real>(LARGE_N),
                              WelchOptions{.nperseg = 256});
    REQUIRE(freqs.size() == 129);   // 256/2 + 1
    REQUIRE(psd.size() == 129);
    for (auto v : psd)
        CHECK(v >= 0.0);
    for (auto v : freqs)
        CHECK(v >= 0.0);
}

TEST_CASE("welch: 131072 samples, two-sided PSD", "[stress][spectral]")
{
    std::vector<Real> x(HUGE_N);
    constexpr double fs = 44100.0;
    for (std::size_t i = 0; i < HUGE_N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 440.0 * i / fs);
    auto [freqs, psd] = welch(std::span<const Real>(x), fs,
                               WelchOptions{.nperseg = 512, .onesided = false});
    CHECK(freqs.size() == 512);
    CHECK(psd.size() == 512);
    for (auto v : psd)
        CHECK(v >= 0.0);
}

TEST_CASE("stft: 65536 samples, correct time-frequency dimensions", "[stress][spectral]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 440.0);
    auto result = stft(std::span<const Real>(x), static_cast<Real>(LARGE_N),
                       STFTOptions{.nperseg = 256});
    REQUIRE(result.freqs.size() == 129);   // 256/2 + 1
    REQUIRE(result.times.size() > 0);
    REQUIRE(result.Zxx.size() == 129);
    for (auto& col : result.Zxx)
        CHECK(col.size() == result.times.size());
}

TEST_CASE("spectrogram: 65536 samples, power values non-negative", "[stress][spectral]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 440.0);
    auto result = spectrogram(std::span<const Real>(x), static_cast<Real>(LARGE_N),
                              STFTOptions{.nperseg = 256});
    for (auto& row : result.power)
        for (auto v : row)
            CHECK(v >= 0.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Peak detection — large and dense signals
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("find_peaks: dense sine (1000 Hz, 65536 samples) finds ~1000 peaks", "[stress][peaks]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 1000.0);
    auto r = find_peaks(std::span<const Real>(x));
    // 1000 Hz × 1 s ≈ 1000 peaks
    CHECK(r.indices.size() > 900);
    CHECK(r.indices.size() < 1100);
    for (auto idx : r.indices)
        CHECK(x[idx] > 0.0);
}

TEST_CASE("find_peaks: distance filter limits peak count on 65536 samples", "[stress][peaks]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 500.0);
    auto r = find_peaks(std::span<const Real>(x), PeakOptions{.distance = 120});
    // With min distance 120, peaks are at most LARGE_N/120 apart
    CHECK(r.indices.size() <= LARGE_N / 120 + 5);
    // But we still detect a reasonable number of peaks (500 Hz * 1s = 500 expected without filter)
    CHECK(r.indices.size() > 10);
}

TEST_CASE("find_peaks: prominence + width filter on 65536 samples", "[stress][peaks]")
{
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, 500.0);
    auto r = find_peaks(std::span<const Real>(x),
                        PeakOptions{.prominence = 0.5, .width = 5.0});
    for (auto idx : r.indices) {
        CHECK(std::isfinite(x[idx]));
        CHECK(x[idx] > 0.0);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Correlation / Convolution — moderate-to-large inputs
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("convolve: 1000 × 100 all-ones, triangular shape", "[stress][correlate]")
{
    std::vector<Real> x(1000, 1.0);
    std::vector<Real> y(100,  1.0);
    auto c = convolve(x, y);
    REQUIRE(c.size() == 1099);
    // Overlap increases linearly up to 100, stays at 100, then decreases
    for (std::size_t i = 0; i < 100; ++i)
        CHECK_THAT(c[i], WithinAbs(static_cast<double>(i + 1), 1e-9));
    for (std::size_t i = 100; i < 1000; ++i)
        CHECK_THAT(c[i], WithinAbs(100.0, 1e-9));
    for (std::size_t i = 1000; i < 1099; ++i)
        CHECK_THAT(c[i], WithinAbs(static_cast<double>(1099 - i), 1e-9));
}

TEST_CASE("correlate: 500×500 autocorrelation, zero-lag peak", "[stress][correlate]")
{
    auto t = linspace(0.0, 1.0, 500, false);
    auto x = sinusoid(t, 10.0);
    auto c = correlate(std::span<const Real>(x), std::span<const Real>(x));
    REQUIRE(c.size() == 999);
    auto peak_it = std::max_element(c.begin(), c.end());
    CHECK(std::distance(c.begin(), peak_it) == 499);  // zero-lag at centre
}

TEST_CASE("correlate Same mode: 500×500, output length = 500", "[stress][correlate]")
{
    auto t = linspace(0.0, 1.0, 500, false);
    auto x = sinusoid(t, 5.0);
    auto c = correlate(std::span<const Real>(x), std::span<const Real>(x),
                       ConvolveMode::Same);
    REQUIRE(c.size() == 500);
}

// ════════════════════════════════════════════════════════════════════════════
// Metrics — large inputs
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("rms: 65536 unit-amplitude sine → rms ≈ 1/√2", "[stress][metrics]")
{
    auto t = linspace(0.0, 1024.0, LARGE_N, false);
    auto x = sinusoid(t, 1.0);
    CHECK_THAT(rms(std::span<const Real>(x)), WithinRel(1.0 / std::sqrt(2.0), 1e-3));
}

TEST_CASE("snr: 65536 samples, sine + weak white noise → SNR > 40 dB", "[stress][metrics]")
{
    constexpr double fs = 65536.0;
    constexpr double f0 = 1000.0;
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, f0);
    auto noise = white_noise(LARGE_N, 0.001, 99u);
    for (std::size_t i = 0; i < LARGE_N; ++i) x[i] += noise[i];
    double s = snr(std::span<const Real>(x), f0, fs);
    CHECK(s > 40.0);
}

TEST_CASE("thd: 65536 sine, very low THD", "[stress][metrics]")
{
    constexpr double fs = 65536.0;
    constexpr double f0 = 1000.0;
    auto t = linspace(0.0, 1.0, LARGE_N, false);
    auto x = sinusoid(t, f0);
    double thd_db = thd(std::span<const Real>(x), f0, fs);
    CHECK(thd_db < -60.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Repeated-call stability
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter: 100 repeated calls return identical results", "[stress][filter]")
{
    auto ref = butter(8, 0.1, FilterType::Lowpass);
    for (int i = 0; i < 100; ++i) {
        auto sos = butter(8, 0.1, FilterType::Lowpass);
        REQUIRE(sos.size() == ref.size());
        for (std::size_t k = 0; k < sos.size(); ++k)
            for (std::size_t j = 0; j < 6; ++j)
                CHECK_THAT(sos[k][j], WithinAbs(ref[k][j], 1e-14));
    }
}

TEST_CASE("rfft: 50 calls with same input return identical results", "[stress][fft]")
{
    std::vector<Real> x(256);
    for (std::size_t i = 0; i < 256; ++i)
        x[i] = std::cos(2.0 * std::numbers::pi * 10.0 * i / 256.0);
    auto ref = rfft(std::span<const Real>(x));
    for (int i = 0; i < 50; ++i) {
        auto r = rfft(std::span<const Real>(x));
        REQUIRE(r.size() == ref.size());
        for (std::size_t k = 0; k < r.size(); ++k) {
            CHECK_THAT(r[k].real(), WithinAbs(ref[k].real(), 1e-14));
            CHECK_THAT(r[k].imag(), WithinAbs(ref[k].imag(), 1e-14));
        }
    }
}

TEST_CASE("white_noise: same seed produces identical output on repeated calls", "[stress][generate]")
{
    auto ref = white_noise(1000, 1.0, 777u);
    for (int i = 0; i < 10; ++i) {
        auto x = white_noise(1000, 1.0, 777u);
        REQUIRE(x.size() == ref.size());
        for (std::size_t k = 0; k < x.size(); ++k)
            CHECK_THAT(x[k], WithinAbs(ref[k], 0.0));
    }
}

TEST_CASE("white_noise: different seeds produce different output", "[stress][generate]")
{
    auto x1 = white_noise(1000, 1.0, 1u);
    auto x2 = white_noise(1000, 1.0, 2u);
    double diff = 0.0;
    for (std::size_t i = 0; i < 1000; ++i) diff += std::abs(x1[i] - x2[i]);
    CHECK(diff > 1.0);
}

TEST_CASE("sosfilt: 20 repeated calls return identical results", "[stress][filter]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    std::vector<Real> x(512);
    for (std::size_t i = 0; i < 512; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 0.05 * static_cast<double>(i));
    auto ref = sosfilt(sos, x);
    for (int i = 0; i < 20; ++i) {
        auto y = sosfilt(sos, x);
        REQUIRE(y.size() == ref.size());
        for (std::size_t k = 0; k < y.size(); ++k)
            CHECK_THAT(y[k], WithinAbs(ref[k], 0.0));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Small data edge cases (1–4 samples)
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("sinusoid: single-sample time array", "[stress][generate]")
{
    std::vector<Real> t = {0.0};
    auto y = sinusoid(t, 100.0, 1.0, 0.0);
    REQUIRE(y.size() == 1);
    CHECK_THAT(y[0], WithinAbs(0.0, 1e-12));   // sin(0) = 0
}

TEST_CASE("sinusoid: two-sample time array", "[stress][generate]")
{
    std::vector<Real> t = {0.0, 0.25};
    auto y = sinusoid(t, 1.0);
    REQUIRE(y.size() == 2);
    CHECK_THAT(y[0], WithinAbs(0.0, 1e-12));
    CHECK_THAT(y[1], WithinAbs(1.0, 1e-9));   // sin(2π * 0.25) = 1
}

TEST_CASE("chirp: single-sample produces finite value", "[stress][generate]")
{
    std::vector<Real> t = {0.0};
    auto y = chirp(t, 10.0, 100.0, 1.0);
    REQUIRE(y.size() == 1);
    CHECK(std::isfinite(y[0]));
    CHECK_THAT(y[0], WithinAbs(1.0, 1e-12));  // cos(0) = 1 at t=0 with phi=0
}

TEST_CASE("square_wave: single-sample", "[stress][generate]")
{
    std::vector<Real> t = {0.0};
    auto y = square_wave(t, 1.0, 0.5);
    REQUIRE(y.size() == 1);
    CHECK(y[0] == 1.0);  // t=0 is in duty half (0 < 0.5)
}

TEST_CASE("sawtooth_wave: single-sample", "[stress][generate]")
{
    std::vector<Real> t = {0.0};
    auto y = sawtooth_wave(t, 1.0, 1.0);
    REQUIRE(y.size() == 1);
    CHECK(std::isfinite(y[0]));
}

TEST_CASE("unit_impulse: n=1 at idx=0", "[stress][generate]")
{
    auto x = unit_impulse(1, 0);
    REQUIRE(x.size() == 1);
    CHECK_THAT(x[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("convolve: single-element inputs, full mode", "[stress][correlate]")
{
    std::vector<Real> x = {3.0};
    std::vector<Real> y = {7.0};
    auto c = convolve(x, y);
    REQUIRE(c.size() == 1);
    CHECK_THAT(c[0], WithinAbs(21.0, 1e-12));
}

TEST_CASE("convolve: two-element inputs produce correct 3-element output", "[stress][correlate]")
{
    std::vector<Real> x = {1.0, 2.0};
    std::vector<Real> y = {3.0, 4.0};
    auto c = convolve(x, y);
    REQUIRE(c.size() == 3);
    CHECK_THAT(c[0], WithinAbs(3.0,  1e-12));   // 1*3
    CHECK_THAT(c[1], WithinAbs(10.0, 1e-12));   // 1*4 + 2*3
    CHECK_THAT(c[2], WithinAbs(8.0,  1e-12));   // 2*4
}

TEST_CASE("convolve: 4-element inputs, Full / Same / Valid modes", "[stress][correlate]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> k = {1.0, 1.0};
    auto full  = convolve(x, k, ConvolveMode::Full);
    auto same  = convolve(x, k, ConvolveMode::Same);
    auto valid = convolve(x, k, ConvolveMode::Valid);
    CHECK(full.size()  == 5);
    CHECK(same.size()  == 4);
    CHECK(valid.size() == 3);
}

TEST_CASE("rfft: single-element input", "[stress][fft]")
{
    std::vector<Real> x = {5.0};
    auto X = rfft(std::span<const Real>(x));
    REQUIRE(X.size() == 1);
    CHECK_THAT(X[0].real(), WithinAbs(5.0, 1e-12));
    CHECK_THAT(X[0].imag(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("rfft: two-element input, DC=0, Nyquist=2", "[stress][fft]")
{
    std::vector<Real> x = {1.0, -1.0};
    auto X = rfft(std::span<const Real>(x));
    REQUIRE(X.size() == 2);
    CHECK_THAT(X[0].real(), WithinAbs(0.0, 1e-12));   // DC = 1 + (-1) = 0
    CHECK_THAT(X[1].real(), WithinAbs(2.0, 1e-12));   // Nyquist = 1 - (-1) = 2
}

TEST_CASE("rfft: four-element input, known spectrum", "[stress][fft]")
{
    // x = [1, 0, -1, 0] → X = [0, 2, 0]
    std::vector<Real> x = {1.0, 0.0, -1.0, 0.0};
    auto X = rfft(std::span<const Real>(x));
    REQUIRE(X.size() == 3);
    CHECK_THAT(X[0].real(), WithinAbs(0.0, 1e-12));
    CHECK_THAT(std::abs(X[1]), WithinAbs(2.0, 1e-12));
    CHECK_THAT(X[2].real(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("find_peaks: exactly 3 samples, peak at centre", "[stress][peaks]")
{
    std::vector<Real> x = {0.0, 1.0, 0.0};
    auto r = find_peaks(x);
    REQUIRE(r.indices.size() == 1);
    CHECK(r.indices[0] == 1);
    CHECK_THAT(r.heights[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("find_peaks: monotone 3-sample input, no peaks", "[stress][peaks]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0};
    auto r = find_peaks(x);
    CHECK(r.indices.empty());
}

TEST_CASE("find_peaks: 2 samples, below minimum of 3 for peak detection", "[stress][peaks]")
{
    std::vector<Real> x = {0.0, 1.0};
    auto r = find_peaks(x);
    CHECK(r.indices.empty());
}

TEST_CASE("rms: single-element signal", "[stress][metrics]")
{
    std::vector<Real> x = {3.0};
    CHECK_THAT(rms(std::span<const Real>(x)), WithinAbs(3.0, 1e-12));
}

TEST_CASE("rms: two-element signal", "[stress][metrics]")
{
    std::vector<Real> x = {3.0, 4.0};
    // sqrt((9 + 16)/2) = sqrt(12.5)
    CHECK_THAT(rms(std::span<const Real>(x)), WithinAbs(std::sqrt(12.5), 1e-12));
}

TEST_CASE("sosfilt: single-sample signal returns single finite value", "[stress][filter]")
{
    auto sos = butter(2, 0.1, FilterType::Lowpass);
    std::vector<Real> x = {1.0};
    auto y = sosfilt(sos, x);
    REQUIRE(y.size() == 1);
    CHECK(std::isfinite(y[0]));
}

TEST_CASE("sosfilt: empty signal returns empty output", "[stress][filter]")
{
    auto sos = butter(2, 0.1, FilterType::Lowpass);
    std::vector<Real> x;
    auto y = sosfilt(sos, x);
    CHECK(y.empty());
}

TEST_CASE("lfilter: empty signal returns empty output", "[stress][filter]")
{
    std::vector<Real> b = {1.0};
    std::vector<Real> a = {1.0};
    std::vector<Real> x;
    auto y = lfilter(b, a, x);
    CHECK(y.empty());
}

TEST_CASE("linspace: n=2 returns exact endpoints", "[stress][generate]")
{
    auto t = linspace(3.0, 7.0, 2);
    REQUIRE(t.size() == 2);
    CHECK_THAT(t[0], WithinAbs(3.0, 1e-12));
    CHECK_THAT(t[1], WithinAbs(7.0, 1e-12));
}

TEST_CASE("linspace: large n, endpoints and monotonicity", "[stress][generate]")
{
    constexpr std::size_t N = 1000001;
    auto t = linspace(0.0, 1.0, N);
    REQUIRE(t.size() == N);
    CHECK_THAT(t.front(), WithinAbs(0.0, 1e-12));
    CHECK_THAT(t.back(),  WithinAbs(1.0, 1e-12));
    for (std::size_t i = 1; i < N; ++i)
        CHECK(t[i] > t[i-1]);
}

TEST_CASE("arange: large range with unit step", "[stress][generate]")
{
    auto t = arange(0.0, 10000.0, 1.0);
    REQUIRE(t.size() == 10000);
    CHECK_THAT(t.front(), WithinAbs(0.0,    1e-12));
    CHECK_THAT(t.back(),  WithinAbs(9999.0, 1e-9));
}

TEST_CASE("make_window: all window types, n=256, values finite and in plausible range", "[stress][spectral]")
{
    for (auto type : {Window::Rectangular, Window::Hann, Window::Hamming,
                      Window::Blackman, Window::BlackmanHarris, Window::FlatTop}) {
        auto w = make_window(type, 256);
        REQUIRE(w.size() == 256);
        for (auto v : w)
            CHECK(std::isfinite(v));
        // Hann, Blackman, BlackmanHarris taper to zero at endpoints; Hamming to ~0.08
        if (type == Window::Hann || type == Window::Blackman || type == Window::BlackmanHarris) {
            CHECK_THAT(w.front(), WithinAbs(0.0, 1e-3));
            CHECK_THAT(w.back(),  WithinAbs(0.0, 1e-3));
        }
    }
}

TEST_CASE("freqz: 1000-point response of 4th-order LP has correct DC/Nyquist behaviour", "[stress][filter]")
{
    auto sos = butter(4, 0.1, FilterType::Lowpass);
    auto [freqs, H] = freqz(sos, 1000);
    REQUIRE(freqs.size() == 1000);
    REQUIRE(H.size() == 1000);
    // DC gain ≈ 1 for lowpass
    CHECK_THAT(std::abs(H[0]), WithinAbs(1.0, 1e-6));
    // Gain near Nyquist should be near zero for LP
    CHECK(std::abs(H.back()) < 0.01);
}

// ════════════════════════════════════════════════════════════════════════════
// High-order filter stress tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter order=12: design and apply to 65536-sample signal", "[stress][filter]")
{
    auto sos = butter(12, 0.1, FilterType::Lowpass);
    REQUIRE(sos.size() == 6);  // 12th-order = 6 biquad sections
    std::vector<Real> x(LARGE_N, 0.0);
    x[0] = 1.0;   // impulse
    auto y = sosfilt(sos, std::span<const Real>(x));
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y)
        CHECK(std::isfinite(v));
    // Impulse response must decay to near zero for a stable LP
    CHECK(std::abs(y.back()) < 1e-6);
}

TEST_CASE("butter order=16: design produces 8 sections and stable response", "[stress][filter]")
{
    auto sos = butter(16, 0.2, FilterType::Lowpass);
    REQUIRE(sos.size() == 8);
    // DC gain should be 1
    auto [freqs, H] = freqz(sos, 512);
    CHECK_THAT(std::abs(H[0]), WithinAbs(1.0, 1e-4));
    CHECK(std::abs(H.back()) < 0.001);
}

TEST_CASE("firwin order=255: design and apply FIR LP at large scale", "[stress][filter]")
{
    auto h = firwin(255, 0.1, Window::Hann);
    REQUIRE(h.size() == 255);
    // Apply to large signal using lfilter with b=h, a=[1]
    std::vector<Real> x(LARGE_N, 0.0);
    for (std::size_t i = 0; i < LARGE_N; i += 100) x[i] = 1.0;  // impulse train
    std::vector<Real> a = {1.0};
    auto y = lfilter(std::span<const Real>(h), std::span<const Real>(a),
                     std::span<const Real>(x));
    REQUIRE(y.size() == LARGE_N);
    for (auto v : y)
        CHECK(std::isfinite(v));
}

// ════════════════════════════════════════════════════════════════════════════
// Large convolution and correlation
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("convolve: two 500-sample rect pulses, triangular output", "[stress][correlate]")
{
    // convolve uses O(N·M) direct summation — keep N manageable
    constexpr std::size_t N = 500;
    std::vector<Real> x(N, 1.0);
    std::vector<Real> h(N, 1.0);
    auto y = convolve(std::span<const Real>(x), std::span<const Real>(h));
    REQUIRE(y.size() == 2*N - 1);
    // Convolution of two rect pulses of length N is a triangle: peak at index N-1 = N
    CHECK_THAT(y[N-1], WithinAbs(static_cast<double>(N), 1e-6));
    for (auto v : y)
        CHECK(std::isfinite(v));
}

TEST_CASE("correlate: auto-correlation of 500-sample cosine peaks at zero lag", "[stress][correlate]")
{
    // O(N²) implementation — keep N manageable
    constexpr std::size_t N = 500;
    auto t = linspace(0.0, static_cast<double>(N-1) / 1000.0, N);
    auto x = sinusoid(std::span<const Real>(t), 10.0, 1.0, std::numbers::pi / 2.0);
    auto r = correlate(std::span<const Real>(x), std::span<const Real>(x));
    REQUIRE(r.size() == 2*N - 1);
    // Peak should be at lag 0 (centre of the output)
    std::size_t centre = N - 1;
    double peak = r[centre];
    for (auto v : r)
        CHECK(v <= peak + 1e-6);
}

TEST_CASE("convolve Valid mode: moving average of 2000-sample signal", "[stress][correlate]")
{
    constexpr std::size_t Nx = 2000, Nk = 50;
    std::vector<Real> x(Nx, 1.0), k(Nk, 1.0 / static_cast<double>(Nk));
    auto y = convolve(std::span<const Real>(x), std::span<const Real>(k),
                      ConvolveMode::Valid);
    REQUIRE(y.size() == Nx - Nk + 1);
    for (auto v : y)
        CHECK_THAT(v, WithinAbs(1.0, 1e-10));
}

// ════════════════════════════════════════════════════════════════════════════
// Metrics stress tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("rms: HUGE_N signal of ones is exactly 1.0", "[stress][metrics]")
{
    std::vector<Real> x(HUGE_N, 1.0);
    double r = rms(std::span<const Real>(x));
    CHECK_THAT(r, WithinAbs(1.0, 1e-10));
}

TEST_CASE("snr(sig,noise): LARGE_N vectors at 40 dB is accurate", "[stress][metrics]")
{
    // Signal power = 1, noise power = 1e-4 → SNR = 40 dB
    constexpr std::size_t N = LARGE_N;
    std::vector<Real> sig(N, 1.0);
    std::vector<Real> noise(N, 0.01);  // RMS = 0.01, power = 1e-4
    double r = snr(std::span<const Real>(sig), std::span<const Real>(noise));
    CHECK_THAT(r, WithinAbs(40.0, 0.01));
}

TEST_CASE("thd: large pure sine has very low THD", "[stress][metrics]")
{
    // Use N=8192 (power-of-2) and fs=8192 so f0=1000 Hz lands on bin 1000 exactly
    constexpr double fs = 8192.0, f0 = 1000.0;
    constexpr std::size_t N = 8192;
    auto t = linspace(0.0, static_cast<double>(N-1) / fs, N);
    auto x = sinusoid(std::span<const Real>(t), f0, 1.0);
    double h = thd(std::span<const Real>(x), f0, fs);
    CHECK(h < -60.0);   // < -60 dB
}

TEST_CASE("thd: repeated calls give identical results", "[stress][metrics]")
{
    constexpr double fs = 8192.0, f0 = 440.0;
    constexpr std::size_t N = 8192;
    auto t = linspace(0.0, static_cast<double>(N-1) / fs, N);
    auto x = sinusoid(std::span<const Real>(t), f0, 1.0);
    double first = thd(std::span<const Real>(x), f0, fs);
    for (int i = 0; i < 20; ++i) {
        double r = thd(std::span<const Real>(x), f0, fs);
        CHECK_THAT(r, WithinAbs(first, 1e-10));
    }
}

TEST_CASE("sinad: sine plus tiny 2nd harmonic has high SINAD (> 50 dB)", "[stress][metrics]")
{
    // sinad requires some noise/distortion in the denominator — add a tiny 2nd harmonic
    constexpr double fs = 8192.0, f0 = 1000.0;
    constexpr std::size_t N = 8192;
    auto t = linspace(0.0, static_cast<double>(N-1) / fs, N);
    std::vector<Real> x(N);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * f0      * t[i])
             + 1e-4 * std::sin(2.0 * std::numbers::pi * 2*f0 * t[i]);
    double s = sinad(std::span<const Real>(x), f0, fs);
    CHECK(s > 50.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Peak detection stress tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("find_peaks: LARGE_N signal with 1000 sinusoidal cycles, peaks all found", "[stress][peaks]")
{
    constexpr std::size_t N = LARGE_N;
    constexpr double fs = 65536.0, f0 = 1000.0;   // 1000 Hz → 1000 cycles
    auto t = linspace(0.0, static_cast<double>(N-1) / fs, N);
    auto x = sinusoid(std::span<const Real>(t), f0, 1.0);
    auto result = find_peaks(std::span<const Real>(x), PeakOptions{.height = 0.5});
    // ~1000 peaks per second × 1 second
    CHECK(result.indices.size() >= 900);
    CHECK(result.indices.size() <= 1100);
    for (auto idx : result.indices) {
        CHECK(idx < N);
        CHECK(x[idx] >= 0.5);
    }
}

TEST_CASE("find_peaks: repeated calls on same signal produce identical results", "[stress][peaks]")
{
    constexpr std::size_t N = 1024;
    auto t = linspace(0.0, 1.0, N, false);
    auto x = sinusoid(std::span<const Real>(t), 10.0, 1.0);
    PeakOptions opts{.height = 0.5};
    auto r0 = find_peaks(std::span<const Real>(x), opts);
    for (int i = 0; i < 50; ++i) {
        auto ri = find_peaks(std::span<const Real>(x), opts);
        REQUIRE(ri.indices.size() == r0.indices.size());
        for (std::size_t j = 0; j < r0.indices.size(); ++j)
            CHECK(ri.indices[j] == r0.indices[j]);
    }
}

TEST_CASE("peak_prominences: large signal with 100 equal-height peaks", "[stress][peaks]")
{
    constexpr std::size_t N = 10000;
    std::vector<Real> signal(N, 0.0);
    std::vector<std::size_t> peaks;
    // Place 100 equal peaks spaced 100 samples apart
    for (std::size_t i = 50; i < N; i += 100) {
        signal[i] = 5.0;
        peaks.push_back(i);
    }
    auto prom = peak_prominences(std::span<const Real>(signal),
                                 std::span<const std::size_t>(peaks));
    REQUIRE(prom.size() == peaks.size());
    for (auto p : prom) {
        CHECK(std::isfinite(p));
        CHECK(p > 0.0);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Welch/STFT with various options
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("welch: different window types all produce valid PSD", "[stress][spectral]")
{
    constexpr double fs = 1000.0;
    std::vector<Real> x(4096, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 100.0 * i / fs);

    for (auto win : {Window::Hann, Window::Hamming, Window::Blackman,
                     Window::BlackmanHarris, Window::FlatTop, Window::Rectangular}) {
        WelchOptions opts{.window = win};
        PSDResult r;
        REQUIRE_NOTHROW(r = welch(std::span<const Real>(x), fs, opts));
        REQUIRE(!r.freqs.empty());
        REQUIRE(r.psd.size() == r.freqs.size());
        for (auto v : r.psd) {
            CHECK(std::isfinite(v));
            CHECK(v >= 0.0);
        }
    }
}

TEST_CASE("welch: explicit noverlap=0 (non-overlapping frames)", "[stress][spectral]")
{
    constexpr double fs = 1000.0;
    std::vector<Real> x(4096, 1.0);
    WelchOptions opts{.noverlap = 0};
    PSDResult r;
    REQUIRE_NOTHROW(r = welch(std::span<const Real>(x), fs, opts));
    REQUIRE(!r.psd.empty());
    for (auto v : r.psd)
        CHECK(std::isfinite(v));
}

TEST_CASE("welch: large nperseg approaches periodogram", "[stress][spectral]")
{
    constexpr std::size_t N = 4096;
    constexpr double fs = 1000.0;
    std::vector<Real> x(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * std::numbers::pi * 50.0 * i / fs);

    // nperseg == N: single frame = periodogram approximation
    WelchOptions opts{.nperseg = N};
    PSDResult r;
    REQUIRE_NOTHROW(r = welch(std::span<const Real>(x), fs, opts));
    // Peak should be near 50 Hz
    std::size_t peak_bin = 0;
    for (std::size_t k = 1; k < r.psd.size(); ++k)
        if (r.psd[k] > r.psd[peak_bin]) peak_bin = k;
    CHECK_THAT(r.freqs[peak_bin], WithinAbs(50.0, 2.0));
}

TEST_CASE("stft: explicit nfft larger than nperseg zero-pads correctly", "[stress][spectral]")
{
    constexpr std::size_t N = 2048;
    constexpr double fs = 1000.0;
    std::vector<Real> x(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = std::cos(2.0 * std::numbers::pi * 100.0 * i / fs);

    STFTOptions opts{.nperseg = 128, .nfft = 256};
    STFTResult r;
    REQUIRE_NOTHROW(r = stft(std::span<const Real>(x), fs, opts));
    // With nfft=256, one-sided: 129 frequency bins
    REQUIRE(r.freqs.size() == 129);
    REQUIRE(!r.times.empty());
    // Frequency axis should span 0 to fs/2
    CHECK_THAT(r.freqs[0],   WithinAbs(0.0,      1e-6));
    CHECK_THAT(r.freqs.back(), WithinAbs(fs / 2.0, 1.0));
}

// ════════════════════════════════════════════════════════════════════════════
// Very large FFT stress
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("rfft: N=65536 (2^16) impulse → all bins equal 1", "[stress][fft]")
{
    // LARGE_N = 65536 (power-of-2 → O(N log N))
    std::vector<Real> x(LARGE_N, 0.0);
    x[0] = 1.0;   // impulse
    std::vector<Complex> X;
    REQUIRE_NOTHROW(X = rfft(std::span<const Real>(x)));
    REQUIRE(X.size() == LARGE_N / 2 + 1);
    // DFT of impulse at 0: |X[k]| = 1 for all k
    for (auto& c : X)
        CHECK_THAT(std::abs(c), WithinAbs(1.0, 1e-10));
}

TEST_CASE("fft: 32768-point complex roundtrip is within precision", "[stress][fft]")
{
    constexpr std::size_t N = 32768;  // 2^15 — fast but not so large as to time out
    std::vector<Complex> x(N, {0.0, 0.0});
    for (std::size_t i = 0; i < N; ++i)
        x[i] = {std::sin(2.0 * std::numbers::pi * i / N), 0.0};
    auto orig = x;
    auto X = fft(std::span<const Complex>(x));
    auto y  = ifft(std::span<const Complex>(X));
    REQUIRE(y.size() == N);
    for (std::size_t i = 0; i < N; ++i)
        CHECK_THAT(y[i].real(), WithinAbs(orig[i].real(), 1e-9));
}
