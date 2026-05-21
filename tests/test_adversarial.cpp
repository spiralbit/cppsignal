// ─────────────────────────────────────────────────────────────────────────────
// test_adversarial.cpp — adversarial and invariant tests for cppsignal
//
// Each test probes an invariant that MUST hold, a boundary that could silently
// break, or a bug we discovered during analysis.  Tests marked [BUG] will fail
// against the current implementation.
// ─────────────────────────────────────────────────────────────────────────────

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cps/cps.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ═════════════════════════════════════════════════════════════════════════════
// WINDOW FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

// BUG: Tukey window with small n and alpha=0.5 computes taper = floor(alpha*(n-1)/2).
// For n ≤ 4 this rounds to 0, and the formula cos(π * i / taper) divides by zero
// (double arithmetic, so the result is NaN, not a crash). These tests FAIL against
// the current implementation — they document the expected correct behaviour.

TEST_CASE("make_window: Tukey n=2 must not produce NaN values [BUG]") {
    auto w = cps::make_window(cps::Window::Tukey, 2, 0.5);
    REQUIRE(w.size() == 2);
    CHECK(!std::isnan(w[0]));
    CHECK(!std::isnan(w[1]));
}

TEST_CASE("make_window: Tukey n=3 must not produce NaN values [BUG]") {
    auto w = cps::make_window(cps::Window::Tukey, 3, 0.5);
    REQUIRE(w.size() == 3);
    for (auto v : w) CHECK(!std::isnan(v));
}

TEST_CASE("make_window: Tukey n=4 must not produce NaN values [BUG]") {
    auto w = cps::make_window(cps::Window::Tukey, 4, 0.5);
    REQUIRE(w.size() == 4);
    for (auto v : w) CHECK(!std::isnan(v));
}

TEST_CASE("make_window: Tukey n=5 has non-NaN values (taper becomes 1)") {
    // n=5, alpha=0.5 → taper = floor(0.5*4/2) = 1 (no division by zero)
    auto w = cps::make_window(cps::Window::Tukey, 5, 0.5);
    REQUIRE(w.size() == 5);
    for (auto v : w) CHECK(!std::isnan(v));
}

TEST_CASE("make_window: Hann n=2 produces all-zero window") {
    // w[i] = 0.5*(1 - cos(2πi/M)) with M=1:  w[0]=0, w[1]=0
    auto w = cps::make_window(cps::Window::Hann, 2);
    REQUIRE(w.size() == 2);
    CHECK_THAT(w[0], WithinAbs(0.0, 1e-12));
    CHECK_THAT(w[1], WithinAbs(0.0, 1e-12));
}

TEST_CASE("firwin: Hann window of size 2 causes normalization to fail") {
    // Hann n=2 → [0,0] → sum=0 → NumericalError in firwin normalisation
    CHECK_THROWS_AS(cps::firwin(2, 0.1, cps::Window::Hann), cps::NumericalError);
}

TEST_CASE("make_window: n=1 always returns {1.0} for all window types") {
    for (auto type : {cps::Window::Rectangular, cps::Window::Hann,
                      cps::Window::Hamming, cps::Window::Blackman,
                      cps::Window::BlackmanHarris, cps::Window::FlatTop}) {
        auto w = cps::make_window(type, 1);
        REQUIRE(w.size() == 1);
        CHECK_THAT(w[0], WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("make_window: Kaiser beta=0 is equivalent to Rectangular") {
    const std::size_t N = 32;
    auto rect   = cps::make_window(cps::Window::Rectangular, N);
    auto kaiser = cps::make_window(cps::Window::Kaiser, N, 0.0);
    REQUIRE(kaiser.size() == rect.size());
    for (std::size_t i = 0; i < N; ++i)
        CHECK_THAT(kaiser[i], WithinAbs(rect[i], 1e-6));
}

TEST_CASE("make_window: all standard windows stay within [0, 1] for large n") {
    const std::size_t N = 64;
    for (auto type : {cps::Window::Hann, cps::Window::Hamming,
                      cps::Window::Blackman, cps::Window::BlackmanHarris,
                      cps::Window::Kaiser}) {
        auto w = cps::make_window(type, N, 5.0);
        for (auto v : w) {
            CHECK(v >= -1e-9);   // no negative values (FlatTop excluded)
            CHECK(v <= 1.0 + 1e-9);
        }
    }
}

TEST_CASE("make_window: symmetric windows satisfy w[i] == w[n-1-i]") {
    const std::size_t N = 17;  // odd length
    for (auto type : {cps::Window::Hann, cps::Window::Hamming,
                      cps::Window::Blackman, cps::Window::BlackmanHarris}) {
        auto w = cps::make_window(type, N);
        for (std::size_t i = 0; i < N / 2; ++i)
            CHECK_THAT(w[i], WithinAbs(w[N - 1 - i], 1e-12));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// FILTER DESIGN INVARIANTS
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter: order=N LP produces ceil(N/2) SOS sections") {
    for (int order : {1, 2, 3, 4, 5, 6, 7, 8}) {
        auto sos = cps::butter(order, 0.3, cps::FilterType::Lowpass);
        std::size_t expected = static_cast<std::size_t>((order + 1) / 2);
        CHECK(sos.size() == expected);
    }
}

TEST_CASE("butter: every SOS row has a0 == 1.0") {
    auto sos = cps::butter(5, 0.25, cps::FilterType::Lowpass);
    for (const auto& row : sos)
        CHECK_THAT(row[3], WithinAbs(1.0, 1e-12));
}

TEST_CASE("butter LP: DC gain (z=1) is 1.0 for all orders") {
    for (int order : {1, 2, 3, 4, 6, 8}) {
        auto sos = cps::butter(order, 0.3, cps::FilterType::Lowpass);
        double gain = 1.0;
        for (const auto& row : sos) {
            double num = row[0] + row[1] + row[2];
            double den = row[3] + row[4] + row[5];
            gain *= num / den;
        }
        CHECK_THAT(gain, WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("butter HP: Nyquist gain (z=-1) is 1.0 for all orders") {
    for (int order : {1, 2, 3, 4, 6}) {
        auto sos = cps::butter(order, 0.3, cps::FilterType::Highpass);
        double gain = 1.0;
        for (const auto& row : sos) {
            double num = row[0] - row[1] + row[2];
            double den = row[3] - row[4] + row[5];
            gain *= num / den;
        }
        CHECK_THAT(gain, WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("butter LP: gain at cutoff is exactly -3 dB (1/sqrt(2))") {
    const double Wn    = 0.3;
    const double omega = std::numbers::pi * Wn;
    auto sos = cps::butter(4, Wn, cps::FilterType::Lowpass);

    cps::Complex H{1.0, 0.0};
    for (const auto& row : sos) {
        cps::Complex z    = std::exp(cps::Complex{0.0, omega});
        cps::Complex zinv = 1.0 / z;
        cps::Complex num  = row[0] + row[1] * zinv + row[2] * zinv * zinv;
        cps::Complex den  = row[3] + row[4] * zinv + row[5] * zinv * zinv;
        H *= num / den;
    }
    CHECK_THAT(std::abs(H), WithinAbs(1.0 / std::sqrt(2.0), 1e-6));
}

TEST_CASE("butter HP: gain at cutoff is exactly -3 dB (1/sqrt(2))") {
    const double Wn    = 0.3;
    const double omega = std::numbers::pi * Wn;
    auto sos = cps::butter(4, Wn, cps::FilterType::Highpass);

    cps::Complex H{1.0, 0.0};
    for (const auto& row : sos) {
        cps::Complex z    = std::exp(cps::Complex{0.0, omega});
        cps::Complex zinv = 1.0 / z;
        cps::Complex num  = row[0] + row[1] * zinv + row[2] * zinv * zinv;
        cps::Complex den  = row[3] + row[4] * zinv + row[5] * zinv * zinv;
        H *= num / den;
    }
    CHECK_THAT(std::abs(H), WithinAbs(1.0 / std::sqrt(2.0), 1e-6));
}

TEST_CASE("butter LP: stopband attenuation grows with order") {
    // At 2x the cutoff frequency, higher-order filter should attenuate more
    const double Wn      = 0.2;
    const double omega2x = std::numbers::pi * 2.0 * Wn;

    auto gain_at_2x = [&](int order) {
        auto sos = cps::butter(order, Wn, cps::FilterType::Lowpass);
        cps::Complex H{1.0, 0.0};
        for (const auto& row : sos) {
            cps::Complex z    = std::exp(cps::Complex{0.0, omega2x});
            cps::Complex zinv = 1.0 / z;
            cps::Complex num  = row[0] + row[1] * zinv + row[2] * zinv * zinv;
            cps::Complex den  = row[3] + row[4] * zinv + row[5] * zinv * zinv;
            H *= num / den;
        }
        return std::abs(H);
    };

    CHECK(gain_at_2x(2) > gain_at_2x(4));
    CHECK(gain_at_2x(4) > gain_at_2x(6));
}

TEST_CASE("firwin: LP coefficients sum to 1.0 (unity DC gain)") {
    auto h = cps::firwin(51, 0.3, cps::Window::Hamming);
    double sum = std::accumulate(h.begin(), h.end(), 0.0);
    CHECK_THAT(sum, WithinAbs(1.0, 1e-6));
}

TEST_CASE("firwin: HP coefficients sum to ~0.0 (zero DC gain)") {
    auto h = cps::firwin(51, 0.3, cps::Window::Hamming, cps::FilterType::Highpass);
    double sum = std::accumulate(h.begin(), h.end(), 0.0);
    CHECK_THAT(sum, WithinAbs(0.0, 1e-6));
}

TEST_CASE("firwin: LP has perfect linear phase — h[i] == h[N-1-i]") {
    int numtaps = 51;
    auto h = cps::firwin(numtaps, 0.3, cps::Window::Hann);
    REQUIRE(static_cast<int>(h.size()) == numtaps);
    for (int i = 0; i < numtaps / 2; ++i)
        CHECK_THAT(h[i], WithinRel(h[numtaps - 1 - i], 1e-10));
}

TEST_CASE("firwin: HP has perfect linear phase — h[i] == h[N-1-i]") {
    int numtaps = 51;
    auto h = cps::firwin(numtaps, 0.3, cps::Window::Hamming, cps::FilterType::Highpass);
    REQUIRE(static_cast<int>(h.size()) == numtaps);
    for (int i = 0; i < numtaps / 2; ++i)
        CHECK_THAT(h[i], WithinRel(h[numtaps - 1 - i], 1e-10));
}

// ═════════════════════════════════════════════════════════════════════════════
// SOSFILT / LFILTER INVARIANTS
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("sosfilt: butter LP step response converges to 1.0") {
    auto sos = cps::butter(4, 0.1, cps::FilterType::Lowpass);
    const int N = 600;
    std::vector<cps::Real> step(N, 1.0);
    auto out = cps::sosfilt(sos, step);
    CHECK_THAT(out[N - 1], WithinAbs(1.0, 1e-4));
}

TEST_CASE("sosfilt: butter HP step response converges to 0.0") {
    // HP blocks DC — steady-state output for constant input must be 0
    auto sos = cps::butter(4, 0.1, cps::FilterType::Highpass);
    const int N = 600;
    std::vector<cps::Real> step(N, 1.0);
    auto out = cps::sosfilt(sos, step);
    CHECK_THAT(out[N - 1], WithinAbs(0.0, 1e-4));
}

TEST_CASE("lfilter: identity [1]/[1] passes signal unchanged") {
    std::vector<cps::Real> sig = {1.0, -2.0, 3.0, 0.5, -1.5, 4.0};
    auto out = cps::lfilter(std::vector<cps::Real>{1.0}, std::vector<cps::Real>{1.0}, sig);
    REQUIRE(out.size() == sig.size());
    for (std::size_t i = 0; i < sig.size(); ++i)
        CHECK_THAT(out[i], WithinAbs(sig[i], 1e-12));
}

TEST_CASE("lfilter: single-pole IIR impulse response is a geometric sequence") {
    // H(z) = 1 / (1 - 0.9 z^{-1})  →  h[n] = 0.9^n
    std::vector<cps::Real> impulse(8, 0.0);
    impulse[0] = 1.0;
    auto out = cps::lfilter(std::vector<cps::Real>{1.0}, std::vector<cps::Real>{1.0, -0.9}, impulse);
    REQUIRE(out.size() == 8);
    for (int i = 0; i < 8; ++i)
        CHECK_THAT(out[i], WithinRel(std::pow(0.9, i), 1e-9));
}

TEST_CASE("lfilter: FIR moving average of constant signal returns constant") {
    const double val = 3.14;
    std::vector<cps::Real> sig(100, val);
    // 5-tap uniform average → output = val at steady state
    std::vector<cps::Real> h(5, 0.2);
    auto out = cps::lfilter(h, std::vector<cps::Real>{1.0}, sig);
    REQUIRE(out.size() == sig.size());
    // After the transient (first 5 samples), output = val
    for (std::size_t i = 5; i < out.size(); ++i)
        CHECK_THAT(out[i], WithinAbs(val, 1e-10));
}

// ═════════════════════════════════════════════════════════════════════════════
// CONVOLUTION / CORRELATION INVARIANTS
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("convolve: Full mode is commutative") {
    std::vector<cps::Real> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<cps::Real> y = {1.0, -1.0, 2.0};
    auto xy = cps::convolve(x, y);
    auto yx = cps::convolve(y, x);
    REQUIRE(xy.size() == yx.size());
    for (std::size_t i = 0; i < xy.size(); ++i)
        CHECK_THAT(xy[i], WithinAbs(yx[i], 1e-12));
}

TEST_CASE("convolve: convolution with unit impulse is identity") {
    std::vector<cps::Real> x     = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<cps::Real> delta = {1.0};
    auto out = cps::convolve(x, delta);
    REQUIRE(out.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK_THAT(out[i], WithinAbs(x[i], 1e-12));
}

TEST_CASE("convolve: convolution with delayed impulse shifts signal") {
    std::vector<cps::Real> x     = {1.0, 2.0, 3.0, 4.0};
    std::vector<cps::Real> delta = {0.0, 0.0, 1.0};  // delay of 2
    // Full output length = 4+3-1 = 6; x appears at positions 2..5
    auto out = cps::convolve(x, delta);
    REQUIRE(out.size() == 6);
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK_THAT(out[i + 2], WithinAbs(x[i], 1e-12));
    CHECK_THAT(out[0], WithinAbs(0.0, 1e-12));
    CHECK_THAT(out[1], WithinAbs(0.0, 1e-12));
}

TEST_CASE("convolve: Valid mode on equal-length vectors is a dot product") {
    std::vector<cps::Real> x = {1.0, 2.0, 3.0};
    std::vector<cps::Real> y = {4.0, 5.0, 6.0};
    auto out = cps::convolve(x, y, cps::ConvolveMode::Valid);
    REQUIRE(out.size() == 1);
    // convolve(x,y)[2] = x[0]*y[2] + x[1]*y[1] + x[2]*y[0] = 6+10+12 = 28
    CHECK_THAT(out[0], WithinAbs(28.0, 1e-10));
}

TEST_CASE("convolve: Same mode output length equals max of input lengths") {
    std::vector<cps::Real> x(7, 1.0);
    std::vector<cps::Real> y(3, 1.0);
    auto out1 = cps::convolve(x, y, cps::ConvolveMode::Same);
    auto out2 = cps::convolve(y, x, cps::ConvolveMode::Same);
    CHECK(out1.size() == 7);
    CHECK(out2.size() == 7);
}

TEST_CASE("convolve: two rectangular pulses give triangular output") {
    // rect * rect = triangle
    std::vector<cps::Real> r(4, 1.0);
    auto tri = cps::convolve(r, r);
    REQUIRE(tri.size() == 7);
    // Expected: [1, 2, 3, 4, 3, 2, 1]
    std::vector<double> expected = {1, 2, 3, 4, 3, 2, 1};
    for (std::size_t i = 0; i < tri.size(); ++i)
        CHECK_THAT(tri[i], WithinAbs(expected[i], 1e-12));
}

TEST_CASE("correlate: auto-correlation peak is always at zero lag") {
    const int N = 64;
    std::vector<cps::Real> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2 * std::numbers::pi * 4 * i / N) + 0.3 * std::cos(2 * std::numbers::pi * 9 * i / N);

    auto ac = cps::correlate(x, x);
    std::size_t zero_lag = static_cast<std::size_t>(N - 1);
    auto max_it  = std::max_element(ac.begin(), ac.end());
    std::size_t max_idx = static_cast<std::size_t>(std::distance(ac.begin(), max_it));
    CHECK(max_idx == zero_lag);
}

TEST_CASE("correlate: cross-correlation with shifted signal peaks at the shift") {
    const int N = 48;
    const int shift = 7;

    // Construct a pulse in x, and y = x shifted right by `shift`
    std::vector<cps::Real> x(N, 0.0), y(N, 0.0);
    for (int i = 4; i < 12; ++i) x[i] = 1.0;
    for (int i = 4 + shift; i < 12 + shift && i < N; ++i) y[i] = 1.0;

    auto corr   = cps::correlate(x, y);
    // zero-lag is at index N-1; lag +shift is at index N-1-shift
    auto max_it  = std::max_element(corr.begin(), corr.end());
    std::size_t max_idx  = static_cast<std::size_t>(std::distance(corr.begin(), max_it));
    std::size_t expected = static_cast<std::size_t>(N - 1 - shift);
    CHECK(max_idx == expected);
}

// ═════════════════════════════════════════════════════════════════════════════
// FFT / IFFT INVARIANTS
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("fft+ifft: complex roundtrip recovers original signal") {
    const int N = 16;
    std::vector<cps::Complex> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = {std::cos(2 * std::numbers::pi * 3.0 * i / N),
                 std::sin(2 * std::numbers::pi * 1.0 * i / N)};

    auto X = cps::fft(std::span<const cps::Complex>(x));
    auto y = cps::ifft(std::span<const cps::Complex>(X));
    REQUIRE(y.size() == static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        CHECK_THAT(y[i].real(), WithinAbs(x[i].real(), 1e-10));
        CHECK_THAT(y[i].imag(), WithinAbs(x[i].imag(), 1e-10));
    }
}

TEST_CASE("rfft+irfft: roundtrip recovers original signal") {
    const int N = 32;
    std::vector<cps::Real> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2 * std::numbers::pi * 3.0 * i / N)
             + 0.5 * std::cos(2 * std::numbers::pi * 7.0 * i / N);

    auto spec = cps::rfft(x);
    auto y    = cps::irfft(spec, static_cast<std::size_t>(N));
    REQUIRE(y.size() == static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i)
        CHECK_THAT(y[i], WithinAbs(x[i], 1e-10));
}

TEST_CASE("rfft: single-sample signal gives DC-only spectrum") {
    std::vector<cps::Real> x = {3.14};
    auto spec = cps::rfft(x);
    REQUIRE(spec.size() == 1);
    CHECK_THAT(spec[0].real(), WithinAbs(3.14, 1e-12));
    CHECK_THAT(spec[0].imag(), WithinAbs(0.0,  1e-12));
}

TEST_CASE("rfft: DC and Nyquist bins are always real for real input") {
    const int N = 16;
    std::vector<cps::Real> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2 * std::numbers::pi * i / N) + i * 0.1;

    auto spec = cps::rfft(x);
    // DC (k=0) is purely real
    CHECK_THAT(spec[0].imag(), WithinAbs(0.0, 1e-10));
    // Nyquist (k=N/2) is purely real for even N
    CHECK_THAT(spec[N / 2].imag(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("rfft: linearity — rfft(a*x + b*y) == a*rfft(x) + b*rfft(y)") {
    const int    N = 16;
    const double a = 2.5, b = -0.7;
    std::vector<cps::Real> x(N), y(N), z(N);
    for (int i = 0; i < N; ++i) {
        x[i] = std::sin(2 * std::numbers::pi * i / N);
        y[i] = std::cos(4 * std::numbers::pi * i / N);
        z[i] = a * x[i] + b * y[i];
    }

    auto Xf = cps::rfft(x);
    auto Yf = cps::rfft(y);
    auto Zf = cps::rfft(z);

    REQUIRE(Xf.size() == Zf.size());
    for (std::size_t k = 0; k < Zf.size(); ++k) {
        cps::Complex expected = a * Xf[k] + b * Yf[k];
        CHECK_THAT(Zf[k].real(), WithinAbs(expected.real(), 1e-9));
        CHECK_THAT(Zf[k].imag(), WithinAbs(expected.imag(), 1e-9));
    }
}

TEST_CASE("rfft: impulse at index 0 gives flat spectrum (all bins magnitude 1)") {
    const int N = 8;
    std::vector<cps::Real> impulse(N, 0.0);
    impulse[0] = 1.0;
    auto spec = cps::rfft(impulse);
    REQUIRE(spec.size() == static_cast<std::size_t>(N / 2 + 1));
    for (const auto& c : spec)
        CHECK_THAT(std::abs(c), WithinAbs(1.0, 1e-10));
}

TEST_CASE("fft: Parseval's theorem holds — sum|X[k]|^2 = N * sum|x[n]|^2") {
    const int N = 16;
    std::vector<cps::Complex> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = {std::sin(2 * std::numbers::pi * 3.0 * i / N), 0.0};

    auto X = cps::fft(std::span<const cps::Complex>(x));

    double time_power = 0.0;
    for (const auto& c : x)  time_power += std::norm(c);
    double freq_power = 0.0;
    for (const auto& c : X)  freq_power += std::norm(c);

    CHECK_THAT(freq_power, WithinRel(N * time_power, 1e-9));
}

// ═════════════════════════════════════════════════════════════════════════════
// PEAK DETECTION EDGE CASES
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("find_peaks: N < 3 returns empty result") {
    CHECK(cps::find_peaks(std::vector<cps::Real>{}).indices.empty());
    CHECK(cps::find_peaks(std::vector<cps::Real>{1.0}).indices.empty());
    CHECK(cps::find_peaks(std::vector<cps::Real>{1.0, 2.0}).indices.empty());
}

TEST_CASE("find_peaks: minimum 3-sample signal with a peak") {
    std::vector<cps::Real> sig = {0.5, 2.0, 0.5};
    auto result = cps::find_peaks(sig);
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 1);
}

TEST_CASE("find_peaks: flat signal has no local maxima") {
    std::vector<cps::Real> flat(20, 3.14);
    CHECK(cps::find_peaks(flat).indices.empty());
}

TEST_CASE("find_peaks: plateau (equal neighbours) is not a strict local maximum") {
    // 0, 1, 1, 1, 0 — centre samples equal, not a peak
    std::vector<cps::Real> sig = {0.0, 1.0, 1.0, 1.0, 0.0};
    CHECK(cps::find_peaks(sig).indices.empty());
}

TEST_CASE("find_peaks: peaks at first and last valid positions are both found") {
    std::vector<cps::Real> sig = {0.0, 5.0, 1.0, 1.0, 4.0, 0.0};
    auto result = cps::find_peaks(sig);
    REQUIRE(result.indices.size() == 2);
    CHECK(result.indices[0] == 1);
    CHECK(result.indices[1] == 4);
}

TEST_CASE("find_peaks: distance=1 is a no-op (no peaks suppressed)") {
    std::vector<cps::Real> sig = {0.0, 2.0, 0.5, 1.0, 0.0};
    auto without_dist = cps::find_peaks(sig);
    auto with_dist1   = cps::find_peaks(sig, {.distance = 1});
    CHECK(with_dist1.indices == without_dist.indices);
}

TEST_CASE("find_peaks: peaks exactly dist apart are both kept") {
    // Peaks at index 1 and 5, gap = 4 = dist; condition is < dist → FALSE → break → both kept
    std::vector<cps::Real> sig = {0.0, 2.0, 0.0, 0.0, 0.0, 1.5, 0.0};
    auto result = cps::find_peaks(sig, {.distance = 4});
    CHECK(result.indices.size() == 2);
}

TEST_CASE("find_peaks: distance filter keeps taller of two conflicting peaks") {
    // Peaks at 1 (h=1) and 3 (h=2), within dist=3 → taller wins
    std::vector<cps::Real> sig = {0.0, 1.0, 0.0, 2.0, 0.0};
    auto result = cps::find_peaks(sig, {.distance = 3});
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 3);
}

TEST_CASE("find_peaks: distance filter with 3-way conflict keeps global maximum") {
    // All three peaks within dist=5 of each other → only the tallest survives
    std::vector<cps::Real> sig = {0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0};
    auto result = cps::find_peaks(sig, {.distance = 5});
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 3);
    CHECK_THAT(result.heights[0], WithinAbs(3.0, 1e-12));
}

TEST_CASE("find_peaks: height filter removes all below threshold") {
    std::vector<cps::Real> sig = {0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0};
    auto result = cps::find_peaks(sig, {.height = 3.0});
    CHECK(result.indices.empty());
}

TEST_CASE("find_peaks: threshold filter requires rise above both neighbours") {
    // Peak at 2 (h=1.0) rises by only 0.1 over left neighbour — below threshold=0.5
    std::vector<cps::Real> sig = {0.9, 0.9, 1.0, 0.5, 0.0};
    auto result_nothr = cps::find_peaks(sig);
    auto result_thr   = cps::find_peaks(sig, {.threshold = 0.5});
    // Without threshold: peak at 2 is found
    REQUIRE(result_nothr.indices.size() == 1);
    // With threshold 0.5: 1.0 - 0.9 = 0.1 < 0.5 → removed
    CHECK(result_thr.indices.empty());
}

TEST_CASE("peak_prominences: isolated peak prominence = height minus signal minimum") {
    std::vector<cps::Real> sig  = {0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0};
    std::vector<std::size_t> pk = {3};
    auto proms = cps::peak_prominences(sig, pk);
    REQUIRE(proms.size() == 1);
    // No higher peak on either side → contour base = min over whole signal = 0
    CHECK_THAT(proms[0], WithinAbs(5.0, 1e-12));
}

TEST_CASE("peak_prominences: lower peak bounded by higher neighbour on left") {
    // Tall peak at index 1, shorter peak at index 5
    std::vector<cps::Real> sig  = {0.0, 10.0, 0.5, 1.0, 0.5, 3.0, 0.0};
    std::vector<std::size_t> pk = {1, 5};
    auto proms = cps::peak_prominences(sig, pk);
    REQUIRE(proms.size() == 2);
    // Peak at 1: no higher peak, contour base = min over full signal = 0, prom = 10
    CHECK_THAT(proms[0], WithinAbs(10.0, 1e-12));
    // Peak at 5 (h=3): left walk finds signal[1]=10 > 3 → left_min = min[1..5] = 0.5
    //                  right walk hits boundary → right_min = min[5..6] = 0.0
    // prom = 3 - max(0.5, 0.0) = 2.5
    CHECK_THAT(proms[1], WithinAbs(2.5, 1e-12));
}

TEST_CASE("find_peaks: width filter does not crash on valid smooth peak") {
    // Symmetric triangular peak; width at half-prominence is well-defined
    std::vector<cps::Real> sig = {0.0, 0.5, 1.0, 2.0, 1.0, 0.5, 0.0};
    auto result = cps::find_peaks(sig, {.width = 1.5});
    REQUIRE(result.indices.size() == 1);
    CHECK(result.indices[0] == 3);
}

// ═════════════════════════════════════════════════════════════════════════════
// METRICS INVARIANTS
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("rms: pure sine amplitude A gives rms = A/sqrt(2)") {
    const int    N = 1024;
    const double A = 3.0;
    std::vector<cps::Real> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = A * std::sin(2 * std::numbers::pi * 5.0 * i / N);
    CHECK_THAT(cps::rms(x), WithinRel(A / std::sqrt(2.0), 1e-4));
}

TEST_CASE("rms: constant signal returns that constant") {
    std::vector<cps::Real> sig(50, 7.5);
    CHECK_THAT(cps::rms(sig), WithinAbs(7.5, 1e-12));
}

TEST_CASE("snr: identical signal and noise gives 0 dB") {
    std::vector<cps::Real> v = {1.0, -2.0, 3.0, 0.5, -1.5};
    CHECK_THAT(cps::snr(v, v), WithinAbs(0.0, 1e-10));
}

TEST_CASE("snr: signal amplitude 10x noise gives 20 dB") {
    const int N = 128;
    std::vector<cps::Real> sig(N, 10.0);
    std::vector<cps::Real> noise(N, 1.0);
    CHECK_THAT(cps::snr(sig, noise), WithinAbs(20.0, 1e-6));
}

TEST_CASE("welch: Parseval — integral of PSD approximates signal variance") {
    const double fs = 1000.0;
    const double A  = 2.0;
    const int    N  = 4096;
    std::vector<cps::Real> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = A * std::sin(2 * std::numbers::pi * 100.0 * i / fs);

    auto [freqs, psd] = cps::welch(x, fs, {.nperseg = 512});

    double df    = (freqs.size() > 1) ? (freqs[1] - freqs[0]) : 1.0;
    double power = std::accumulate(psd.begin(), psd.end(), 0.0) * df;

    // Variance of A*sin is A^2/2
    CHECK_THAT(power, WithinRel(0.5 * A * A, 0.1));
}

TEST_CASE("welch: all PSD values are non-negative") {
    const int N = 512;
    std::vector<cps::Real> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2 * std::numbers::pi * 50.0 * i / 1000.0) + 0.1 * (i % 7 - 3);

    auto [freqs, psd] = cps::welch(x, 1000.0);
    for (auto p : psd)
        CHECK(p >= 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// ERROR / VALIDATION PATHS
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter: order=0 throws ValueError") {
    CHECK_THROWS_AS(cps::butter(0, 0.3, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: negative order throws ValueError") {
    CHECK_THROWS_AS(cps::butter(-1, 0.3, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: Wn=0 throws ValueError") {
    CHECK_THROWS_AS(cps::butter(2, 0.0, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: Wn=1 throws ValueError") {
    CHECK_THROWS_AS(cps::butter(2, 1.0, cps::FilterType::Lowpass), cps::ValueError);
}

TEST_CASE("butter: Bandpass throws NotImplemented") {
    CHECK_THROWS_AS(cps::butter(2, 0.3, cps::FilterType::Bandpass), cps::NotImplemented);
}

TEST_CASE("firwin: numtaps=0 throws ValueError") {
    CHECK_THROWS_AS(cps::firwin(0, 0.3), cps::ValueError);
}

TEST_CASE("firwin: even numtaps with Highpass throws ValueError") {
    CHECK_THROWS_AS(
        cps::firwin(10, 0.3, cps::Window::Hamming, cps::FilterType::Highpass),
        cps::ValueError);
}

TEST_CASE("sosfilt: empty SOS throws ValueError") {
    cps::SOS empty;
    std::vector<cps::Real> sig = {1.0, 2.0, 3.0};
    CHECK_THROWS_AS(cps::sosfilt(empty, sig), cps::ValueError);
}

TEST_CASE("lfilter: a[0]=0 throws ValueError") {
    std::vector<cps::Real> sig = {1.0, 2.0, 3.0};
    CHECK_THROWS_AS(
        cps::lfilter(std::vector<cps::Real>{1.0}, std::vector<cps::Real>{0.0, 1.0}, sig),
        cps::ValueError);
}

TEST_CASE("rms: empty signal throws ValueError") {
    CHECK_THROWS_AS(cps::rms(std::vector<cps::Real>{}), cps::ValueError);
}

TEST_CASE("snr (two-vector): mismatched lengths throws ValueError") {
    std::vector<cps::Real> a = {1.0, 2.0, 3.0};
    std::vector<cps::Real> b = {1.0, 2.0};
    CHECK_THROWS_AS(cps::snr(a, b), cps::ValueError);
}

TEST_CASE("snr (two-vector): zero noise throws NumericalError") {
    std::vector<cps::Real> sig   = {1.0, 2.0, 3.0};
    std::vector<cps::Real> noise = {0.0, 0.0, 0.0};
    CHECK_THROWS_AS(cps::snr(sig, noise), cps::NumericalError);
}

TEST_CASE("thd: n_harmonics=0 throws ValueError") {
    const int N = 256;
    std::vector<cps::Real> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2 * std::numbers::pi * 100.0 * i / 1000.0);
    CHECK_THROWS_AS(cps::thd(sig, 100.0, 1000.0, 0), cps::ValueError);
}

TEST_CASE("fftfreq: n=0 throws ValueError") {
    CHECK_THROWS_AS(cps::fftfreq(0), cps::ValueError);
}

TEST_CASE("rfftfreq: n=0 throws ValueError") {
    CHECK_THROWS_AS(cps::rfftfreq(0), cps::ValueError);
}

TEST_CASE("irfft: spectrum size mismatch throws ValueError") {
    auto spec = cps::rfft(std::vector<cps::Real>(8, 1.0));
    // spec.size() == 5 (= 8/2+1); n=10 requires 6 → mismatch
    CHECK_THROWS_AS(cps::irfft(spec, 10), cps::ValueError);
}

TEST_CASE("convolve: empty x throws ValueError") {
    std::vector<cps::Real> empty;
    std::vector<cps::Real> y = {1.0, 2.0};
    CHECK_THROWS_AS(cps::convolve(empty, y), cps::ValueError);
}

TEST_CASE("welch: signal shorter than nperseg throws ValueError") {
    std::vector<cps::Real> sig(50, 1.0);
    CHECK_THROWS_AS(cps::welch(sig, 1000.0, {.nperseg = 100}), cps::ValueError);
}

TEST_CASE("make_window: n=0 throws ValueError") {
    CHECK_THROWS_AS(cps::make_window(cps::Window::Hann, 0), cps::ValueError);
}

TEST_CASE("fft: empty input throws ValueError") {
    CHECK_THROWS_AS(cps::fft(std::vector<cps::Real>{}), cps::ValueError);
}
