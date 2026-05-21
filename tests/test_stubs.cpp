// test_stubs.cpp — verify every NotImplemented stub throws cps::NotImplemented
//
// These tests document the full set of planned-but-unimplemented functions and
// ensure they signal their status clearly rather than silently returning garbage.
//
// Stubs covered:
//   filter/design.hpp  : butter(Bandpass), butter(Bandstop), cheby1, cheby2, ellip, bessel
//                        firwin(Kaiser window), firwin(Tukey window)
//   filter/apply.hpp   : filtfilt
//   filter/analysis.hpp: group_delay
//   spectral/psd.hpp   : periodogram, csd
//   signal/resample.hpp: decimate, interpolate, resample
#include <catch2/catch_test_macros.hpp>
#include <cps/cps.hpp>
#include <vector>
#include <span>

using namespace cps;

// ════════════════════════════════════════════════════════════════════════════
// butter — unimplemented filter types
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("butter Bandpass throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(butter(4, 0.3, FilterType::Bandpass), NotImplemented);
}

TEST_CASE("butter Bandstop throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(butter(4, 0.3, FilterType::Bandstop), NotImplemented);
}

TEST_CASE("butter Bandpass with any order throws NotImplemented", "[stubs][filter]")
{
    for (int order : {1, 2, 3, 6, 10}) {
        CHECK_THROWS_AS(butter(order, 0.3, FilterType::Bandpass), NotImplemented);
    }
}

TEST_CASE("butter Bandstop with any order throws NotImplemented", "[stubs][filter]")
{
    for (int order : {1, 2, 3, 6}) {
        CHECK_THROWS_AS(butter(order, 0.3, FilterType::Bandstop), NotImplemented);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Alternative IIR design functions
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("cheby1 throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(cheby1(4, 1.0, 0.3, FilterType::Lowpass), NotImplemented);
}

TEST_CASE("cheby1 with different parameters throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(cheby1(2, 0.5, 0.5, FilterType::Highpass), NotImplemented);
    CHECK_THROWS_AS(cheby1(6, 3.0, 0.2, FilterType::Lowpass,  {.fs = 1000.0}), NotImplemented);
}

TEST_CASE("cheby2 throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(cheby2(4, 40.0, 0.3, FilterType::Lowpass), NotImplemented);
}

TEST_CASE("cheby2 with different parameters throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(cheby2(2, 20.0, 0.5, FilterType::Highpass), NotImplemented);
    CHECK_THROWS_AS(cheby2(6, 60.0, 0.2, FilterType::Lowpass,  {.fs = 1000.0}), NotImplemented);
}

TEST_CASE("ellip throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(ellip(4, 1.0, 40.0, 0.3, FilterType::Lowpass), NotImplemented);
}

TEST_CASE("ellip with different parameters throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(ellip(2, 0.5, 20.0, 0.4, FilterType::Highpass), NotImplemented);
    CHECK_THROWS_AS(ellip(6, 2.0, 60.0, 0.2, FilterType::Lowpass, {.fs = 1000.0}), NotImplemented);
}

TEST_CASE("bessel throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(bessel(4, 0.3, FilterType::Lowpass), NotImplemented);
}

TEST_CASE("bessel with different parameters throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(bessel(2, 0.5, FilterType::Highpass), NotImplemented);
    CHECK_THROWS_AS(bessel(6, 0.2, FilterType::Lowpass, {.fs = 1000.0}), NotImplemented);
}

// ════════════════════════════════════════════════════════════════════════════
// firwin — unsupported window types
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("firwin with Kaiser window throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(firwin(11, 0.2, Window::Kaiser), NotImplemented);
}

TEST_CASE("firwin with Tukey window throws NotImplemented", "[stubs][filter]")
{
    CHECK_THROWS_AS(firwin(11, 0.2, Window::Tukey), NotImplemented);
}

TEST_CASE("firwin Kaiser throws NotImplemented for any length", "[stubs][filter]")
{
    CHECK_THROWS_AS(firwin(5,   0.2, Window::Kaiser), NotImplemented);
    CHECK_THROWS_AS(firwin(101, 0.2, Window::Kaiser), NotImplemented);
    CHECK_THROWS_AS(firwin(51,  0.2, Window::Kaiser, FilterType::Highpass), NotImplemented);
}

// ════════════════════════════════════════════════════════════════════════════
// filtfilt
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("filtfilt throws NotImplemented", "[stubs][filter]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    std::vector<Real> x = {1.0, 2.0, 3.0};
    CHECK_THROWS_AS(filtfilt(sos, x), NotImplemented);
}

TEST_CASE("filtfilt with empty signal throws NotImplemented (stub before guard)", "[stubs][filter]")
{
    auto sos = butter(2, 0.1, FilterType::Lowpass);
    std::vector<Real> x;
    CHECK_THROWS_AS(filtfilt(sos, x), NotImplemented);
}

// ════════════════════════════════════════════════════════════════════════════
// group_delay
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("group_delay throws NotImplemented", "[stubs][filter]")
{
    auto sos = butter(4, 0.2, FilterType::Lowpass);
    CHECK_THROWS_AS(group_delay(sos), NotImplemented);
    CHECK_THROWS_AS(group_delay(sos, 1024), NotImplemented);
}

// ════════════════════════════════════════════════════════════════════════════
// periodogram
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("periodogram throws NotImplemented", "[stubs][spectral]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    CHECK_THROWS_AS(periodogram(std::span<const Real>(x), 1000.0), NotImplemented);
}

TEST_CASE("periodogram with window argument throws NotImplemented", "[stubs][spectral]")
{
    std::vector<Real> x(256, 1.0);
    CHECK_THROWS_AS(periodogram(std::span<const Real>(x), 1000.0, Window::Hann), NotImplemented);
    CHECK_THROWS_AS(periodogram(std::span<const Real>(x), 1000.0, Window::Hamming), NotImplemented);
}

// ════════════════════════════════════════════════════════════════════════════
// csd
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("csd throws NotImplemented", "[stubs][spectral]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0};
    std::vector<Real> y = {4.0, 5.0, 6.0};
    CHECK_THROWS_AS(csd(std::span<const Real>(x), std::span<const Real>(y), 1000.0),
                    NotImplemented);
}

TEST_CASE("csd with WelchOptions throws NotImplemented", "[stubs][spectral]")
{
    std::vector<Real> x(256, 1.0);
    std::vector<Real> y(256, 1.0);
    CHECK_THROWS_AS(
        csd(std::span<const Real>(x), std::span<const Real>(y), 1000.0,
            WelchOptions{.nperseg = 64}),
        NotImplemented);
}

// ════════════════════════════════════════════════════════════════════════════
// decimate / interpolate / resample
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("decimate throws NotImplemented", "[stubs][resample]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    CHECK_THROWS_AS(decimate(x, 2), NotImplemented);
}

TEST_CASE("decimate with different factors throws NotImplemented", "[stubs][resample]")
{
    std::vector<Real> x(100, 1.0);
    for (int q : {2, 3, 4, 8, 10}) {
        CHECK_THROWS_AS(decimate(x, q), NotImplemented);
    }
}

TEST_CASE("interpolate throws NotImplemented", "[stubs][resample]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0};
    CHECK_THROWS_AS(interpolate(x, 2), NotImplemented);
}

TEST_CASE("interpolate with different factors throws NotImplemented", "[stubs][resample]")
{
    std::vector<Real> x(50, 1.0);
    for (int p : {2, 3, 5, 8}) {
        CHECK_THROWS_AS(interpolate(x, p), NotImplemented);
    }
}

TEST_CASE("resample throws NotImplemented", "[stubs][resample]")
{
    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    CHECK_THROWS_AS(resample(x, 3, 2), NotImplemented);
}

TEST_CASE("resample with different p/q ratios throws NotImplemented", "[stubs][resample]")
{
    std::vector<Real> x(100, 1.0);
    CHECK_THROWS_AS(resample(x, 1, 2), NotImplemented);   // downsample
    CHECK_THROWS_AS(resample(x, 2, 1), NotImplemented);   // upsample
    CHECK_THROWS_AS(resample(x, 3, 4), NotImplemented);   // rational
    CHECK_THROWS_AS(resample(x, 7, 3), NotImplemented);   // fractional up
}

// ════════════════════════════════════════════════════════════════════════════
// NotImplemented exception type is a cps::Error subtype
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("NotImplemented is catchable as cps::Error", "[stubs]")
{
    bool caught = false;
    try {
        std::vector<Real> x = {1.0, 2.0};
        decimate(x, 2);
    } catch (const cps::Error&) {
        caught = true;
    }
    CHECK(caught);
}

TEST_CASE("NotImplemented is catchable as std::exception", "[stubs]")
{
    bool caught = false;
    try {
        std::vector<Real> x = {1.0, 2.0};
        decimate(x, 2);
    } catch (const std::exception& e) {
        caught = true;
        // Message should contain "not implemented"
        std::string msg = e.what();
        CHECK(msg.find("not implemented") != std::string::npos);
    }
    CHECK(caught);
}

TEST_CASE("NotImplemented message contains function name", "[stubs]")
{
    bool caught = false;
    try {
        auto sos = butter(4, 0.2, FilterType::Lowpass);
        std::vector<Real> x = {1.0};
        filtfilt(sos, x);
    } catch (const cps::NotImplemented& e) {
        caught = true;
        std::string msg = e.what();
        CHECK(msg.find("filtfilt") != std::string::npos);
    }
    CHECK(caught);
}
