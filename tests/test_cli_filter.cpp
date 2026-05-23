#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <sstream>
#include <cli/format.hpp>
#include <cli/generate.hpp>
#include <cli/filter.hpp>

using namespace cps::cli;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Test helpers ──────────────────────────────────────────────────────────────

static std::pair<StreamHeader, std::vector<float>>
sine_through_filter(double freq, uint32_t sr, FilterOptions fopts, double duration = 2.0)
{
    GenerateOptions gopts;
    gopts.type        = GenerateType::Sine;
    gopts.freq        = freq;
    gopts.sample_rate = sr;
    gopts.duration    = duration;

    std::ostringstream gen_buf;
    generate(gen_buf, gopts);

    std::istringstream gen_in(gen_buf.str());
    std::ostringstream filt_buf;
    filter_stream(gen_in, filt_buf, fopts);

    std::istringstream filt_in(filt_buf.str());
    auto hdr     = read_header(filt_in);
    auto samples = read_samples_f32(filt_in, hdr);
    return {hdr, samples};
}

// Measure steady-state RMS by skipping the first skip_seconds of transient.
static double steady_rms(const std::vector<float>& samples, uint32_t sr,
                          double skip_seconds = 0.2)
{
    auto skip = static_cast<std::size_t>(skip_seconds * sr);
    if (skip >= samples.size()) return 0.0;
    double sum = 0.0;
    for (std::size_t i = skip; i < samples.size(); ++i)
        sum += static_cast<double>(samples[i]) * samples[i];
    return std::sqrt(sum / static_cast<double>(samples.size() - skip));
}

// ── Header passthrough ────────────────────────────────────────────────────────

TEST_CASE("filter passes sample_rate through header unchanged", "[cli][filter]")
{
    for (uint32_t sr : {8000u, 44100u, 48000u}) {
        FilterOptions fopts;
        fopts.shape  = FilterShape::Lowpass;
        fopts.cutoff = static_cast<double>(sr) / 4.0;

        auto [hdr, samples] = sine_through_filter(100.0, sr, fopts);
        CHECK(hdr.sample_rate == sr);
    }
}

TEST_CASE("filter preserves sample count", "[cli][filter]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 2000.0;

    auto [hdr, samples] = sine_through_filter(440.0, 44100, fopts, 1.0);
    CHECK(samples.size() == 44100u);
}

TEST_CASE("filter passes stream format through unchanged", "[cli][filter]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 2000.0;

    auto [hdr, samples] = sine_through_filter(440.0, 44100, fopts);
    CHECK(hdr.format == SampleFormat::Float32);
}

// ── Lowpass ───────────────────────────────────────────────────────────────────

TEST_CASE("lowpass: tone well below cutoff passes with near-unity gain", "[cli][filter][lp]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 4000.0;
    fopts.order  = 4;

    // 100 Hz is far below 4 kHz — gain should be ~1, RMS of unit sine ≈ 0.707
    auto [hdr, samples] = sine_through_filter(100.0, 44100, fopts);
    CHECK_THAT(steady_rms(samples, hdr.sample_rate),
               WithinAbs(1.0 / std::sqrt(2.0), 0.05));
}

TEST_CASE("lowpass: tone well above cutoff is strongly attenuated", "[cli][filter][lp]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 500.0;
    fopts.order  = 4;

    // 8 kHz is a decade above 500 Hz — 4th-order Butterworth gives > 80 dB attenuation
    auto [hdr, samples] = sine_through_filter(8000.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.01);
}

TEST_CASE("lowpass: tone at cutoff is approximately -3 dB (RMS halved)", "[cli][filter][lp]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 1000.0;
    fopts.order  = 4;

    // At cutoff, Butterworth gain = 1/sqrt(2), so RMS of unit sine ≈ 0.5
    auto [hdr, samples] = sine_through_filter(1000.0, 44100, fopts);
    CHECK_THAT(steady_rms(samples, hdr.sample_rate), WithinAbs(0.5, 0.05));
}

TEST_CASE("lowpass: higher order gives steeper rolloff", "[cli][filter][lp]")
{
    auto measure = [&](int order) {
        FilterOptions fopts;
        fopts.shape  = FilterShape::Lowpass;
        fopts.cutoff = 1000.0;
        fopts.order  = order;
        auto [hdr, samples] = sine_through_filter(3000.0, 44100, fopts);
        return steady_rms(samples, hdr.sample_rate);
    };

    // A 6th-order filter should reject 3x cutoff more than a 2nd-order filter
    CHECK(measure(6) < measure(2));
}

// ── Highpass ──────────────────────────────────────────────────────────────────

TEST_CASE("highpass: tone well below cutoff is strongly attenuated", "[cli][filter][hp]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Highpass;
    fopts.cutoff = 4000.0;
    fopts.order  = 4;

    auto [hdr, samples] = sine_through_filter(100.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.01);
}

TEST_CASE("highpass: tone well above cutoff passes with near-unity gain", "[cli][filter][hp]")
{
    FilterOptions fopts;
    fopts.shape  = FilterShape::Highpass;
    fopts.cutoff = 500.0;
    fopts.order  = 4;

    auto [hdr, samples] = sine_through_filter(8000.0, 44100, fopts);
    CHECK_THAT(steady_rms(samples, hdr.sample_rate),
               WithinAbs(1.0 / std::sqrt(2.0), 0.05));
}

// ── Bandpass ──────────────────────────────────────────────────────────────────

TEST_CASE("bandpass: tone within passband is not strongly attenuated", "[cli][filter][bp]")
{
    FilterOptions fopts;
    fopts.shape      = FilterShape::Bandpass;
    fopts.cutoff_low  = 800.0;
    fopts.cutoff_high = 1200.0;
    fopts.order      = 4;

    auto [hdr, samples] = sine_through_filter(1000.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) > 0.2);
}

TEST_CASE("bandpass: tone below passband is attenuated", "[cli][filter][bp]")
{
    FilterOptions fopts;
    fopts.shape      = FilterShape::Bandpass;
    fopts.cutoff_low  = 800.0;
    fopts.cutoff_high = 1200.0;
    fopts.order      = 4;

    auto [hdr, samples] = sine_through_filter(100.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.1);
}

TEST_CASE("bandpass: tone above passband is attenuated", "[cli][filter][bp]")
{
    FilterOptions fopts;
    fopts.shape      = FilterShape::Bandpass;
    fopts.cutoff_low  = 800.0;
    fopts.cutoff_high = 1200.0;
    fopts.order      = 4;

    auto [hdr, samples] = sine_through_filter(8000.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.1);
}

// ── Bandstop ──────────────────────────────────────────────────────────────────

TEST_CASE("bandstop: tone within stopband is attenuated", "[cli][filter][bs]")
{
    FilterOptions fopts;
    fopts.shape      = FilterShape::Bandstop;
    fopts.cutoff_low  = 800.0;
    fopts.cutoff_high = 1200.0;
    fopts.order      = 4;

    auto [hdr, samples] = sine_through_filter(1000.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.2);
}

TEST_CASE("bandstop: tone outside stopband passes through", "[cli][filter][bs]")
{
    FilterOptions fopts;
    fopts.shape      = FilterShape::Bandstop;
    fopts.cutoff_low  = 800.0;
    fopts.cutoff_high = 1200.0;
    fopts.order      = 4;

    auto [hdr, samples] = sine_through_filter(200.0, 44100, fopts);
    CHECK(steady_rms(samples, hdr.sample_rate) > 0.3);
}

// ── FIR ───────────────────────────────────────────────────────────────────────

TEST_CASE("firwin lowpass attenuates a tone well above cutoff", "[cli][filter][firwin]")
{
    FilterOptions fopts;
    fopts.impl   = FilterImpl::FirWin;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 500.0;
    fopts.taps   = 101;

    // FIR has a longer group delay; use a longer signal and skip more transient
    auto [hdr, samples] = sine_through_filter(5000.0, 44100, fopts, 3.0);
    CHECK(steady_rms(samples, hdr.sample_rate, 0.5) < 0.05);
}

TEST_CASE("firwin lowpass passes a tone well below cutoff", "[cli][filter][firwin]")
{
    FilterOptions fopts;
    fopts.impl   = FilterImpl::FirWin;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 4000.0;
    fopts.taps   = 101;

    auto [hdr, samples] = sine_through_filter(200.0, 44100, fopts, 3.0);
    CHECK_THAT(steady_rms(samples, hdr.sample_rate, 0.5),
               WithinAbs(1.0 / std::sqrt(2.0), 0.1));
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST_CASE("filter_stream throws on invalid stream magic", "[cli][filter][error]")
{
    std::string bad(12, '\0');  // zeroed — wrong magic
    std::istringstream in(bad);
    std::ostringstream out;

    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 1000.0;
    CHECK_THROWS_AS(filter_stream(in, out, fopts), std::runtime_error);
}

TEST_CASE("filter_stream throws on empty input", "[cli][filter][error]")
{
    std::istringstream in("");
    std::ostringstream out;

    FilterOptions fopts;
    fopts.shape  = FilterShape::Lowpass;
    fopts.cutoff = 1000.0;
    CHECK_THROWS_AS(filter_stream(in, out, fopts), std::runtime_error);
}

// ── Argument parsing (RED — parse_filter_args not yet implemented) ─────────────

TEST_CASE("parse_filter_args throws when no cutoff is given for lowpass", "[cli][filter][args]")
{
    const char* argv[] = {"cps", "filter", "--shape", "lowpass"};
    CHECK_THROWS_AS(
        parse_filter_args(4, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_filter_args throws when bandpass cutoff-low >= cutoff-high", "[cli][filter][args]")
{
    const char* argv[] = {"cps", "filter", "--shape", "bandpass",
                          "--cutoff-low", "2000", "--cutoff-high", "1000"};
    CHECK_THROWS_AS(
        parse_filter_args(8, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_filter_args throws on non-positive order", "[cli][filter][args]")
{
    const char* argv[] = {"cps", "filter", "--order", "0", "--cutoff", "1000"};
    CHECK_THROWS_AS(
        parse_filter_args(6, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_filter_args throws on unknown filter shape", "[cli][filter][args]")
{
    const char* argv[] = {"cps", "filter", "--shape", "bandreject", "--cutoff", "1000"};
    CHECK_THROWS_AS(
        parse_filter_args(6, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_filter_args accepts valid lowpass options and returns correct defaults", "[cli][filter][args]")
{
    const char* argv[] = {"cps", "filter", "--shape", "lowpass", "--cutoff", "1000"};
    auto opts = parse_filter_args(6, const_cast<char**>(argv));

    CHECK(opts.shape  == FilterShape::Lowpass);
    CHECK_THAT(opts.cutoff, WithinAbs(1000.0, 1e-9));
    CHECK(opts.impl   == FilterImpl::Butter);  // default
    CHECK(opts.order  == 4);                   // default
}

TEST_CASE("parse_filter_args accepts --type firwin", "[cli][filter][args]")
{
    const char* argv[] = {"cps", "filter", "--type", "firwin",
                          "--shape", "lowpass", "--cutoff", "1000", "--taps", "51"};
    auto opts = parse_filter_args(10, const_cast<char**>(argv));

    CHECK(opts.impl == FilterImpl::FirWin);
    CHECK(opts.taps == 51);
}
