#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <sstream>
#include <cli/format.hpp>
#include <cli/generate.hpp>
#include <cli/filter.hpp>

using namespace cps::cli;
using Catch::Matchers::WithinAbs;

// ── Test helpers ──────────────────────────────────────────────────────────────

// Pipe a GenerateOptions through a sequence of FilterOptions and return the result.
static std::pair<StreamHeader, std::vector<float>>
pipeline(GenerateOptions gopts, std::vector<FilterOptions> filters)
{
    std::ostringstream buf;
    generate(buf, gopts);

    for (const auto& fopts : filters) {
        std::istringstream in(buf.str());
        buf = std::ostringstream{};
        filter_stream(in, buf, fopts);
    }

    std::istringstream in(buf.str());
    auto hdr     = read_header(in);
    auto samples = read_samples_f32(in, hdr);
    return {hdr, samples};
}

static double steady_rms(const std::vector<float>& v, uint32_t sr,
                          double skip_seconds = 0.3)
{
    auto skip = static_cast<std::size_t>(skip_seconds * sr);
    if (skip >= v.size()) return 0.0;
    double sum = 0.0;
    for (std::size_t i = skip; i < v.size(); ++i)
        sum += static_cast<double>(v[i]) * v[i];
    return std::sqrt(sum / static_cast<double>(v.size() - skip));
}

// ── Stream integrity ──────────────────────────────────────────────────────────

TEST_CASE("generate | filter produces a readable CPS stream", "[cli][pipeline]")
{
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 440.0; g.duration = 1.0; g.sample_rate = 44100;

    FilterOptions f;
    f.shape = FilterShape::Lowpass; f.cutoff = 2000.0;

    auto [hdr, samples] = pipeline(g, {f});
    CHECK(hdr.sample_rate == 44100u);
    CHECK(samples.size()  == 44100u);
    CHECK_FALSE(samples.empty());
}

TEST_CASE("pipeline preserves sample_rate through two filter stages", "[cli][pipeline]")
{
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 440.0; g.duration = 1.0; g.sample_rate = 48000;

    FilterOptions f1; f1.shape = FilterShape::Lowpass;  f1.cutoff = 8000.0;
    FilterOptions f2; f2.shape = FilterShape::Highpass; f2.cutoff =  200.0;

    auto [hdr, samples] = pipeline(g, {f1, f2});
    CHECK(hdr.sample_rate == 48000u);
}

TEST_CASE("pipeline preserves sample count through two filter stages", "[cli][pipeline]")
{
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 440.0; g.duration = 2.0; g.sample_rate = 44100;

    FilterOptions f1; f1.shape = FilterShape::Lowpass;  f1.cutoff = 4000.0;
    FilterOptions f2; f2.shape = FilterShape::Lowpass;  f2.cutoff = 2000.0;

    auto [hdr, samples] = pipeline(g, {f1, f2});
    CHECK(samples.size() == 2u * 44100u);
}

// ── Functional: LP | HP acts as a bandpass ────────────────────────────────────

TEST_CASE("generate | LP | HP passes a midband tone with near-unity gain", "[cli][pipeline]")
{
    // 1 kHz tone, LP at 4 kHz, HP at 200 Hz — both filters' passbands contain 1 kHz
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 1000.0; g.duration = 2.0; g.sample_rate = 44100;

    FilterOptions lp; lp.shape = FilterShape::Lowpass;  lp.cutoff = 4000.0; lp.order = 4;
    FilterOptions hp; hp.shape = FilterShape::Highpass; hp.cutoff =  200.0; hp.order = 4;

    auto [hdr, samples] = pipeline(g, {lp, hp});
    // Unit sine RMS ≈ 0.707; both filters pass 1 kHz, so expect > 0.5
    CHECK(steady_rms(samples, hdr.sample_rate) > 0.5);
}

TEST_CASE("generate | LP | HP eliminates a tone above both passbands", "[cli][pipeline]")
{
    // 10 kHz tone, LP at 4 kHz — the LP stage should remove it before HP runs
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 10000.0; g.duration = 2.0; g.sample_rate = 44100;

    FilterOptions lp; lp.shape = FilterShape::Lowpass;  lp.cutoff = 4000.0; lp.order = 6;
    FilterOptions hp; hp.shape = FilterShape::Highpass; hp.cutoff =  200.0; hp.order = 4;

    auto [hdr, samples] = pipeline(g, {lp, hp});
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.02);
}

TEST_CASE("generate | LP | HP eliminates a tone below both passbands", "[cli][pipeline]")
{
    // 50 Hz tone, HP at 200 Hz — the HP stage should remove it
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 50.0; g.duration = 2.0; g.sample_rate = 44100;

    FilterOptions lp; lp.shape = FilterShape::Lowpass;  lp.cutoff = 4000.0; lp.order = 4;
    FilterOptions hp; hp.shape = FilterShape::Highpass; hp.cutoff =  200.0; hp.order = 6;

    auto [hdr, samples] = pipeline(g, {lp, hp});
    CHECK(steady_rms(samples, hdr.sample_rate) < 0.02);
}

// ── Cascaded lowpass: narrower cutoff reduces output further ──────────────────

TEST_CASE("cascaded lowpass filters have additive rolloff in dB", "[cli][pipeline]")
{
    // A 3 kHz tone through one 1 kHz LP vs two 1 kHz LPs — the cascade should attenuate more
    GenerateOptions g;
    g.type = GenerateType::Sine; g.freq = 3000.0; g.duration = 2.0; g.sample_rate = 44100;

    FilterOptions lp; lp.shape = FilterShape::Lowpass; lp.cutoff = 1000.0; lp.order = 4;

    auto [h1, s1] = pipeline(g, {lp});
    auto [h2, s2] = pipeline(g, {lp, lp});

    CHECK(steady_rms(s2, h2.sample_rate) < steady_rms(s1, h1.sample_rate));
}

// ── Noise pipeline ────────────────────────────────────────────────────────────

TEST_CASE("generate white noise | lowpass reduces high-frequency energy", "[cli][pipeline]")
{
    // Without filter: wideband; with LP at 2 kHz: should reduce power significantly
    // because sr=44100 means the noise spans 0–22050 Hz and we keep only 0–2000 Hz.
    GenerateOptions g;
    g.type = GenerateType::White; g.amplitude = 1.0; g.duration = 2.0;
    g.sample_rate = 44100; g.seed = 42;

    FilterOptions lp; lp.shape = FilterShape::Lowpass; lp.cutoff = 2000.0; lp.order = 6;

    auto [h_raw,  s_raw]  = pipeline(g, {});
    auto [h_filt, s_filt] = pipeline(g, {lp});

    // The LP passes ~2/22 ≈ 9% of the bandwidth; power should drop noticeably
    CHECK(steady_rms(s_filt, h_filt.sample_rate) < steady_rms(s_raw, h_raw.sample_rate));
}
