#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cps/cps.hpp>
#include <cli/format.hpp>
#include <cli/generate.hpp>

using namespace cps::cli;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Test helpers ──────────────────────────────────────────────────────────────

static std::pair<StreamHeader, std::vector<float>>
run_generate(GenerateOptions opts)
{
    std::ostringstream buf;
    generate(buf, opts);
    std::istringstream in(buf.str());
    auto hdr     = read_header(in);
    auto samples = read_samples_f32(in, hdr);
    return {hdr, samples};
}

// Return the dominant frequency in Hz found via rfft magnitude peak.
static double dominant_freq(const std::vector<float>& samples, uint32_t sr)
{
    std::vector<cps::Real> x(samples.begin(), samples.end());
    auto spectrum = cps::rfft(x);
    auto freqs    = cps::rfftfreq(x.size(), 1.0 / static_cast<double>(sr));

    auto peak = std::max_element(spectrum.begin(), spectrum.end(),
        [](const auto& a, const auto& b){ return std::abs(a) < std::abs(b); });
    return freqs[static_cast<std::size_t>(std::distance(spectrum.begin(), peak))];
}

static double rms(const std::vector<float>& v)
{
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    return std::sqrt(sum / static_cast<double>(v.size()));
}

// ── Header ────────────────────────────────────────────────────────────────────

TEST_CASE("generate writes a valid stream header", "[cli][generate]")
{
    GenerateOptions opts;
    opts.type        = GenerateType::Sine;
    opts.freq        = 440.0;
    opts.duration    = 0.1;
    opts.sample_rate = 44100;

    auto [hdr, samples] = run_generate(opts);

    CHECK(hdr.sample_rate == 44100u);
    CHECK(hdr.channels    == 1u);
    CHECK(hdr.format      == SampleFormat::Float32);
    CHECK_FALSE(samples.empty());
}

// ── Sample count ──────────────────────────────────────────────────────────────

TEST_CASE("generate produces exactly floor(duration * sample_rate) samples", "[cli][generate]")
{
    for (double dur : {0.1, 0.5, 1.0, 2.0}) {
        GenerateOptions opts;
        opts.type        = GenerateType::Sine;
        opts.freq        = 440.0;
        opts.duration    = dur;
        opts.sample_rate = 44100;

        auto [hdr, samples] = run_generate(opts);
        auto expected = static_cast<std::size_t>(dur * 44100);
        CHECK(samples.size() == expected);
    }
}

TEST_CASE("generate respects non-default sample rates", "[cli][generate]")
{
    for (uint32_t sr : {8000u, 22050u, 48000u, 96000u}) {
        GenerateOptions opts;
        opts.type        = GenerateType::Sine;
        opts.freq        = 440.0;
        opts.duration    = 1.0;
        opts.sample_rate = sr;

        auto [hdr, samples] = run_generate(opts);
        CHECK(hdr.sample_rate == sr);
        CHECK(samples.size()  == static_cast<std::size_t>(sr));
    }
}

// ── Sine ──────────────────────────────────────────────────────────────────────

TEST_CASE("generate sine dominant frequency matches --freq", "[cli][generate][sine]")
{
    for (double freq : {220.0, 440.0, 1000.0, 4000.0}) {
        GenerateOptions opts;
        opts.type        = GenerateType::Sine;
        opts.freq        = freq;
        opts.duration    = 1.0;
        opts.sample_rate = 44100;

        auto [hdr, samples] = run_generate(opts);
        CHECK_THAT(dominant_freq(samples, hdr.sample_rate), WithinRel(freq, 0.01));
    }
}

TEST_CASE("generate sine peak amplitude matches --amplitude", "[cli][generate][sine]")
{
    for (double amp : {0.1, 0.5, 0.8, 1.0}) {
        GenerateOptions opts;
        opts.type      = GenerateType::Sine;
        opts.freq      = 440.0;
        opts.amplitude = amp;
        opts.duration  = 1.0;

        auto [hdr, samples] = run_generate(opts);
        float peak = *std::max_element(samples.begin(), samples.end());
        CHECK_THAT(static_cast<double>(peak), WithinAbs(amp, 0.01));
    }
}

TEST_CASE("generate sine RMS is amplitude / sqrt(2)", "[cli][generate][sine]")
{
    GenerateOptions opts;
    opts.type      = GenerateType::Sine;
    opts.freq      = 440.0;
    opts.amplitude = 1.0;
    opts.duration  = 1.0;

    auto [hdr, samples] = run_generate(opts);
    CHECK_THAT(rms(samples), WithinAbs(1.0 / std::sqrt(2.0), 0.01));
}

TEST_CASE("generate sine samples are bounded by amplitude", "[cli][generate][sine]")
{
    GenerateOptions opts;
    opts.type      = GenerateType::Sine;
    opts.freq      = 440.0;
    opts.amplitude = 0.9;
    opts.duration  = 1.0;

    auto [hdr, samples] = run_generate(opts);
    for (float s : samples)
        CHECK(std::abs(s) <= 0.9f + 1e-5f);
}

// ── Chirp ─────────────────────────────────────────────────────────────────────

TEST_CASE("generate chirp dominant frequency at start is near freq_start", "[cli][generate][chirp]")
{
    GenerateOptions opts;
    opts.type        = GenerateType::Chirp;
    opts.freq_start  = 200.0;
    opts.freq_end    = 2000.0;
    opts.duration    = 2.0;
    opts.sample_rate = 44100;

    auto [hdr, samples] = run_generate(opts);

    // Examine the first 50 ms — at this point the chirp is still near freq_start
    std::size_t window = hdr.sample_rate / 20;
    std::vector<float> head(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(window));
    CHECK_THAT(dominant_freq(head, hdr.sample_rate), WithinRel(opts.freq_start, 0.20));
}

TEST_CASE("generate chirp dominant frequency at end is near freq_end", "[cli][generate][chirp]")
{
    GenerateOptions opts;
    opts.type        = GenerateType::Chirp;
    opts.freq_start  = 200.0;
    opts.freq_end    = 2000.0;
    opts.duration    = 2.0;
    opts.sample_rate = 44100;

    auto [hdr, samples] = run_generate(opts);

    std::size_t window = hdr.sample_rate / 20;
    std::vector<float> tail(samples.end() - static_cast<std::ptrdiff_t>(window), samples.end());
    CHECK_THAT(dominant_freq(tail, hdr.sample_rate), WithinRel(opts.freq_end, 0.20));
}

// ── White noise ───────────────────────────────────────────────────────────────

TEST_CASE("generate white noise RMS is approximately equal to amplitude (std_dev)", "[cli][generate][noise]")
{
    // white_noise uses amplitude as Gaussian std_dev — so RMS ≈ amplitude
    GenerateOptions opts;
    opts.type      = GenerateType::White;
    opts.amplitude = 0.5;
    opts.duration  = 2.0;
    opts.seed      = 42;

    auto [hdr, samples] = run_generate(opts);
    CHECK_THAT(rms(samples), WithinRel(0.5, 0.15));  // within 15%
}

TEST_CASE("generate white noise with same seed produces identical output", "[cli][generate][noise]")
{
    GenerateOptions opts;
    opts.type     = GenerateType::White;
    opts.duration = 0.1;
    opts.seed     = 123;

    auto [h1, s1] = run_generate(opts);
    auto [h2, s2] = run_generate(opts);

    REQUIRE(s1.size() == s2.size());
    for (std::size_t i = 0; i < s1.size(); ++i)
        CHECK(s1[i] == s2[i]);
}

TEST_CASE("generate white noise with different seeds produces different output", "[cli][generate][noise]")
{
    GenerateOptions a, b;
    a.type = b.type = GenerateType::White;
    a.duration = b.duration = 0.5;
    a.seed = 1;
    b.seed = 2;

    auto [h1, s1] = run_generate(a);
    auto [h2, s2] = run_generate(b);

    bool any_different = false;
    for (std::size_t i = 0; i < s1.size(); ++i)
        if (s1[i] != s2[i]) { any_different = true; break; }
    CHECK(any_different);
}

// ── Pink noise ────────────────────────────────────────────────────────────────

TEST_CASE("generate pink noise has more spectral energy in low frequencies than high", "[cli][generate][noise]")
{
    GenerateOptions opts;
    opts.type        = GenerateType::Pink;
    opts.duration    = 4.0;
    opts.seed        = 42;
    opts.sample_rate = 44100;

    auto [hdr, samples] = run_generate(opts);
    std::vector<cps::Real> x(samples.begin(), samples.end());
    auto spectrum = cps::rfft(x);

    std::size_t n = spectrum.size();
    double low_energy = 0.0, high_energy = 0.0;
    for (std::size_t i = 1;     i < n / 4;  ++i) low_energy  += std::abs(spectrum[i]);
    for (std::size_t i = n * 3 / 4; i < n;  ++i) high_energy += std::abs(spectrum[i]);

    CHECK(low_energy > high_energy);
}

// ── Argument parsing (RED — parse_generate_args not yet implemented) ───────────

TEST_CASE("parse_generate_args throws on zero duration", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--type", "sine", "--duration", "0"};
    CHECK_THROWS_AS(
        parse_generate_args(6, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_generate_args throws on negative duration", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--type", "sine", "--duration", "-1"};
    CHECK_THROWS_AS(
        parse_generate_args(6, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_generate_args throws on zero sample rate", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--sample-rate", "0"};
    CHECK_THROWS_AS(
        parse_generate_args(4, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_generate_args throws on unknown generator type", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--type", "triangle"};
    CHECK_THROWS_AS(
        parse_generate_args(4, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_generate_args throws when sine frequency exceeds Nyquist", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--type", "sine",
                          "--freq", "30000", "--sample-rate", "44100"};
    CHECK_THROWS_AS(
        parse_generate_args(8, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_generate_args throws on non-positive amplitude", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--amplitude", "0"};
    CHECK_THROWS_AS(
        parse_generate_args(4, const_cast<char**>(argv)),
        std::invalid_argument);
}

TEST_CASE("parse_generate_args accepts valid sine options and returns correct defaults", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--type", "sine",
                          "--freq", "440", "--duration", "1.0"};
    auto opts = parse_generate_args(8, const_cast<char**>(argv));

    CHECK(opts.type        == GenerateType::Sine);
    CHECK_THAT(opts.freq,     WithinAbs(440.0, 1e-9));
    CHECK_THAT(opts.duration, WithinAbs(1.0,   1e-9));
    CHECK(opts.sample_rate == 44100u);  // default
    CHECK(opts.channels    == 1u);      // default
    CHECK_THAT(opts.amplitude, WithinAbs(1.0, 1e-9)); // default
}

TEST_CASE("parse_generate_args accepts --sample-rate override", "[cli][generate][args]")
{
    const char* argv[] = {"cps", "generate", "--type", "sine",
                          "--freq", "440", "--duration", "1",
                          "--sample-rate", "48000"};
    auto opts = parse_generate_args(10, const_cast<char**>(argv));
    CHECK(opts.sample_rate == 48000u);
}
