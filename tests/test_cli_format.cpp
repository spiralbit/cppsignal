#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <cli/format.hpp>

using namespace cps::cli;

// ── Header round-trip ─────────────────────────────────────────────────────────

TEST_CASE("stream header round-trips through binary encode/decode", "[cli][format]")
{
    StreamHeader original;
    original.sample_rate = 48000;
    original.channels    = 2;
    original.format      = SampleFormat::Float32;

    std::ostringstream out;
    write_header(out, original);

    std::istringstream in(out.str());
    auto decoded = read_header(in);

    CHECK(decoded.sample_rate == 48000u);
    CHECK(decoded.channels    == 2u);
    CHECK(decoded.format      == SampleFormat::Float32);
}

TEST_CASE("stream header encodes as exactly 12 bytes", "[cli][format]")
{
    std::ostringstream out;
    write_header(out, StreamHeader{});
    // magic(4) + version(1) + format(1) + channels(2) + sample_rate(4) = 12
    CHECK(out.str().size() == 12u);
}

TEST_CASE("stream header round-trips float64 format", "[cli][format]")
{
    StreamHeader h;
    h.format = SampleFormat::Float64;

    std::ostringstream out;
    write_header(out, h);
    std::istringstream in(out.str());
    CHECK(read_header(in).format == SampleFormat::Float64);
}

TEST_CASE("stream header round-trips non-default sample rates", "[cli][format]")
{
    for (uint32_t sr : {8000u, 22050u, 44100u, 48000u, 96000u}) {
        StreamHeader h;
        h.sample_rate = sr;

        std::ostringstream out;
        write_header(out, h);
        std::istringstream in(out.str());
        CHECK(read_header(in).sample_rate == sr);
    }
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST_CASE("read_header rejects wrong magic bytes", "[cli][format]")
{
    StreamHeader h;
    std::ostringstream buf;
    write_header(buf, h);
    std::string raw = buf.str();
    raw[0] = 'X';  // corrupt first magic byte

    std::istringstream in(raw);
    CHECK_THROWS_AS(read_header(in), std::runtime_error);
}

TEST_CASE("read_header rejects unsupported version", "[cli][format]")
{
    StreamHeader h;
    std::ostringstream buf;
    write_header(buf, h);
    std::string raw = buf.str();
    raw[4] = static_cast<char>(99);  // version byte is at offset 4

    std::istringstream in(raw);
    CHECK_THROWS_AS(read_header(in), std::runtime_error);
}

TEST_CASE("read_header rejects a truncated stream", "[cli][format]")
{
    StreamHeader h;
    std::ostringstream buf;
    write_header(buf, h);
    std::string raw = buf.str().substr(0, 6);  // cut short

    std::istringstream in(raw);
    CHECK_THROWS_AS(read_header(in), std::runtime_error);
}

TEST_CASE("read_header rejects an empty stream", "[cli][format]")
{
    std::istringstream in("");
    CHECK_THROWS_AS(read_header(in), std::runtime_error);
}

// ── Sample encoding ───────────────────────────────────────────────────────────

TEST_CASE("float32 samples round-trip through write/read", "[cli][format]")
{
    StreamHeader h;
    h.format = SampleFormat::Float32;

    const std::vector<float> original = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};

    std::ostringstream buf;
    write_header(buf, h);
    write_samples(buf, std::span<const float>(original), h);

    std::istringstream in(buf.str());
    auto hdr     = read_header(in);
    auto decoded = read_samples_f32(in, hdr);

    REQUIRE(decoded.size() == original.size());
    for (std::size_t i = 0; i < original.size(); ++i)
        CHECK(decoded[i] == original[i]);
}

TEST_CASE("float64 samples round-trip through write/read as float32", "[cli][format]")
{
    StreamHeader h;
    h.format = SampleFormat::Float64;

    const std::vector<double> original = {0.0, 0.25, -0.25, 0.75, -0.75};

    std::ostringstream buf;
    write_header(buf, h);
    write_samples(buf, std::span<const double>(original), h);

    std::istringstream in(buf.str());
    auto hdr     = read_header(in);
    auto decoded = read_samples_f32(in, hdr);

    REQUIRE(decoded.size() == original.size());
    for (std::size_t i = 0; i < original.size(); ++i)
        CHECK(decoded[i] == static_cast<float>(original[i]));
}

TEST_CASE("read_samples_f32 returns empty vector for stream with no body", "[cli][format]")
{
    StreamHeader h;
    std::ostringstream buf;
    write_header(buf, h);

    std::istringstream in(buf.str());
    auto hdr     = read_header(in);
    auto decoded = read_samples_f32(in, hdr);

    CHECK(decoded.empty());
}
