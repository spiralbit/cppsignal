#pragma once
#include <cstdint>
#include <iosfwd>
#include <span>
#include <stdexcept>
#include <vector>

namespace cps::cli {

// ── Wire format ───────────────────────────────────────────────────────────────
//
// A CPS stream is a binary blob that flows between pipeline stages.
//
// Header (12 bytes, little-endian):
//   [0-3]  magic      "CPS\0"
//   [4]    version    uint8  — currently 1
//   [5]    format     uint8  — 0 = float32, 1 = float64
//   [6-7]  channels   uint16 — 1 = mono, 2 = stereo
//   [8-11] sample_rate uint32 — Hz
//
// Body: N * channels * sizeof(format) bytes, interleaved samples, little-endian.

enum class SampleFormat : uint8_t { Float32 = 0, Float64 = 1 };

struct StreamHeader {
    uint32_t     sample_rate = 44100;
    uint16_t     channels    = 1;
    SampleFormat format      = SampleFormat::Float32;
};

// Write a 12-byte header to out.
void write_header(std::ostream& out, const StreamHeader& h);

// Read and validate a header from in.
// Throws std::runtime_error on bad magic, unsupported version, or truncation.
StreamHeader read_header(std::istream& in);

// Write samples to out in the format recorded in h.
void write_samples(std::ostream& out, std::span<const float>  samples, const StreamHeader& h);
void write_samples(std::ostream& out, std::span<const double> samples, const StreamHeader& h);

// Read all remaining bytes from in and return them as float32.
// Converts from float64 if h.format == Float64.
std::vector<float> read_samples_f32(std::istream& in, const StreamHeader& h);

} // namespace cps::cli
