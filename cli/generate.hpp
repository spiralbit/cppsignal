#pragma once
#include "format.hpp"
#include <iosfwd>

namespace cps::cli {

enum class GenerateType { Sine, Chirp, White, Pink };

struct GenerateOptions {
    GenerateType type        = GenerateType::Sine;
    double       freq        = 440.0;   // Hz — sine only
    double       freq_start  = 100.0;   // Hz — chirp start
    double       freq_end    = 1000.0;  // Hz — chirp end
    double       amplitude   = 1.0;     // linear peak amplitude
    double       duration    = 1.0;     // seconds
    uint32_t     sample_rate = 44100;   // Hz
    uint16_t     channels    = 1;
    unsigned     seed        = 0;       // noise reproducibility
    SampleFormat format      = SampleFormat::Float32;
};

// Generate a signal and write header + samples to out.
// Returns the number of samples written per channel.
std::size_t generate(std::ostream& out, const GenerateOptions& opts);

// Parse CLI argv into GenerateOptions.
// Throws std::invalid_argument on missing or out-of-range arguments.
GenerateOptions parse_generate_args(int argc, char** argv);

} // namespace cps::cli
