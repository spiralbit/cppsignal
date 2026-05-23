#pragma once
#include "format.hpp"
#include <iosfwd>

namespace cps::cli {

enum class FilterShape { Lowpass, Highpass, Bandpass, Bandstop };
enum class FilterImpl  { Butter, FirWin };

struct FilterOptions {
    FilterImpl  impl        = FilterImpl::Butter;
    FilterShape shape       = FilterShape::Lowpass;
    int         order       = 4;    // IIR filter order
    int         taps        = 101;  // FIR tap count (odd recommended)
    double      cutoff      = 0.0;  // Hz — lowpass / highpass
    double      cutoff_low  = 0.0;  // Hz — bandpass / bandstop lower edge
    double      cutoff_high = 0.0;  // Hz — bandpass / bandstop upper edge
};

// Read a CPS stream from in, apply the filter, write the result to out.
// Reads sample_rate from the stream header; cutoff frequencies must be in Hz.
// Throws std::runtime_error if the stream header is invalid.
// Returns the number of samples written per channel.
std::size_t filter_stream(std::istream& in, std::ostream& out, const FilterOptions& opts);

// Parse CLI argv into FilterOptions.
// Throws std::invalid_argument on missing, ambiguous, or out-of-range arguments.
FilterOptions parse_filter_args(int argc, char** argv);

} // namespace cps::cli
