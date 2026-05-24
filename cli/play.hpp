#pragma once
#include "format.hpp"
#include <iosfwd>

namespace cps::cli {

// Read a CPS stream from `in` and play it on the default audio device.
// Blocks until playback is complete.
void play_stream(std::istream& in);

} // namespace cps::cli
