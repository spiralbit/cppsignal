#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/signal/resample.hpp — sample-rate conversion
//
// STATUS: STUB — not yet implemented.
//
// Resampling by a rational factor p/q requires:
//   1. Upsample by p (insert p-1 zeros between each sample)
//   2. Anti-alias lowpass filter at min(1/p, 1/q) * Nyquist
//   3. Downsample by q (keep every q-th sample)
//
// An efficient implementation uses a polyphase filter bank to avoid computing
// the filtered zeros (which are always zero after upsampling).
//
// PLAN:
//   decimate(x, q)          — integer downsampling with anti-alias filter
//   interpolate(x, p)       — integer upsampling with interpolation filter
//   resample(x, p, q)       — rational p/q resampling via polyphase filter bank
//   resample_poly(x, p, q)  — polyphase implementation (more efficient)
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <vector>
#include <span>

namespace cps {

// Downsample by integer factor q, with anti-alias lowpass filter applied first.
// The filter prevents aliasing of frequencies above fs/(2*q).
inline std::vector<Real> decimate(std::span<const Real> /*x*/, int /*q*/)
{
    throw NotImplemented("decimate");
}

// Upsample by integer factor p, filling new samples by linear or sinc interpolation.
inline std::vector<Real> interpolate(std::span<const Real> /*x*/, int /*p*/)
{
    throw NotImplemented("interpolate");
}

// Resample by rational factor p/q (= new_rate / old_rate).
// Equivalent to upsample by p → filter → downsample by q.
inline std::vector<Real> resample(std::span<const Real> /*x*/, int /*p*/, int /*q*/)
{
    throw NotImplemented("resample");
}

} // namespace cps
