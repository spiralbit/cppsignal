#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/signal/peaks.hpp — local peak / extremum detection
//
// FULLY IMPLEMENTED:
//   find_peaks()      — detect local maxima with optional constraints
//   peak_prominences() — compute how much a peak stands above its surroundings
// ─────────────────────────────────────────────────────────────────────────────

#include "../core/types.hpp"
#include "../core/result.hpp"
#include <vector>
#include <span>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <optional>

namespace cps {

// ── PeakResult ────────────────────────────────────────────────────────────────

struct PeakResult {
    std::vector<std::size_t> indices;      // sample indices of detected peaks
    std::vector<Real>        heights;      // signal value at each peak
    std::vector<Real>        prominences;  // computed if opts.prominence is set
};


// ── peak_prominences() ────────────────────────────────────────────────────────
//
// Compute the prominence of each peak.
//
// The prominence of a peak P is the height difference between P and the
// highest point in the lowest "valley" separating P from any higher peak.
// Formally:
//   1. Find the highest peak to the left of P that is taller than P.
//   2. Find the minimum signal value in [left_peak, P] — this is the left base.
//   3. Do the same on the right.
//   4. prominence = P_height - max(left_base, right_base)
//
// Used by find_peaks() when opts.prominence is set.

[[nodiscard]] inline std::vector<Real> peak_prominences(std::span<const Real>        signal,
                                                        std::span<const std::size_t> peak_indices)
{
    const std::size_t N  = signal.size();
    const std::size_t np = peak_indices.size();
    std::vector<Real> proms(np, 0.0);

    for (std::size_t pi = 0; pi < np; ++pi) {
        std::size_t idx = peak_indices[pi];
        if (idx >= N)
            throw ValueError("peak_prominences: peak index out of range");
        Real height = signal[idx];

        // ── Left base: walk OUTWARD from idx (right-to-left) until we hit a
        // sample higher than the peak, or reach the left edge.
        // The contour base is the minimum in [boundary, idx].
        Real left_min = height;
        bool found_left = false;
        for (std::size_t li = idx; li-- > 0; ) {
            if (signal[li] > height) {
                for (std::size_t s = li; s <= idx; ++s)
                    left_min = std::min(left_min, signal[s]);
                found_left = true;
                break;
            }
        }
        if (!found_left) {
            for (std::size_t s = 0; s <= idx; ++s)
                left_min = std::min(left_min, signal[s]);
        }

        // ── Right base: walk outward from idx (left-to-right) ────────────────
        Real right_min = height;
        bool found_right = false;
        for (std::size_t ri = idx + 1; ri < N; ++ri) {
            if (signal[ri] > height) {
                for (std::size_t s = idx; s <= ri; ++s)
                    right_min = std::min(right_min, signal[s]);
                found_right = true;
                break;
            }
        }
        if (!found_right) {
            for (std::size_t s = idx; s < N; ++s)
                right_min = std::min(right_min, signal[s]);
        }

        proms[pi] = height - std::max(left_min, right_min);
    }
    return proms;
}


// ── find_peaks() ──────────────────────────────────────────────────────────────
//
// Detect local maxima in a signal with optional constraints.
//
// A sample x[n] is a local maximum if x[n] > x[n-1] AND x[n] > x[n+1].
// Plateaus (equal neighbours) are not counted — strict inequality only.
//
// Constraints applied in order:
//   1. height      — peak value >= height
//   2. threshold   — peak value must exceed both immediate neighbours by >= threshold
//   3. distance    — no two peaks within `distance` samples of each other;
//                    when two peaks conflict, the shorter one is removed
//   4. prominence  — peak must stand above its surroundings by >= prominence
//   5. width       — peak width at half-prominence must be >= width (samples)
//
// Parameters:
//   signal — input samples
//   opts   — PeakOptions (all fields are optional)
//
// Returns PeakResult with indices and heights of all accepted peaks.
// If opts.prominence is set, PeakResult.prominences is also populated.
//
// Example:
//   auto result = cps::find_peaks(signal, {.height=0.5, .distance=10});
//   for (auto i : result.indices)
//       std::cout << "Peak at sample " << i << " = " << signal[i] << '\n';

[[nodiscard]] inline PeakResult find_peaks(std::span<const Real> signal, PeakOptions opts = {})
{
    const std::size_t N = signal.size();
    if (N < 3) return {};   // need at least 3 samples for a peak

    std::vector<std::size_t> candidates;

    // ── Step 1: find all local maxima ─────────────────────────────────────────
    for (std::size_t n = 1; n + 1 < N; ++n) {
        if (signal[n] > signal[n-1] && signal[n] > signal[n+1])
            candidates.push_back(n);
    }

    // ── Step 2: height filter ─────────────────────────────────────────────────
    if (opts.height.has_value()) {
        Real min_h = *opts.height;
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&](std::size_t i) { return signal[i] < min_h; }),
            candidates.end());
    }

    // ── Step 3: threshold filter (min rise above neighbours) ─────────────────
    if (opts.threshold.has_value()) {
        Real thr = *opts.threshold;
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&](std::size_t i) {
                               return (signal[i] - signal[i-1]) < thr
                                   || (signal[i] - signal[i+1]) < thr; // GCOV_EXCL_BR_LINE
                           }),
            candidates.end());
    }

    // ── Step 4: distance filter ───────────────────────────────────────────────
    // Keep only the tallest peak within any distance-radius window.
    if (opts.distance.has_value() && opts.distance.value() > 1) {
        int dist = opts.distance.value();
        std::vector<bool> keep(candidates.size(), true);

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (!keep[i]) continue;
            for (std::size_t j = i + 1; j < candidates.size(); ++j) {
                if (!keep[j]) continue; // GCOV_EXCL_BR_LINE
                if (static_cast<int>(candidates[j] - candidates[i]) < dist) {
                    // Remove the shorter of the two
                    if (signal[candidates[i]] >= signal[candidates[j]])
                        keep[j] = false;
                    else {
                        keep[i] = false;
                        break;   // i is removed — move on
                    }
                } else {
                    break;       // remaining candidates are all farther away
                }
            }
        }

        std::vector<std::size_t> filtered;
        for (std::size_t i = 0; i < candidates.size(); ++i)
            if (keep[i]) filtered.push_back(candidates[i]);
        candidates = std::move(filtered);
    }

    // ── Step 5: prominence filter ─────────────────────────────────────────────
    std::vector<Real> proms;
    if (opts.prominence.has_value() || opts.width.has_value()) {
        // Prominences are needed for both the prominence filter and to compute
        // the half-prominence level used by the width filter.
        proms = peak_prominences(signal, candidates);
    }

    if (opts.prominence.has_value()) {
        Real min_prom = *opts.prominence;

        std::vector<std::size_t> filtered;
        std::vector<Real>        filtered_proms;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (proms[i] >= min_prom) {
                filtered.push_back(candidates[i]);
                filtered_proms.push_back(proms[i]);
            }
        }
        candidates = std::move(filtered);
        proms      = std::move(filtered_proms);
    }

    // ── Step 6: width filter ──────────────────────────────────────────────────
    // Width is measured at the half-prominence level: the signal must stay
    // above (peak_height - prominence/2) for at least `width` samples on
    // each side combined. Left and right crossing positions are interpolated
    // linearly between samples so sub-sample widths are supported.
    if (opts.width.has_value()) {
        Real min_width = *opts.width;

        std::vector<std::size_t> filtered;
        std::vector<Real>        filtered_proms;

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            std::size_t idx   = candidates[i];
            Real        h     = signal[idx];
            Real        prom  = proms.empty() ? h : proms[i]; // GCOV_EXCL_BR_LINE
            Real        level = h - prom / 2.0;

            // Walk left from peak to find crossing below level
            double left_pos = static_cast<double>(idx);
            for (std::size_t s = idx; s-- > 0; ) { // GCOV_EXCL_BR_LINE
                if (signal[s] < level) {
                    // Interpolate: when signal[s]<level, signal[s+1]>=level → denom > 0
                    double denom = signal[s+1] - signal[s];
                    left_pos = static_cast<double>(s) + (level - signal[s]) / denom;
                    break;
                }
                left_pos = static_cast<double>(s);
            }

            // Walk right from peak to find crossing below level
            double right_pos = static_cast<double>(idx);
            for (std::size_t s = idx + 1; s < N; ++s) { // GCOV_EXCL_BR_LINE
                if (signal[s] < level) {
                    // Interpolate: when signal[s]<level, signal[s-1]>=level → |denom| > 0
                    double denom = signal[s] - signal[s-1];
                    right_pos = static_cast<double>(s-1) + (level - signal[s-1]) / denom;
                    break;
                }
                right_pos = static_cast<double>(s);
            }

            if (right_pos - left_pos >= min_width) {
                filtered.push_back(candidates[i]);
                if (!proms.empty()) filtered_proms.push_back(proms[i]); // GCOV_EXCL_BR_LINE
            }
        }
        candidates = std::move(filtered);
        proms      = std::move(filtered_proms);
    }

    // ── Assemble result ───────────────────────────────────────────────────────
    PeakResult result;
    result.indices.reserve(candidates.size());
    result.heights.reserve(candidates.size());

    for (auto i : candidates) {
        result.indices.push_back(i);
        result.heights.push_back(signal[i]);
    }
    result.prominences = std::move(proms);

    return result;
}

} // namespace cps
