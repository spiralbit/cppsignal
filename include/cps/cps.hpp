#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/cps.hpp — single-header convenience include for the whole library
//
// Usage: #include <cps/cps.hpp>
//
// If you only need part of the library, include the specific headers instead
// to keep compile times low.
// ─────────────────────────────────────────────────────────────────────────────

// Core
#include "core/types.hpp"
#include "core/concepts.hpp"
#include "core/result.hpp"

// Backends (default: PocketFFT, Sequential, StandardAlloc)
#include "backends/fft/pocketfft.hpp"
#include "backends/threading/sequential.hpp"
#include "backends/alloc/standard.hpp"

// Filter design, application, and analysis
#include "filter/design.hpp"
#include "filter/apply.hpp"
#include "filter/analysis.hpp"

// Spectral analysis
#include "spectral/windows.hpp"
#include "spectral/fft.hpp"
#include "spectral/psd.hpp"
#include "spectral/stft.hpp"

// Signal utilities
#include "signal/generate.hpp"
#include "signal/peaks.hpp"
#include "signal/correlate.hpp"
#include "signal/resample.hpp"

// Measurement metrics
#include "measure/metrics.hpp"
