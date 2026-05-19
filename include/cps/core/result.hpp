#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/core/result.hpp — error types and exception hierarchy
//
// CppSignal uses exceptions for error reporting. All exceptions derive from
// cps::Error so callers can catch the whole library with a single handler:
//
//   try {
//       auto sos = cps::butter(4, 600.0, cps::FilterType::Lowpass, {.fs=1000});
//   } catch (const cps::Error& e) {
//       std::cerr << "cps error: " << e.what() << '\n';
//   }
//
// In the future this header may also expose a non-throwing API using
// std::expected (C++23). For now, exceptions keep the code simple.
// ─────────────────────────────────────────────────────────────────────────────

#include <stdexcept>
#include <string>

namespace cps {

// ── Base exception ───────────────────────────────────────────────────────────
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ── Specific error kinds ─────────────────────────────────────────────────────

// Bad argument value: e.g. filter order <= 0, Wn outside (0,1), etc.
struct ValueError : Error {
    using Error::Error;
};

// Operation not yet implemented (stub functions throw this).
struct NotImplemented : Error {
    explicit NotImplemented(std::string_view what)
        : Error(std::string("not implemented: ") + std::string(what)) {}
};

// Numerical failure: e.g. filter design produced unstable poles.
struct NumericalError : Error {
    using Error::Error;
};

// Backend not available: e.g. calling FFTW backend when not compiled in.
struct BackendUnavailable : Error {
    using Error::Error;
};

} // namespace cps
