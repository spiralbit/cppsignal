#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/backends/threading/std_exec.hpp — C++23 std::execution threading backend
//
// STATUS: STUB — not yet implemented.
//
// C++23 adds std::execution with parallel algorithms:
//   std::for_each(std::execution::par_unseq, ...);
//
// When compiler support matures (GCC 14+, Clang 18+ with TBB or oneDPL),
// this backend will distribute parallel_for across hardware threads using
// the standard library, with no TBB/OpenMP dependency.
//
// For now, enable CPS_ENABLE_STD_EXECUTION at your own risk — it falls back
// to Sequential until this file is filled in.
// ─────────────────────────────────────────────────────────────────────────────

#include "sequential.hpp"   // fall back to sequential for now

namespace cps::backends {

// TODO: replace this alias with a real parallel implementation once
// std::execution::par is available on all three target platforms.
using StdExecution = Sequential;

} // namespace cps::backends
