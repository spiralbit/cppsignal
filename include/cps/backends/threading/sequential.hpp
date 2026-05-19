#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/backends/threading/sequential.hpp — single-threaded execution backend
//
// The default threading backend. Simply iterates in a plain for-loop.
// Used as the default because it adds zero overhead and has no dependencies.
//
// To add parallelism, swap in the std_exec.hpp backend or write your own
// that satisfies the cps::ThreadingBackend concept.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <concepts>

namespace cps::backends {

struct Sequential {
    // Call f(i) for each i in [0, n), in order, on the calling thread.
    template<std::invocable<std::size_t> Fn>
    void parallel_for(std::size_t n, Fn&& f) const {
        for (std::size_t i = 0; i < n; ++i)
            f(i);
    }
};

} // namespace cps::backends
