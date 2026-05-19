#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/backends/alloc/standard.hpp — default heap allocator backend
//
// Wraps new[]/delete[] so internal buffers can be swapped out for pool
// allocators on embedded targets without changing any other code.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <new>

namespace cps::backends {

struct StandardAlloc {
    template<typename T>
    T* allocate(std::size_t n) {
        return new T[n]{};   // zero-initialise
    }

    template<typename T>
    void deallocate(T* ptr, std::size_t /*n*/) {
        delete[] ptr;
    }
};

} // namespace cps::backends
