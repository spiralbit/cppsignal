#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/core/concepts.hpp — C++20 concepts that define pluggable backend contracts
//
// Every backend in cps/backends/ must satisfy one of these concepts. The
// concepts are checked at compile time via static_assert or requires-clauses,
// giving clean error messages when a custom backend is ill-formed.
//
// HOW PLUGGABLE BACKENDS WORK
// ───────────────────────────
// API functions are templated on a backend type that defaults to the built-in
// (PocketFFT, sequential threading, std::allocator). To swap in a different
// backend, pass it explicitly:
//
//   // default: PocketFFT
//   auto spec = cps::fft(signal);
//
//   // explicit: FFTW (requires CPS_ENABLE_FFTW and FFTW licence)
//   auto spec = cps::fft<cps::backends::FFTW>(signal);
//
// ─────────────────────────────────────────────────────────────────────────────

#include <span>
#include <complex>
#include <cstddef>
#include <concepts>

namespace cps {

// ── FFTBackend concept ───────────────────────────────────────────────────────
//
// A type B satisfies FFTBackend if it provides:
//
//   forward(in, out)  — complex DFT, no normalisation (matches numpy/scipy
//                       convention: inverse divides by N, forward does not)
//   inverse(in, out)  — complex IDFT, divides each element by N
//   rfft(in, out)     — real-to-complex DFT, out has length N/2+1
//   irfft(in, out, N) — complex-to-real IDFT; N is the original signal length
//                       (needed because N and N-1 map to the same rfft size)
//   name()            — human-readable string for diagnostics
//
// All spans must already be the right size; backends do not allocate.

template<typename B>
concept FFTBackend = requires(
    B                                          backend,
    std::span<const std::complex<double>>      cx_in,
    std::span<std::complex<double>>            cx_out,
    std::span<const double>                    real_in,
    std::span<std::complex<double>>            cx_out2,
    std::span<const std::complex<double>>      cx_in2,
    std::span<double>                          real_out,
    std::size_t                                n)
{
    { backend.forward(cx_in,  cx_out)      } -> std::same_as<void>;
    { backend.inverse(cx_in,  cx_out)      } -> std::same_as<void>;
    { backend.rfft   (real_in, cx_out2)    } -> std::same_as<void>;
    { backend.irfft  (cx_in2, real_out, n) } -> std::same_as<void>;
    { B::name()                            } -> std::convertible_to<std::string_view>;
};

// ── ThreadingBackend concept ─────────────────────────────────────────────────
//
// A threading backend provides a parallel_for that iterates [0, n) and calls
// f(i) for each i. The sequential backend just calls f in a plain loop.
// A TBB or std::execution backend would distribute work across threads.

template<typename B>
concept ThreadingBackend = requires(B backend, std::size_t n)
{
    // parallel_for(n, f): call f(i) for i in [0, n), possibly in parallel
    { backend.parallel_for(n, [](std::size_t){}) } -> std::same_as<void>;
};

// ── AllocBackend concept ─────────────────────────────────────────────────────
//
// An allocator backend wraps memory allocation. The default uses the heap.
// An embedded target might supply a pool allocator here.
//
// allocate<T>(n)     — return a T* pointing to n default-initialised elements
// deallocate(ptr, n) — release memory previously obtained from allocate

template<typename B>
concept AllocBackend = requires(B backend, std::size_t n)
{
    { backend.template allocate<double>(n)          } -> std::same_as<double*>;
    { backend.deallocate(std::declval<double*>(), n) } -> std::same_as<void>;
};

// ── ResampleBackend concept (forward declaration for resample module) ────────

template<typename B>
concept ResampleBackend = requires(
    B                         backend,
    std::span<const double>   in,
    std::span<double>         out,
    int up, int down)
{
    // rational(in, out, up, down): resample by the rational factor up/down
    { backend.rational(in, out, up, down) } -> std::same_as<void>;
};

} // namespace cps
