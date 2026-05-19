#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/backends/fft/pocketfft.hpp — default FFT backend
//
// Preferred path: wraps PocketFFT (BSD-3) by Martin Reinecke (Max Planck
// Institute). Used inside NumPy since v1.17 — production-quality, handles
// any size N (not just powers of two), single header, no dependencies.
//
// Fallback path: if PocketFFT is not on the include path (e.g. building
// offline without CMake FetchContent), the name `PocketFFT` is aliased to
// `BuiltinFFT` — our self-contained Cooley-Tukey implementation that has no
// external dependencies at all.  Power-of-2 sizes are O(N log N); all others
// fall back to O(N²) naive DFT (see builtin.hpp for details).
//
// This detection is done at compile time via __has_include so no CMake
// variables or preprocessor flags are required — it just works.
// ─────────────────────────────────────────────────────────────────────────────

#if __has_include("pocketfft_hdronly.h")

// ── Preferred: PocketFFT available ───────────────────────────────────────────

#include "pocketfft_hdronly.h"
#include <span>
#include <vector>
#include <complex>
#include <string_view>
#include <cstddef>
#include <cassert>

namespace cps::backends {

struct PocketFFT {

    // ── FFTBackend concept interface ─────────────────────────────────────────

    // Complex forward DFT: out[k] = sum_n in[n] * exp(-2πi*k*n/N)
    // No normalisation (matches numpy/scipy forward convention).
    void forward(std::span<const std::complex<double>> in,
                 std::span<std::complex<double>>       out) const
    {
        assert(in.size() == out.size());
        const std::size_t N = in.size();

        pocketfft::shape_t  shape = {N};
        pocketfft::stride_t s_in  = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
        pocketfft::stride_t s_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
        pocketfft::shape_t  axes  = {0};

        pocketfft::c2c(shape, s_in, s_out, axes,
                       /*forward=*/true,
                       in.data(), out.data(),
                       /*fct=*/1.0);
    }

    // Complex inverse DFT: out[n] = (1/N) * sum_k in[k] * exp(+2πi*k*n/N)
    void inverse(std::span<const std::complex<double>> in,
                 std::span<std::complex<double>>       out) const
    {
        assert(in.size() == out.size());
        const std::size_t N = in.size();

        pocketfft::shape_t  shape = {N};
        pocketfft::stride_t s_in  = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
        pocketfft::stride_t s_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
        pocketfft::shape_t  axes  = {0};

        pocketfft::c2c(shape, s_in, s_out, axes,
                       /*forward=*/false,
                       in.data(), out.data(),
                       /*fct=*/1.0 / static_cast<double>(N));
    }

    // Real-to-complex forward DFT.
    // Output length is N/2+1 (non-redundant positive frequencies only).
    void rfft(std::span<const double>         in,
              std::span<std::complex<double>> out) const
    {
        const std::size_t N = in.size();
        assert(out.size() == N / 2 + 1);

        pocketfft::shape_t  shape = {N};
        pocketfft::stride_t s_in  = {static_cast<std::ptrdiff_t>(sizeof(double))};
        pocketfft::stride_t s_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};

        pocketfft::r2c(shape, s_in, s_out, /*axis=*/0,
                       /*forward=*/true,
                       in.data(), out.data(),
                       /*fct=*/1.0);
    }

    // Complex-to-real inverse DFT.
    // n_original: length of the original real signal (cannot be inferred from
    // the rfft output length alone — both N=8 and N=9 give length-5 spectra).
    void irfft(std::span<const std::complex<double>> in,
               std::span<double>                    out,
               std::size_t                          n_original) const
    {
        assert(out.size() == n_original);

        pocketfft::shape_t  shape = {n_original};
        pocketfft::stride_t s_in  = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
        pocketfft::stride_t s_out = {static_cast<std::ptrdiff_t>(sizeof(double))};

        pocketfft::c2r(shape, s_in, s_out, /*axis=*/0,
                       /*forward=*/false,
                       in.data(), out.data(),
                       /*fct=*/1.0 / static_cast<double>(n_original));
    }

    static constexpr std::string_view name() { return "PocketFFT"; }
};

} // namespace cps::backends

#else // PocketFFT not found — fall back to built-in Cooley-Tukey

// ── Fallback: built-in FFT (no external dependencies) ────────────────────────

#include "builtin.hpp"

namespace cps::backends {
    // Alias so that all code using `PocketFFT` as the default backend type
    // transparently picks up the built-in implementation when PocketFFT is
    // not available.  Performance note: power-of-2 sizes are O(N log N);
    // non-power-of-2 sizes fall back to O(N²) — see builtin.hpp.
    using PocketFFT = BuiltinFFT;
}

#endif // __has_include("pocketfft_hdronly.h")
