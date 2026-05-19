#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/backends/fft/builtin.hpp — self-contained Cooley-Tukey FFT
//
// Used automatically when PocketFFT cannot be found (detected via __has_include
// in pocketfft.hpp). No dependencies beyond the C++ standard library.
//
// PERFORMANCE
// ───────────
// • Power-of-2 sizes   : O(N log N) radix-2 Cooley-Tukey (fast path)
// • Non-power-of-2 sizes: O(N²) naive DFT (correct but slow — prefer power-of-2
//   segment lengths when using Welch/STFT with the built-in backend)
//
// ACCURACY
// ────────
// Double precision throughout. Round-trip error (ifft(fft(x)) - x) is
// typically < 1e-12 for N ≤ 65536.
// ─────────────────────────────────────────────────────────────────────────────

#include <span>
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>
#include <string_view>
#include <cstddef>
#include <cassert>
#include <algorithm>

namespace cps::backends {

namespace detail_builtin {

// ── Utilities ─────────────────────────────────────────────────────────────────

inline bool is_pow2(std::size_t n) { return n > 0 && (n & (n - 1)) == 0; }

// ── Bit-reversal permutation ───────────────────────────────────────────────
// Reorders x so that x[i] ends up at x[bit_reverse(i)].
// Required before the Cooley-Tukey butterfly stages.
inline void bit_reverse(std::vector<std::complex<double>>& x)
{
    const std::size_t N = x.size();
    for (std::size_t i = 1, j = 0; i < N; ++i) {
        std::size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
}

// ── Radix-2 Cooley-Tukey DIT FFT ──────────────────────────────────────────
// In-place. N must be a power of 2.
// inverse=false : forward DFT  (no normalisation)
// inverse=true  : inverse DFT  (divides by N)
inline void cooley_tukey(std::vector<std::complex<double>>& x, bool inverse)
{
    const std::size_t N = x.size();
    if (N <= 1) return;

    bit_reverse(x);

    // Butterfly stages — len doubles each pass: 2, 4, 8, … N
    for (std::size_t len = 2; len <= N; len <<= 1) {
        // Twiddle factor for this stage: e^(-j2π/len) for forward,
        //                               e^(+j2π/len) for inverse
        const double ang = 2.0 * std::numbers::pi
                           / static_cast<double>(len)
                           * (inverse ? 1.0 : -1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));

        for (std::size_t i = 0; i < N; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const auto u = x[i + j];
                const auto v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;   // even output
                x[i + j + len / 2] = u - v;   // odd output
                w *= wlen;
            }
        }
    }

    // Normalise inverse transform by 1/N (scipy/numpy convention)
    if (inverse) {
        const double inv_N = 1.0 / static_cast<double>(N);
        for (auto& c : x) c *= inv_N;
    }
}

// ── Naive O(N²) DFT ───────────────────────────────────────────────────────
// Used as a fallback for non-power-of-2 sizes.
// Correct for any N but slow — only practical for small N (< ~512).
inline void naive_dft(std::vector<std::complex<double>>& x, bool inverse)
{
    const std::size_t N = x.size();
    std::vector<std::complex<double>> out(N, {0.0, 0.0});

    const double sign = inverse ? 1.0 : -1.0;
    for (std::size_t k = 0; k < N; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            const double angle = sign * 2.0 * std::numbers::pi
                                 * static_cast<double>(k)
                                 * static_cast<double>(n)
                                 / static_cast<double>(N);
            out[k] += x[n] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
    }

    if (inverse) {
        const double inv_N = 1.0 / static_cast<double>(N);
        for (auto& c : out) c *= inv_N;
    }
    x = std::move(out);
}

// ── Dispatch: fast path for power-of-2, slow path otherwise ───────────────
inline void fft_impl(std::vector<std::complex<double>>& x, bool inverse)
{
    if (is_pow2(x.size()))
        cooley_tukey(x, inverse);
    else
        naive_dft(x, inverse);   // correct but O(N²) — see header comment
}

} // namespace detail_builtin


// ── BuiltinFFT — satisfies cps::FFTBackend ────────────────────────────────────

struct BuiltinFFT {

    void forward(std::span<const std::complex<double>> in,
                 std::span<std::complex<double>>       out) const
    {
        assert(in.size() == out.size());
        std::vector<std::complex<double>> buf(in.begin(), in.end());
        detail_builtin::fft_impl(buf, /*inverse=*/false);
        std::copy(buf.begin(), buf.end(), out.begin());
    }

    void inverse(std::span<const std::complex<double>> in,
                 std::span<std::complex<double>>       out) const
    {
        assert(in.size() == out.size());
        std::vector<std::complex<double>> buf(in.begin(), in.end());
        detail_builtin::fft_impl(buf, /*inverse=*/true);
        std::copy(buf.begin(), buf.end(), out.begin());
    }

    // Real-to-complex FFT: promote real input to complex, run FFT,
    // return only the non-redundant (positive-frequency) half.
    void rfft(std::span<const double>         in,
              std::span<std::complex<double>> out) const
    {
        const std::size_t N    = in.size();
        const std::size_t Nout = N / 2 + 1;
        assert(out.size() == Nout);

        // Promote real → complex
        std::vector<std::complex<double>> buf(N);
        for (std::size_t i = 0; i < N; ++i)
            buf[i] = {in[i], 0.0};

        detail_builtin::fft_impl(buf, /*inverse=*/false);

        // Copy only the first N/2+1 bins (rest are conjugate mirrors)
        std::copy(buf.begin(), buf.begin() + Nout, out.begin());
    }

    // Complex-to-real IFFT.
    // Reconstructs the full two-sided spectrum using conjugate symmetry,
    // then applies the inverse FFT.
    void irfft(std::span<const std::complex<double>> in,
               std::span<double>                    out,
               std::size_t                          n_original) const
    {
        assert(out.size() == n_original);
        assert(in.size() == n_original / 2 + 1);

        // Reconstruct full two-sided spectrum
        std::vector<std::complex<double>> buf(n_original);
        for (std::size_t k = 0; k < in.size(); ++k)
            buf[k] = in[k];
        // Mirror: buf[N-k] = conj(buf[k]) for k = 1 … floor((N-1)/2).
        // The bound n_original - in.size() + 1 gives the correct limit for both even and odd N.
        for (std::size_t k = 1; k < n_original - in.size() + 1; ++k)
            buf[n_original - k] = std::conj(in[k]);

        detail_builtin::fft_impl(buf, /*inverse=*/true);

        // Output should be real — discard any floating-point imaginary residual
        for (std::size_t i = 0; i < n_original; ++i)
            out[i] = buf[i].real();
    }

    static constexpr std::string_view name() { return "BuiltinFFT (Cooley-Tukey)"; }
};

} // namespace cps::backends
