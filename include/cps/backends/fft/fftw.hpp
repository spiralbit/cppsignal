#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cps/backends/fft/fftw.hpp — optional FFTW3 backend
//
// FFTW (Fastest Fourier Transform in the West) by Matteo Frigo & Steven Johnson
// at MIT. Extremely fast on large, power-of-two sizes due to machine-tuned
// codelets generated at install time.
//
// LICENCE WARNING
// ───────────────
// FFTW is GPL v2 or later. Using this backend makes any binary that links it
// subject to the GPL unless you hold a commercial FFTW licence ($12,500 from
// MIT Technology Licensing Office as of 2026).
//
// This backend is compiled in only when CMake option CPS_ENABLE_FFTW=ON.
// By default it is OFF so the library remains MIT-clean.
//
// HOW TO ENABLE
// ─────────────
//   cmake -DCPS_ENABLE_FFTW=ON ..
//
// Then use it explicitly per call-site:
//   #include <cps/backends/fft/fftw.hpp>
//   auto spectrum = cps::fft<cps::backends::FFTW>(signal);
//
// THREAD SAFETY
// ─────────────
// FFTW plan *creation* is NOT thread-safe — it mutates internal global state.
// All plan creation in this backend is serialised through a module-level mutex.
// FFTW plan *execution* via the new-array API (fftw_execute_dft, etc.) IS
// thread-safe and is used here without any locking.
//
// One plan per transform direction is cached per thread. For workloads with a
// fixed transform size (Welch, STFT), each thread pays the planning cost once.
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CPS_HAS_FFTW
#  error "FFTW backend not enabled. Build with -DCPS_ENABLE_FFTW=ON and ensure FFTW3 is installed."
#endif

#include <fftw3.h>
#include <span>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <cstddef>
#include <limits>
#include "../../core/result.hpp"

namespace cps::backends {

namespace detail_fftw {

// Serialises all FFTW plan creation and destruction across threads.
inline std::mutex s_plan_mtx;

struct PlanEntry {
    std::size_t n    = 0;
    fftw_plan   plan = nullptr;
};

struct PlanCache {
    PlanEntry fwd, inv, rfwd, rinv;

    ~PlanCache()
    {
        std::lock_guard lk(s_plan_mtx);
        for (auto* e : {&fwd, &inv, &rfwd, &rinv})
            if (e->plan) { fftw_destroy_plan(e->plan); e->plan = nullptr; }
    }
};

inline PlanCache& this_thread_cache()
{
    thread_local PlanCache cache;
    return cache;
}

// Returns the cached plan for this thread.
// Fast path (cache hit): no lock taken — the entry is thread-local.
// Slow path (cache miss): acquires s_plan_mtx, creates a new plan via `maker`.
template<typename Maker>
inline fftw_plan get_plan(PlanEntry& e, std::size_t n, Maker maker)
{
    if (e.n == n) return e.plan;
    std::lock_guard lk(s_plan_mtx);
    if (e.plan) fftw_destroy_plan(e.plan);
    e.plan = nullptr; e.n = 0;   // clear before maker so a throw leaves a clean state
    e.plan = maker(n);           // throws ValueError on allocation or planning failure
    e.n    = n;
    return e.plan;
}

} // namespace detail_fftw


// ── FFTW — satisfies cps::FFTBackend ─────────────────────────────────────────

struct FFTW {

    void forward(std::span<const std::complex<double>> in,
                 std::span<std::complex<double>>       out) const
    {
        const std::size_t N = in.size();
        if (N > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw cps::ValueError("FFTW: transform size exceeds INT_MAX");
        fftw_plan plan = detail_fftw::get_plan(
            detail_fftw::this_thread_cache().fwd, N,
            [](std::size_t n) {
                auto* a = reinterpret_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
                auto* b = reinterpret_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
                if (!a || !b) { fftw_free(a); fftw_free(b); throw cps::ValueError("FFTW: out of memory"); }
                fftw_plan p = fftw_plan_dft_1d(static_cast<int>(n), a, b,
                                               FFTW_FORWARD, FFTW_ESTIMATE);
                fftw_free(a); fftw_free(b);
                if (!p) throw cps::ValueError("FFTW: plan creation failed");
                return p;
            });

        // new-array execute: safe to call concurrently with different buffers
        auto* p_in  = reinterpret_cast<fftw_complex*>(
                          const_cast<std::complex<double>*>(in.data()));
        auto* p_out = reinterpret_cast<fftw_complex*>(out.data());
        fftw_execute_dft(plan, p_in, p_out);
    }

    void inverse(std::span<const std::complex<double>> in,
                 std::span<std::complex<double>>       out) const
    {
        const std::size_t N = in.size();
        if (N > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw cps::ValueError("FFTW: transform size exceeds INT_MAX");
        fftw_plan plan = detail_fftw::get_plan(
            detail_fftw::this_thread_cache().inv, N,
            [](std::size_t n) {
                auto* a = reinterpret_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
                auto* b = reinterpret_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
                if (!a || !b) { fftw_free(a); fftw_free(b); throw cps::ValueError("FFTW: out of memory"); }
                fftw_plan p = fftw_plan_dft_1d(static_cast<int>(n), a, b,
                                               FFTW_BACKWARD, FFTW_ESTIMATE);
                fftw_free(a); fftw_free(b);
                if (!p) throw cps::ValueError("FFTW: plan creation failed");
                return p;
            });

        auto* p_in  = reinterpret_cast<fftw_complex*>(
                          const_cast<std::complex<double>*>(in.data()));
        auto* p_out = reinterpret_cast<fftw_complex*>(out.data());
        fftw_execute_dft(plan, p_in, p_out);

        // Normalise by N to match numpy/scipy convention
        const double inv_N = 1.0 / static_cast<double>(N);
        for (auto& v : out) v *= inv_N;
    }

    void rfft(std::span<const double>         in,
              std::span<std::complex<double>> out) const
    {
        const std::size_t N = in.size();
        if (N > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw cps::ValueError("FFTW: transform size exceeds INT_MAX");
        fftw_plan plan = detail_fftw::get_plan(
            detail_fftw::this_thread_cache().rfwd, N,
            [](std::size_t n) {
                auto* a = static_cast<double*>(fftw_malloc(n * sizeof(double)));
                auto* b = reinterpret_cast<fftw_complex*>(
                              fftw_malloc((n / 2 + 1) * sizeof(fftw_complex)));
                if (!a || !b) { fftw_free(a); fftw_free(b); throw cps::ValueError("FFTW: out of memory"); }
                fftw_plan p = fftw_plan_dft_r2c_1d(static_cast<int>(n), a, b, FFTW_ESTIMATE);
                fftw_free(a); fftw_free(b);
                if (!p) throw cps::ValueError("FFTW: plan creation failed");
                return p;
            });

        auto* p_in  = const_cast<double*>(in.data());
        auto* p_out = reinterpret_cast<fftw_complex*>(out.data());
        fftw_execute_dft_r2c(plan, p_in, p_out);
    }

    void irfft(std::span<const std::complex<double>> in,
               std::span<double>                    out,
               std::size_t                          n_original) const
    {
        if (n_original > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw cps::ValueError("FFTW: transform size exceeds INT_MAX");
        fftw_plan plan = detail_fftw::get_plan(
            detail_fftw::this_thread_cache().rinv, n_original,
            [](std::size_t n) {
                auto* a = reinterpret_cast<fftw_complex*>(
                              fftw_malloc((n / 2 + 1) * sizeof(fftw_complex)));
                auto* b = static_cast<double*>(fftw_malloc(n * sizeof(double)));
                if (!a || !b) { fftw_free(a); fftw_free(b); throw cps::ValueError("FFTW: out of memory"); }
                fftw_plan p = fftw_plan_dft_c2r_1d(static_cast<int>(n), a, b, FFTW_ESTIMATE);
                fftw_free(a); fftw_free(b);
                if (!p) throw cps::ValueError("FFTW: plan creation failed");
                return p;
            });

        auto* p_in  = reinterpret_cast<fftw_complex*>(
                          const_cast<std::complex<double>*>(in.data()));
        auto* p_out = out.data();
        fftw_execute_dft_c2r(plan, p_in, p_out);

        // FFTW c2r does not normalise; divide by N
        const double inv_N = 1.0 / static_cast<double>(n_original);
        for (auto& v : out) v *= inv_N;
    }

    static constexpr std::string_view name() { return "FFTW3"; }
};

} // namespace cps::backends
