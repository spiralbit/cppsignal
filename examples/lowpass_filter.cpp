// ─────────────────────────────────────────────────────────────────────────────
// examples/lowpass_filter.cpp
//
// Demonstrates designing and applying a Butterworth lowpass filter.
//
// Scenario: a sensor signal at 50 Hz is buried in 200 Hz noise.
// We design a 4th-order Butterworth LP at 100 Hz (fs=1000 Hz) and filter it.
// ─────────────────────────────────────────────────────────────────────────────

#include <cps/cps.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

int main()
{
    constexpr double fs      = 1000.0;  // sample rate (Hz)
    constexpr double f_sig   = 50.0;    // signal frequency (Hz)
    constexpr double f_noise = 200.0;   // noise frequency (Hz)
    constexpr int    N       = 500;     // number of samples

    // ── 1. Generate a noisy signal ────────────────────────────────────────────
    auto t     = cps::linspace(0.0, (N - 1) / fs, N);
    auto clean = cps::sinusoid(t, f_sig,   1.0);   // 50 Hz, amplitude 1
    auto noise = cps::sinusoid(t, f_noise, 0.5);   // 200 Hz, amplitude 0.5
    std::vector<cps::Real> noisy(N);
    for (int i = 0; i < N; ++i)
        noisy[i] = clean[i] + noise[i];

    std::cout << "Noisy signal RMS:  " << cps::rms(noisy) << '\n';

    // ── 2. Design a 4th-order Butterworth LP at 100 Hz ───────────────────────
    // The SOS (second-order sections) representation is returned — more
    // numerically stable than b/a polynomial form for high orders.
    auto sos = cps::butter(4, 100.0, cps::FilterType::Lowpass, {.fs = fs});
    std::cout << "Filter order:      4 (Butterworth LP)\n";
    std::cout << "Cutoff:            100 Hz\n";
    std::cout << "SOS sections:      " << sos.size() << '\n';

    // ── 3. Apply the filter ───────────────────────────────────────────────────
    auto filtered = cps::sosfilt(sos, noisy);

    // ── 4. Measure the output ─────────────────────────────────────────────────
    // Skip the first 100 samples (transient settling of the filter)
    std::vector<cps::Real> tail(filtered.begin() + 100, filtered.end());
    std::vector<cps::Real> clean_tail(clean.begin() + 100, clean.end());

    std::cout << "Filtered RMS:      " << cps::rms(tail) << '\n';
    std::cout << "Expected clean RMS:" << cps::rms(clean_tail) << '\n';

    // ── 5. Print frequency response at a few key points ──────────────────────
    auto [f, H] = cps::freqz(sos, 512, fs);
    std::cout << "\nFrequency response (selected):\n";
    std::cout << std::fixed << std::setprecision(1);
    for (std::size_t k = 0; k < f.size(); ++k) {
        double freq = f[k];
        if (freq < 10   || (freq > 90  && freq < 110) ||
            freq > 190 && freq < 210   || freq > 490) {
            double mag_db = 20.0 * std::log10(std::abs(H[k]) + 1e-12);
            std::cout << "  " << std::setw(6) << freq << " Hz  "
                      << std::setprecision(1) << mag_db << " dB\n";
        }
    }

    return 0;
}
