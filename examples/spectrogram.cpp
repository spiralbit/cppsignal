// ─────────────────────────────────────────────────────────────────────────────
// examples/spectrogram.cpp
//
// Demonstrates the STFT and spectrogram on a chirp signal.
//
// A linear chirp sweeps from 10 Hz to 400 Hz over 2 seconds (fs=1000 Hz).
// The STFT shows how the frequency content evolves over time.
// We print a simple ASCII representation of the power spectrogram.
// ─────────────────────────────────────────────────────────────────────────────

#include <cps/cps.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <string>

int main()
{
    constexpr double fs  = 1000.0;   // sample rate (Hz)
    constexpr double dur = 2.0;      // signal duration (seconds)
    const int        N   = static_cast<int>(fs * dur);

    // ── 1. Generate a linear chirp: 10 Hz → 400 Hz over 2 seconds ────────────
    auto t     = cps::linspace(0.0, dur, N);
    auto chirp = cps::chirp(t, 10.0, 400.0, dur);

    std::cout << "Signal length:  " << N << " samples\n";
    std::cout << "Duration:       " << dur << " s\n";
    std::cout << "Frequency:      sweeps 10 → 400 Hz\n\n";

    // ── 2. Compute STFT ───────────────────────────────────────────────────────
    cps::STFTOptions opts;
    opts.nperseg  = 128;             // 128-point FFT → 0.128 s per frame
    opts.noverlap = 64;              // 50% overlap
    opts.window   = cps::Window::Hann;

    auto sg = cps::spectrogram(chirp, fs, opts);

    std::cout << "STFT frames:    " << sg.times.size() << '\n';
    std::cout << "Frequency bins: " << sg.freqs.size() << '\n';
    std::cout << "Freq resolution:" << (fs / opts.nperseg) << " Hz/bin\n\n";

    // ── 3. ASCII spectrogram (time on X axis, frequency on Y axis) ────────────
    // Subsample to fit terminal: show every 4th time frame, every 4th freq bin
    // and map power to grayscale characters.
    const std::string palette = " .:-=+*#%@";

    // Find max power for normalisation
    double max_power = 0.0;
    for (auto& row : sg.power)
        for (double v : row)
            max_power = std::max(max_power, v);

    if (max_power == 0.0) { std::cout << "Empty spectrogram.\n"; return 1; }

    constexpr std::size_t freq_step = 4;
    constexpr std::size_t time_step = 4;

    // Print header
    std::cout << "Spectrogram (frequency 0 → " << sg.freqs.back() << " Hz, "
              << "time 0 → " << sg.times.back() << " s)\n";
    std::cout << std::string(60, '-') << '\n';

    // Print rows from high frequency to low (so low freq is at the bottom)
    for (int fi = static_cast<int>(sg.freqs.size()) - 1; fi >= 0;
         fi -= static_cast<int>(freq_step))
    {
        if (fi < 0) break;
        std::cout << std::setw(5) << std::fixed << std::setprecision(0)
                  << sg.freqs[fi] << " Hz |";
        for (std::size_t ti = 0; ti < sg.times.size(); ti += time_step) {
            double norm = sg.power[fi][ti] / max_power;
            int    idx  = static_cast<int>(norm * (palette.size() - 1));
            idx = std::clamp(idx, 0, static_cast<int>(palette.size()) - 1);
            std::cout << palette[idx];
        }
        std::cout << '\n';
    }
    std::cout << std::string(60, '-') << '\n';
    std::cout << "       |";
    // Print rough time labels
    for (std::size_t ti = 0; ti < sg.times.size(); ti += time_step) {
        if (ti % (time_step * 5) == 0)
            std::cout << '|';
        else
            std::cout << ' ';
    }
    std::cout << "\n       Time →\n";

    return 0;
}
