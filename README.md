# CppSignal

A modern C++20 signal processing library modelled on [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html).

CppSignal provides a clean function-style API for IIR/FIR filter design, FFT, spectral analysis, peak finding, and signal generation — with no mandatory framework dependencies and an MIT licence suitable for commercial use.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Building](#building)
  - [CMake options](#cmake-options)
- [Code coverage](#code-coverage)
- [Windows](#windows)
  - [Code coverage on Windows — OpenCppCoverage](#code-coverage-on-windows--opencppcoverage)
- [macOS](#macos)
  - [Code coverage on macOS — gcovr](#code-coverage-on-macos--gcovr)
- [Linux](#linux)
  - [Code coverage on Linux — gcovr](#code-coverage-on-linux--gcovr)
- [Using CppSignal in your project (FetchContent)](#using-cppsignal-in-your-project-fetchcontent)
- [Project layout](#project-layout)
- [Licence](#licence)

## Features

| Category               | Functions |
|------------------------|-----------|
| **Convolution**        | `convolve`, `correlate` |
| **FFT**                | `fft`, `ifft`, `rfft`, `irfft`, `fftfreq`, `rfftfreq` |
| **Filter analysis**    | `freqz` |
| **Filter&nbsp;application** | `sosfilt`, `lfilter` |
| **Filter design**      | `butter`, `firwin` |
| **Metrics**            | `rms`, `snr`, `thd`, `sinad` |
| **Peak finding**       | `find_peaks`, `peak_prominences` |
| **Signal generation**  | `linspace`, `arange`, `sinusoid`, `chirp`, `gausspulse`, `unit_impulse`, `square_wave`, `sawtooth_wave`, `white_noise` |
| **Spectral analysis**  | `welch`, `stft`, `spectrogram` |
| **Windows**            | `hann_window`, `hamming_window`, `blackman_window`, `kaiser_window`, `flattop_window`, `tukey_window`, `make_window` |

### Scipy-style API

```cpp
#include <cps/cps.hpp>

// Design a 4th-order Butterworth lowpass at 100 Hz (fs = 1000 Hz)
auto sos = cps::butter(4, 100.0, cps::FilterType::Lowpass, {.fs = 1000.0});

// Apply it
auto filtered = cps::sosfilt(sos, signal);

// Compute power spectral density via Welch's method
auto [freqs, psd] = cps::welch(signal, 1000.0);

// Find peaks with prominence filtering
auto peaks = cps::find_peaks(signal, {.prominence = 0.5});
```

### Pluggable backends

All compute-heavy operations are templated on interchangeable backends defined by C++20 concepts. The defaults require no configuration:

```cpp
// Default: PocketFFT backend (fetched automatically by CMake)
auto spectrum = cps::rfft(signal);

// Explicit override per call-site
auto spectrum = cps::rfft<cps::backends::BuiltinFFT>(signal);
```

| Backend | Type | Default | Notes |
|---|---|---|---|
| `PocketFFT` | FFT | yes | BSD-3; used by NumPy. Auto-fetched via CMake. Falls back to `BuiltinFFT` if unavailable. |
| `BuiltinFFT` | FFT | fallback | Self-contained Cooley-Tukey; no dependencies. O(N log N) for power-of-2 sizes. |
| `FFTW` | FFT | no | Enable with `-DCPS_ENABLE_FFTW=ON`. GPL licence — see below. |
| `SequentialThreading` | Threading | yes | Single-threaded; always available. |
| `StdExecution` | Threading | no | C++23 `std::execution`. Enable with `-DCPS_ENABLE_STD_EXECUTION=ON`. |
| `StandardAlloc` | Allocator | yes | Wraps `std::allocator`. |

## Requirements

- C++20 compiler (MSVC 19.29+, GCC 11+, Clang 13+)
- CMake 3.20+
- Internet access at configure time (CMake fetches PocketFFT and Catch2 automatically)

## Building

Configure and build:

```
cmake -S cppsignal -B cppsignal-build
cmake --build cppsignal-build
```

Run the tests (Catch2, built by default):

```
ctest --test-dir cppsignal-build --output-on-failure
```

Platform-specific notes are in the sections below.

### CMake options

| Option | Default | Description |
|---|---|---|
| `CPS_BUILD_TESTS` | `ON` | Build the Catch2 test suite |
| `CPS_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `CPS_ENABLE_COVERAGE` | `OFF` | Add `coverage` build target (see [Code coverage](#code-coverage)) |
| `CPS_ENABLE_FFTW` | `OFF` | Enable FFTW3 backend (GPL licence) |
| `CPS_ENABLE_STD_EXECUTION` | `OFF` | Enable C++23 `std::execution` threading backend |

## Code coverage

The `coverage` target runs the full test suite and writes an HTML report to `<build>/coverage_report/index.html` plus a Cobertura XML file for CI systems.

Configure with coverage enabled, then build the `coverage` target:

```
cmake -S cppsignal -B cppsignal-cov -DCPS_ENABLE_COVERAGE=ON
cmake --build cppsignal-cov --target coverage
```

The report appears at `cppsignal-cov/coverage_report/index.html`. Platform-specific tool installation is described below.

---

## Windows

The Visual Studio generator is multi-config and has no default configuration, so you must pass `-C <Config>` to both `cmake --build` and `ctest`:

```powershell
cmake -S cppsignal -B cppsignal-build
cmake --build cppsignal-build --config Debug
ctest --test-dir cppsignal-build -C Debug --output-on-failure
```

### Code coverage on Windows — OpenCppCoverage

OpenCppCoverage instruments test executables at runtime using the Windows debug API, so no special compiler flags are needed.

**Install OpenCppCoverage** (pick one method):

Option A — installer (no prerequisites):
1. Download the latest `.exe` installer from the [releases page](https://github.com/OpenCppCoverage/OpenCppCoverage/releases)
2. Run it and accept the defaults (installs to `C:\Program Files\OpenCppCoverage\`)

Option B — Chocolatey:
```powershell
# Install Chocolatey if you don't have it (run in an elevated PowerShell):
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then install OpenCppCoverage:
choco install opencppcoverage
```

**Configure and run:**

```powershell
cmake -S cppsignal -B cppsignal-cov -DCPS_ENABLE_COVERAGE=ON
cmake --build cppsignal-cov --config Debug --target coverage
# Report: cppsignal-cov\coverage_report\index.html
```

> CMake will fail at configure time with installation instructions if OpenCppCoverage is not found, so you will not get a silently empty report.

---

## macOS

The generic build commands above apply without modification. Use any generator (Ninja is recommended for speed):

```bash
cmake -S cppsignal -B cppsignal-build -G Ninja
cmake --build cppsignal-build
ctest --test-dir cppsignal-build --output-on-failure
```

### Code coverage on macOS — gcovr

Tests are recompiled with `--coverage` flags. gcovr harvests the `.gcda` files after the run.

```bash
brew install gcovr        # or: pip install gcovr

cmake -S cppsignal -B cppsignal-cov -DCPS_ENABLE_COVERAGE=ON
cmake --build cppsignal-cov --target coverage
# Report: cppsignal-cov/coverage_report/index.html
```

---

## Linux

The generic build commands above apply without modification:

```bash
cmake -S cppsignal -B cppsignal-build
cmake --build cppsignal-build
ctest --test-dir cppsignal-build --output-on-failure
```

### Code coverage on Linux — gcovr

```bash
pip install gcovr          # or: sudo apt install gcovr

cmake -S cppsignal -B cppsignal-cov -DCPS_ENABLE_COVERAGE=ON
cmake --build cppsignal-cov --target coverage
# Report: cppsignal-cov/coverage_report/index.html
```

---

## Using CppSignal in your project (FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(
    cppsignal
    GIT_REPOSITORY https://github.com/spiralbit/cppsignal.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(cppsignal)

target_link_libraries(my_app PRIVATE cppsignal::cppsignal)
```

## Project layout

```
include/cps/
  cps.hpp                  # Single master include
  core/
    types.hpp              # Real, Complex, SOS, ZPK, FilterType, Window, …
    concepts.hpp           # FFTBackend, ThreadingBackend, AllocBackend concepts
    result.hpp             # Exception types (ValueError, NotImplemented, …)
  backends/
    fft/
      pocketfft.hpp        # PocketFFT wrapper (falls back to builtin)
      builtin.hpp          # Self-contained Cooley-Tukey FFT
      fftw.hpp             # FFTW3 wrapper (optional, GPL)
    threading/
      sequential.hpp       # Single-threaded backend
      std_exec.hpp         # std::execution backend (C++23)
    alloc/
      standard.hpp         # std::allocator wrapper
  filter/
    design.hpp             # butter, firwin, cheby1/2, ellip, bessel (stubs)
    apply.hpp              # sosfilt, lfilter, filtfilt (stub)
    analysis.hpp           # freqz, group_delay (stub)
  spectral/
    windows.hpp            # All window functions
    fft.hpp                # fft, ifft, rfft, irfft, fftfreq, rfftfreq
    psd.hpp                # welch, periodogram (stub), csd (stub)
    stft.hpp               # stft, spectrogram
  signal/
    generate.hpp           # linspace, arange, sinusoid, chirp, gausspulse, …
    peaks.hpp              # find_peaks, peak_prominences
    correlate.hpp          # convolve, correlate
    resample.hpp           # decimate, interpolate, resample (stubs)
  measure/
    metrics.hpp            # rms, snr, thd, sinad
tests/                     # Catch2 test suite
examples/
  lowpass_filter.cpp       # Butterworth LP design and application
  spectrogram.cpp          # STFT / ASCII spectrogram of a chirp
```

## Licence

MIT — see [LICENSE](LICENSE).

**FFTW exception:** the optional FFTW backend (`-DCPS_ENABLE_FFTW=ON`) links against FFTW3 which is GPL-licensed. Enabling it makes the combined work GPL. It is disabled by default so the library remains MIT-clean. See [fftw.org/doc/License-and-Copyright.html](http://fftw.org/doc/License-and-Copyright.html) for commercial licence options.
