# cps — Command-Line Signal Processing Tool

`cps` is a UNIX-pipeline tool built on top of the cppsignal library. Each invocation reads or writes a self-describing binary stream, so multiple stages can be chained with `|` to build signal-processing pipelines without temporary files.

```
cps generate [options] | cps filter [options] | <audio-player>
```

## Contents

- [Wire format](#wire-format)
- [Building](#building)
- [Exit codes](#exit-codes)
- [cps generate](#cps-generate)
- [cps filter](#cps-filter)
- [Examples](#examples)

---

## Wire format

Every `cps` stream is a 12-byte little-endian header followed by raw sample data.

```
Offset  Size  Type    Field
──────  ────  ──────  ──────────────────────────────────────────
  0       4   char    Magic bytes: "CPS\0"
  4       1   uint8   Version (currently 1)
  5       1   uint8   Sample format: 0 = float32, 1 = float64
  6       2   uint16  Channel count (1 = mono, 2 = stereo)
  8       4   uint32  Sample rate in Hz
 12       …   float   Interleaved samples, little-endian
```

Samples are raw IEEE-754 values — no compression, no padding. Interleaving follows the standard layout: for stereo, sample 0 is [L₀, R₀], sample 1 is [L₁, R₁], etc.

The header is validated on read: a wrong magic, unsupported version, or truncated header causes the tool to exit with an error.

---

## Building

```sh
cmake -B build -DCPS_BUILD_CLI=ON
cmake --build build --config Release
# Executable: build/Release/cps  (or build/cps on Linux/macOS)
```

`CPS_BUILD_CLI` is `ON` by default. Pass `-DCPS_BUILD_CLI=OFF` to skip the CLI when building the library only.

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| 1    | Runtime error (bad stream, I/O failure) |
| 2    | Invalid argument (bad option, out-of-range value) |

---

## cps generate

Synthesises a signal and writes a CPS stream to stdout.

```
cps generate --type <type> [options]
```

### Required

| Option | Values | Description |
|--------|--------|-------------|
| `--type` | `sine` `chirp` `white` `pink` | Signal type to generate |

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--amplitude <float>` | `1.0` | Peak amplitude (linear). Must be > 0. |
| `--duration <seconds>` | `1.0` | Length of the generated signal. Must be > 0. |
| `--sample-rate <Hz>` | `44100` | Output sample rate. |
| `--channels <n>` | `1` | Channel count written to the stream header. |

### Type-specific options

#### `--type sine`

Generates a pure sinusoid at a fixed frequency.

| Option | Default | Description |
|--------|---------|-------------|
| `--freq <Hz>` | `440.0` | Frequency of the sine wave. Must be < Nyquist (sample-rate / 2). |

#### `--type chirp`

Generates a linear frequency sweep (chirp) from `freq-start` to `freq-end` over the full duration.

| Option | Default | Description |
|--------|---------|-------------|
| `--freq-start <Hz>` | `100.0` | Starting frequency. Must be < Nyquist. |
| `--freq-end <Hz>` | `1000.0` | Ending frequency. Must be < Nyquist. |

#### `--type white`

Generates Gaussian white noise. Amplitude controls the standard deviation, so RMS ≈ amplitude.

| Option | Default | Description |
|--------|---------|-------------|
| `--seed <uint>` | `0` | RNG seed for reproducibility. Same seed + length → identical output. |

#### `--type pink`

Generates 1/f (pink) noise using Paul Kellet's approximation. The output is peak-normalised so the maximum absolute sample equals `amplitude`.

| Option | Default | Description |
|--------|---------|-------------|
| `--seed <uint>` | `0` | RNG seed for reproducibility. |

### Validation

- `--type` is required; omitting it is an error.
- `--amplitude` must be > 0.
- `--duration` must be > 0.
- `--sample-rate` must be > 0.
- `--freq` (sine) must be strictly less than `sample-rate / 2`.
- `--freq-start` and `--freq-end` (chirp) must each be strictly less than `sample-rate / 2`.
- `duration × sample-rate` must produce at least one sample.

---

## cps filter

Reads a CPS stream from stdin, applies a digital filter, and writes the filtered stream to stdout. The sample rate is read from the stream header; all cutoff frequencies are specified in Hz.

```
cps filter --shape <shape> [options]
```

### Shape and cutoff options

| Option | Values | Description |
|--------|--------|-------------|
| `--shape <shape>` | `lowpass` `highpass` `bandpass` `bandstop` | Filter topology. Defaults to `lowpass`. |
| `--cutoff <Hz>` | — | -3 dB frequency for `lowpass` or `highpass`. Required for those shapes. |
| `--cutoff-low <Hz>` | — | Lower -3 dB edge for `bandpass` or `bandstop`. Required for those shapes. Must be > 0. |
| `--cutoff-high <Hz>` | — | Upper -3 dB edge for `bandpass` or `bandstop`. Required for those shapes. |

### Implementation options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--type <impl>` | `butter` `firwin` | `butter` | Filter implementation. |
| `--order <n>` | positive integer | `4` | IIR filter order (Butterworth only). Higher order → steeper rolloff. |
| `--taps <n>` | odd positive integer | `101` | FIR tap count (firwin only). Higher taps → narrower transition band. |
| `--sample-rate <Hz>` | — | — | Hint used only for pre-parse Nyquist validation. Does not override the rate in the stream header. |

### Filter implementations

#### `--type butter` (default)

Butterworth IIR filter — maximally flat magnitude response with no passband ripple.

- Uses second-order sections (SOS) for numerical stability at higher orders.
- Lowpass and highpass: single-cutoff design via the standard bilinear transform.
- Bandpass and bandstop: proper LP→BP / LP→BS analog prototype transformation followed by the bilinear transform, mirroring `scipy.signal.butter`. Both even and odd filter orders are supported.
- Applied with `sosfilt` (single-pass, causal).

#### `--type firwin`

Windowed-sinc FIR filter using a Hamming window.

- Linear phase — constant group delay across the passband.
- Supports `lowpass` and `highpass` only. Use `--type butter` for `bandpass`/`bandstop`.
- `--taps` should be odd (even tap counts with `highpass` are rejected).
- Applied with `lfilter` (single-pass, causal).

### Validation

- `--order` must be > 0 (butter).
- `--cutoff` is required and must be > 0 for `lowpass`/`highpass`; must be < Nyquist when `--sample-rate` is provided.
- `--cutoff-low` and `--cutoff-high` are both required for `bandpass`/`bandstop`.
- `--cutoff-low` must be > 0 and strictly less than `--cutoff-high`.
- `--cutoff-high` must be < Nyquist when `--sample-rate` is provided.
- `--type firwin` with `--shape bandpass` or `bandstop` is rejected; use `--type butter`.

---

## Examples

### 1. Generate a 1-second 440 Hz sine and play it

```sh
cps generate --type sine --freq 440 --duration 1 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 2. Generate 5 seconds of pink noise

```sh
cps generate --type pink --duration 5 --amplitude 0.5 --seed 42 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 3. Apply a 4th-order lowpass at 1 kHz

```sh
cps generate --type white --duration 3 \
  | cps filter --shape lowpass --cutoff 1000 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 4. Highpass filter to remove DC and rumble (< 80 Hz)

```sh
cps generate --type pink --duration 4 \
  | cps filter --shape highpass --cutoff 80 --order 6 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 5. Notch filter — remove 50 Hz mains hum

```sh
cps generate --type sine --freq 50 --duration 2 \
  | cps filter --shape bandstop --cutoff-low 48 --cutoff-high 52 --order 4 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 6. Bandpass filter — isolate a frequency band (800–1200 Hz)

```sh
cps generate --type white --duration 3 \
  | cps filter --shape bandpass --cutoff-low 800 --cutoff-high 1200 --order 4 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 7. Cascaded filters — LP then HP acts as a bandpass

```sh
cps generate --type white --duration 3 \
  | cps filter --shape lowpass  --cutoff 4000 --order 4 \
  | cps filter --shape highpass --cutoff  200 --order 4 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 8. Chirp sweep through a lowpass filter

```sh
cps generate --type chirp --freq-start 100 --freq-end 10000 --duration 5 \
  | cps filter --shape lowpass --cutoff 2000 --order 6 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 9. Higher sample rate (48 kHz) and higher-order filter

```sh
cps generate --type white --duration 2 --sample-rate 48000 \
  | cps filter --shape lowpass --cutoff 8000 --order 8 \
  | play -t raw -r 48000 -e float -b 32 -c 1 -
```

### 10. FIR lowpass — linear phase, useful for signal analysis

```sh
cps generate --type white --duration 3 \
  | cps filter --type firwin --shape lowpass --cutoff 2000 --taps 201 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 11. Save output to a file instead of playing

```sh
cps generate --type sine --freq 261.63 --duration 2 > middle_c.cps
cps filter --shape lowpass --cutoff 5000 < middle_c.cps > filtered.cps
```

### 12. Inspect a stream header with `xxd`

```sh
cps generate --type sine --freq 440 --duration 0.001 | xxd | head -2
# 00000000: 4350 5300 0100 0100 44ac 0000 ...   CPS.....D...
#           ^^^^       ^^   ^^^^  ^^^^^^^^
#           magic  ver fmt  ch=1  sr=44100 (LE)
```

### 13. Three-stage pipeline: generate → notch → lowpass → play

```sh
cps generate --type white --duration 5 --amplitude 0.8 \
  | cps filter --shape bandstop --cutoff-low 990 --cutoff-high 1010 --order 4 \
  | cps filter --shape lowpass  --cutoff 4000 --order 4 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```

### 14. Reproduce identical noise with a fixed seed

```sh
# Both commands produce byte-for-byte identical output:
cps generate --type white --duration 1 --seed 1234 > a.cps
cps generate --type white --duration 1 --seed 1234 > b.cps
diff a.cps b.cps  # no output — identical
```

### 15. Steeper rolloff with higher filter order

```sh
# Compare 2nd-order vs 8th-order lowpass on a tone just above cutoff:
cps generate --type sine --freq 1500 --duration 2 \
  | cps filter --shape lowpass --cutoff 1000 --order 2 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -

cps generate --type sine --freq 1500 --duration 2 \
  | cps filter --shape lowpass --cutoff 1000 --order 8 \
  | play -t raw -r 44100 -e float -b 32 -c 1 -
```
