#include "filter.hpp"
#include <cps/cps.hpp>
#include <complex>
#include <cmath>
#include <istream>
#include <numbers>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

namespace cps::cli {

static cps::FilterType to_cps_type(FilterShape shape)
{
    switch (shape) {
    case FilterShape::Lowpass:  return cps::FilterType::Lowpass;
    case FilterShape::Highpass: return cps::FilterType::Highpass;
    case FilterShape::Bandpass: return cps::FilterType::Bandpass;
    case FilterShape::Bandstop: return cps::FilterType::Bandstop;
    }
    throw std::invalid_argument("unknown filter shape");
}

// ── Butterworth bandstop/bandpass via LP prototype + bilinear ─────────────────
//
// The LP+HP summation approach fails for IIR because the two filters have
// different phase responses — the residual at the target frequency doesn't
// cancel. Instead we derive proper bandstop/bandpass SOS directly.
//
// Algorithm (mirrors scipy.signal.butter with btype='bandstop'/'bandpass'):
//   1. Pre-warp edge frequencies for the bilinear transform.
//   2. Compute Butterworth LP analog prototype poles (order N).
//   3. Apply LP->BS or LP->BP frequency transformation (each LP pole -> 2 poles).
//   4. Apply bilinear transform s->z.
//   5. Pair complex-conjugate poles into biquad sections.
//   6. Normalise gain.

static cps::SOS butter_bandstop_sos(int N, double fc1, double fc2, double fs)
{
    using Cx = std::complex<double>;
    const double pi = std::numbers::pi;

    // Step 1: pre-warp edge frequencies
    const double W1  = 2.0 * fs * std::tan(pi * fc1 / fs);
    const double W2  = 2.0 * fs * std::tan(pi * fc2 / fs);
    const double W0  = std::sqrt(W1 * W2);    // analog center frequency
    const double BW  = W2 - W1;               // analog bandwidth
    const double kbt = 2.0 * fs;              // bilinear constant

    // Digital notch frequency: th0 = 2*atan(W0 / 2fs)
    const double th0     = 2.0 * std::atan2(W0, kbt);
    const double cos_th0 = std::cos(th0);

    cps::SOS sos;

    // Add one biquad section from a complex BS pole zp (and its conjugate)
    auto add_biquad = [&](Cx zp) {
        const double a1 = -2.0 * zp.real();
        const double a2 = std::norm(zp);
        // BS numerator: (z - exp(j*th0))(z - exp(-j*th0)) = z^2 - 2*cos(th0)*z + 1
        sos.push_back({1.0, -2.0 * cos_th0, 1.0, 1.0, a1, a2});
    };

    // Steps 3+4: LP->BS transformation then bilinear, for one upper LP pole
    // Each LP pole p -> 2 BS analog poles -> 2 digital biquad sections
    auto process_bs = [&](Cx p) {
        const Cx bp   = BW * p;
        const Cx disc = std::sqrt(bp * bp - Cx{4.0 * W0 * W0, 0.0});
        add_biquad((kbt + (bp + disc) / 2.0) / (kbt - (bp + disc) / 2.0));
        add_biquad((kbt + (bp - disc) / 2.0) / (kbt - (bp - disc) / 2.0));
    };

    // Upper half-plane LP poles (conjugates handled by real-coefficient biquad)
    const int n_pairs = N / 2;
    for (int k = 0; k < n_pairs; ++k) {
        const double angle = pi * (2*(k+1) + N - 1) / (2.0 * N);
        process_bs({std::cos(angle), std::sin(angle)});
    }
    // Odd N: one real LP pole at exp(j*pi) = -1
    if (N % 2 == 1) {
        // discriminant = BW^2 - 4*W0^2
        // < 0 (narrow band): two complex-conjugate BS poles — one biquad suffices
        // > 0 (wide band):   two distinct real BS poles — build biquad explicitly
        const double discriminant = BW * BW - 4.0 * W0 * W0;
        if (discriminant >= 0.0) {
            const double d  = std::sqrt(discriminant);
            const double s1 = (-BW + d) / 2.0;
            const double s2 = (-BW - d) / 2.0;
            const double z1 = (kbt + s1) / (kbt - s1);
            const double z2 = (kbt + s2) / (kbt - s2);
            sos.push_back({1.0, -2.0 * cos_th0, 1.0, 1.0, -(z1 + z2), z1 * z2});
        } else {
            const Cx bp   = Cx{-BW, 0.0};
            const Cx disc = std::sqrt(bp * bp - Cx{4.0 * W0 * W0, 0.0});
            add_biquad((kbt + (bp + disc) / 2.0) / (kbt - (bp + disc) / 2.0));
        }
    }

    // Step 6: normalise so DC gain = 1
    double dc_gain = 1.0;
    for (const auto& row : sos) {
        dc_gain *= (row[0] + row[1] + row[2]) / (1.0 + row[4] + row[5]);
    }
    const double scale = std::pow(1.0 / dc_gain, 1.0 / static_cast<double>(sos.size()));
    for (auto& row : sos) { row[0] *= scale; row[1] *= scale; row[2] *= scale; }

    return sos;
}

static cps::SOS butter_bandpass_sos(int N, double fc1, double fc2, double fs)
{
    using Cx = std::complex<double>;
    const double pi = std::numbers::pi;

    const double W1  = 2.0 * fs * std::tan(pi * fc1 / fs);
    const double W2  = 2.0 * fs * std::tan(pi * fc2 / fs);
    const double W0  = std::sqrt(W1 * W2);
    const double BW  = W2 - W1;
    const double kbt = 2.0 * fs;

    cps::SOS sos;

    // BP biquad: zeros at z=+1 and z=-1 -> numerator [1, 0, -1]
    auto add_biquad = [&](Cx zp) {
        const double a1 = -2.0 * zp.real();
        const double a2 = std::norm(zp);
        sos.push_back({1.0, 0.0, -1.0, 1.0, a1, a2});
    };

    auto process_bp = [&](Cx p) {
        const Cx bp   = BW * p;
        const Cx disc = std::sqrt(bp * bp - Cx{4.0 * W0 * W0, 0.0});
        add_biquad((kbt + (bp + disc) / 2.0) / (kbt - (bp + disc) / 2.0));
        add_biquad((kbt + (bp - disc) / 2.0) / (kbt - (bp - disc) / 2.0));
    };

    const int n_pairs = N / 2;
    for (int k = 0; k < n_pairs; ++k) {
        const double angle = pi * (2*(k+1) + N - 1) / (2.0 * N);
        process_bp({std::cos(angle), std::sin(angle)});
    }
    if (N % 2 == 1) {
        const double discriminant = BW * BW - 4.0 * W0 * W0;
        if (discriminant >= 0.0) {
            const double d  = std::sqrt(discriminant);
            const double s1 = (-BW + d) / 2.0;
            const double s2 = (-BW - d) / 2.0;
            const double z1 = (kbt + s1) / (kbt - s1);
            const double z2 = (kbt + s2) / (kbt - s2);
            sos.push_back({1.0, 0.0, -1.0, 1.0, -(z1 + z2), z1 * z2});
        } else {
            const Cx bp   = Cx{-BW, 0.0};
            const Cx disc = std::sqrt(bp * bp - Cx{4.0 * W0 * W0, 0.0});
            const Cx s1   = (bp + disc) / 2.0;
            add_biquad((kbt + s1) / (kbt - s1));
        }
    }

    // Normalise to unity gain at the digital centre frequency
    const double th_centre = 2.0 * std::atan2(W0, kbt);
    const Cx z_c = std::polar(1.0, th_centre);
    Cx num{1.0, 0.0}, den{1.0, 0.0};
    for (const auto& row : sos) {
        const Cx zi = 1.0 / z_c;
        num *= row[0] + row[1]*zi + row[2]*zi*zi;
        den *= 1.0   + row[4]*zi + row[5]*zi*zi;
    }
    const double centre_gain = std::abs(num) / std::abs(den);
    const double scale = std::pow(1.0 / centre_gain, 1.0 / static_cast<double>(sos.size()));
    for (auto& row : sos) { row[0] *= scale; row[1] *= scale; row[2] *= scale; }

    return sos;
}

// ─────────────────────────────────────────────────────────────────────────────

std::size_t filter_stream(std::istream& in, std::ostream& out, const FilterOptions& opts)
{
    StreamHeader hdr = read_header(in);
    std::vector<float> raw = read_samples_f32(in, hdr);

    const double fs  = static_cast<double>(hdr.sample_rate);
    const double nyq = fs / 2.0;

    std::vector<cps::Real> signal(raw.begin(), raw.end());
    std::vector<cps::Real> filtered;
    cps::FilterOptions fopts{.fs = fs};

    if (opts.impl == FilterImpl::Butter) {
        cps::SOS sos;
        if (opts.shape == FilterShape::Bandpass)
            sos = butter_bandpass_sos(opts.order, opts.cutoff_low, opts.cutoff_high, fs);
        else if (opts.shape == FilterShape::Bandstop)
            sos = butter_bandstop_sos(opts.order, opts.cutoff_low, opts.cutoff_high, fs);
        else
            sos = cps::butter(opts.order, opts.cutoff, to_cps_type(opts.shape), fopts);
        filtered = cps::sosfilt(sos, signal);
    } else {
        if (opts.shape == FilterShape::Bandpass || opts.shape == FilterShape::Bandstop)
            throw std::invalid_argument("FIR bandpass/bandstop is not supported; use --type butter");
        // FirWin: cutoff normalised to [0,1] where 1 = Nyquist
        const double norm_cutoff = opts.cutoff / nyq;
        auto h = cps::firwin(opts.taps, norm_cutoff, cps::Window::Hamming,
                             to_cps_type(opts.shape));
        std::vector<cps::Real> a = {1.0};
        filtered = cps::lfilter(h, a, signal);
    }

    std::vector<float> out_f32(filtered.begin(), filtered.end());
    write_header(out, hdr);
    write_samples(out, std::span<const float>(out_f32), hdr);
    return out_f32.size();
}

FilterOptions parse_filter_args(int argc, char** argv)
{
    FilterOptions opts;
    bool cutoff_set      = false;
    bool cutoff_low_set  = false;
    bool cutoff_high_set = false;
    std::optional<double> sample_rate_hint;

    for (int i = 2; i + 1 < argc; i += 2) {
        std::string_view key{argv[i]};
        std::string_view val{argv[i + 1]};

        if (key == "--type") {
            if      (val == "butter") opts.impl = FilterImpl::Butter;
            else if (val == "firwin") opts.impl = FilterImpl::FirWin;
            else throw std::invalid_argument("unknown filter type: " + std::string(val));
        } else if (key == "--shape") {
            if      (val == "lowpass")  opts.shape = FilterShape::Lowpass;
            else if (val == "highpass") opts.shape = FilterShape::Highpass;
            else if (val == "bandpass") opts.shape = FilterShape::Bandpass;
            else if (val == "bandstop") opts.shape = FilterShape::Bandstop;
            else throw std::invalid_argument("unknown filter shape: " + std::string(val));
        } else if (key == "--order") {
            opts.order = std::stoi(std::string(val));
        } else if (key == "--taps") {
            opts.taps = std::stoi(std::string(val));
        } else if (key == "--cutoff") {
            opts.cutoff = std::stod(std::string(val));
            cutoff_set = true;
        } else if (key == "--cutoff-low") {
            opts.cutoff_low = std::stod(std::string(val));
            cutoff_low_set = true;
        } else if (key == "--cutoff-high") {
            opts.cutoff_high = std::stod(std::string(val));
            cutoff_high_set = true;
        } else if (key == "--sample-rate") {
            sample_rate_hint = std::stod(std::string(val));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(key));
        }
    }

    if (opts.order <= 0)
        throw std::invalid_argument("--order must be > 0");

    if (opts.shape == FilterShape::Lowpass || opts.shape == FilterShape::Highpass) {
        if (!cutoff_set || opts.cutoff <= 0.0)
            throw std::invalid_argument("--cutoff required for lowpass/highpass and must be > 0");
        if (sample_rate_hint && opts.cutoff >= *sample_rate_hint / 2.0)
            throw std::invalid_argument("--cutoff exceeds Nyquist frequency");
    }

    if (opts.shape == FilterShape::Bandpass || opts.shape == FilterShape::Bandstop) {
        if (!cutoff_low_set || !cutoff_high_set)
            throw std::invalid_argument("--cutoff-low and --cutoff-high required for bandpass/bandstop");
        if (opts.cutoff_low <= 0.0)
            throw std::invalid_argument("--cutoff-low must be > 0");
        if (opts.cutoff_low >= opts.cutoff_high)
            throw std::invalid_argument("--cutoff-low must be less than --cutoff-high");
        if (sample_rate_hint && opts.cutoff_high >= *sample_rate_hint / 2.0)
            throw std::invalid_argument("--cutoff-high exceeds Nyquist frequency");
    }

    return opts;
}

} // namespace cps::cli
