#include "filter.hpp"
#include <cps/cps.hpp>
#include <istream>
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
        if (opts.shape == FilterShape::Bandpass) {
            // Approximate bandpass as cascaded LP + HP
            auto lp  = cps::butter(opts.order, opts.cutoff_high, cps::FilterType::Lowpass,  fopts);
            auto hp  = cps::butter(opts.order, opts.cutoff_low,  cps::FilterType::Highpass, fopts);
            auto tmp = cps::sosfilt(lp, signal);
            filtered = cps::sosfilt(hp, tmp);
        } else if (opts.shape == FilterShape::Bandstop) {
            // Bandstop = sum of LP (below stop) and HP (above stop) outputs
            auto lp     = cps::butter(opts.order, opts.cutoff_low,  cps::FilterType::Lowpass,  fopts);
            auto hp     = cps::butter(opts.order, opts.cutoff_high, cps::FilterType::Highpass, fopts);
            auto lp_out = cps::sosfilt(lp, signal);
            auto hp_out = cps::sosfilt(hp, signal);
            filtered.resize(signal.size());
            for (std::size_t i = 0; i < signal.size(); ++i)
                filtered[i] = lp_out[i] + hp_out[i];
        } else {
            auto sos = cps::butter(opts.order, opts.cutoff, to_cps_type(opts.shape), fopts);
            filtered = cps::sosfilt(sos, signal);
        }
    } else {
        // FirWin: cutoff normalised to [0,1] where 1 = Nyquist
        double norm_cutoff = opts.cutoff / nyq;
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
    bool cutoff_set     = false;
    bool cutoff_low_set = false;
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
        if (opts.cutoff_low >= opts.cutoff_high)
            throw std::invalid_argument("--cutoff-low must be less than --cutoff-high");
    }

    return opts;
}

} // namespace cps::cli
