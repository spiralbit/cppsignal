#include "filter.hpp"
#include <cps/cps.hpp>
#include <istream>
#include <ostream>
#include <stdexcept>

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
    // TODO: implement argument parsing
    (void)argc; (void)argv;
    throw std::logic_error("parse_filter_args: not yet implemented");
}

} // namespace cps::cli
