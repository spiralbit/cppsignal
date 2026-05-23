#include "generate.hpp"
#include <cps/cps.hpp>
#include <ostream>
#include <random>
#include <stdexcept>

namespace cps::cli {

// Simple 1/f (pink) noise generator using Paul Kellet's approximation.
static std::vector<float> pink_noise_generate(std::size_t n, double amplitude, unsigned seed)
{
    std::mt19937 rng{seed};
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double b0=0, b1=0, b2=0, b3=0, b4=0, b5=0, b6=0;
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double w = dist(rng);
        b0 = 0.99886*b0 + w*0.0555179;
        b1 = 0.99332*b1 + w*0.0750759;
        b2 = 0.96900*b2 + w*0.1538520;
        b3 = 0.86650*b3 + w*0.3104856;
        b4 = 0.55000*b4 + w*0.5329522;
        b5 = -0.7616*b5 - w*0.0168980;
        double pink = b0+b1+b2+b3+b4+b5+b6 + w*0.5362;
        b6 = w*0.115926;
        out[i] = static_cast<float>(pink);
    }
    // Normalise to [-amplitude, +amplitude]
    float peak = 0.0f;
    for (float s : out) peak = std::max(peak, std::abs(s));
    if (peak > 0.0f)
        for (float& s : out) s *= static_cast<float>(amplitude) / peak;
    return out;
}

std::size_t generate(std::ostream& out, const GenerateOptions& opts)
{
    const std::size_t n = static_cast<std::size_t>(opts.duration * opts.sample_rate);
    auto t = cps::linspace(0.0, static_cast<double>(n - 1) / opts.sample_rate, n);

    StreamHeader hdr;
    hdr.sample_rate = opts.sample_rate;
    hdr.channels    = opts.channels;
    hdr.format      = opts.format;
    write_header(out, hdr);

    switch (opts.type) {
    case GenerateType::Sine: {
        auto sig = cps::sinusoid(t, opts.freq, opts.amplitude);
        std::vector<float> s(sig.begin(), sig.end());
        write_samples(out, std::span<const float>(s), hdr);
        break;
    }
    case GenerateType::Chirp: {
        auto sig = cps::chirp(t, opts.freq_start, opts.freq_end, opts.duration);
        std::vector<float> s(sig.size());
        for (std::size_t i = 0; i < sig.size(); ++i)
            s[i] = static_cast<float>(sig[i] * opts.amplitude);
        write_samples(out, std::span<const float>(s), hdr);
        break;
    }
    case GenerateType::White: {
        // white_noise generates Gaussian noise with given std_dev;
        // we use amplitude as std_dev so RMS ≈ amplitude.
        auto sig = cps::white_noise(n, opts.amplitude, opts.seed);
        std::vector<float> s(sig.begin(), sig.end());
        write_samples(out, std::span<const float>(s), hdr);
        break;
    }
    case GenerateType::Pink: {
        auto s = pink_noise_generate(n, opts.amplitude, opts.seed);
        write_samples(out, std::span<const float>(s), hdr);
        break;
    }
    }

    return n;
}

GenerateOptions parse_generate_args(int argc, char** argv)
{
    // TODO: implement argument parsing
    (void)argc; (void)argv;
    throw std::logic_error("parse_generate_args: not yet implemented");
}

} // namespace cps::cli
