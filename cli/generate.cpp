#include "generate.hpp"
#include <cps/cps.hpp>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

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
    GenerateOptions opts;
    bool type_set = false;

    for (int i = 2; i + 1 < argc; i += 2) {
        std::string_view key{argv[i]};
        std::string_view val{argv[i + 1]};

        if (key == "--type") {
            type_set = true;
            if      (val == "sine")  opts.type = GenerateType::Sine;
            else if (val == "chirp") opts.type = GenerateType::Chirp;
            else if (val == "white") opts.type = GenerateType::White;
            else if (val == "pink")  opts.type = GenerateType::Pink;
            else throw std::invalid_argument("unknown generator type: " + std::string(val));
        } else if (key == "--freq") {
            opts.freq = std::stod(std::string(val));
        } else if (key == "--freq-start") {
            opts.freq_start = std::stod(std::string(val));
        } else if (key == "--freq-end") {
            opts.freq_end = std::stod(std::string(val));
        } else if (key == "--amplitude") {
            opts.amplitude = std::stod(std::string(val));
        } else if (key == "--duration") {
            opts.duration = std::stod(std::string(val));
        } else if (key == "--sample-rate") {
            opts.sample_rate = static_cast<uint32_t>(std::stoul(std::string(val)));
        } else if (key == "--channels") {
            opts.channels = static_cast<uint16_t>(std::stoul(std::string(val)));
        } else if (key == "--seed") {
            opts.seed = static_cast<unsigned>(std::stoul(std::string(val)));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(key));
        }
    }

    if (opts.amplitude <= 0.0)
        throw std::invalid_argument("--amplitude must be > 0");
    if (opts.duration <= 0.0)
        throw std::invalid_argument("--duration must be > 0");
    if (opts.sample_rate == 0)
        throw std::invalid_argument("--sample-rate must be > 0");

    const double nyquist = opts.sample_rate / 2.0;
    if (opts.type == GenerateType::Sine && opts.freq >= nyquist)
        throw std::invalid_argument("--freq " + std::to_string(opts.freq) +
                                    " Hz exceeds Nyquist (" + std::to_string(nyquist) + " Hz)");

    return opts;
}

} // namespace cps::cli
