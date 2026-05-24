#include "format.hpp"
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace cps::cli {

static constexpr char     kMagic[4] = {'C', 'P', 'S', '\0'};
static constexpr uint8_t  kVersion  = 1;

void write_header(std::ostream& out, const StreamHeader& h)
{
    out.write(kMagic, 4);
    uint8_t ver = kVersion;
    out.write(reinterpret_cast<const char*>(&ver), 1);
    uint8_t fmt = static_cast<uint8_t>(h.format);
    out.write(reinterpret_cast<const char*>(&fmt), 1);
    out.write(reinterpret_cast<const char*>(&h.channels),    2);
    out.write(reinterpret_cast<const char*>(&h.sample_rate), 4);
    if (!out)
        throw std::runtime_error("write_header: I/O error (broken pipe?)");
}

StreamHeader read_header(std::istream& in)
{
    char magic[4] = {};
    if (!in.read(magic, 4) || std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("invalid CPS stream: bad magic");

    uint8_t ver = 0;
    if (!in.read(reinterpret_cast<char*>(&ver), 1) || ver != kVersion)
        throw std::runtime_error("invalid CPS stream: unsupported version");

    uint8_t fmt = 0;
    if (!in.read(reinterpret_cast<char*>(&fmt), 1))
        throw std::runtime_error("invalid CPS stream: truncated header");

    StreamHeader h;
    h.format = static_cast<SampleFormat>(fmt);
    if (!in.read(reinterpret_cast<char*>(&h.channels),    2) ||
        !in.read(reinterpret_cast<char*>(&h.sample_rate), 4))
        throw std::runtime_error("invalid CPS stream: truncated header");

    return h;
}

void write_samples(std::ostream& out, std::span<const float> s, const StreamHeader&)
{
    out.write(reinterpret_cast<const char*>(s.data()), static_cast<std::streamsize>(s.size_bytes()));
    if (!out)
        throw std::runtime_error("write_samples: I/O error (broken pipe?)");
}

void write_samples(std::ostream& out, std::span<const double> s, const StreamHeader&)
{
    out.write(reinterpret_cast<const char*>(s.data()), static_cast<std::streamsize>(s.size_bytes()));
    if (!out)
        throw std::runtime_error("write_samples: I/O error (broken pipe?)");
}

std::vector<float> read_samples_f32(std::istream& in, const StreamHeader& h)
{
    std::vector<float> out;
    if (h.format == SampleFormat::Float32) {
        float f;
        while (in.read(reinterpret_cast<char*>(&f), sizeof(float)))
            out.push_back(f);
        if (in.gcount() > 0)
            throw std::runtime_error("CPS stream: truncated sample data (partial float32)");
    } else {
        double d;
        while (in.read(reinterpret_cast<char*>(&d), sizeof(double)))
            out.push_back(static_cast<float>(d));
        if (in.gcount() > 0)
            throw std::runtime_error("CPS stream: truncated sample data (partial float64)");
    }
    return out;
}

} // namespace cps::cli
