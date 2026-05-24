#include "play.hpp"
#include <miniaudio.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace cps::cli {

namespace {

struct PlaybackState {
    const float*             data;
    std::size_t              total;    // total frames (samples / channels)
    std::atomic<std::size_t> pos{0};
    ma_uint32                channels;
};

void data_callback(ma_device* device, void* output, const void*, ma_uint32 frameCount)
{
    auto* s       = static_cast<PlaybackState*>(device->pUserData);
    std::size_t cur  = s->pos.load(std::memory_order_relaxed);
    std::size_t remaining = s->total - cur;
    std::size_t take      = static_cast<std::size_t>(frameCount) < remaining
                            ? static_cast<std::size_t>(frameCount)
                            : remaining;

    std::memcpy(output,
                s->data + cur * s->channels,
                take * s->channels * sizeof(float));

    if (take < static_cast<std::size_t>(frameCount))
        std::memset(static_cast<float*>(output) + take * s->channels, 0,
                    (static_cast<std::size_t>(frameCount) - take) * s->channels * sizeof(float));

    s->pos.store(cur + take, std::memory_order_release);
}

} // namespace

void play_stream(std::istream& in)
{
    auto hdr     = read_header(in);
    auto samples = read_samples_f32(in, hdr);

    if (samples.empty())
        throw std::runtime_error("cps play: stream contains no samples");

    PlaybackState state;
    state.data     = samples.data();
    state.total    = samples.size() / hdr.channels;
    state.channels = hdr.channels;

    ma_device_config cfg  = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = hdr.channels;
    cfg.sampleRate        = hdr.sample_rate;
    cfg.dataCallback      = data_callback;
    cfg.pUserData         = &state;

    ma_device device;
    ma_result result = ma_device_init(nullptr, &cfg, &device);
    if (result != MA_SUCCESS)
        throw std::runtime_error(
            std::string("cps play: failed to open audio device: ") +
            ma_result_description(result));

    result = ma_device_start(&device);
    if (result != MA_SUCCESS) {
        ma_device_uninit(&device);
        throw std::runtime_error(
            std::string("cps play: failed to start audio device: ") +
            ma_result_description(result));
    }

    while (state.pos.load(std::memory_order_acquire) < state.total)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // All frames have been handed to the device; wait for the internal
    // buffer to drain before tearing down (typically 2-3 device periods).
    ma_uint32 period_frames = device.playback.internalPeriodSizeInFrames;
    ma_uint32 period_count  = device.playback.internalPeriods;
    unsigned  drain_ms      = (period_frames && device.sampleRate)
                              ? (period_frames * period_count * 1000u) / device.sampleRate + 50u
                              : 200u;
    std::this_thread::sleep_for(std::chrono::milliseconds(drain_ms));

    ma_device_uninit(&device);
}

} // namespace cps::cli
