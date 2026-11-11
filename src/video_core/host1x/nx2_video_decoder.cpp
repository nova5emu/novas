// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <optional>
#include <set>

#ifdef __ANDROID__
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#endif

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host1x/ffmpeg.h"
#include "video_core/host1x/h264_packet.h"
#include "video_core/host1x/nx2_video_decoder.h"

extern "C" {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#include <libswscale/swscale.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

namespace Tegra::Host1x {

namespace {

std::optional<NvdecCommon::VideoCodec> CodecFromMime(std::string_view mime) {
    if (mime.find("vp9") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP9;
    }
    if (mime.find("vp8") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP8;
    }
    // "video/av01" or "video/av1". FFmpeg decodes AV1 with libdav1d.
    if (mime.find("av01") != std::string_view::npos || mime.find("av1") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::AV1;
    }
    if (mime.find("avc") != std::string_view::npos || mime.find("264") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::H264;
    }
    return std::nullopt;
}

#ifdef __ANDROID__
/// Decodes with the platform's MediaCodec into NV12 FFmpeg frames, for codecs the bundled Android
/// FFmpeg has no decoder for (AV1: Switch 2 movies). Most phones decode AV1 in hardware.
class MediaCodecDecoder {
public:
    static std::unique_ptr<MediaCodecDecoder> Create(const char* mime, u32 width, u32 height) {
        AMediaCodec* codec = AMediaCodec_createDecoderByType(mime);
        if (codec == nullptr) {
            LOG_ERROR(HW_GPU, "nx2 video: MediaCodec has no decoder for \"{}\"", mime);
            return nullptr;
        }
        auto decoder = std::unique_ptr<MediaCodecDecoder>(new MediaCodecDecoder(codec));
        decoder->width = width != 0 ? static_cast<s32>(width) : 1920;
        decoder->height = height != 0 ? static_cast<s32>(height) : 1080;
        decoder->stride = decoder->width;
        decoder->slice_height = decoder->height;

        AMediaFormat* format = AMediaFormat_new();
        AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, mime);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, decoder->width);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, decoder->height);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, ColorFormatYuv420Flexible);
        const media_status_t configured = AMediaCodec_configure(codec, format, nullptr, nullptr, 0);
        AMediaFormat_delete(format);
        if (configured != AMEDIA_OK || AMediaCodec_start(codec) != AMEDIA_OK) {
            LOG_ERROR(HW_GPU, "nx2 video: MediaCodec \"{}\" failed to start ({})", mime,
                      static_cast<int>(configured));
            return nullptr;
        }
        decoder->started = true;
        char* name = nullptr;
        if (AMediaCodec_getName(codec, &name) == AMEDIA_OK && name != nullptr) {
            LOG_INFO(HW_GPU, "nx2 video: MediaCodec decoder {} for \"{}\" {}x{}", name, mime,
                     decoder->width, decoder->height);
            AMediaCodec_releaseName(codec, name);
        }
        return decoder;
    }

    ~MediaCodecDecoder() {
        if (started) {
            AMediaCodec_stop(codec);
        }
        AMediaCodec_delete(codec);
    }

    void Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
        // While every input slot is taken, collect finished frames to free one.
        for (int attempt = 0; attempt < 250; ++attempt) {
            const ssize_t index = AMediaCodec_dequeueInputBuffer(codec, 2000);
            if (index >= 0) {
                size_t capacity{};
                u8* buffer = AMediaCodec_getInputBuffer(codec, static_cast<size_t>(index), &capacity);
                const size_t size = std::min(capacity, data.size());
                if (size < data.size()) {
                    LOG_WARNING(HW_GPU, "nx2 video: frame of {} bytes cut to MediaCodec's {}",
                                data.size(), capacity);
                }
                std::memcpy(buffer, data.data(), size);
                AMediaCodec_queueInputBuffer(codec, static_cast<size_t>(index), 0, size, pts, 0);
                ++in_flight;
                // Frames finished later are collected through Poll.
                Receive(pictures, 0);
                return;
            }
            Receive(pictures, 0);
        }
        LOG_WARNING(HW_GPU, "nx2 video: MediaCodec took no input, frame dropped");
    }

    void Drain(std::vector<Nx2Picture>& pictures) {
        for (int attempt = 0; attempt < 250; ++attempt) {
            const ssize_t index = AMediaCodec_dequeueInputBuffer(codec, 2000);
            if (index >= 0) {
                AMediaCodec_queueInputBuffer(codec, static_cast<size_t>(index), 0, 0, 0,
                                             AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
                break;
            }
            Receive(pictures, 0);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        while (!end_of_stream && std::chrono::steady_clock::now() < deadline) {
            Receive(pictures, 10000);
        }
        Flush();
    }

    void Flush() {
        AMediaCodec_flush(codec);
        end_of_stream = false;
        in_flight = 0;
    }

    bool HasPendingOutput() const {
        return in_flight > 0;
    }

    void Poll(std::vector<Nx2Picture>& pictures) {
        Receive(pictures, 0);
    }

private:
    static constexpr s32 ColorFormatYuv420Planar = 19;
    static constexpr s32 ColorFormatYuv420PackedPlanar = 20;
    static constexpr s32 ColorFormatYuv420Flexible = 0x7F420888;

    explicit MediaCodecDecoder(AMediaCodec* codec_) : codec{codec_} {}

    /// Collects finished frames: waits up to first_timeout_us for the first, then takes only
    /// what is ready.
    void Receive(std::vector<Nx2Picture>& pictures, s64 first_timeout_us) {
        s64 timeout = first_timeout_us;
        while (true) {
            AMediaCodecBufferInfo info{};
            const ssize_t index = AMediaCodec_dequeueOutputBuffer(codec, &info, timeout);
            timeout = 0;
            if (index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                ReadOutputFormat();
                continue;
            }
            if (index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                continue;
            }
            if (index < 0) {
                return;
            }
            if ((info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0) {
                end_of_stream = true;
            }
            if (info.size > 0) {
                size_t capacity{};
                const u8* buffer =
                    AMediaCodec_getOutputBuffer(codec, static_cast<size_t>(index), &capacity);
                if (buffer != nullptr && static_cast<size_t>(info.offset) + info.size <= capacity) {
                    if (auto picture = Convert(buffer + info.offset, static_cast<size_t>(info.size),
                                               info.presentationTimeUs)) {
                        pictures.push_back(std::move(*picture));
                    }
                }
                in_flight = std::max(in_flight - 1, 0);
            }
            AMediaCodec_releaseOutputBuffer(codec, static_cast<size_t>(index), false);
            if (end_of_stream) {
                return;
            }
        }
    }

    void ReadOutputFormat() {
        AMediaFormat* format = AMediaCodec_getOutputFormat(codec);
        if (format == nullptr) {
            return;
        }
        s32 value{};
        if (AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_WIDTH, &value)) {
            width = value;
        }
        if (AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_HEIGHT, &value)) {
            height = value;
        }
        stride = AMediaFormat_getInt32(format, "stride", &value) && value > 0 ? value : width;
        slice_height =
            AMediaFormat_getInt32(format, "slice-height", &value) && value > 0 ? value : height;
        if (AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, &value)) {
            color_format = value;
        }
        s32 left{}, top{}, right{}, bottom{};
        if (AMediaFormat_getRect(format, "crop", &left, &top, &right, &bottom) && right > left &&
            bottom > top) {
            crop_left = left;
            crop_top = top;
            visible_width = right - left + 1;
            visible_height = bottom - top + 1;
        } else {
            crop_left = crop_top = 0;
            visible_width = width;
            visible_height = height;
        }
        stride = std::max(stride, width);
        slice_height = std::max(slice_height, height);
        LOG_INFO(HW_GPU,
                 "nx2 video: MediaCodec output {}x{} visible {}x{}+{}+{} stride {} slice {} "
                 "color format {:#x}",
                 width, height, visible_width, visible_height, crop_left, crop_top, stride,
                 slice_height, color_format);
        AMediaFormat_delete(format);
    }

    std::optional<Nx2Picture> Convert(const u8* src, size_t size, s64 pts) {
        const s32 out_width = visible_width > 0 ? visible_width : width;
        const s32 out_height = visible_height > 0 ? visible_height : height;
        const bool planar = color_format == ColorFormatYuv420Planar ||
                            color_format == ColorFormatYuv420PackedPlanar;
        const size_t luma_size = static_cast<size_t>(stride) * slice_height;
        // One past the last byte read: decoders leave the last chroma row unpadded (Exynos AV1:
        // stride 2048, slice height 1088, a buffer 128 bytes short of a full plane).
        const size_t last_chroma_row = static_cast<size_t>(crop_top / 2 + (out_height + 1) / 2 - 1);
        const size_t chroma_bytes = static_cast<size_t>((out_width + 1) / 2);
        const size_t needed =
            planar ? luma_size + static_cast<size_t>(stride / 2) * (slice_height / 2) +
                         last_chroma_row * static_cast<size_t>(stride / 2) + crop_left / 2 +
                         chroma_bytes
                   : luma_size + last_chroma_row * static_cast<size_t>(stride) +
                         static_cast<size_t>(crop_left & ~1) + chroma_bytes * 2;
        if (out_width <= 0 || out_height <= 0 || size < needed) {
            if (!warned_layout) {
                warned_layout = true;
                LOG_ERROR(HW_GPU, "nx2 video: MediaCodec frame of {} bytes, expected {} ({}x{})",
                          size, needed, out_width, out_height);
            }
            return std::nullopt;
        }

        auto frame = std::make_shared<FFmpeg::Frame>();
        AVFrame* f = frame->GetFrame();
        f->format = AV_PIX_FMT_NV12;
        f->width = out_width;
        f->height = out_height;
        if (av_frame_get_buffer(f, 0) < 0) {
            return std::nullopt;
        }
        f->pts = pts;
        f->best_effort_timestamp = pts;
        f->color_range = AVCOL_RANGE_MPEG;

        const u8* luma = src + static_cast<size_t>(crop_top) * stride + crop_left;
        for (s32 y = 0; y < out_height; ++y) {
            std::memcpy(f->data[0] + static_cast<size_t>(y) * f->linesize[0],
                        luma + static_cast<size_t>(y) * stride, static_cast<size_t>(out_width));
        }
        const s32 chroma_width = (out_width + 1) / 2;
        const s32 chroma_height = (out_height + 1) / 2;
        if (planar) {
            const s32 chroma_stride = stride / 2;
            const u8* u = src + luma_size + static_cast<size_t>(crop_top / 2) * chroma_stride +
                          crop_left / 2;
            const u8* v = u + static_cast<size_t>(chroma_stride) * (slice_height / 2);
            for (s32 y = 0; y < chroma_height; ++y) {
                u8* dst = f->data[1] + static_cast<size_t>(y) * f->linesize[1];
                const u8* u_row = u + static_cast<size_t>(y) * chroma_stride;
                const u8* v_row = v + static_cast<size_t>(y) * chroma_stride;
                for (s32 x = 0; x < chroma_width; ++x) {
                    dst[x * 2] = u_row[x];
                    dst[x * 2 + 1] = v_row[x];
                }
            }
        } else {
            const u8* uv = src + luma_size + static_cast<size_t>(crop_top / 2) * stride +
                           (crop_left & ~1);
            for (s32 y = 0; y < chroma_height; ++y) {
                std::memcpy(f->data[1] + static_cast<size_t>(y) * f->linesize[1],
                            uv + static_cast<size_t>(y) * stride,
                            static_cast<size_t>(chroma_width) * 2);
            }
        }
        return Nx2Picture{static_cast<u32>(out_width), static_cast<u32>(out_height),
                          static_cast<u64>(pts), std::move(frame)};
    }

    AMediaCodec* codec{};
    bool started{};
    bool end_of_stream{};
    bool warned_layout{};
    s32 in_flight{};
    s32 width{};
    s32 height{};
    s32 stride{};
    s32 slice_height{};
    s32 color_format{ColorFormatYuv420Flexible};
    s32 crop_left{};
    s32 crop_top{};
    s32 visible_width{};
    s32 visible_height{};
};
#endif

} // namespace

bool IsH264Stream(std::span<const u8> data) {
    if (data.size() >= 4) {
        // Annex-B 3-byte start code: 00 00 01 <nal_header>
        if (data[0] == 0 && data[1] == 0 && data[2] == 1) {
            const u8 nal_type = data[3] & 0x1F;
            if ((data[3] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 5) {
        // Annex-B 4-byte start code: 00 00 00 01 <nal_header>
        if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
        // avcC / MP4 length-prefixed NAL unit: [u32 be length] <nal_header>
        const u32 nalu_len = (static_cast<u32>(data[0]) << 24) |
                             (static_cast<u32>(data[1]) << 16) |
                             (static_cast<u32>(data[2]) << 8) |
                             static_cast<u32>(data[3]);
        if (nalu_len > 0 && nalu_len <= data.size() - 4) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 7) {
        // avcC configuration record: version=1, profile, compat, level, lengthSizeMinusOne | 0xFC, numSPS | 0xE0
        if (data[0] == 1 && (data[4] & 0xFC) == 0xFC && (data[5] & 0xE0) == 0xE0 && (data[5] & 0x1F) > 0) {
            return true;
        }
    }
    return false;
}

struct Nx2VideoDecoder::Impl {
    std::string name;
    std::optional<FFmpeg::Decoder> decoder;
    std::optional<FFmpeg::HardwareContext> hardware;
    std::optional<FFmpeg::DecoderContext> context;
    SwsContext* scaler{};
    bool open{};
#ifdef __ANDROID__
    std::unique_ptr<MediaCodecDecoder> media_codec;
#endif
    H264PacketConverter h264_packets;
    /// Timestamps of submitted frames not yet output. Frames copied back from a host GPU decoder
    /// lose theirs and take the earliest one instead.
    std::multiset<u64> pending_pts;
    std::vector<u8> packet; ///< FFmpeg reads up to AV_INPUT_BUFFER_PADDING_SIZE past the data

    ~Impl() {
        sws_freeContext(scaler);
        context.reset();
        hardware.reset();
    }

    void Receive(std::vector<Nx2Picture>& pictures) {
        while (auto frame = context->ReceiveFrame()) {
            const AVFrame* f = frame->GetFrame();
            const s64 stamp = f->best_effort_timestamp != AV_NOPTS_VALUE ? f->best_effort_timestamp
                                                                         : f->pts;
            u64 pts{};
            if (stamp != AV_NOPTS_VALUE) {
                pts = static_cast<u64>(stamp);
                if (const auto it = pending_pts.find(pts); it != pending_pts.end()) {
                    pending_pts.erase(it);
                }
            } else if (!pending_pts.empty()) {
                pts = *pending_pts.begin();
                pending_pts.erase(pending_pts.begin());
            }
            pictures.push_back({static_cast<u32>(f->width), static_cast<u32>(f->height), pts,
                                std::move(frame)});
        }
    }
};

Nx2VideoDecoder::Nx2VideoDecoder(std::string_view mime, [[maybe_unused]] u32 width,
                                 [[maybe_unused]] u32 height)
    : impl{std::make_unique<Impl>()} {
    impl->name = std::string{mime};
    const auto codec = CodecFromMime(mime);
    if (!codec) {
        LOG_ERROR(HW_GPU, "nx2 video: no decoder for codec \"{}\"", mime);
        return;
    }
    impl->decoder.emplace(*codec);
    if (impl->decoder->GetCodec() == nullptr) {
#ifdef __ANDROID__
        if (*codec == NvdecCommon::VideoCodec::AV1) {
            impl->media_codec = MediaCodecDecoder::Create("video/av01", width, height);
            impl->open = impl->media_codec != nullptr;
            LOG_INFO(HW_GPU, "nx2 video: MediaCodec decoder for \"{}\" {}", mime,
                     impl->open ? "open" : "failed to open");
            return;
        }
#endif
        LOG_ERROR(HW_GPU, "nx2 video: FFmpeg has no decoder for \"{}\"", mime);
        return;
    }
    impl->context.emplace(*impl->decoder);
    if (Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu) {
        impl->hardware.emplace();
        impl->hardware->InitializeForDecoder(*impl->context, *impl->decoder);
    } else {
        // From PokeDinner: one decoder thread falls below 60 fps on 1080p-1440p movies (AV1 with
        // dav1d, VP9), and the client drops the movie when no frame is ready. Frame threads add a
        // few frames of latency, covered by the client's queued inputs.
        AVCodecContext* codec_context = impl->context->GetCodecContext();
        codec_context->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        codec_context->thread_count = 4;
    }
    impl->open = impl->context->OpenContext(*impl->decoder);
    LOG_INFO(HW_GPU, "nx2 video: decoder for \"{}\" {}", mime, impl->open ? "open" : "failed to open");
}

Nx2VideoDecoder::~Nx2VideoDecoder() = default;

bool Nx2VideoDecoder::IsOpen() const {
    return impl->open;
}

const std::string& Nx2VideoDecoder::Name() const {
    return impl->name;
}

void Nx2VideoDecoder::Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
    if (!impl->open || data.empty()) {
        return;
    }
#ifdef __ANDROID__
    if (impl->media_codec) {
        impl->media_codec->Decode(data, pts, pictures);
        return;
    }
#endif
    impl->packet.assign(data.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);
    std::memcpy(impl->packet.data(), data.data(), data.size());

    // Convert MP4/avcC length-prefixed NALUs to Annex-B start codes for FFmpeg if needed.
    if (impl->name.find("avc") != std::string::npos || impl->name.find("264") != std::string::npos) {
        impl->h264_packets.Convert(std::span<u8>{impl->packet.data(), data.size()});
    }

    FFmpeg::Packet packet{std::span<const u8>{impl->packet.data(), data.size()}};
    packet.GetPacket()->pts = static_cast<s64>(pts);
    // Hidden VP9 frames never come out: keep the set from growing over a long movie.
    if (impl->pending_pts.size() >= 64) {
        impl->pending_pts.erase(impl->pending_pts.begin());
    }
    impl->pending_pts.insert(pts);
    impl->context->SendPacket(packet);
    impl->Receive(pictures);
}

void Nx2VideoDecoder::Drain(std::vector<Nx2Picture>& pictures) {
    if (!impl->open) {
        return;
    }
#ifdef __ANDROID__
    if (impl->media_codec) {
        impl->media_codec->Drain(pictures);
        return;
    }
#endif
    avcodec_send_packet(impl->context->GetCodecContext(), nullptr);
    impl->Receive(pictures);
    Reset();
}

bool Nx2VideoDecoder::HasPendingOutput() const {
#ifdef __ANDROID__
    return impl->media_codec && impl->media_codec->HasPendingOutput();
#else
    return false;
#endif
}

void Nx2VideoDecoder::Poll([[maybe_unused]] std::vector<Nx2Picture>& pictures) {
#ifdef __ANDROID__
    if (impl->media_codec) {
        impl->media_codec->Poll(pictures);
    }
#endif
}

void Nx2VideoDecoder::Reset() {
#ifdef __ANDROID__
    if (impl->media_codec) {
        impl->media_codec->Flush();
        impl->pending_pts.clear();
        return;
    }
#endif
    if (impl->open) {
        avcodec_flush_buffers(impl->context->GetCodecContext());
    }
    impl->pending_pts.clear();
    impl->h264_packets.Reset();
}

void Nx2VideoDecoder::Render(const Nx2Picture& picture, Nx2PictureFormat format, u32 width,
                             u32 height, u64 pitch, u64 chroma_offset, std::span<u8> out) {
    const AVFrame* f = picture.frame->GetFrame();
    const u64 needed = format == Nx2PictureFormat::Nv12
                           ? chroma_offset + pitch * ((u64{height} + 1) / 2)
                           : pitch * height;
    const u64 min_pitch = format == Nx2PictureFormat::Nv12 ? ((u64{width} + 1) / 2) * 2 : 0;
    if (width == 0 || height == 0 || pitch < min_pitch || out.size() < needed) {
        return;
    }
    const AVPixelFormat target = format == Nx2PictureFormat::Nv12    ? AV_PIX_FMT_NV12
                                 : format == Nx2PictureFormat::Rgba8 ? AV_PIX_FMT_RGBA
                                                                     : AV_PIX_FMT_BGRA;
    // Frames copied back from a host GPU decoder are already NV12. Copy them as they are:
    // FFmpeg 6.1's swscale leaves the chroma plane of an NV12 to NV12 copy at zero, which shows
    // the movies green.
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_NV12 &&
        static_cast<u32>(f->width) == width && static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            std::memcpy(out.data() + chroma_offset + y * pitch, f->data[1] + y * f->linesize[1],
                        min_pitch);
        }
        return;
    }
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_YUV420P &&
        f->color_range != AVCOL_RANGE_JPEG && static_cast<u32>(f->width) == width &&
        static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        const u32 chroma_width = static_cast<u32>(min_pitch / 2);
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            const u8* u = f->data[1] + y * f->linesize[1];
            const u8* v = f->data[2] + y * f->linesize[2];
            u8* dst = out.data() + chroma_offset + y * pitch;
            for (u32 x = 0; x < chroma_width; ++x) {
                dst[x * 2] = u[x];
                dst[x * 2 + 1] = v[x];
            }
        }
        return;
    }
    impl->scaler = sws_getCachedContext(impl->scaler, f->width, f->height,
                                        static_cast<AVPixelFormat>(f->format),
                                        static_cast<int>(width), static_cast<int>(height), target,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (impl->scaler == nullptr) {
        LOG_ERROR(HW_GPU, "nx2 video: cannot convert {} to {}",
                  av_get_pix_fmt_name(static_cast<AVPixelFormat>(f->format)),
                  av_get_pix_fmt_name(target));
        return;
    }
    // The stream's matrix and range; untagged HD video is BT.709.
    int space = SWS_CS_ITU601;
    if (f->colorspace == AVCOL_SPC_BT709 ||
        (f->colorspace == AVCOL_SPC_UNSPECIFIED && f->height >= 720)) {
        space = SWS_CS_ITU709;
    } else if (f->colorspace == AVCOL_SPC_BT2020_NCL || f->colorspace == AVCOL_SPC_BT2020_CL) {
        space = SWS_CS_BT2020;
    }
    const bool full_range_source = f->color_range == AVCOL_RANGE_JPEG ||
                                  f->format == AV_PIX_FMT_YUVJ420P ||
                                  f->format == AV_PIX_FMT_YUVJ422P ||
                                  f->format == AV_PIX_FMT_YUVJ444P ||
                                  f->format == AV_PIX_FMT_YUVJ440P ||
                                  f->format == AV_PIX_FMT_YUVJ411P;
    const int* coefficients = sws_getCoefficients(space);
    sws_setColorspaceDetails(impl->scaler, coefficients, full_range_source, coefficients,
                             format != Nx2PictureFormat::Nv12, 0, 1 << 16, 1 << 16);
    std::array<u8*, 4> planes{out.data(), nullptr, nullptr, nullptr};
    std::array<int, 4> strides{static_cast<int>(pitch), 0, 0, 0};
    if (format == Nx2PictureFormat::Nv12) {
        planes[1] = out.data() + chroma_offset;
        strides[1] = static_cast<int>(pitch);
    }
    sws_scale(impl->scaler, f->data, f->linesize, 0, f->height, planes.data(), strides.data());
}

} // namespace Tegra::Host1x
