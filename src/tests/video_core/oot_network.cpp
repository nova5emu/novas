// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "video_core/oot_network.h"

using namespace VideoCommon::OotNetwork;

namespace {
CopyEngine::Surface CopySurface(u32 width, u32 height, GPUVAddr address) {
    CopyEngine::Surface surface{};
    surface.format = Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    surface.linear = CopyEngine::MemoryLayout::BlockLinear;
    surface.block_height.Assign(4);
    surface.depth = 1;
    surface.width = width;
    surface.height = height;
    surface.addr_upper = static_cast<u32>(address >> 32);
    surface.addr_lower = static_cast<u32>(address);
    return surface;
}
} // namespace

TEST_CASE("Ocarina of Time anti-aliasing default follows the platform", "[video_core][oot_network]") {
#ifdef __ANDROID__
    CHECK(Settings::values.oot_anti_aliasing.GetDefault() == Settings::OotAntiAliasing::NoSharpening);
#else
    CHECK(Settings::values.oot_anti_aliasing.GetDefault() == Settings::OotAntiAliasing::Game);
#endif
}

TEST_CASE("Ocarina of Time tonemap shaders are recognized", "[video_core][oot_network]") {
    STATIC_REQUIRE(IsTonemapShader(0xbded5ce917183214ULL));
    STATIC_REQUIRE(IsTonemapShader(0x6cb06edf14be1d79ULL));
    STATIC_REQUIRE(IsTonemapShader(0x3ac011bd2d772994ULL));
    STATIC_REQUIRE_FALSE(IsTonemapShader(0));
    STATIC_REQUIRE_FALSE(IsTonemapShader(PrepassProgramKey));
}

TEST_CASE("Ocarina of Time network bypass follows the setting and the game",
          "[video_core][oot_network]") {
    auto& setting = Settings::values.oot_anti_aliasing;
    const auto previous = setting.GetValue();

    setting.SetValue(Settings::OotAntiAliasing::Game);
    CHECK_FALSE(BypassEnabled(ProgramId));

    setting.SetValue(Settings::OotAntiAliasing::NoSharpening);
    CHECK(BypassEnabled(ProgramId));
    CHECK_FALSE(BypassEnabled(0x0400C3F00006E000ULL));

    setting.SetValue(previous);
}

TEST_CASE("Ocarina of Time intermediate passes require a replaced output",
          "[video_core][oot_network]") {
    for (u64 key : {PrepassProgramKey, NetworkProgramKey}) {
        CHECK_FALSE(ShouldSkipIntermediate(key, false));
        CHECK(ShouldSkipIntermediate(key, true));
    }
    CHECK_FALSE(ShouldSkipIntermediate(OutputProgramKey, true));
    CHECK_FALSE(ShouldSkipIntermediate(HandheldOutputShaderHash, true));
    CHECK_FALSE(ShouldSkipIntermediate(0xd9ee412f00ce399aULL, true));
    CHECK_FALSE(ShouldSkipIntermediate(0x03bf0ffc7d2d0bedULL, true));
    CHECK_FALSE(ShouldSkipIntermediate(0x56aef55a14b812c5ULL, true));
    CHECK_FALSE(ShouldSkipIntermediate(0, true));
}

TEST_CASE("Ocarina of Time docked and handheld output passes are replaced", "[video_core][oot_network]") {
    for (u64 key : {u64{0xda6df3e54c716df2ULL}, u64{0x6f9367bd41dc0b58ULL},
                    u64{0xd8efd9f0c95abd80ULL}}) {
        CHECK(IsOutputProgram(key));
        CHECK_FALSE(ShouldSkipIntermediate(key, true));
        CHECK_FALSE(ShouldSkipIntermediate(key, false));
    }
    CHECK_FALSE(IsOutputProgram(PrepassProgramKey));
    CHECK_FALSE(IsOutputProgram(NetworkProgramKey));
    CHECK_FALSE(IsOutputProgram(HandheldOutputShaderHash));
    CHECK_FALSE(IsOutputProgram(0xd9ee412f00ce399aULL));
}

TEST_CASE("Ocarina of Time copy uses actual source dimensions and layout", "[video_core][oot_network]") {
    auto source = CopySurface(1440, 810, 0x8114786000ULL);
    auto destination = CopySurface(1920, 1080, 0x81252ff000ULL);
    destination.block_height.Assign(5);
    const auto scaled = MakeCopyConfig(source, destination);
    REQUIRE(scaled);
    CHECK(scaled->filter == CopyEngine::Filter::Bilinear);
    CHECK(scaled->must_accelerate);
    CHECK(scaled->src_x1 == 1440);
    CHECK(scaled->src_y1 == 810);
    CHECK(scaled->dst_x1 == 1920);
    CHECK(scaled->dst_y1 == 1080);

    source.width = destination.width;
    source.height = destination.height;
    const auto equal = MakeCopyConfig(source, destination);
    REQUIRE(equal);
    CHECK(equal->filter == CopyEngine::Filter::Point);

    source = CopySurface(2560, 1440, source.Address());
    const auto downscaled = MakeCopyConfig(source, destination);
    REQUIRE(downscaled);
    CHECK(downscaled->filter == CopyEngine::Filter::Bilinear);
}

TEST_CASE("Ocarina of Time copy rejects invalid surfaces and unreasonable sizes", "[video_core][oot_network]") {
    auto destination = CopySurface(1920, 1080, 0x81252ff000ULL);
    auto source = CopySurface(1440, 810, 0x8114786000ULL);
    SECTION("wrong format") {
        source.format = Tegra::RenderTargetFormat::R32_FLOAT;
    }
    SECTION("pitch linear") {
        source.linear = CopyEngine::MemoryLayout::Pitch;
    }
    SECTION("zero width") {
        source.width = 0;
    }
    SECTION("zero height") {
        source.height = 0;
    }
    SECTION("oversized") {
        source.width = 8193;
    }
    SECTION("aspect mismatch") {
        source.height = 1080;
    }
    SECTION("excessive upscale") {
        source.width = 384;
        source.height = 216;
    }
    SECTION("excessive downscale") {
        source.width = 5760;
        source.height = 3240;
    }
    SECTION("missing address") {
        source.addr_upper = 0;
        source.addr_lower = 0;
    }
    SECTION("same address") {
        source.addr_upper = destination.addr_upper;
        source.addr_lower = destination.addr_lower;
    }
    SECTION("array layer") {
        source.layer = 1;
    }
    SECTION("multiple layers") {
        source.depth = 2;
    }
    SECTION("3D tiling") {
        source.block_depth.Assign(1);
    }
    SECTION("unsupported block width") {
        source.block_width.Assign(1);
    }
    SECTION("unsupported block height") {
        source.block_height.Assign(6);
    }
    SECTION("invalid destination") {
        destination.format = Tegra::RenderTargetFormat::R32_FLOAT;
    }
    CHECK_FALSE(MakeCopyConfig(source, destination));
}

TEST_CASE("Ocarina of Time dynamic resolution accepts a small relative aspect difference",
          "[video_core][oot_network]") {
    const auto source = CopySurface(864, 480, 0x8113949000ULL);
    const auto destination = CopySurface(1920, 1080, 0x81252ff000ULL);
    const auto config = MakeCopyConfig(source, destination);
    REQUIRE(config);
    CHECK(config->filter == CopyEngine::Filter::Bilinear);
    CHECK(config->src_x0 == 0);
    CHECK(config->src_y0 == 0);
    CHECK(config->src_x1 == 864);
    CHECK(config->src_y1 == 480);
    CHECK(config->dst_x1 == 1920);
    CHECK(config->dst_y1 == 1080);
}

TEST_CASE("Ocarina of Time dynamic resolution enforces relative aspect and scale bounds",
          "[video_core][oot_network]") {
    const auto destination = CopySurface(1920, 1080, 0x81252ff000ULL);
    SECTION("exactly four times per axis") {
        CHECK(MakeCopyConfig(CopySurface(480, 270, 0x8113949000ULL), destination));
    }
    SECTION("above four times on both axes") {
        CHECK_FALSE(MakeCopyConfig(CopySurface(448, 252, 0x8113949000ULL), destination));
    }
    SECTION("above four times on the vertical axis only") {
        CHECK_FALSE(MakeCopyConfig(CopySurface(480, 269, 0x8113949000ULL), destination));
    }
    SECTION("above four times on the horizontal axis only") {
        CHECK_FALSE(MakeCopyConfig(CopySurface(479, 270, 0x8113949000ULL), destination));
    }
    SECTION("two percent upper boundary") {
        CHECK(MakeCopyConfig(CopySurface(1632, 900, 0x8113949000ULL), destination));
        CHECK_FALSE(MakeCopyConfig(CopySurface(1633, 900, 0x8113949000ULL), destination));
    }
    SECTION("two percent lower boundary") {
        CHECK(MakeCopyConfig(CopySurface(1568, 900, 0x8113949000ULL), destination));
        CHECK_FALSE(MakeCopyConfig(CopySurface(1567, 900, 0x8113949000ULL), destination));
    }
    SECTION("four to three aspect remains rejected") {
        CHECK_FALSE(MakeCopyConfig(CopySurface(640, 480, 0x8113949000ULL), destination));
    }
}

TEST_CASE("Ocarina of Time tonemap snapshots retain each render target descriptor", "[video_core][oot_network]") {
    BypassState state;
    CHECK_FALSE(state.GetTonemappedSurface());
    const auto first = CopySurface(1440, 810, 0x8114786000ULL);
    state.SetTonemappedSurface(first);
    state.output_replaced.store(true);
    auto second = CopySurface(1920, 1080, 0x8113949000ULL);
    second.block_height.Assign(5);
    state.SetTonemappedSurface(second);
    const auto snapshot = state.GetTonemappedSurface();
    REQUIRE(snapshot);
    CHECK(snapshot->Address() == second.Address());
    CHECK(snapshot->width == second.width);
    CHECK(snapshot->height == second.height);
    CHECK(snapshot->block_height.Value() == 5);
    CHECK(state.tonemapped_image.load() == second.Address());
    CHECK(state.output_replaced.load());
    state.SetTonemappedSurface(std::nullopt);
    CHECK_FALSE(state.GetTonemappedSurface());
    CHECK(state.tonemapped_image.load() == 0);
    CHECK_FALSE(state.output_replaced.load());
}
