/**************************************************************
 *   █████╗ ██╗     ███╗   ███╗   ███╗   ██╗    ██╗██████╗    *
 *  ██╔══██╗██║     ████╗ ████║ ██╔═══██╗████╗  ██║██╔══██╗   *
 *  ███████║██║     ██╔████╔██║ ██║   ██║██╔██╗ ██║██║  ██║   *
 *  ██╔══██║██║     ██║╚██╔╝██║ ██║   ██║██║╚██╗██║██║  ██║   *
 *  ██║  ██║███████╗██║ ╚═╝ ██║ ╚██████╔╝██║ ╚████║██████╔╝   *
 *  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝    *
 *                                                            *
 *   This file is part of the Almond Project.                 *
 *   AlmondShell - Modular C++ Framework                      *
 *                                                            *
 *   SPDX-License-Identifier: LicenseRef-MIT-NoSell           *
 *                                                            *
 *   Provided "AS IS", without warranty of any kind.          *
 *   Use permitted for Non-Commercial Purposes ONLY,          *
 *   without prior commercial licensing agreement.            *
 *                                                            *
 *   Redistribution Allowed with This Notice and              *
 *   LICENSE file. No obligation to disclose modifications.   *
 *                                                            *
 *   See LICENSE file for full terms.                         *
 *                                                            *
 **************************************************************/
 // agui.cpp
#include "pch.h"

#include "agui.hpp"

#include "aatlasmanager.hpp"
#include "acontext.hpp"
#include "aspritepool.hpp"
#include "aspriteregistry.hpp"
#include "atexture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace almondnamespace::gui
{
namespace
{
    using Vec2 = almondnamespace::gui::Vec2;
    using Color = ::almondnamespace::gui::Color;
    using EventType = almondnamespace::gui::EventType;
    using InputEvent = ::almondnamespace::gui::InputEvent;

    // ASCII glyphs 0x20-0x7F from the public domain font8x8_basic set by Daniel Hepper.
    constexpr std::array<std::array<std::uint8_t, 8>, 96> kFont8x8Basic = { {
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // space
        { 0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00 }, // !
        { 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // "
        { 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00 }, // #
        { 0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00 }, // $
        { 0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00 }, // %
        { 0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00 }, // &
        { 0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00 }, // '
        { 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00 }, // (
        { 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00 }, // )
        { 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00 }, // *
        { 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00 }, // +
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06 }, // ,
        { 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00 }, // -
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00 }, // .
        { 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00 }, // /
        { 0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00 }, // 0
        { 0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00 }, // 1
        { 0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00 }, // 2
        { 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00 }, // 3
        { 0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00 }, // 4
        { 0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00 }, // 5
        { 0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00 }, // 6
        { 0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00 }, // 7
        { 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00 }, // 8
        { 0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00 }, // 9
        { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00 }, // :
        { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06 }, // ;
        { 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00 }, // <
        { 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00 }, // =
        { 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00 }, // >
        { 0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00 }, // ?
        { 0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00 }, // @
        { 0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00 }, // A
        { 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00 }, // B
        { 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00 }, // C
        { 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00 }, // D
        { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00 }, // E
        { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00 }, // F
        { 0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00 }, // G
        { 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00 }, // H
        { 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00 }, // I
        { 0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00 }, // J
        { 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00 }, // K
        { 0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00 }, // L
        { 0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00 }, // M
        { 0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00 }, // N
        { 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00 }, // O
        { 0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00 }, // P
        { 0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00 }, // Q
        { 0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00 }, // R
        { 0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00 }, // S
        { 0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00 }, // T
        { 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00 }, // U
        { 0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00 }, // V
        { 0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00 }, // W
        { 0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00 }, // X
        { 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00 }, // Y
        { 0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00 }, // Z
        { 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00 }, // [
        { 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00 }, // \

        { 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00 }, // ]
        { 0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00 }, // ^
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF }, // _
        { 0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00 }, // `
        { 0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00 }, // a
        { 0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00 }, // b
        { 0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00 }, // c
        { 0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00 }, // d
        { 0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00 }, // e
        { 0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00 }, // f
        { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F }, // g
        { 0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00 }, // h
        { 0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00 }, // i
        { 0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E }, // j
        { 0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00 }, // k
        { 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00 }, // l
        { 0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00 }, // m
        { 0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00 }, // n
        { 0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00 }, // o
        { 0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F }, // p
        { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78 }, // q
        { 0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00 }, // r
        { 0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00 }, // s
        { 0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00 }, // t
        { 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00 }, // u
        { 0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00 }, // v
        { 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00 }, // w
        { 0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00 }, // x
        { 0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F }, // y
        { 0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00 }, // z
        { 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00 }, // {
        { 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00 }, // |
        { 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00 }, // }
        { 0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ~
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }  // DEL
    } };

    struct GuiResources {
        bool atlasBuilt = false;
        TextureAtlas* atlas = nullptr;
        SpriteHandle windowBackground{};
        SpriteHandle buttonNormal{};
        SpriteHandle buttonHover{};
        SpriteHandle buttonActive{};
        std::array<SpriteHandle, 96> glyphs{};
        float glyphWidth = 8.0f;
        float glyphHeight = 8.0f;
    };

    GuiResources g_resources{};
    std::mutex g_resourceMutex;

    struct PtrHash {
        size_t operator()(const void* p) const noexcept {
            return std::hash<const void*>{}(p);
        }
    };
    static std::unordered_set<const void*, PtrHash> g_uploadedContexts;
    static std::mutex g_uploadMutex;

    struct FrameState {
        // thread's active render context (must be set in begin_frame)
        core::Context* ctx = nullptr;

        Vec2 cursor{};
        Vec2 origin{};
        Vec2 windowSize{};
        Vec2 mousePos{};

        bool mouseDown = false;
        bool prevMouseDown = false;
        bool insideWindow = false;
        bool justPressed = false;
    };

    thread_local FrameState g_frame{};

    // Cheap tripwire so this doesn't regress again:
    static_assert(std::is_same_v<decltype(g_frame.ctx), core::Context*>,
        "FrameState must carry core::Context* ctx");

    constexpr const char* kAtlasName = "__agui_builtin";
    constexpr float kContentPadding = 8.0f;
    constexpr float kLineSpacing = 4.0f;
    constexpr float kFontScale = 2.0f;
    constexpr float kTitleScale = 2.5f;

    [[nodiscard]] std::vector<std::uint8_t> make_solid_pixels(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a, std::uint32_t w, std::uint32_t h)
    {
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(w) * h * 4, 0);
        for (std::size_t idx = 0; idx < pixels.size(); idx += 4) {
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = a;
        }
        return pixels;
    }

    [[nodiscard]] std::vector<std::uint8_t> make_glyph_pixels(const std::array<std::uint8_t, 8>& glyph)
    {
        constexpr std::uint32_t w = 8;
        constexpr std::uint32_t h = 8;
        std::vector<std::uint8_t> pixels(w * h * 4, 0);
        for (std::uint32_t row = 0; row < h; ++row) {
            const std::uint8_t bits = glyph[row];
            for (std::uint32_t col = 0; col < w; ++col) {
                const bool on = (bits >> col) & 0x1u;
                const std::size_t index = static_cast<std::size_t>(row) * w * 4 + col * 4;
                pixels[index + 0] = 255;
                pixels[index + 1] = 255;
                pixels[index + 2] = 255;
                pixels[index + 3] = on ? 255 : 0;
            }
        }
        return pixels;
    }

    [[nodiscard]] SpriteHandle add_sprite(TextureAtlas& atlas, const std::string& name, const std::vector<std::uint8_t>& pixels, std::uint32_t w, std::uint32_t h)
    {
        Texture texture{};
        texture.name = name;
        texture.width = w;
        texture.height = h;
        texture.channels = 4;
        texture.pixels = pixels;

        auto entry = atlas.add_entry(name, texture);
        if (!entry) {
            throw std::runtime_error("[agui] Failed to add atlas entry: " + name);
        }

        if (almondnamespace::spritepool::capacity == 0) {
            almondnamespace::spritepool::initialize(2048);
        }

        SpriteHandle handle = almondnamespace::spritepool::allocate();
        if (!handle.is_valid()) {
            throw std::runtime_error("[agui] Sprite pool exhausted while registering GUI sprite");
        }

        handle.atlasIndex = static_cast<std::uint32_t>(atlas.get_index());
        handle.localIndex = static_cast<std::uint32_t>(entry->index);

        almondnamespace::atlasmanager::registry.add(name, handle,
            entry->region.u1,
            entry->region.v1,
            entry->region.u2 - entry->region.u1,
            entry->region.v2 - entry->region.v1);

        return handle;
    }

    void ensure_resources()
    {
        std::scoped_lock lock(g_resourceMutex);
        if (g_resources.atlasBuilt)
            return;

        auto atlasIt = almondnamespace::atlasmanager::atlas_map.find(kAtlasName);
        if (atlasIt == almondnamespace::atlasmanager::atlas_map.end()) {
            almondnamespace::atlasmanager::create_atlas({
                .name = kAtlasName,
                .width = 512,
                .height = 512,
                .generate_mipmaps = false
            });
            atlasIt = almondnamespace::atlasmanager::atlas_map.find(kAtlasName);
            if (atlasIt == almondnamespace::atlasmanager::atlas_map.end()) {
                throw std::runtime_error("[agui] Unable to create GUI atlas");
            }
        }

        TextureAtlas& atlas = atlasIt->second;
        g_resources.atlas = &atlas;

        g_resources.windowBackground = add_sprite(atlas, "__agui/window_bg",
            make_solid_pixels(0x33, 0x35, 0x38, 0xFF, 8, 8), 8, 8);
        g_resources.buttonNormal = add_sprite(atlas, "__agui/button_normal",
            make_solid_pixels(0x5B, 0x5F, 0x66, 0xFF, 8, 8), 8, 8);
        g_resources.buttonHover = add_sprite(atlas, "__agui/button_hover",
            make_solid_pixels(0x76, 0x7C, 0x85, 0xFF, 8, 8), 8, 8);
        g_resources.buttonActive = add_sprite(atlas, "__agui/button_active",
            make_solid_pixels(0x94, 0x9A, 0xA3, 0xFF, 8, 8), 8, 8);

        for (std::size_t i = 0; i < kFont8x8Basic.size(); ++i) {
            const int codepoint = static_cast<int>(i) + 32;
            const std::string name = "__agui/glyph_" + std::to_string(codepoint);
            auto pixels = make_glyph_pixels(kFont8x8Basic[i]);
            g_resources.glyphs[i] = add_sprite(atlas, name, pixels, 8, 8);
        }

        g_resources.atlasBuilt = true;
    }

    void ensure_backend_upload(core::Context& ctx)
    {
        ensure_resources();
        if (!g_resources.atlas) return;

        std::scoped_lock lock(g_uploadMutex);
        if (g_uploadedContexts.contains(&ctx)) return;

        const std::uint32_t handle = ctx.add_atlas_safe(*g_resources.atlas);
        if (handle == 0) {
            throw std::runtime_error("[agui] Failed to upload GUI atlas for this context");
        }
        g_uploadedContexts.insert(&ctx);
    }

    void draw_sprite(const SpriteHandle& handle, float x, float y, float w, float h)
    {
        if (!handle.is_valid() || !g_frame.ctx)
            return;

        const auto& atlases = almondnamespace::atlasmanager::get_atlas_vector();
        g_frame.ctx->draw_sprite_safe(handle, atlases, x, y, w, h);
    }

    void draw_text_line(std::string_view text, float x, float y, float scale)
    {
        ensure_resources();
        if (!g_frame.ctx)
            return;

        const float glyphW = g_resources.glyphWidth * scale;
        const float glyphH = g_resources.glyphHeight * scale;
        const SpriteHandle fallback = g_resources.glyphs['?' - 32];

        for (char ch : text) {
            if (ch == '\n') {
                y += glyphH + kLineSpacing;
                x = g_frame.origin.x + kContentPadding;
                continue;
            }

            unsigned char uch = static_cast<unsigned char>(ch);
            SpriteHandle handle = fallback;
            if (uch >= 32 && uch < 128) {
                handle = g_resources.glyphs[uch - 32];
            }

            draw_sprite(handle, x, y, glyphW, glyphH);
            x += glyphW;
        }
    }

    void reset_frame()
    {
        g_frame.cursor = {};
        g_frame.origin = {};
        g_frame.windowSize = {};
        g_frame.insideWindow = false;
    }

} // namespace

void push_input(const almondnamespace::gui::InputEvent& e) noexcept
{
    g_frame.mousePos = e.mouse_pos;
    if (e.type == EventType::MouseDown) {
        g_frame.mouseDown = true;
    }
    else if (e.type == EventType::MouseUp) {
        g_frame.mouseDown = false;
    }
}

void begin_frame(core::Context& ctx, float, Vec2 mouse_pos, bool mouse_down) noexcept
{
    try {
        ensure_backend_upload(ctx);
    }
    catch (...) {
        // swallow: GUI is optional, avoid propagating exceptions across frames
    }

    g_frame.ctx = &ctx;
    g_frame.prevMouseDown = g_frame.mouseDown;
    g_frame.mouseDown = mouse_down;
    g_frame.justPressed = (!g_frame.prevMouseDown && g_frame.mouseDown);
    g_frame.mousePos = mouse_pos;
    reset_frame();
}

void end_frame() noexcept
{
    g_frame.ctx = nullptr;
    g_frame.insideWindow = false;
}

void begin_window(std::string_view title, Vec2 position, Vec2 size) noexcept
{
    if (!g_frame.ctx)
        return;

    ensure_resources();

    g_frame.origin = position;
    g_frame.windowSize = size;
    g_frame.cursor = { position.x + kContentPadding, position.y + kContentPadding };
    g_frame.insideWindow = true;

    draw_sprite(g_resources.windowBackground, position.x, position.y, size.x, size.y);

    const float titleHeight = g_resources.glyphHeight * kTitleScale;
    draw_text_line(title, position.x + kContentPadding, position.y + kContentPadding, kTitleScale);

    g_frame.cursor.y = position.y + kContentPadding + titleHeight + kContentPadding;
}

void end_window() noexcept
{
    g_frame.insideWindow = false;
}

bool button(std::string_view label, Vec2 size) noexcept
{
    if (!g_frame.insideWindow || !g_frame.ctx)
        return false;

    const Vec2 pos = g_frame.cursor;
    const float width = std::max<float>(static_cast<float>(size.x), g_resources.glyphWidth * kFontScale + 2.0f * kContentPadding);
    const float height = std::max<float>(static_cast<float>(size.y), g_resources.glyphHeight * kFontScale + 2.0f * kContentPadding);

    const bool hovered = (g_frame.mousePos.x >= pos.x && g_frame.mousePos.x <= pos.x + width &&
        g_frame.mousePos.y >= pos.y && g_frame.mousePos.y <= pos.y + height);

    const SpriteHandle background = hovered
        ? (g_frame.mouseDown ? g_resources.buttonActive : g_resources.buttonHover)
        : g_resources.buttonNormal;

    draw_sprite(background, pos.x, pos.y, width, height);

    const float textWidth = static_cast<float>(label.size()) * g_resources.glyphWidth * kFontScale;
    const float textHeight = g_resources.glyphHeight * kFontScale;
    const float textX = pos.x + std::max(0.0f, (width - textWidth) * 0.5f);
    const float textY = pos.y + std::max(0.0f, (height - textHeight) * 0.5f);
    draw_text_line(label, textX, textY, kFontScale);

    g_frame.cursor.y += height + kContentPadding;
    return hovered && g_frame.justPressed;
}

void label(std::string_view text) noexcept
{
    if (!g_frame.insideWindow || !g_frame.ctx)
        return;

    draw_text_line(text, g_frame.cursor.x, g_frame.cursor.y, kFontScale);
    g_frame.cursor.y += g_resources.glyphHeight * kFontScale + kLineSpacing;
}

float line_height() noexcept
{
    try {
        ensure_resources();
    }
    catch (...) {
        return 16.0f;
    }
    return g_resources.glyphHeight * kFontScale + kLineSpacing;
}

float glyph_width() noexcept
{
    try {
        ensure_resources();
    }
    catch (...) {
        return 8.0f * kFontScale;
    }
    return g_resources.glyphWidth * kFontScale;
}

} // namespace almondnamespace::gui
