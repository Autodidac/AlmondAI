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
// agui.hpp
#pragma once

#include "aplatform.hpp"
#include "aengineconfig.hpp"

#include <cstdint>
#include <string_view>

namespace almondnamespace::core {
    struct Context;
}

namespace almondnamespace::gui
{
    struct Vec2 {
        float x = 0.0f;
        float y = 0.0f;

        constexpr Vec2() noexcept = default;
        constexpr Vec2(float x_, float y_) noexcept : x(x_), y(y_) {}

        [[nodiscard]] constexpr Vec2 operator+(const Vec2& other) const noexcept
        {
            return Vec2{x + other.x, y + other.y};
        }

        [[nodiscard]] constexpr Vec2& operator+=(const Vec2& other) noexcept
        {
            x += other.x;
            y += other.y;
            return *this;
        }
    };

    struct Color {
        float r = 1.0f;
        float g = 1.0f;
        float b = 1.0f;
        float a = 1.0f;
    };

    enum class EventType {
        None,
        MouseDown,
        MouseUp,
        MouseMove
    };

    struct InputEvent {
        EventType type = EventType::None;
        Vec2 mouse_pos{};
        std::uint8_t mouse_button = 0;
    };

    void push_input(const InputEvent& e) noexcept;

    void begin_frame(core::Context& ctx, float dt, Vec2 mouse_pos, bool mouse_down) noexcept;
    void end_frame() noexcept;

    void begin_window(std::string_view title, Vec2 position, Vec2 size) noexcept;
    void end_window() noexcept;

    [[nodiscard]] bool button(std::string_view label, Vec2 size) noexcept;
    void label(std::string_view text) noexcept;

    [[nodiscard]] float line_height() noexcept;
    [[nodiscard]] float glyph_width() noexcept;

} // namespace almondnamespace::gui

namespace almondnamespace::ui
{
    using vec2 = gui::Vec2;
    using color = gui::Color;
    using event_type = gui::EventType;
    using input_event = gui::InputEvent;

    using gui::begin_frame;
    using gui::end_frame;
    using gui::begin_window;
    using gui::end_window;
    using gui::button;
    using gui::label;
    using gui::push_input;
    using gui::line_height;
    using gui::glyph_width;
}
