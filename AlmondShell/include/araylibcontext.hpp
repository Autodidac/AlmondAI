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
 // araylibcontext.hpp
#pragma once

#include "aengineconfig.hpp"

#if defined(ALMOND_USING_RAYLIB)

#include "acontext.hpp"
#include "awindowdata.hpp"

#include "acontextwindow.hpp"
#include "arobusttime.hpp"
#include "aimageloader.hpp"
#include "araylibtextures.hpp" // AtlasGPU, gpu_atlases, ensure_uploaded
#include "araylibstate.hpp"    // brings in the actual inline s_raylibstate
#include "aatlasmanager.hpp"
#include "araylibrenderer.hpp" // RendererContext, RenderMode
#include "acommandline.hpp"    // cli::window_width, cli::window_height
#include "araylibcontextinput.hpp"

#if defined(_WIN32)
namespace almondnamespace::core { void MakeDockable(HWND hwnd, HWND parent); }
#endif

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

// Raylib last
#include <raylib.h>
#if !defined(RAYLIB_NO_WINDOW)
extern "C" void CloseWindow(void);
#endif

#if defined(_WIN32)
#include <windows.h>
#include <gl/gl.h>
#endif

namespace almondnamespace::raylibcontext
{
    // Usually at the start of your program, before window creation:
    static almondnamespace::contextwindow::WindowData* g_raylibwindowContext;

    inline GuiFitViewport compute_fit_viewport(int fbW, int fbH, int refW, int refH) noexcept
    {
        GuiFitViewport r{};
        r.fbW = std::max(1, fbW);
        r.fbH = std::max(1, fbH);
        r.refW = std::max(1, refW);
        r.refH = std::max(1, refH);

        const float sx = float(r.fbW) / float(r.refW);
        const float sy = float(r.fbH) / float(r.refH);
        r.scale = (std::max)(0.0001f, (std::min)(sx, sy)); // fit

        r.vpW = std::max(1, int(std::lround(r.refW * r.scale)));
        r.vpH = std::max(1, int(std::lround(r.refH * r.scale)));
        r.vpX = (r.fbW - r.vpW) / 2;
        r.vpY = (r.fbH - r.vpH) / 2;
        return r;
    }

    // --- Helper: apply size to Win32 child + raylib/GLFW (after docking) ---
    inline void apply_native_resize(int w, int h, bool updateNativeWindow, bool updateRaylibWindow)
    {
        const int W = std::max(1, w);
        const int H = std::max(1, h);

#if defined(_WIN32)
        if (updateNativeWindow && s_raylibstate.hwnd) {
            ::SetWindowPos(s_raylibstate.hwnd, nullptr, 0, 0, W, H,
                SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
        }
#endif

#if !defined(RAYLIB_NO_WINDOW)
        if (updateRaylibWindow && ::IsWindowReady()) {
            // Let raylib manage DPI internally; pass physical-ish intent.
            ::SetWindowSize(W, H);
        }
#endif
    }

    inline std::pair<int, int> framebuffer_size() noexcept
    {
#if !defined(RAYLIB_NO_WINDOW)
        if (!::IsWindowReady()) return { 1,1 };
        return { std::max(1, GetRenderWidth()), std::max(1, GetRenderHeight()) };
#else
        return { 1,1 };
#endif
    }

    inline void update_mouse_scale()
    {
#if !defined(RAYLIB_NO_WINDOW)
        if (!::IsWindowReady()) return;
        const auto [sx, sy] = framebuffer_scale();
        ::SetMouseScale(sx, sy);

        const Vector2 offset = ::GetRenderOffset();
        ::SetMouseOffset(static_cast<int>(std::lround(offset.x)),
            static_cast<int>(std::lround(offset.y)));
#endif
    }

    inline void dispatch_resize(const std::shared_ptr<core::Context>& ctx,
        unsigned int fbWidth,
        unsigned int fbHeight,
        bool updateRaylibWindow,
        bool notifyClient = true,
        bool skipNativeApply = false)
    {
        unsigned int nextFbW = fbWidth;
        unsigned int nextFbH = fbHeight;
        bool nextUpdateWindow = updateRaylibWindow;
        bool nextNotifyClient = notifyClient;
        bool nextSkipNativeApply = skipNativeApply;

        if (s_raylibstate.dispatchingResize)
        {
            s_raylibstate.pendingWidth = nextFbW;
            s_raylibstate.pendingHeight = nextFbH;
            s_raylibstate.pendingUpdateWindow = s_raylibstate.pendingUpdateWindow || nextUpdateWindow;
            s_raylibstate.pendingNotifyClient = s_raylibstate.pendingNotifyClient || nextNotifyClient;
            s_raylibstate.pendingSkipNativeApply = nextSkipNativeApply;
            s_raylibstate.hasPendingResize = true;
            return;
        }

        s_raylibstate.dispatchingResize = true;

        for (;;)
        {
            const unsigned int safeFbW = std::max(1u, nextFbW);
            const unsigned int safeFbH = std::max(1u, nextFbH);

            s_raylibstate.width = safeFbW;
            s_raylibstate.height = safeFbH;

            // Logical window size (for info only)
            unsigned int logicalW = safeFbW, logicalH = safeFbH;
#if !defined(RAYLIB_NO_WINDOW)
            if (::IsWindowReady()) {
                logicalW = std::max(1, ::GetScreenWidth());
                logicalH = std::max(1, ::GetScreenHeight());
            }
#endif
            s_raylibstate.logicalWidth = logicalW;
            s_raylibstate.logicalHeight = logicalH;

            // Decide the virtual canvas
            int refW = (core::cli::window_width > 0) ? core::cli::window_width : 1920;
            int refH = (core::cli::window_height > 0) ? core::cli::window_height : 1080;

            // Mirror VIRTUAL size into ctx for GUI/sprite math
            if (ctx) {
                ctx->width = refW;
                ctx->height = refH;
            }

            bool hasNativeParent = false;
#if defined(_WIN32)
            HWND observedParent = nullptr;
            if (s_raylibstate.hwnd) {
                observedParent = ::GetParent(s_raylibstate.hwnd);
                if (observedParent && observedParent != ::GetDesktopWindow()) {
                    hasNativeParent = true;
                }
                else {
                    observedParent = nullptr;
                }
                s_raylibstate.parent = observedParent;
            }
            else if (s_raylibstate.parent) {
                hasNativeParent = true;
            }
#endif

            const bool updateNativeWindow = !nextSkipNativeApply && hasNativeParent;
            const bool updateFramebuffer = !nextSkipNativeApply && (nextUpdateWindow || hasNativeParent);

            if (!nextSkipNativeApply) {
                apply_native_resize(static_cast<int>(safeFbW),
                    static_cast<int>(safeFbH),
                    updateNativeWindow,
                    updateFramebuffer);
            }

            if (nextNotifyClient && s_raylibstate.clientOnResize) {
                try {
                    s_raylibstate.clientOnResize(static_cast<int>(safeFbW),
                        static_cast<int>(safeFbH));
                }
                catch (const std::exception& e) {
                    std::cerr << "[Raylib] onResize client callback threw: " << e.what() << "\n";
                }
                catch (...) {
                    std::cerr << "[Raylib] onResize client callback threw unknown exception.\n";
                }
            }

            if (!s_raylibstate.hasPendingResize) break;

            // Drain pending coalesced resize
            nextFbW = s_raylibstate.pendingWidth;
            nextFbH = s_raylibstate.pendingHeight;
            nextUpdateWindow = s_raylibstate.pendingUpdateWindow;
            nextNotifyClient = s_raylibstate.pendingNotifyClient;
            nextSkipNativeApply = s_raylibstate.pendingSkipNativeApply;

            s_raylibstate.hasPendingResize = false;
            s_raylibstate.pendingUpdateWindow = false;
            s_raylibstate.pendingNotifyClient = false;
            s_raylibstate.pendingSkipNativeApply = false;
        }

        s_raylibstate.dispatchingResize = false;
    }

    inline void sync_framebuffer_size(const std::shared_ptr<core::Context>& ctx,
        bool notifyClient)
    {
#if !defined(RAYLIB_NO_WINDOW)
        if (!::IsWindowReady()) return;

        const int rw = std::max(1, ::GetRenderWidth());
        const int rh = std::max(1, ::GetRenderHeight());

        dispatch_resize(ctx, (unsigned)rw, (unsigned)rh,
            /*updateRaylibWindow=*/false,
            notifyClient,
            /*skipNativeApply=*/true);
#endif
    }

    // ──────────────────────────────────────────────
    // Initialize Raylib window and context
    // ──────────────────────────────────────────────
    inline bool raylib_initialize(std::shared_ptr<core::Context> ctx,
        HWND parentWnd = nullptr,
        unsigned int w = 800,
        unsigned int h = 600,
        std::function<void(int, int)> onResize = nullptr,
        std::string windowTitle = {})
    {
        const unsigned int clampedWidth = std::max(1u, w);
        const unsigned int clampedHeight = std::max(1u, h);

        s_raylibstate.clientOnResize = std::move(onResize);
        s_raylibstate.parent = parentWnd;
        s_raylibstate.dispatchingResize = false;
        s_raylibstate.hasPendingResize = false;
        s_raylibstate.pendingUpdateWindow = false;
        s_raylibstate.pendingNotifyClient = false;
        s_raylibstate.pendingSkipNativeApply = false;
        s_raylibstate.pendingWidth = 0;
        s_raylibstate.pendingHeight = 0;

        std::weak_ptr<core::Context> ctxWeak = ctx;
        s_raylibstate.onResize = [ctxWeak](int fbW, int fbH)
            {
                const int safeW = std::max(1, fbW);
                const int safeH = std::max(1, fbH);
                if (auto locked = ctxWeak.lock()) {
                    dispatch_resize(locked, (unsigned)safeW, (unsigned)safeH,
                        /*updateRaylibWindow=*/false,
                        /*notifyClient=*/false);
                }
            };
        if (ctx) ctx->onResize = s_raylibstate.onResize;

        // Seed sizes; treat given w/h as framebuffer pixels to start
        dispatch_resize(ctx, clampedWidth, clampedHeight, /*updateRaylibWindow=*/false, /*notifyClient=*/false);

        static bool initialized = false;
        if (initialized) return true;
        initialized = true;

        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIGHDPI);
        if (windowTitle.empty()) windowTitle = "Raylib Window";

        InitWindow((int)s_raylibstate.width, (int)s_raylibstate.height, windowTitle.c_str());
        SetWindowTitle(windowTitle.c_str());

        // First sync to actual framebuffer + logical sizes
        sync_framebuffer_size(ctx, /*notifyClient=*/false);

#if defined(_WIN32)
        s_raylibstate.hwnd = (HWND)GetWindowHandle();
        s_raylibstate.hdc = GetDC(s_raylibstate.hwnd);
        s_raylibstate.glContext = wglGetCurrentContext();
#endif

        if (ctx) {
            ctx->hwnd = s_raylibstate.hwnd;
            ctx->hdc = s_raylibstate.hdc;
            ctx->hglrc = s_raylibstate.glContext;
            // ctx->width/height already set to virtual in dispatch_resize
        }

        // If docking into a parent, reparent + single hard resize pass
        if (s_raylibstate.parent) {
            ::SetParent(s_raylibstate.hwnd, s_raylibstate.parent);
            ::ShowWindow(s_raylibstate.hwnd, SW_SHOW);

            LONG_PTR style = ::GetWindowLongPtr(s_raylibstate.hwnd, GWL_STYLE);
            style &= ~WS_OVERLAPPEDWINDOW;
            style |= (WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN);
            ::SetWindowLongPtr(s_raylibstate.hwnd, GWL_STYLE, style);

            LONG_PTR ex = ::GetWindowLongPtr(s_raylibstate.hwnd, GWL_EXSTYLE);
            ex &= ~WS_EX_APPWINDOW;
            ::SetWindowLongPtr(s_raylibstate.hwnd, GWL_EXSTYLE, ex);

#if defined(_WIN32)
            almondnamespace::core::MakeDockable(s_raylibstate.hwnd, s_raylibstate.parent);
#endif

            if (!windowTitle.empty()) {
                const std::wstring wideTitle(windowTitle.begin(), windowTitle.end());
                ::SetWindowTextW(s_raylibstate.hwnd, wideTitle.c_str());
            }

            RECT client{};
            ::GetClientRect(s_raylibstate.parent, &client);
            const int pw = std::max<LONG>(1, client.right - client.left);
            const int ph = std::max<LONG>(1, client.bottom - client.top);

            apply_native_resize(pw, ph, /*native*/true, /*raylib*/true);

            // Sync from real framebuffer
            sync_framebuffer_size(ctx, /*notifyClient=*/true);
        }

        s_raylibstate.running = true;

        // Hook atlas uploads
        atlasmanager::register_backend_uploader(core::ContextType::RayLib,
            [](const TextureAtlas& atlas) {
                raylibtextures::ensure_uploaded(atlas);
            });

        return true;
    }

    inline void Raylib_CloseWindow() { /* no-op placeholder */ }

    // ──────────────────────────────────────────────
    // Per-frame event processing
    // ──────────────────────────────────────────────
    inline bool raylib_process(std::shared_ptr<core::Context> ctx, core::CommandQueue& queue)
    {
        if (!s_raylibstate.running || WindowShouldClose()) {
            s_raylibstate.running = false;
            return true;
        }

        atlasmanager::process_pending_uploads(core::ContextType::RayLib);

        // Observe current framebuffer size
        const unsigned int prevFbW = s_raylibstate.width;
        const unsigned int prevFbH = s_raylibstate.height;

        const int rw = std::max(1, GetRenderWidth());
        const int rh = std::max(1, GetRenderHeight());

        if ((unsigned)rw != prevFbW || (unsigned)rh != prevFbH) {
            dispatch_resize(ctx, (unsigned)rw, (unsigned)rh,
                /*updateRaylibWindow=*/false,
                /*notifyClient=*/true,
                /*skipNativeApply=*/true);
        }

#if defined(_WIN32)
        if (!wglMakeCurrent(s_raylibstate.hdc, s_raylibstate.glContext)) {
            s_raylibstate.running = false;
            std::cerr << "[Raylib] Failed to make Raylib GL context current\n";
            return true;
        }
#endif

        // ----- VIRTUAL FIT VIEWPORT + MOUSE MAPPING -----
        const int fbW = std::max(1, GetRenderWidth());
        const int fbH = std::max(1, GetRenderHeight());
        const int refW = (core::cli::window_width > 0) ? core::cli::window_width : 1920;
        const int refH = (core::cli::window_height > 0) ? core::cli::window_height : 1080;

        // Ensure ctx continues to see VIRTUAL size
        if (ctx) { ctx->width = refW; ctx->height = refH; }

        const GuiFitViewport fit = compute_fit_viewport(fbW, fbH, refW, refH);

        // Viewport for rendering (letterbox/pillarbox)
        s_raylibstate.lastViewport = fit;

        // Mouse: map framebuffer -> virtual
#if !defined(RAYLIB_NO_WINDOW)
        ::SetMouseOffset(-fit.vpX, -fit.vpY);
        const float invScale = (fit.scale > 0.0f) ? (1.0f / fit.scale) : 1.0f;
        ::SetMouseScale(invScale, invScale);
#endif
        // -----------------------------------------------

        // Animated bg (helps verify viewport)
        static auto* bgTimer = almondnamespace::time::getTimer("menu", "bg_color");
        if (!bgTimer) bgTimer = &almondnamespace::time::createNamedTimer("menu", "bg_color");
        const double t = almondnamespace::time::elapsed(*bgTimer);

        const unsigned char r = static_cast<unsigned char>((0.5 + 0.5 * std::sin(t * 1.0)) * 255);
        const unsigned char g = static_cast<unsigned char>((0.5 + 0.5 * std::sin(t * 0.7 + 2.0)) * 255);
        const unsigned char b = static_cast<unsigned char>((0.5 + 0.5 * std::sin(t * 1.3 + 4.0)) * 255);

        BeginDrawing();
        ClearBackground(Color{ r, g, b, 255 });

        // NOTE: Your draw_sprite path uses ctx->width/height → now virtual.
        // That guarantees consistent atlas button sizing/placement.
        queue.drain();

        EndDrawing();

        return true; // continue running
    }

    inline void raylib_clear()
    {
        BeginDrawing();
        ClearBackground(DARKPURPLE);
        EndDrawing();
    }

    inline void raylib_present() { /* EndDrawing() presents already */ }

    // ──────────────────────────────────────────────
    // Cleanup and shutdown
    // ──────────────────────────────────────────────
    inline void raylib_cleanup(std::shared_ptr<almondnamespace::core::Context>& ctx)
    {
        if (!s_raylibstate.running) return;

#if defined(_WIN32)
        if (s_raylibstate.hdc && s_raylibstate.glContext) {
            wglMakeCurrent(nullptr, nullptr);
        }
#endif

#if !defined(RAYLIB_NO_WINDOW)
        if (IsWindowReady()) {
            CloseWindow();
        }
#endif

#if defined(_WIN32)
        if (s_raylibstate.hwnd && ::IsWindow(s_raylibstate.hwnd)) {
            if (s_raylibstate.hdc) {
                ::ReleaseDC(s_raylibstate.hwnd, s_raylibstate.hdc);
                s_raylibstate.hdc = nullptr;
            }
            ::DestroyWindow(s_raylibstate.hwnd);
        }
        s_raylibstate.hwnd = nullptr;
        s_raylibstate.glContext = nullptr;
#endif

        s_raylibstate.running = false;

        if (ctx) {
            ctx->hwnd = nullptr;
            ctx->hdc = nullptr;
            ctx->hglrc = nullptr;
        }
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────
    inline int  raylib_get_width()  noexcept { return static_cast<int>(s_raylibstate.logicalWidth); }
    inline int  raylib_get_height() noexcept { return static_cast<int>(s_raylibstate.logicalHeight); }
    inline void raylib_set_window_title(const std::string& title) { SetWindowTitle(title.c_str()); }
    inline bool RaylibIsRunning(std::shared_ptr<core::Context>) { return s_raylibstate.running; }

} // namespace almondnamespace::raylibcontext

#endif // ALMOND_USING_RAYLIB
