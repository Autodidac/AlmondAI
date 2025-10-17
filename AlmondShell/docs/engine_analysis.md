# AlmondShell Engine Analysis

## Architecture Highlights
- **Configurable Runtime Flags** – `include/aengineconfig.hpp` centralises renderer and context toggles (SDL, Raylib, OpenGL, software renderer) alongside headless/WinMain entry-point switches, making it the primary control surface for build-time composition.
- **Versioned Core** – `include/aversion.hpp` exposes inline getters for the semantic version components, enabling compile-time inspection of the running revision.
- **Context Management** – `src/acontext.cpp` and `include/acontextmultiplexer.hpp` orchestrate the per-backend render contexts and their window lifecycles, while `include/acontextwindow.hpp` encapsulates platform window data.
- **Hot-Reload Pipeline** – `src/ascriptingsystem.cpp` drives compilation and loading of `.ascript.cpp` files via the task graph scheduler, handing control to `ScriptScheduler` nodes created in `include/ascriptingsystem.hpp`.

## Observed Gaps
- **Fragile Script Loading** – Before this update, missing script sources or compilation artefacts would silently cascade into load failures. The scripting system now validates inputs and outputs before attempting to load DLLs.
- **Roadmap Traceability** – The existing `roadmap.txt` lacked granular prompts or acceptance checks per phase, making automation hand-offs hard to script.
- **Testing Surface** – No automated smoke tests or CI hooks are defined for the critical updater and renderer paths, leaving regression risk high during phase transitions.

## Recent Progress (v0.61.0)
- Raylib caches its virtual-fit viewport each frame and reprojects atlas sprites through that transform, eliminating the clipping and drifting that previously affected menu buttons in letterboxed windows.
- SDL3 keeps its context dimensions in lockstep with the renderer output size so GUI layouts always see the correct canvas and remain visually centered after resizes.
- The menu overlay now honours a configurable column cap (defaulting to four via `--menu-columns`), preventing overly wide grids while still allowing narrower layouts for constrained displays.

## Recent Progress (v0.60.0)
- Raylib now mirrors its internal viewport offset when drawing atlas-backed GUI sprites and when scaling mouse input, so letterboxed windows keep buttons aligned with their visual positions.
- SDL3 rendering queries the active viewport before mapping normalized GUI coordinates, ensuring atlas-driven overlays remain centered when logical presentation introduces padding.

## Recent Progress (v0.59.10)
- Introduced `agui.hpp` as a backend-agnostic immediate mode GUI layer that seeds a shared atlas and drives Raylib/SDL draw calls through the existing sprite pipeline, giving every context text and button primitives out of the box.

## Recent Progress (v0.59.9)
- Stabilised Raylib UI scaling across docked and standalone windows by preserving logical dimensions alongside framebuffer sizes, so atlas-driven GUI buttons keep their intended proportions after DPI-aware resizes in the Raylib backend.

## Recent Progress (v0.59.8)
- Adopted backend-owned Raylib and SDL window handles inside the multiplexer, replacing the temporary shell HWNDs with the real
  render surfaces so docking keeps working without spawning blank white host windows.
- Normalised Raylib GUI scaling by driving sprite metrics from the framebuffer-to-context ratio, keeping menu buttons sized with
  the window instead of shrinking as the viewport grows.

## Recent Progress (v0.59.3)
- Cached narrow window titles alongside HWND handles in the multiplexer so SDL and Raylib initialisers can read consistent UTF-8
  names when contexts spawn, fixing the Windows build regression caused by the loop-scoped `narrowTitle` identifier.

## Recent Progress (v0.59.1)
- The Raylib backend now coalesces OS-driven resize events and forwards them through guarded callbacks, keeping context dimensions and client hooks in sync with the window manager.

## Recent Progress (v0.59.0)
- The Win32-specific multi-context manager is now compiled only on Windows, and a portable stub keeps non-Windows builds linking while the docked UI remains a platform-specific feature.
- `awindowdata.hpp` no longer drags in `<windows.h>` on every platform, replacing the raw handle types with lightweight aliases so headless tools and POSIX builds stop failing during preprocessing.

## Recent Progress (v0.58.3)
- `RunEngine()` now launches the same multi-context render loop that powers `wWinMain`, letting non-Windows entry points exercise
  the full renderer/task orchestration stack without stubbing out the runtime.
- The Win32 bootstrap delegates to a shared helper so there is a single source of truth for context startup, menu routing, and
  scene lifecycle management.

## Recent Progress (v0.58.2)
- Version helpers now use thread-local buffers and expose string views, making it safe to query build metadata from concurrent tools.
- Updater configuration derives its semantic version directly from the version helpers, eliminating manual sync errors.
- Added `docs/aengineconfig_flags.md` to catalogue supported engine configuration switches for multi-backend builds.

## Recent Progress (v0.58.0)
- Added `ScriptLoadReport` to expose per-stage reload diagnostics, capture failure reasons, and emit success flags for automation to consume.
- Task graph workers now destroy coroutine frames after execution and prune completed nodes, preventing latent reload handles from accumulating between runs.
- Script reload orchestration awaits completion synchronously, guaranteeing that editors and tooling observe a consistent state before proceeding.

## Recommended Focus Areas
1. **Viewport Regression Harness** – Capture golden images for Raylib, SDL3, and software renderers to lock in the new fit-viewport logic and catch drift before it reaches production builds.
2. **Shared Layout Utilities** – Centralise the letterbox-aware size helpers so every backend (and future UI code) consumes the same conversion logic instead of duplicating clamps in multiple files.
3. **Menu Configuration UX** – Surface the new `--menu-columns` knob inside in-engine settings so column limits can be adjusted without restarting, then validate keyboard/gamepad navigation against dynamic changes.
4. **Cross-Context Resize Telemetry** – Instrument resize callbacks to log logical vs framebuffer dimensions, making it easier to diagnose when a backend falls out of sync with the multiplexer’s expectations.

## Context Cleanup Watchlist
- See `docs/context_audit.md` for a full census of "context" modules.  The audit tags `include/araylibcontext_win32.hpp` and `include/avulkanglfwcontext.hpp` as safe deletion candidates once external dependencies are ruled out, while `include/acontextrenderer.hpp` remains a "maybe" pending deeper archaeology.

## Next-Step Prompt Template
> "You are advancing AlmondShell through Phase {N} of the roadmap. Inspect the outstanding tasks for this phase, create a mini plan listing concrete PR-sized steps, implement them, and return a report with code diffs, tests, and docs updated."

Use this template alongside the refreshed roadmap to keep future assistant runs aligned with the delivery cadence.
