// utils.hpp
#pragma once

#include <iostream>
#include <fmt/core.h>

// When building as a static (header‑only) library, define ULTIMATECPP_STATIC in your project settings
#ifdef ULTIMATECPP_STATIC
    // Header‑only or static library: no import/export attributes required.
#define ULTIMATECPP_API
#else
#ifdef ULTIMATECPP_EXPORTS
    // Building the DLL.
#define ULTIMATECPP_API __declspec(dllexport)
#else
    // Consuming the DLL.
    // NOTE: For inline functions, we remove __declspec(dllimport) to avoid linker errors.
#define ULTIMATECPP_API
#endif
#endif

namespace safe_io {

    // Returns the output stream (e.g., std::cout).
    ULTIMATECPP_API inline std::ostream& out() noexcept {
        return std::cout;
    }

    // Dummy function to force linking of this translation unit.
    ULTIMATECPP_API inline void force_link_out() noexcept {
        (void)out();
    }

    // Templated print function using fmt for formatting.
    template <typename... Args>
    inline void print(fmt::format_string<Args...> fmt_str, Args&&... args) {
        out() << fmt::vformat(fmt_str, fmt::make_format_args(args...)) << '\n';
    }
}
