// main.cpp
#include "../include/utils.hpp"
#include <iostream>
#include <limits>
#include <string>

int main()
{
    // Print messages using {fmt} formatting.
    safe_io::print("Production code: {}", "Hello, world!");
    safe_io::print("This output is buffered for {}.", "efficiency");

    // Explicit flushing can be done if required.
    // safe_io::out().flush();

#ifdef _WIN32
    // Wait for console input to keep the window open on Windows.
    safe_io::print("Press Enter to exit...");
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
#else
    return 0; // Explicit return for non-Windows platforms.
#endif
}
