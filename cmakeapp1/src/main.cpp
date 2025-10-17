// main.cpp
#include "../include/utils.hpp"

int main()
{
    // Print messages using {fmt} formatting.
    safe_io::print("Production code: {}", "Hello, world!");
    safe_io::print("This output is buffered for {}.", "efficiency");

    // Explicit flushing can be done if required.
    // safe_io::out().flush();

#ifndef _WIN32
    return 0; // Explicit return for non-Windows platforms.
#endif
}
