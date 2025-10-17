// main.cpp

#include "utils.hpp"  // This brings in safe_io::force_link_out() and other functions.

#include <iostream>
#include <limits>
#include <string>

int main()
{
    // Force the static library object to be linked.
    // Use the function from the safe_io namespace—not a separately declared one.
    safe_io::force_link_out();

    // Now you can use the functions from the library.
    safe_io::print("Production code: {}", "Hello, world!");
    safe_io::print("This output is buffered for {}.", "efficiency");

#ifdef _WIN32
    // Wait for console input to keep the window open on Windows.
    safe_io::print("Press Enter to exit...");
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
#else
    return 0;
#endif
}
