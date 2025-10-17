
# Set the vcpkg toolchain file
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "../${CMAKE_SOURCE_DIR}/vcpkg_installed/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

# Create the static library target
add_library(StaticLib1 STATIC)

# Specify include directories for StaticLib1
target_include_directories(StaticLib1
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${VCPKG_INSTALLATION_ROOT}/x64-linux/include
        target_precompile_headers(StaticLib1 PRIVATE "../pch.h")
)

# Add all the source files for the static library
target_sources(StaticLib1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/StaticLib1.cpp
   # ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)

# Find and link dependencies
find_package(fmt CONFIG QUIET)
find_package(Threads QUIET)
# Link fmt and threads


if(fmt_FOUND)
    message(STATUS "Using fmt.")

    target_link_libraries(StaticLib1 PRIVATE fmt::fmt Threads::Threads)
else()
    message(FATAL_ERROR "Required packages (fmt) not found. Please install them via your package manager or vcpkg.")
endif()


# Compiler-specific options for GNU/Clang/MSVC
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(StaticLib1 PRIVATE -Wall -Wextra -Wpedantic -std=c++20)
elseif(MSVC)
    target_compile_options(StaticLib1 PRIVATE /W4 /std:c++20)
endif()

# Installation rules for StaticLib1
install(TARGETS StaticLib1
    ARCHIVE DESTINATION lib    # For static libraries on Unix
    LIBRARY DESTINATION lib    # For shared libraries (if any)
    RUNTIME DESTINATION bin    # For executables (if any)
)

# Install public headers from the "include" directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
