# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
#cmake_minimum_required(VERSION 3.3[...<3.8>] [FATAL_ERROR])

project(MyCrossPlatformProject VERSION 1.0
                  DESCRIPTION "My Cross Platform Project Description"
                  LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# (Optional) Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Enforce install location at top-level (in the project root)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/built" CACHE PATH "Install path prefix" FORCE)
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")

# Global output directories
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/lib/Debug")
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/Release")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG     "${CMAKE_BINARY_DIR}/lib/Debug")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE   "${CMAKE_BINARY_DIR}/lib/Release")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG     "${CMAKE_BINARY_DIR}/bin/Debug")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE   "${CMAKE_BINARY_DIR}/bin/Release")

# Configure vcpkg installation root
if(DEFINED ENV{VCPKG_INSTALLATION_ROOT})
    set(VCPKG_INSTALLATION_ROOT "$ENV{VCPKG_INSTALLATION_ROOT}/vcpkg_installed")
else()
    set(VCPKG_INSTALLATION_ROOT "${CMAKE_SOURCE_DIR}/vcpkg_installed")
endif()

# Include sub-projects
option(ALMONDAI_BUILD_LEGACY "Build legacy sample targets" OFF)
if (ALMONDAI_BUILD_LEGACY)
    add_subdirectory("cmakeapp1")
    add_subdirectory("app1")
    add_subdirectory ("Application1")
    add_subdirectory ("StaticLib1")
endif()

set(ALMONDAI_SOURCES
    src/tensor.cpp
    src/adapter.cpp
    src/tokenizer_word.cpp
    src/retrieval.cpp
    src/governor.cpp
    src/ingest.cpp
    src/train.cpp
    src/model.cpp
    src/eval.cpp
    src/mcp.cpp
    src/serve.cpp
    src/buildparse.cpp
)

add_library(almondai STATIC ${ALMONDAI_SOURCES})
target_include_directories(almondai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(almondai PUBLIC cxx_std_20)

add_executable(almondai_console src/main.cpp)
target_link_libraries(almondai_console PRIVATE almondai)

set_target_properties(almondai PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
set_target_properties(almondai_console PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)















