
# Set the vcpkg toolchain file
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg_installed/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

# Create executable for vulkanapp1
add_executable(app1
    src/main.cpp
    src/utils.cpp
    include/utils.hpp
)

find_package(fmt CONFIG REQUIRED)

# Include headers and link to library
target_include_directories(app1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${VCPKG_INSTALLATION_ROOT}/x64-linux/include
)

target_link_directories(app1 PRIVATE
    ${VCPKG_INSTALLATION_ROOT}/x64-linux/lib
)

target_link_libraries(app1 PRIVATE
    fmt
)

# Installation rule
install(TARGETS app1
    RUNTIME DESTINATION bin
)

# Fix missing symbols for GNU/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(app1 PRIVATE -Wl,--allow-shlib-undefined)
endif()
