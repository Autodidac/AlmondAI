
# Set the vcpkg toolchain file
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg_installed/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

# Add executable
add_executable(Application1
    src/Application1.cpp
)

# Include headers and link to library
target_include_directories(Application1 PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${VCPKG_INSTALLATION_ROOT}/x64-linux/include
    ${VCPKG_INSTALLATION_ROOT}/x64-linux/lib
    ${VCPKG_INSTALLATION_ROOT}/x64-windows/include
    ${VCPKG_INSTALLATION_ROOT}/x64-windows/lib
)

# Specify include directories for Application1
target_include_directories(Application1 PRIVATE 
    ${CMAKE_SOURCE_DIR}../../StaticLib1
    ${CMAKE_SOURCE_DIR}../../StaticLib1/include
)

# Link StaticLib1
target_link_libraries(Application1 PRIVATE
    StaticLib1
    fmt
)

target_link_directories(Application1 PRIVATE
    ${VCPKG_INSTALLATION_ROOT}/x64-linux/lib
)

# If you're targeting X11, make sure that the X11-specific source is added conditionally
if(UNIX AND NOT APPLE)  # Assuming this targets Linux/X11
   # target_sources(Static PRIVATE ../../src/Linux.cpp)
endif()

# Additional linker options for GNU/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
   # target_link_options(StaticLib1 PRIVATE -Wl,--allow-shlib-undefined)
endif()

# Installation rule for the executable
install(TARGETS Application1
    RUNTIME DESTINATION bin
)
