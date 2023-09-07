# Clipper (https://github.com/LLNL/PolyClipper)
# License: BSL1.0

if(TARGET PolyClipper::PolyClipper)
    return()
endif()

message(STATUS "Third-party: creating target 'PolyClipper::PolyClipper'")

option(ENABLE_STATIC_CXXONLY "enable C++ only build with static libraries" OFF)
option(ENABLE_CXXONLY "enable C++ only build without python bindings" ON)

include(CPM)
CPMAddPackage(
    NAME polyclipper
    GITHUB_REPOSITORY LLNL/PolyClipper
    GIT_TAG 2a727ebce1a5c500c0d3e23a6f22e2ecf941c6d5
    DOWNLOAD_ONLY TRUE
)

# PoltClipper's CMake is a bit of a mess, so we have to do this manually
set(POLYCLIPPER_INCLUDE_DIRS "${polyclipper_SOURCE_DIR}/src")

add_library(PolyClipper INTERFACE)
add_library(PolyClipper::PolyClipper ALIAS PolyClipper)

target_include_directories(PolyClipper SYSTEM INTERFACE
    $<BUILD_INTERFACE:${POLYCLIPPER_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)