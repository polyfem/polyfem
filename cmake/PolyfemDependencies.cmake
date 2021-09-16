# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.

### Configuration
set(POLYFEM_ROOT     "${CMAKE_CURRENT_LIST_DIR}/..")
set(POLYFEM_EXTERNAL ${THIRD_PARTY_DIR})

# Download and update 3rdparty libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(PolyfemDownloadExternal)

################################################################################
# Required libraries
################################################################################

# Sanitizers
if(POLYFEM_WITH_SANITIZERS)
    polyfem_download_sanitizers()
    find_package(Sanitizers)
endif()

# spdlog
if(NOT TARGET spdlog::spdlog)
    polyfem_download_spdlog()
    add_subdirectory(${POLYFEM_EXTERNAL}/spdlog)
endif()

# CL11
if(${POLYFEM_TOPLEVEL_PROJECT})
    if(NOT TARGET CLI11::CLI11)
        polyfem_download_cli11()
        add_subdirectory(${POLYFEM_EXTERNAL}/cli11)
    endif()
endif()

# Clipper
if(POLYFEM_WITH_CLIPPER AND NOT TARGET clipper::clipper)
    polyfem_download_clipper()
    add_library(clipper_clipper ${POLYFEM_EXTERNAL}/clipper/cpp/clipper.cpp)
    add_library(clipper::clipper ALIAS clipper_clipper)
    target_include_directories(clipper_clipper PUBLIC ${POLYFEM_EXTERNAL}/clipper/cpp)
endif()

# GHC Filesystem
if(NOT TARGET ghc::filesystem)
    polyfem_download_filesystem()
    add_subdirectory(${POLYFEM_EXTERNAL}/filesystem)
    add_library(ghc::filesystem ALIAS ghc_filesystem)
endif()
