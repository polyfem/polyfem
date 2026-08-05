cmake_minimum_required(VERSION 3.18)
project(hocgv_miso)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes")
include(eigen)

# The generated/ CMakeLists.txt creates the 'miso' target, fetches misolib,
# and adds all generated sources via target_sources(miso PRIVATE ...).
add_subdirectory(generated)

# Expose generated headers and the EigenAdapters.hpp in the repo root to consumers.
target_include_directories(miso PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_subdirectory(tests)
