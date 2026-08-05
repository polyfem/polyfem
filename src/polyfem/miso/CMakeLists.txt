# miso: miso-generated Jacobian validity solvers, vendored from
# https://github.com/fsichetti/hocgv-miso (see generate.py / README.md to regenerate).
# Depends on misolib (cmake/recipes/misolib.cmake), which is fetched separately.

include(misolib)

add_library(miso)
add_library(miso::miso ALIAS miso)
target_compile_features(miso PRIVATE cxx_std_23)
target_link_libraries(miso PRIVATE misolib::misolib)

# Expose generated headers and EigenAdapters.hpp to consumers (e.g. polyfem/utils/Jacobian.cpp).
target_include_directories(miso PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
)

# The generated/ CMakeLists.txt adds all generated sources via target_sources(miso PRIVATE ...).
add_subdirectory(generated)

target_link_libraries(polyfem PRIVATE miso)
target_compile_definitions(polyfem PUBLIC POLYFEM_WITH_MISO)
