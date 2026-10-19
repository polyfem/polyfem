# miso: miso-generated Jacobian validity solvers, vendored from
# https://github.com/fsichetti/hocgv-miso (see generate.py / README.md to regenerate).
# Depends on misolib (cmake/recipes/misolib.cmake), which is fetched separately.

include(misolib)

add_library(miso)
add_library(miso::miso ALIAS miso)
target_compile_features(miso PRIVATE cxx_std_23)
target_link_libraries(miso PRIVATE misolib::misolib)

# Forward misolib's include dir (not its cxx_std_23 requirement) so quoted includes
# like "core/RealVector.hpp" resolve for consumers of miso.
target_include_directories(miso SYSTEM PUBLIC $<TARGET_PROPERTY:misolib::misolib,INTERFACE_INCLUDE_DIRECTORIES>)

# Expose generated headers and EigenAdapters.hpp to consumers. SYSTEM to keep
# machine-generated code exempt from polyfem::warnings' -Werror=...
target_include_directories(miso SYSTEM PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
)

# The generated/ CMakeLists.txt adds all generated sources via target_sources(miso PRIVATE ...).
add_subdirectory(generated)

target_link_libraries(polyfem PRIVATE miso)
target_compile_definitions(polyfem PUBLIC POLYFEM_WITH_MISO)

# Sources outside miso that pass misolib's intervals to it must be compiled
# with misolib's own compile options and definitions, which miso keeps private.
# The interval type (tight, built on NFG's interval_number) changes its memory
# layout with the SIMD macros: one __m128d holding (high, -low) when __SSE2__,
# __AVX2__ or NEON is defined, two doubles in the opposite order otherwise.
# GCC and Clang on x86-64 always define __SSE2__ and arm64 always has NEON,
# but MSVC defines none of them without /arch, and tight sets /arch:AVX2 (and
# /fp:strict, which interval arithmetic in a directed rounding mode needs
# anyway) only on the targets that link it. Without this, those sources and
# miso read each other's bounds swapped.
function(polyfem_use_miso_intervals target)
	set_source_files_properties(${ARGN} TARGET_DIRECTORY ${target} PROPERTIES
		COMPILE_OPTIONS "$<TARGET_PROPERTY:misolib::misolib,INTERFACE_COMPILE_OPTIONS>"
		COMPILE_DEFINITIONS "$<TARGET_PROPERTY:misolib::misolib,INTERFACE_COMPILE_DEFINITIONS>")
endfunction()
polyfem_use_miso_intervals(polyfem "${PROJECT_SOURCE_DIR}/src/polyfem/utils/Jacobian.cpp")

# Some generated functions (P3TetCGV's subdivision and conversion routines) use
# several hundred KB of stack each, and the search nests them. MSVC's default
# 1 MB main-thread stack overflows; Linux and macOS default to 8 MB. Give every
# executable linking polyfem the same 8 MB on Windows.
if(MSVC)
	target_link_options(polyfem INTERFACE /STACK:8388608)
endif()
