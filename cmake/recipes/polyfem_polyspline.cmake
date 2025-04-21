# data (https://github.com/polyfem/polyfem-data)
# License: MIT

if(TARGET polyfem::polyspline)
    return()
endif()

include(ExternalProject)

set(POLYFEM_POLYSPLINE_DIR "${PROJECT_SOURCE_DIR}/polyspline-data/" CACHE PATH "Where should polyfem download and look for polyspline data?")


ExternalProject_Add(
    polyfem_polyspline_download
    PREFIX ${FETCHCONTENT_BASE_DIR}/polyfem-test-polyspline
    SOURCE_DIR ${POLYFEM_POLYSPLINE_DIR}
    GIT_REPOSITORY https://github.com/polyfem/Poly-Spline-Finite-Element-Method
    GIT_TAG e413c27a1aaab9fccf521e608d390453d4c4ceb7
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

# Create a dummy target for convenience
add_library(polyfem_polyspline INTERFACE)
add_library(polyfem::polyspline ALIAS polyfem_polyspline)

add_dependencies(polyfem_polyspline polyfem_polyspline_download)

target_compile_definitions(polyfem_polyspline INTERFACE POLYFEM_POLYSPLINE_DIR=\"${POLYFEM_POLYSPLINE_DIR}\")
