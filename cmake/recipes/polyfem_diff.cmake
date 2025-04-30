# data (https://github.com/polyfem/polyfem-data)
# License: MIT

if(TARGET polyfem::differentiable)
    return()
endif()

include(ExternalProject)

set(POLYFEM_DIFF_DIR "${PROJECT_SOURCE_DIR}/diff-data/" CACHE PATH "Where should polyfem download and look for diff data?")

ExternalProject_Add(
    polyfem_diff_download
    PREFIX ${FETCHCONTENT_BASE_DIR}/polyfem-test-diff
    SOURCE_DIR ${POLYFEM_DIFF_DIR}
    GIT_REPOSITORY https://github.com/polyfem/differentiability-data
    GIT_TAG 7e49f248417987f0187a8bd0171954f486152a6d
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

# Create a dummy target for convenience
add_library(polyfem_diff INTERFACE)
add_library(polyfem::diff ALIAS polyfem_diff)

add_dependencies(polyfem_diff polyfem_diff_download)

target_compile_definitions(polyfem_diff INTERFACE POLYFEM_DIFF_DIR=\"${POLYFEM_DIFF_DIR}\")
