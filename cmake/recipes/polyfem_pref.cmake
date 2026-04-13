# data (https://github.com/polyfem/polyfem-data)
# License: MIT

if(TARGET polyfem::pref)
    return()
endif()

include(ExternalProject)

set(POLYFEM_PREF_DIR "${PROJECT_SOURCE_DIR}/pref-data/" CACHE PATH "Where should polyfem download and look for pref data?")


ExternalProject_Add(
    polyfem_pref_download
    PREFIX ${FETCHCONTENT_BASE_DIR}/polyfem-test-pref
    SOURCE_DIR ${POLYFEM_PREF_DIR}
    GIT_REPOSITORY https://github.com/polyfem/Decoupling-Simulation-Accuracy-from-Mesh-Quality
    GIT_TAG 4129da2b2c453c70f07ab37a0a717b90c801276e
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

# Create a dummy target for convenience
add_library(polyfem_pref INTERFACE)
add_library(polyfem::pref ALIAS polyfem_pref)

add_dependencies(polyfem_pref polyfem_pref_download)

target_compile_definitions(polyfem_pref INTERFACE POLYFEM_PREF_DIR=\"${POLYFEM_PREF_DIR}\")
