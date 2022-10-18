# data
# License: MIT

if(TARGET polyfem::data)
    return()
endif()

include(ExternalProject)

set(POLYFEM_DATA_DIR "${PROJECT_SOURCE_DIR}/data/" CACHE PATH "Where should polyfem download and look for test data?")

ExternalProject_Add(
    polyfem_data_download
    PREFIX ${FETCHCONTENT_BASE_DIR}/polyfem-test-data
    SOURCE_DIR ${POLYFEM_DATA_DIR}
    GIT_REPOSITORY https://github.com/polyfem/polyfem-data
    GIT_TAG f9d6a461a4cdc341f4623db04a34c806427b2cb4
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

# Create a dummy target for convenience
add_library(polyfem_data INTERFACE)
add_library(polyfem::data ALIAS polyfem_data)

add_dependencies(polyfem_data polyfem_data_download)

target_compile_definitions(polyfem_data INTERFACE  POLYFEM_DATA_DIR=\"${POLYFEM_DATA_DIR}\")