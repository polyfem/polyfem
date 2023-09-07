# data (https://github.com/polyfem/polyfem-data)
# License: MIT

if(TARGET polyfem::data)
    return()
endif()

include(ExternalProject)

set(POLYFEM_DATA_DIR "${PROJECT_SOURCE_DIR}/data/" CACHE PATH "Where should polyfem download and look for test data?")
option(POLYFEM_USE_EXISTING_DATA_DIR "Use and existing data directory instead of downloading it" OFF)

if(POLYFEM_USE_EXISTING_DATA_DIR)
    ExternalProject_Add(
        polyfem_data_download
        PREFIX ${FETCHCONTENT_BASE_DIR}/polyfem-test-data
        SOURCE_DIR ${POLYFEM_DATA_DIR}
        # NOTE: No download step
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
    )
else()
    ExternalProject_Add(
        polyfem_data_download
        PREFIX ${FETCHCONTENT_BASE_DIR}/polyfem-test-data
        SOURCE_DIR ${POLYFEM_DATA_DIR}
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG 1a224cc7c7b1fe76ee9d033e194bdf78e5b4b7c9
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
    )
endif()

# Create a dummy target for convenience
add_library(polyfem_data INTERFACE)
add_library(polyfem::data ALIAS polyfem_data)

add_dependencies(polyfem_data polyfem_data_download)

target_compile_definitions(polyfem_data INTERFACE  POLYFEM_DATA_DIR=\"${POLYFEM_DATA_DIR}\")
