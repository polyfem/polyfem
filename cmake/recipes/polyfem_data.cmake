# data
# License: MIT

message(STATUS "Third-party: fetching 'polyfem data'")

include(FetchContent)
FetchContent_Declare(
    polyfem_data
    GIT_REPOSITORY https://github.com/polyfem/polyfem-data
    GIT_TAG 57b13fe93bcdf89d7d2f30ca02bc39ed3d2cd7c7
    GIT_SHALLOW FALSE
    SOURCE_DIR ${POLYFEM_DATA_ROOT}
)
FetchContent_GetProperties(polyfem_data)
if(NOT polyfem_data_POPULATED)
  FetchContent_Populate(polyfem_data)
  SET(POLYFEM_DATA_DIR ${polyfem_data_SOURCE_DIR})
endif()
