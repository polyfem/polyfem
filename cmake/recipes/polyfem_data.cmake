# data
# License: MIT

message(STATUS "Third-party: fetching 'polyfem data'")

include(FetchContent)
FetchContent_Declare(
    polyfem_data
    GIT_REPOSITORY https://github.com/polyfem/polyfem-data
    GIT_TAG 0e2dca0e92c644534c2a1f279f864bbc352b8c7c
    GIT_SHALLOW FALSE
    SOURCE_DIR ${POLYFEM_DATA_ROOT}
)
FetchContent_GetProperties(polyfem_data)
if(NOT polyfem_data_POPULATED)
  FetchContent_Populate(polyfem_data)
  SET(POLYFEM_DATA_DIR ${polyfem_data_SOURCE_DIR})
endif()
