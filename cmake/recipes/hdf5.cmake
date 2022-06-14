# hdf5
# License: ???

if(TARGET hdf5::hdf5)
    return()
endif()

message(STATUS "Third-party: creating target 'hdf5'")

option(HDF5_BUILD_EXAMPLES OFF)
option(HDF5_BUILD_TOOLS OFF)
option(HDF5_BUILD_UTILS OFF)
option(HDF5_BUILD_HL_TOOLS OFF)
option(HDF5_TEST_CPP OFF)
option(HDF5_TEST_EXAMPLES OFF)
option(HDF5_TEST_FORTRAN OFF)
option(HDF5_TEST_JAVA OFF)
option(HDF5_TEST_PARALLEL OFF)
option(HDF5_TEST_SERIAL OFF)
option(HDF5_TEST_SWMR OFF)
option(HDF5_TEST_TOOLS OFF)
option(HDF5_TEST_VFD OFF)

#To prevent changes in the oput dirs
set (HDF5_EXTERNALLY_CONFIGURED 1)

include(FetchContent)
FetchContent_Declare(
    hdf5
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG 338c2c6cd4bfb9d3001668ef74db178f10fe0abd
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(hdf5)

add_library(hdf5::hdf5 ALIAS hdf5-static)