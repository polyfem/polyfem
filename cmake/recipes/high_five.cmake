# HighFive (https://github.com/BlueBrain/HighFive)
# License: BSL-1.0 License

if(TARGET HighFive::HighFive)
    return()
endif()

message(STATUS "Third-party: creating target 'HighFive'")

option(HIGHFIVE_USE_BOOST "Enable Boost Support" OFF)
# option(HIGHFIVE_USE_EIGEN "Enable Eigen testing" OFF) # Set later
set(HIGHFIVE_USE_EIGEN OFF CACHE BOOL "Enable Eigen testing" FORCE)
mark_as_advanced(HIGHFIVE_USE_EIGEN)
option(HIGHFIVE_USE_XTENSOR "Enable xtensor testing" OFF)
option(HIGHFIVE_USE_OPENCV "Enable OpenCV testing" OFF)
option(HIGHFIVE_UNIT_TESTS "Enable unit tests" OFF)
option(HIGHFIVE_EXAMPLES "Compile examples" OFF)
option(HIGHFIVE_PARALLEL_HDF5 "Enable Parallel HDF5 support" OFF)
option(HIGHFIVE_BUILD_DOCS "Enable documentation building" OFF)
#Used to prevend looking for sys libraries
set(HDF5_C_LIBRARIES "")

include(FetchContent)
FetchContent_Declare(
    high_five
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG v2.6.2
)
FetchContent_MakeAvailable(high_five)

# Eigen
include(eigen)
FetchContent_GetProperties(eigen)
target_include_directories(libdeps SYSTEM INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
target_compile_definitions(libdeps INTERFACE H5_USE_EIGEN)


add_library(HighFive::HighFive ALIAS HighFive)