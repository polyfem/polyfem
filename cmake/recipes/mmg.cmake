# WMTK
# License: LGPL

if(TARGET mmg::mmg)
    return()
endif()

message(STATUS "Third-party: creating target 'mmg::mmg'")

option(BUILD_TESTING "Enable/Disable continuous integration" OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    mmg
    GIT_REPOSITORY https://github.com/MmgTools/mmg.git
    GIT_TAG v5.6.0
    # GIT_TAG 88e2dd6cc773c43141b137fd0972c0eb2f4bbd2a
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(mmg)

add_library(mmg::mmg ALIAS libmmg_a)
add_library(mmg::mmgs ALIAS libmmgs_a)
add_library(mmg::mmg2d ALIAS libmmg2d_a)
add_library(mmg::mmg3d ALIAS libmmg3d_a)
