# MMG (https://github.com/MmgTools/mmg)
# License: LGPL

if(TARGET mmg::mmg)
    return()
endif()

message(STATUS "Third-party: creating target 'mmg::mmg'")

option(BUILD_TESTING "Enable/Disable continuous integration" OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(CPM)
CPMAddPackage("gh:MmgTools/mmg@5.6.0")

add_library(mmg::mmg ALIAS libmmg_a)
add_library(mmg::mmgs ALIAS libmmgs_a)
add_library(mmg::mmg2d ALIAS libmmg2d_a)
add_library(mmg::mmg3d ALIAS libmmg3d_a)