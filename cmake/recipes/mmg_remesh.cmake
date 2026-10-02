# MMG (https://github.com/MmgTools/mmg)
# License: LGPL

# Do not rename this file to mmg.cmake. MMG calls include(mmg) for its own
# internal module, and PolyFEM's recipe directory is on CMAKE_MODULE_PATH.
# Using that name would recursively include this recipe while MMG is configuring.

if(TARGET mmg::mmg)
    return()
endif()

message(STATUS "Third-party: creating target 'mmg::mmg'")

option(BUILD_TESTING "Enable/Disable continuous integration" OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(USE_SCOTCH OFF CACHE STRING "" FORCE)
set(USE_ELAS OFF CACHE STRING "" FORCE)
set(USE_VTK OFF CACHE STRING "" FORCE)

include(CPM)
CPMAddPackage("gh:MmgTools/mmg@5.8.0")

add_library(mmg::mmg ALIAS libmmg_a)
