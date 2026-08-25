# pybind11
# License: MIT

if(TARGET pybind11::embed)
    return()
endif()

message(STATUS "Third-party: creating target 'pybind11::embed'")

set(BUILD_TESTS		OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_GTEST	OFF CACHE BOOL "" FORCE)
set(PYBIND11_FINDPYTHON ON)

include(CPM)
CPMAddPackage(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.3
    GIT_SHALLOW FALSE
)
