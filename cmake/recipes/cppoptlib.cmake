# CppNumericalSolvers (https://github.com/PatWie/CppNumericalSolvers.git)
# License: MIT

if(TARGET cppoptlib)
    return()
endif()

message(STATUS "Third-party: creating target 'cppoptlib'")

include(CPM)
CPMAddPackage(
    NAME cppoptlib
    GITHUB_REPOSITORY PatWie/CppNumericalSolvers
    GIT_TAG 7eddf28fa5a8872a956d3c8666055cac2f5a535d
    DOWNLOAD_ONLY TRUE
)

add_library(cppoptlib INTERFACE)
target_include_directories(cppoptlib SYSTEM INTERFACE ${cppoptlib_SOURCE_DIR}/include)