# CppNumericalSolvers
# License: MIT

if(TARGET cppoptlib)
    return()
endif()

message(STATUS "Third-party: creating target 'cppoptlib'")


include(FetchContent)
FetchContent_Declare(
    cppoptlib
    GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
    GIT_TAG 7eddf28fa5a8872a956d3c8666055cac2f5a535d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(cppoptlib)


add_library(cppoptlib INTERFACE)
target_include_directories(cppoptlib SYSTEM INTERFACE ${cppoptlib_SOURCE_DIR}/include)