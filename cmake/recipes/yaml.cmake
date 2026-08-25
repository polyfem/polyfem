# yaml-cpp (https://github.com/jbeder/yaml-cpp)
# License: MIT

if(TARGET yaml-cpp::yaml-cpp)
    return()
endif()

message(STATUS "Third-party: creating target 'yaml-cpp::yaml-cpp'")

include(CPM)
CPMAddPackage(
    NAME yaml-cpp
    GITHUB_REPOSITORY jbeder/yaml-cpp
    VERSION 0.9.0
    GIT_TAG yaml-cpp-0.9.0
)
