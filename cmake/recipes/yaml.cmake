# yaml-cpp (https://github.com/jbeder/yaml-cpp)
# License: MIT

if(TARGET yaml-cpp::yaml-cpp)
    return()
endif()

message(STATUS "Third-party: creating target 'yaml-cpp::yaml-cpp'")

include(CPM)
CPMAddPackage("gh:jbeder/yaml-cpp#0.8.0")
