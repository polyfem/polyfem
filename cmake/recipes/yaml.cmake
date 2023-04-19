# yaml-cpp (https://github.com/jbeder/yaml-cpp)
# License: MIT

if(TARGET yaml-cpp::yaml-cpp)
    return()
endif()

message(STATUS "Third-party: creating target 'yaml-cpp::yaml-cpp'")

include(FetchContent)
FetchContent_Declare(
    yaml
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
    GIT_TAG 0e6e28d1a38224fc8172fae0109ea7f673c096db
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(yaml)
