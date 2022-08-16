# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(FetchContent)
FetchContent_Declare(
    jse
    GIT_REPOSITORY https://github.com/zfergus/json-spec-engine.git
    GIT_TAG 6fde77ff4b7f820e57a6da0a5a6d5e1024b1c62c
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(jse)
