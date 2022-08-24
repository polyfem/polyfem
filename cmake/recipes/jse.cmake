# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(FetchContent)
FetchContent_Declare(
    jse
    GIT_REPOSITORY https://github.com/geometryprocessing/json-spec-engine.git
    GIT_TAG 93d74f7f5807a99c0aa28b6f11815e703dfed5d7
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(jse)
