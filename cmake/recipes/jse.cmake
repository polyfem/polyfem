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
    GIT_TAG 1261dc89478c7646ff99cbed8bc5357c2813565d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(jse)
