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
    GIT_TAG b50e0f0b9fbd57129c61786180cd0a89f58e0f87
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(jse)
