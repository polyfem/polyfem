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
    GIT_TAG c8856f1855148c38e93a63eb1f3db5a8bed5dfe1
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(jse)
