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
    GIT_TAG 571a02d26b2a6b85b947907b2fe1646de8855b8d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(jse)
