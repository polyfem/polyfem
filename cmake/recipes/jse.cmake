# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(CPM)
CPMAddPackage("gh:geometryprocessing/json-spec-engine#49f1a30f8c2912814916ec3d6108a649b23cb243")