# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(CPM)
CPMAddPackage("gh:geometryprocessing/json-spec-engine#5a9e13b9fee6f59a38cdcd811267dc88a87be0f1")