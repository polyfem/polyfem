# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(CPM)
CPMAddPackage("gh:geometryprocessing/json-spec-engine#81c4dc4092115f37a841285dede3d39559f7fb86")
