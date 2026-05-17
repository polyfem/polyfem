# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(CPM)
CPMAddPackage("gh:geometryprocessing/json-spec-engine#7c8330309d36746d8eedae763f492351a266c6d5")