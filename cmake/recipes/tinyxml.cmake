# tinyxml (https://github.com/leethomason/tinyxml2)
# License: zlib

if(TARGET tinyxml2)
    return()
endif()

message(STATUS "Third-party: creating target 'tinyxml'")

SET(BUILD_SHARED_LIBS OFF)
SET(BUILD_STATIC_LIBS ON)
SET(BUILD_TESTING OFF)

include(CPM)
CPMAddPackage("gh:leethomason/tinyxml2#9.0.0")