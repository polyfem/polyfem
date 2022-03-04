# tinyxml
# License: zlib

if(TARGET tinyxml2)
    return()
endif()

message(STATUS "Third-party: creating target 'tinyxml'")

SET(BUILD_SHARED_LIBS OFF)
SET(BUILD_STATIC_LIBS ON)
SET(BUILD_TESTING OFF)

include(FetchContent)
FetchContent_Declare(
    tinyxml
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG 9.0.0
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(tinyxml)
