# TinyExpr (https://github.com/codeplea/tinyexpr)
# License: zlib

if(TARGET tinyexpr::tinyexpr)
    return()
endif()

message(STATUS "Third-party: creating target 'tinyexpr::tinyexpr'")

include(FetchContent)
FetchContent_Declare(
    tinyexpr
    GIT_REPOSITORY https://github.com/codeplea/tinyexpr.git
    GIT_TAG 4e8cc0067a1e2378faae23eb2dfdd21e9e9907c2
    GIT_SHALLOW FALSE
)

FetchContent_GetProperties(tinyexpr)
if(NOT tinyexpr_POPULATED)
    FetchContent_Populate(tinyexpr)
endif()

add_library(tinyexpr "${tinyexpr_SOURCE_DIR}/tinyexpr.c")
target_include_directories(tinyexpr SYSTEM PUBLIC "${tinyexpr_SOURCE_DIR}")
add_library(tinyexpr::tinyexpr ALIAS tinyexpr)
