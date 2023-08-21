# TinyExpr (https://github.com/codeplea/tinyexpr)
# License: zlib

if(TARGET tinyexpr::tinyexpr)
    return()
endif()

message(STATUS "Third-party: creating target 'tinyexpr::tinyexpr'")

include(CPM)
CPMAddPackage(
    NAME tinyexpr
    GITHUB_REPOSITORY codeplea/tinyexpr
    GIT_TAG 4e8cc0067a1e2378faae23eb2dfdd21e9e9907c2
    DOWNLOAD_ONLY TRUE
)

add_library(tinyexpr "${tinyexpr_SOURCE_DIR}/tinyexpr.c")
target_include_directories(tinyexpr SYSTEM PUBLIC "${tinyexpr_SOURCE_DIR}")
add_library(tinyexpr::tinyexpr ALIAS tinyexpr)
