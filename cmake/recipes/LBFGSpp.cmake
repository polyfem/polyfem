# LBFGSpp (https://github.com/yixuan/LBFGSpp)
# License: MIT

if(TARGET LBGFSpp::LBGFSpp)
    return()
endif()

message(STATUS "Third-party: creating target 'LBGFSpp::LBGFSpp'")

include(CPM)
CPMAddPackage(
    NAME lbfgspp
    GITHUB_REPOSITORY yixuan/LBFGSpp
    GIT_TAG v0.2.0
    DOWNLOAD_ONLY TRUE
)

add_library(LBFGSpp INTERFACE)
target_include_directories(LBFGSpp INTERFACE "${lbfgspp_SOURCE_DIR}/include")
add_library(LBFGSpp::LBFGSpp ALIAS LBFGSpp)
