# LBFGSpp (https://github.com/yixuan/LBFGSpp)
# License: MIT

if(TARGET LBGFSpp::LBGFSpp)
    return()
endif()

message(STATUS "Third-party: creating target 'LBGFSpp::LBGFSpp'")

include(FetchContent)
FetchContent_Declare(
    lbfgspp
    GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
    GIT_TAG v0.1.0
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(lbfgspp)
if(NOT lbfgspp_POPULATED)
    FetchContent_Populate(lbfgspp)
endif()

add_library(LBFGSpp INTERFACE)
target_include_directories(LBFGSpp INTERFACE "${lbfgspp_SOURCE_DIR}/include")
add_library(LBFGSpp::LBFGSpp ALIAS LBFGSpp)
