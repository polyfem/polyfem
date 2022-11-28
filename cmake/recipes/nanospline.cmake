# Nanospline (https://github.com/qnzhou/nanospline)
# License: Mozilla Public License 2.0

if(TARGET nanospline)
    return()
endif()

message(STATUS "Third-party: creating target 'nanospline'")

option(NANOSPLINE_BUILD_TESTS "Build Tests" OFF)

include(FetchContent)
FetchContent_Declare(
    nanospline
    GIT_REPOSITORY https://github.com/qnzhou/nanospline.git
    GIT_TAG de2e4d4daceb5e5058ea3092ddf1c6e5ba447c38
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(nanospline)
