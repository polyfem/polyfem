# MshIO
# License: Apache-2.0

if(TARGET mshio)
    return()
endif()

message(STATUS "Third-party: creating target 'mshio'")


include(FetchContent)
FetchContent_Declare(
    mshio
    GIT_REPOSITORY https://github.com/qnzhou/MshIO.git
    GIT_TAG dbfe01f072a90d067a25c5e962ea1f87e34c4fd3
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(mshio)
