# MshIO
# License: Apache-2.0

if(TARGET mshio)
    return()
endif()

message(STATUS "Third-party: creating target 'mshio'")


include(FetchContent)
FetchContent_Declare(
    mshio
    GIT_REPOSITORY https://github.com/zfergus/MshIO.git
    GIT_TAG cc161de1447827aa401f799c4c3034d2ea1a2bea
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(mshio)
