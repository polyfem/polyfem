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
    GIT_TAG 9dc616b3e04ff9383aa60e0aba0be07bc3b39a87
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(mshio)
