# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/Huangzizhou/ipc-toolkit.git
    GIT_TAG 0bddf92f19726ea4ac616497d0104fa502b8d50d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
