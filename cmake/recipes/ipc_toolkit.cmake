# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG 7c5fc9928e2e0f5aa4c31446fd8615f072c21a64
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
