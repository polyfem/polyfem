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
    GIT_TAG e557b3547cd80f3fcd1c4243852aa4a3c6a3abd0
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
