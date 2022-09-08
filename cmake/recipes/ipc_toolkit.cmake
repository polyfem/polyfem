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
    GIT_TAG 4004049ccb6930c3b8340d1901f10a30eced6fe7
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
