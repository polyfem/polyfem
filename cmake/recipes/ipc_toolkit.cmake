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
    GIT_TAG 6df1d25a51ee1ece6960cb7b6b14d811c5d252fa
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
