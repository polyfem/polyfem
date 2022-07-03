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
    GIT_TAG 913f3886a7c1c41e94ef6dcc2bda6215da92edbf
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
