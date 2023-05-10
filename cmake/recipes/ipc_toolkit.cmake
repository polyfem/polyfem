# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

set(IPC_TOOLKIT_WITH_BROADMARK ON CACHE BOOL "Enable Broadmark" FORCE)

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG 02042b1539e45d7a90819e71ba44cca3d6713718
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
