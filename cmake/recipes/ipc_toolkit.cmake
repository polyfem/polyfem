# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

set(IPC_TOOLKIT_WITH_SIMD ON CACHE BOOL "Enable SIMD" FORCE)
set(IPC_TOOLKIT_WITH_CUDA ON CACHE BOOL "Enable CUDA CCD" FORCE)
set(IPC_TOOLKIT_WITH_BROADMARK ON CACHE BOOL "Enable Broadmark" FORCE)

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG 333b66f3a6485c73f181402f6ad40d660b7d53f0
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
