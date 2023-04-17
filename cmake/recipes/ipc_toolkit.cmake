# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

# WARNING: This forces the use of the floating point CCD used in the original IPC paper.
set(IPC_TOOLKIT_WITH_CORRECT_CCD OFF CACHE BOOL "Use the TightInclusion CCD" FORCE)

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
    GIT_TAG 6aaf0ca84917d595a16a1d37dc1e6cd98ca0467c
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
