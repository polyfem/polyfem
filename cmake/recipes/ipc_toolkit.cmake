# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

if(EXISTS "${POLYFEM_IPC_TOOLKIT_PATH}")
    message(STATUS "Using IPC Toolkit at: ${POLYFEM_IPC_TOOLKIT_PATH}")
    add_subdirectory("${POLYFEM_IPC_TOOLKIT_PATH}" "${PROJECT_BINARY_DIR}/ipc-toolkit")
else()
    include(FetchContent)
    FetchContent_Declare(
        ipc_toolkit
        # GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
        # GIT_TAG 1a5cd55c2e8bd6f21c25e65b006c224bf94c7639
        GIT_REPOSITORY git@github.com:zfergus/ipc-toolkit.git
        GIT_TAG 9b3269b913d6b7f5d7c62ff3d5707aa2789324f5
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(ipc_toolkit)
endif()
