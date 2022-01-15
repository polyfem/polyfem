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
        GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
        GIT_TAG 14a08b1bf1c994311a88ce99973f98001f0b74ce
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(ipc_toolkit)
endif()
