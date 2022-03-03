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
        GIT_TAG 4b18b301d4065fa0c0b4d767e2d6e59a06a06dc6
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(ipc_toolkit)
endif()
