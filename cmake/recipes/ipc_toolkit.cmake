# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
CPMAddPackage(
	NAME "ipc-toolkit"
	GIT_REPOSITORY https://github.com/fsichetti/ipc-toolkit.git
	GIT_TAG "77b518a0"
	OPTIONS "IPC_TOOLKIT_WITH_GEOGRAM ON IPC_TOOLKIT_WITH_CUDA OFF"
)
