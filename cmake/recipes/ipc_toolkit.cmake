# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
option(IPC_TOOLKIT_WITH_ABSEIL OFF)
option(IPC_TOOLKIT_WITH_ROBIN_MAP OFF)
CPMAddPackage(
	NAME "ipc-toolkit"
	GIT_REPOSITORY git@github.com:fsichetti/ipc-toolkit.git
	GIT_TAG "main"
)
