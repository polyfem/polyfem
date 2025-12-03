# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
CPMAddPackage(
	NAME "ipc-toolkit"
	GIT_REPOSITORY git@github.com:fsichetti/ipc-toolkit.git
	GIT_TAG "eae6ff7"
)
