# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
#CPMAddPackage("gh:ipc-sim/ipc-toolkit#6138afafc1bb0045da35176b18252ee5f3811fd9")
CPMAddPackage("gh:maxpaik16/ipc-toolkit#374bb4cf0d2172f4a8e1b8c3434311c0da8eed68")