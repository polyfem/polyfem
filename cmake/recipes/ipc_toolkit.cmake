# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
#CPMAddPackage("gh:ipc-sim/ipc-toolkit#6138afafc1bb0045da35176b18252ee5f3811fd9")
CPMAddPackage("gh:maxpaik16/ipc-toolkit#900108cfdf60fe71408f9ba0ae98f90e0506664a")