# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
CPMAddPackage("gh:Huangzizhou/ipc-toolkit#df76e6aa03a0588e358708fec7eb5e76d1b9c88f")
