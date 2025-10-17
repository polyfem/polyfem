# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
CPMAddPackage("gh:Huangzizhou/ipc-toolkit#e1c63d54797e318cf45e04cc2c738a2eccc4192c")
