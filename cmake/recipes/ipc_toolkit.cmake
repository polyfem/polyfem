# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
# TODO: revert to ipc-sim/ipc-toolkit main once PR #239 lands
# https://github.com/ipc-sim/ipc-toolkit/pull/239
CPMAddPackage("gh:Huangzizhou/ipc-toolkit#7b970316ec3cf2e04fb1cdbf8141ecfd28efd4f0")
