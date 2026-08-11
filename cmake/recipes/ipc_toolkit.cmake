# IPC Toolkit (https://github.com/sdast9/ipc-toolkit)
# Fork of ipc-sim/ipc-toolkit with per-collision stiffness_scale and
# NormalCollisions::compute_avg_distance for the semi-implicit barrier mode.
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(CPM)
CPMAddPackage("gh:sdast9/ipc-toolkit#9da3094a46bcc054cc19024a5c748557c5bb6b9e")
