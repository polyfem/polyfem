# BVH (https://github.com/geometryprocessing/SimpleBVH)
# License: MIT

if(TARGET BVH_lib)
    return()
endif()

message(STATUS "Third-party: creating target 'BVH_lib'")

include(CPM)
CPMAddPackage("gh:geometryprocessing/SimpleBVH#15574502f6cb8039b0bfa4a85ccad04e09deaf05")