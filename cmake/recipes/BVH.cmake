# BVH
# License: MIT

if(TARGET BVH_lib)
    return()
endif()

message(STATUS "Third-party: creating target 'BVH_lib'")


include(FetchContent)
FetchContent_Declare(
    BVH_lib
    GIT_REPOSITORY https://github.com/geometryprocessing/SimpleBVH.git
    GIT_TAG 15574502f6cb8039b0bfa4a85ccad04e09deaf05
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(BVH_lib)
