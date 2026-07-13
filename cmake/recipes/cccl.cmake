# cccl (https://github.com/NVIDIA/cccl)
# License: Apache

if(TARGET CCCL::CCCL)
    return()
endif()

message(STATUS "Third-party: creating target 'CCCL::CCCL'")

include(CPM)
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY NVIDIA/cccl
    VERSION 3.3.0
)
