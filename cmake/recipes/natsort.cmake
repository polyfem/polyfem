# natsort
# License: zlib

if(TARGET natsort::natsort)
    return()
endif()

message(STATUS "Third-party: creating target 'natsort::natsort'")

include(FetchContent)
FetchContent_Declare(
    natsort
    GIT_REPOSITORY https://github.com/sourcefrog/natsort.git
    GIT_TAG f8a6b0cda6de846b2a3b741967f35809de00b083
    GIT_SHALLOW FALSE
)

FetchContent_GetProperties(natsort)
if(NOT natsort_POPULATED)
    FetchContent_Populate(natsort)
endif()

add_library(natsort "${natsort_SOURCE_DIR}/strnatcmp.c")
target_include_directories(natsort PUBLIC ${natsort_SOURCE_DIR})
add_library(natsort::natsort ALIAS natsort)
