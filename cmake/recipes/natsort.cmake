# natsort (https://github.com/sourcefrog/natsort)
# License: zlib

if(TARGET natsort::natsort)
    return()
endif()

message(STATUS "Third-party: creating target 'natsort::natsort'")

include(CPM)
CPMAddPackage(
    NAME natsort
    GITHUB_REPOSITORY sourcefrog/natsort
    GIT_TAG f8a6b0cda6de846b2a3b741967f35809de00b083
    DOWNLOAD_ONLY TRUE
)

add_library(natsort "${natsort_SOURCE_DIR}/strnatcmp.c")
target_include_directories(natsort PUBLIC ${natsort_SOURCE_DIR})
add_library(natsort::natsort ALIAS natsort)
