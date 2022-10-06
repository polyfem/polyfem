# glob
# License: MIT

if(TARGET Glob::Glob)
    return()
endif()

message(STATUS "Third-party: creating target 'Glob::Glob'")

include(FetchContent)
FetchContent_Declare(
    glob
    GIT_REPOSITORY https://github.com/p-ranav/glob.git
    GIT_TAG a5f32776f93aaf827c8deb5ecd2857c19b9ba4a7
    GIT_SHALLOW FALSE
)

FetchContent_MakeAvailable(glob)
add_library(Glob::Glob ALIAS Glob)
