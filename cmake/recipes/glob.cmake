# glob (https://github.com/p-ranav/glob)
# License: MIT

if(TARGET Glob::Glob)
    return()
endif()

message(STATUS "Third-party: creating target 'Glob::Glob'")

include(CPM)
CPMAddPackage("gh:p-ranav/glob#a5f32776f93aaf827c8deb5ecd2857c19b9ba4a7")

add_library(Glob::Glob ALIAS Glob)
