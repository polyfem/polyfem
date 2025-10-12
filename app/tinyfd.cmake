# tinyfd::tinyfd
# License: Zlib

if(TARGET tinyfd::tinyfd)
    return()
endif()

message(STATUS "Third-party: creating target 'tinyfd::tinyfd'")

include(CPM)
CPMAddPackage("gh:polyfem/tinyfiledialogs#a1f53211039a7dd605e9c7983c4b118b0f69f4fc")
