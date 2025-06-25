# tinyfd::tinyfd
# License: Zlib

if(TARGET tinyfd::tinyfd)
    return()
endif()

message(STATUS "Third-party: creating target 'tinyfd::tinyfd'")

include(CPM)
CPMAddPackage(
    NAME tinyfiledialogs
    URL https://sourceforge.net/projects/tinyfiledialogs/files/latest/download
    URL_HASH SHA256=580d959bebd6f068867077968f8ec3e401c3bd74c062c87cca232411112190b6
)

add_library(tinyfiledialogs ${tinyfiledialogs_SOURCE_DIR}/tinyfiledialogs.c)
target_include_directories(tinyfiledialogs INTERFACE ${tinyfiledialogs_SOURCE_DIR})
add_library(tinyfd::tinyfd ALIAS tinyfiledialogs)