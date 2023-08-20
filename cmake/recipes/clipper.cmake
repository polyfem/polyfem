# Clipper (https://sourceforge.net/projects/polyclipping)
# License: (BSL1.0)

if(TARGET clipper::clipper)
    return()
endif()

message(STATUS "Third-party: creating target 'clipper::clipper'")

include(CPM)
CPMAddPackage(
    NAME clipper_clipper
    URL https://sourceforge.net/projects/polyclipping/files/clipper_ver6.4.2.zip
    URL_MD5 100b4ec56c5308bac2d10f3966e35e11
    DOWNLOAD_ONLY TRUE
)

add_library(clipper_clipper ${clipper_clipper_SOURCE_DIR}/cpp/clipper.cpp)
target_include_directories(clipper_clipper PUBLIC ${clipper_clipper_SOURCE_DIR}/cpp)
add_library(clipper::clipper ALIAS clipper_clipper)