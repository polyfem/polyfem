# Clipper
# License: (BSL1.0)

if(TARGET clipper::clipper)
    return()
endif()

message(STATUS "Third-party: creating target 'clipper::clipper'")


include(FetchContent)
FetchContent_Declare(
    clipper_clipper
    URL https://sourceforge.net/projects/polyclipping/files/clipper_ver6.4.2.zip
    URL_MD5 100b4ec56c5308bac2d10f3966e35e11
)
FetchContent_MakeAvailable(clipper_clipper)



add_library(clipper_clipper ${clipper_clipper_SOURCE_DIR}/cpp/clipper.cpp)
target_include_directories(clipper_clipper PUBLIC ${clipper_clipper_SOURCE_DIR}/cpp)
add_library(clipper::clipper ALIAS clipper_clipper)
