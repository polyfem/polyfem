cmake_minimum_required(VERSION 3.16)
project(EasyPolyFEM LANGUAGES CXX)

add_executable(easy_polyfem
    src/main.cpp
    src/cli.cpp
    src/json_writer.cpp
    src/runner.cpp
    src/validator.cpp
)

target_include_directories(easy_polyfem PRIVATE include)
target_compile_features(easy_polyfem PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(easy_polyfem PRIVATE /W4)
else()
    target_compile_options(easy_polyfem PRIVATE -Wall -Wextra -Wpedantic)
endif()