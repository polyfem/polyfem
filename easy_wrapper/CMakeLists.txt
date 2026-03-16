cmake_minimum_required(VERSION 3.16)
project(EasyPolyFEM LANGUAGES CXX)

add_executable(easy_polyfem easy_polyfem.cpp)
target_compile_features(easy_polyfem PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(easy_polyfem PRIVATE /W4)
else()
    target_compile_options(easy_polyfem PRIVATE -Wall -Wextra -Wpedantic)
endif()