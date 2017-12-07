cmake_minimum_required(VERSION 3.1)
project(navigation)

add_executable(${PROJECT_NAME}
	main.cpp
	navigation.cpp
	navigation.hpp
)

target_link_libraries(${PROJECT_NAME} geogram geogram_gfx)

# Let's get a little bit paranoid
include(SetWarnings)
target_compile_options(${PROJECT_NAME} PRIVATE ${ALL_WARNINGS})

# Use C++11
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

# Output directory for binaries
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
