# Microstructure inflator

if(TARGET Microstructures)
    return()
endif()

message(STATUS "Third-party: creating target 'Microstructures'")


include(FetchContent)
FetchContent_Declare(
    Microstructures
    GIT_REPOSITORY https://github.com/Huangzizhou/microstructure_inflators.git
    GIT_TAG f15035ac8581fe26572a6bccad77eaf02f9da04c
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(Microstructures)

add_library(Microstructures INTERFACE)
include_directories(${Microstructures_SOURCE_DIR}/include)
# target_include_directories(Microstructures INTERFACE ${Microstructures_SOURCE_DIR}/include)
target_link_libraries(Microstructures INTERFACE micro_isosurface_inflator)
