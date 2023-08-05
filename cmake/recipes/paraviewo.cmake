# Polyfem Solvers
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")


include(FetchContent)
FetchContent_Declare(
    paraviewo
    GIT_REPOSITORY https://github.com/polyfem/paraviewo.git
    GIT_TAG 61685ee7f924873d1fd13068b6f0b691b295992d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(paraviewo)
