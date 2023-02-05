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
    GIT_TAG e33b4eeecb9bd847fa630f9f12132971076a251d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(paraviewo)
