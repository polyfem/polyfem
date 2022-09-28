# Polyfem Solvers
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")


include(FetchContent)
FetchContent_Declare(
    polysolve
    GIT_REPOSITORY https://github.com/polyfem/polysolve.git
    GIT_TAG 421ad443b1d1f652f9ac2acf9b2816215f5a7d4f
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
