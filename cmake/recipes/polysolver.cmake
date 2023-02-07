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
    GIT_TAG e53ab52e334ade50520961d2a11431baf52ea08d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
