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
    GIT_TAG 85a2affca44bd93c2e893e614e08ce80f1a35499
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
