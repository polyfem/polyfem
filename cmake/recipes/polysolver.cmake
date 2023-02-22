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
    GIT_TAG ec7490e91fca00ae04b8a3415a2e996b37bd125e
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
