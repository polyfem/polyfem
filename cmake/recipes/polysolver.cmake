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
    GIT_TAG 5da3ab94ff86933a53a87afd5f7f7eb37da1e9c5
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
