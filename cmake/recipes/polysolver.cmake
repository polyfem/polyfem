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
    GIT_TAG b2c639c94b5550ee6e8e780b0e02cf0689a59eda
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
