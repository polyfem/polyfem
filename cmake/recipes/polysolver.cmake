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
    GIT_TAG 90fdc63af4630fab1babcf4011ed9258d6946c36
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
