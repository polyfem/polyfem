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
    GIT_TAG 6e10448b3de92b7c2559fd2085d38ae49a1cdbbe
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
