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
    GIT_TAG daaa0c4904ccb2767860e77a8b74c6514f55d821
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
