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
    GIT_TAG 42dd524fba08b8ebb54791eb257f4fb405cc0a14
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
