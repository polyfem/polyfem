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
    GIT_TAG f6c889822647cb040a16bdfd79f46cc6e01767d9
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
