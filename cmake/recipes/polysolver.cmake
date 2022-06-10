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
    GIT_TAG 58dd11fe7c05adc88497861743bd42ce440b3e03
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
