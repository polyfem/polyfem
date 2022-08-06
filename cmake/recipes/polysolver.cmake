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
    GIT_TAG 0044479f5e1f835f311f9fd64902ce6d17977f83
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
