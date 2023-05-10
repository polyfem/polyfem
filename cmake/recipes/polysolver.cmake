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
    GIT_TAG 77c10aadc2c79343ec732b02cf8dc86c0e963699
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
