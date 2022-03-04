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
    GIT_TAG d7766f83a9c554a798625c244c7cec33b192c5d1
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
