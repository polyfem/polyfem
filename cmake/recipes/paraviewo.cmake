# Polyfem Solvers
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")


include(FetchContent)
FetchContent_Declare(
    paraviewo
    GIT_REPOSITORY https://github.com/polyfem/paraviewo.git
    GIT_TAG b3a167554c61952d6cc36b7a2ce4b9a5da5e5d6b
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(paraviewo)
