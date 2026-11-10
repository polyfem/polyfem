# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)

CPMDeclarePackage(eigen NAME eigen GIT_REPOSITORY https://eigen.googlesource.com/mirror GIT_TAG 5.0.1 DOWNLOAD_ONLY ON)
CPMAddPackage("gh:polyfem/polysolve#621e8fb3889717a33c4fa33a608052e9d7459e63")
