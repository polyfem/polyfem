# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)

CPMDeclarePackage(eigen NAME eigen GIT_REPOSITORY https://eigen.googlesource.com/mirror GIT_TAG 5.0.1 DOWNLOAD_ONLY ON)
CPMAddPackage("gh:Kaiyao-Ge/polysolve#cf33dd11c33cf279cc69f9ee6788a551ef48bf45")
