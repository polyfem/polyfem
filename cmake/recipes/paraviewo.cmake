# Polyfem Solvers (https://github.com/polyfem/paraviewo)
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")

# include(CPM)# CPMAddPackage("gh:polyfem/paraviewo#119e825ee3907b34a7d2f0efaa9b0f7790f8c2d6")
CPMAddPackage("gh:xDarkLemon/paraviewo#a7b7d18dfce7aa5b2109364e0b012dab0994bb56")
