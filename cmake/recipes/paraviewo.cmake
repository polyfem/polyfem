# Polyfem Solvers (https://github.com/polyfem/paraviewo)
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")

include(CPM)
# CPMAddPackage("gh:polyfem/paraviewo#119e825ee3907b34a7d2f0efaa9b0f7790f8c2d6")
# CPMAddPackage(
#     NAME paraviewo
#     SOURCE_DIR /home/liuyibo/myrepo/prism/paraviewo
# )
CPMAddPackage("gh:xDarkLemon/paraviewo#c8508e45c0ee60314ea2bca4da630a074a826cfe")
