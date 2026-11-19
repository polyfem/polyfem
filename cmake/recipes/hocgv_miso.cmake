# hocgv-miso: miso-generated Jacobian validity solvers
# License: see hocgv-miso repo

if(TARGET miso)
    return()
endif()

message(STATUS "Third-party: creating target 'miso' via hocgv-miso")

include(CPM)
CPMAddPackage("gh:fsichetti/hocgv-miso#ddd0b3469efc2c0f26abd38ec4cb072801620d13")
