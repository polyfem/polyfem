# hocgv-miso: miso-generated Jacobian validity solvers
# License: see hocgv-miso repo

if(TARGET miso)
    return()
endif()

message(STATUS "Third-party: creating target 'miso' via hocgv-miso")

include(CPM)
CPMAddPackage("gh:fsichetti/hocgv-miso#34439c33183a6ecce3dba202c3777572b894bf6f")
