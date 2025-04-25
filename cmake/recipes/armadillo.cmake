# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET armadillo::armadillo)
    return()
endif()

message(STATUS "Third-party: creating target 'armadillo::armadillo'")

include(CPM)
CPMAddPackage("gl:conradsnicta/armadillo-code#3a23ca275b9fe30a7ef3132fc9c91c4b2ab29f9d")
add_library(armadillo::armadillo ALIAS armadillo)