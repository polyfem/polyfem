################################################################################

if(TARGET RBF)
	return()
endif()

find_package(OpenCL)
if(NOT ${OPENCL_FOUND})
	MESSAGE(WARNING "not opencl")
	return()
endif()

################################################################################

add_library(RBF
	${THIRD_PARTY_DIR}/rbf/rbf_interpolate.cpp
	${THIRD_PARTY_DIR}/rbf/rbf_interpolate.hpp
)

target_include_directories(HYPRE PUBLIC ${OPENCL_INCLUDE_DIRS})
target_link_libraries(polyfem PUBLIC ${OPENCL_LIBRARIES})

