################################################################################

if(TARGET RBF)
	return()
endif()

find_package(OpenCL)
if(NOT ${OPENCL_FOUND})
	message(WARNING "OpenCL not found; RBF interpolation will not be compiled")
	return()
endif()

################################################################################

# add_library(RBF
# 	${THIRD_PARTY_DIR}/rbf/rbf_interpolate.cpp
# 	${THIRD_PARTY_DIR}/rbf/rbf_interpolate.hpp
# )

# target_include_directories(polyfem PUBLIC ${OpenCL_INCLUDE_DIR})
# target_link_libraries(polyfem PUBLIC ${OpenCL_LIBRARY})
