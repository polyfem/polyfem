################################################################################

if(TARGET HYPRE)
	return()
endif()

################################################################################

set(HYPRE_SEQUENTIAL    ON CACHE INTERNAL "" FORCE)
set(HYPRE_PRINT_ERRORS  ON CACHE INTERNAL "" FORCE)
set(HYPRE_BIGINT        ON CACHE INTERNAL "" FORCE)
set(HYPRE_USING_FEI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_USING_OPENMP OFF CACHE INTERNAL "" FORCE)
set(HYPRE_SHARED       OFF CACHE INTERNAL "" FORCE)
# set(HYPRE_LONG_DOUBLE ON)

set(HYPRE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE INTERNAL "" FORCE)

add_subdirectory(${THIRD_PARTY_DIR}/hypre/src hypre)
set_property(TARGET HYPRE PROPERTY FOLDER "dependencies")

target_include_directories(HYPRE PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/hypre)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/blas)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/lapack)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/utilities)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/multivector)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/krylov)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/seq_mv)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/parcsr_mv)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/parcsr_block_mv)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/distributed_matrix)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/IJ_mv)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/matrix_matrix)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/distributed_ls)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/distributed_ls/Euclid)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/distributed_ls/ParaSails)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/parcsr_ls)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/struct_mv)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/struct_ls)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/sstruct_mv)
target_include_directories(HYPRE PUBLIC ${THIRD_PARTY_DIR}/hypre/src/sstruct_ls)

if(HYPRE_USING_OPENMP)
	find_package(OpenMP QUIET REQUIRED)
	target_link_libraries(polyfem PUBLIC OpenMP::OpenMP_CXX)
endif()

if(NOT HYPRE_SEQUENTIAL)
	find_package(MPI)
	if(MPI_CXX_FOUND)
		target_link_libraries(polyfem PUBLIC MPI::MPI_CXX)
	endif()
endif()
