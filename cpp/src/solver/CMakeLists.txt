set(SOURCES
	FEMSolver.cpp
	LinearSolver.cpp
	LinearSolverHypre.cpp
	LinearSolverPardiso.cpp
	NLProblem.cpp
	NLProblem.hpp
	SparseNewtonDescentSolver.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
