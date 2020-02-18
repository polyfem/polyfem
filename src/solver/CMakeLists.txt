set(SOURCES
	FEMSolver.cpp
	FEMSolver.hpp
	LbfgsSolver.hpp
	LinearSolver.cpp
	LinearSolver.hpp
	LinearSolverAMGCL.cpp
	LinearSolverAMGCL.hpp
	LinearSolverEigen.h
	LinearSolverEigen.hpp
	LinearSolverHypre.cpp
	LinearSolverHypre.hpp
	LinearSolverPardiso.cpp
	NLProblem.cpp
	NLProblem.hpp
	SparseNewtonDescentSolver.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
