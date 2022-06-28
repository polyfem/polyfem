set(SOURCES
	NonlinearSolver.hpp
	SparseNewtonDescentSolver.hpp
	LBFGSSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	ALNLProblem.cpp
	ALNLProblem.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(line_search)
