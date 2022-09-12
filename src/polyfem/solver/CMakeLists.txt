set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	NonlinearSolver.hpp
	SparseNewtonDescentSolver.hpp
	LBFGSSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
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
add_subdirectory(forms)
