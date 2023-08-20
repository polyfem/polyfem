set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	LBFGSSolver.hpp
	LBFGSSolver.tpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	NonlinearSolver.hpp
	NonlinearSolver.tpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	SolveData.cpp
	SolveData.hpp
	SparseNewtonDescentSolver.hpp
	SparseNewtonDescentSolver.tpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(line_search)
add_subdirectory(forms)
