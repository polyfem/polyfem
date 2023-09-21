set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	LBFGSSolver.hpp
	LBFGSSolver.tpp
	LBFGSBSolver.hpp
	BFGSSolver.hpp
	GradientDescentSolver.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	NonlinearSolver.hpp
	NonlinearSolver.tpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	Optimizations.hpp
	Optimizations.cpp
	SolveData.cpp
	SolveData.hpp
	DiffCache.hpp
	SolverWithBoxConstraints.hpp
	SparseNewtonDescentSolver.hpp
	SparseNewtonDescentSolver.tpp
	DenseNewtonDescentSolver.hpp
	DenseNewtonDescentSolver.tpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
	AdjointTools.cpp
	AdjointTools.hpp
	AdjointNLProblem.hpp
	AdjointNLProblem.cpp
	MMASolver.hpp
	MMASolver.tpp
	MMASolverAux.hpp
	MMASolverAux.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(line_search)
add_subdirectory(forms)
