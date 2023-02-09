set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	LBFGSSolver.hpp
	LBFGSSolver.tpp
	LBFGSBSolver.hpp
	BFGSSolver.hpp
	MMASolver.hpp
	MMASolverAux.hpp
	MMASolverAux.cpp
	GradientDescentSolver.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	NLHomoProblem.cpp
	NLHomoProblem.hpp
	NonlinearSolver.hpp
	NonlinearSolver.tpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	Optimizations.hpp
	Optimizations.cpp
	SolveData.cpp
	SolveData.hpp
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
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(line_search)
add_subdirectory(forms)
add_subdirectory(constraints)
