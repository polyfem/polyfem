set(SOURCES
	NonlinearSolver.hpp
	SparseNewtonDescentSolver.hpp
	LBFGSSolver.hpp
	MMASolver.hpp
	MMASolverAux.hpp
	MMASolverAux.cpp
	GradientDescentSolver.hpp
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
	ShapeProblem.hpp
	ShapeProblem.cpp
	MaterialProblem.hpp
	MaterialProblem.cpp
	TopologyOptimizationProblem.hpp
	TopologyOptimizationProblem.cpp
	OptimizationProblem.hpp
	OptimizationProblem.cpp
	Optimizations.hpp
	Optimizations.cpp
	GeneralOptimizationProblem.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(line_search)
