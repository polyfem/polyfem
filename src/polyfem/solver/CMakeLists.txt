set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	Optimizations.hpp
	Optimizations.cpp
	SolveData.cpp
	SolveData.hpp
	DiffCache.hpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
	AdjointTools.cpp
	AdjointTools.hpp
	AdjointNLProblem.hpp
	AdjointNLProblem.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(forms)
