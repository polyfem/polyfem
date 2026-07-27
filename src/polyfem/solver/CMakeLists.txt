set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	NLProblem.cpp
	NLProblem.hpp
	NLHomoProblem.cpp
	NLHomoProblem.hpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	SolveData.cpp
	SolveData.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(forms)
