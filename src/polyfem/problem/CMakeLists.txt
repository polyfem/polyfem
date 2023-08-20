set(SOURCES
	PointBasedProblem.cpp
	PointBasedProblem.hpp
	ElasticProblem.cpp
	ElasticProblem.hpp
	FrankeProblem.cpp
	FrankeProblem.hpp
	KernelProblem.cpp
	KernelProblem.hpp
	MiscProblem.cpp
	MiscProblem.hpp
	NodeProblem.cpp
	NodeProblem.hpp
	TestProblem.cpp
	TestProblem.hpp
	ProblemFactory.cpp
	ProblemFactory.hpp
	ProblemWithSolution.cpp
	ProblemWithSolution.hpp
	StokesProblem.cpp
	StokesProblem.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
