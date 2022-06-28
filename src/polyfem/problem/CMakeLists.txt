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

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
