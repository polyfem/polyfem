set(SOURCES
	PointBasedProblem.cpp
	PointBasedProblem.hpp
	GenericProblem.cpp
	GenericProblem.hpp
	ElasticProblem.cpp
	ElasticProblem.hpp
	FrankeProblem.cpp
	FrankeProblem.hpp
	KernelProblem.cpp
	KernelProblem.hpp
	MiscProblem.cpp
	MiscProblem.hpp
	TestProblem.cpp
	TestProblem.hpp
	Problem.cpp
	Problem.hpp
	ProblemWithSolution.cpp
	ProblemWithSolution.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
