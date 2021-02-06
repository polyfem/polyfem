set(SOURCES
	LbfgsSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	ALNLProblem.cpp
	ALNLProblem.hpp
	SparseNewtonDescentSolver.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
