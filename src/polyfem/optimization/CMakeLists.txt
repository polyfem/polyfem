set(SOURCES
	OptState.cpp
	OptState.hpp
	StateDiff.cpp
	Optimizations.hpp
	Optimizations.cpp
	AdjointNLProblem.hpp
	AdjointNLProblem.cpp
	AdjointTools.hpp
	AdjointTools.cpp
	DiffCache.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(forms)
add_subdirectory(force_derivatives)
add_subdirectory(parametrization)
