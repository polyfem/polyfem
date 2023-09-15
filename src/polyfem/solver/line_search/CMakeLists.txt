set(SOURCES
	LineSearch.hpp
	LineSearch.tpp
	ArmijoLineSearch.hpp
	BacktrackingLineSearch.hpp
	CppOptArmijoLineSearch.hpp
	MoreThuenteLineSearch.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
