set(SOURCES
	LineSearch.hpp
	LineSearch.tpp
	ArmijoLineSearch.hpp
	BacktrackingLineSearch.hpp
	CppOptArmijoLineSearch.hpp
	MoreThuenteLineSearch.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
