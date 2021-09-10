set(SOURCES
	LineSearch.hpp
	ArmijoLineSearch.hpp
	BisectionLineSearch.hpp
	CppOptArmijoLineSearch.hpp
	MoreThuenteLineSearch.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
