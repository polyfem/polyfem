set(SOURCES
	Parameterization.hpp
	Parameterizations.hpp
	Parameterizations.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
