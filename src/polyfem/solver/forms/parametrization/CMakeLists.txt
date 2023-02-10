set(SOURCES
	Parametrization.hpp
	Parametrizations.hpp
	Parametrizations.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
