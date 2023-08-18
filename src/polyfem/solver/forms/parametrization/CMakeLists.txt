set(SOURCES
	Parametrization.hpp
	Parametrization.cpp
	Parametrizations.hpp
	Parametrizations.cpp
	SDFParametrizations.hpp
	SDFParametrizations.cpp
	NodeCompositeParametrizations.hpp
	NodeCompositeParametrizations.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
