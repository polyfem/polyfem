set(SOURCES
	Parametrization.hpp
	Parametrizations.hpp
	Parametrizations.cpp
	SplineParametrizations.hpp
	SplineParametrizations.cpp
	SDFParametrizations.hpp
	SDFParametrizations.cpp
	NodeCompositeParametrizations.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
