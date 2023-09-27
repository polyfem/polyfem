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

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
