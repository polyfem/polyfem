set(SOURCES
	ImplicitTimeIntegrator.cpp
	ImplicitTimeIntegrator.hpp
	ImplicitEuler.cpp
	ImplicitEuler.hpp
	ImplicitNewmark.cpp
	ImplicitNewmark.hpp
	BDF.cpp
	BDF.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
