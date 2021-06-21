set(SOURCES
	ImplicitTimeIntegrator.cpp
	ImplicitTimeIntegrator.hpp
	ImplicitEuler.cpp
	ImplicitEuler.hpp
	ImplicitNewmark.cpp
	ImplicitNewmark.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
