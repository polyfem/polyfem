set(SOURCES
	Form.hpp
	BodyForm.hpp
	BodyForm.cpp
	ElasticForm.hpp
	ElasticForm.cpp
	InertiaForm.hpp
	InertiaForm.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

