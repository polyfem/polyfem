set(SOURCES
	AMIPSForm.cpp
	AMIPSForm.hpp
	Smooth.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
