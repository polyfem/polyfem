set(SOURCES
	AMIPSForm.cpp
	AMIPSForm.hpp
	Collapse.cpp
	Smooth.cpp
	Split.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
