set(SOURCES
	UIState.cpp
	UIState.hpp
	UIMenu.cpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(${PROJECT_NAME}_bin PRIVATE ${SOURCES})
