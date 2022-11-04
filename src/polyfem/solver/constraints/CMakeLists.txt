set(SOURCES
    Constraints.hpp
    ShapeConstraints.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
