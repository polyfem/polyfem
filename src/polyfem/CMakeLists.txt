set(SOURCES
	Common.hpp
	State.cpp
	State.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(assembler)
add_subdirectory(autogen)
add_subdirectory(basis)
add_subdirectory(io)
add_subdirectory(mesh)
add_subdirectory(problem)
add_subdirectory(quadrature)
add_subdirectory(refinement)
add_subdirectory(solver)
add_subdirectory(state)
add_subdirectory(time_integrator)
add_subdirectory(utils)
