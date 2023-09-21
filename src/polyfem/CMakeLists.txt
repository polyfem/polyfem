set(SOURCES
	Common.hpp
	State.cpp
	State.hpp
	Units.cpp
	Units.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
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
