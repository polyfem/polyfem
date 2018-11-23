set(SOURCES
	Common.hpp
	State.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(assembler)
add_subdirectory(autogen)
add_subdirectory(basis)
add_subdirectory(mesh)
add_subdirectory(mesh2D)
add_subdirectory(mesh3D)
add_subdirectory(problem)
add_subdirectory(quadrature)
add_subdirectory(solver)
add_subdirectory(utils)
if(POLYFEM_TOPLEVEL_PROJECT)
	add_subdirectory(viewer)
endif()
