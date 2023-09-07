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

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
