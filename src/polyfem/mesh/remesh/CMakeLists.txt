set(SOURCES
	L2Projection.cpp
	L2Projection.hpp
	MMGRemesh.cpp
	MMGRemesh.hpp
	Remesher.cpp
	Remesher.hpp
	SizingFieldRemesher.cpp
	SizingFieldRemesher.hpp
	WildRemesher.cpp
	WildRemesher.hpp
	WildTetRemesher.cpp
	WildTriRemesher.cpp
	PhysicsRemesher.cpp
	PhysicsRemesher.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(wild_remesh)