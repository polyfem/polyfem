set(SOURCES)

if(POLYFEM_WITH_MMG)
	list(APPEND SOURCES
		MMGRemesh.cpp
		MMGRemesh.hpp
	)
endif()

if(POLYFEM_WITH_ITR)
	list(APPEND SOURCES
		L2Projection.cpp
		L2Projection.hpp
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
	add_subdirectory(wild_remesh)
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
