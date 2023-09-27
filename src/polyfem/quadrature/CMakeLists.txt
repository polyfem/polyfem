set(SOURCES
	HexQuadrature.cpp
	HexQuadrature.hpp
	LineQuadrature.cpp
	LineQuadrature.hpp
	PolygonQuadrature.cpp
	PolygonQuadrature.hpp
	PolyhedronQuadrature.cpp
	PolyhedronQuadrature.hpp
	QuadQuadrature.cpp
	QuadQuadrature.hpp
	Quadrature.hpp
	TetQuadrature.cpp
	TetQuadrature.hpp
	TriQuadrature.cpp
	TriQuadrature.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
